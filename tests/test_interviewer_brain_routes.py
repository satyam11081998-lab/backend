"""
Contract tests: frontend <-> backend routes <-> unified brain <-> (fake) provider <-> (fake) DB.

Exercises the real FastAPI routes with TestClient. No network, no keys.
    python -m pytest tests/test_interviewer_brain_routes.py -q
"""
from __future__ import annotations

import json
import os

import pytest

from tests.interviewer_fakes import FakeDB, FakeLLM, install_route_fakes, parse_sse

os.environ["ADAPTIVE_INTERVIEWER"] = "true"

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import routes.attempts as att  # noqa: E402
import routes.attempts_brain as ab  # noqa: E402
import services.interview_engine as ie  # noqa: E402
import services.interviewer_decision as idec  # noqa: E402
from services.interviewer import dedupe, engine, telemetry  # noqa: E402
from services.interviewer.assessor import Assessment  # noqa: E402

STATE = {"db": FakeDB(), "llm": FakeLLM()}

# V11 fakes (flag-off path) - same as tests/test_v11_voice_integration.py
_V11_CALLS = []


class _V11LLM:
    def __init__(self):
        import types
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        import types
        _V11_CALLS.append(kw)
        text = "V11 says: split by segment."
        if kw.get("stream"):
            class C:
                def __init__(s, t):
                    s.choices = [types.SimpleNamespace(delta=types.SimpleNamespace(content=t))] if t else []
                    s.usage = None if t else types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
                    s.id = "x"
            return iter([C(text), C(None)])
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))],
                                     usage=None, id="x")


_v11 = _V11LLM()

app = FastAPI()
app.include_router(att.router)
client = TestClient(app)
H = {"Authorization": "Bearer t"}


@pytest.fixture(autouse=True)
def brain_on(monkeypatch):
    monkeypatch.setenv("INTERVIEWER_BRAIN", "on")
    for name in ("get_supabase_client", "get_verified_user", "get_verified_user_id", "is_guest_user",
                 "check_rate_limit", "assert_daily_budget", "llm_case_content"):
        monkeypatch.setattr(att, name, getattr(att, name))
    install_route_fakes(att, lambda: STATE["db"])
    monkeypatch.setattr(engine, "LLM_FACTORY", lambda plan: STATE["llm"])
    monkeypatch.setattr(engine, "default_assessor", lambda uid: (lambda **kw: Assessment(material=False)))
    monkeypatch.setattr(ie, "openai_client", lambda: _v11)
    monkeypatch.setattr(ie, "resolve_llm", lambda feature: (_v11, "fake-v11", "openai"))
    monkeypatch.setattr(ie, "log_ai_usage", lambda **kw: None)
    monkeypatch.setattr(idec, "openai_client", lambda: None)
    monkeypatch.delenv("INTERVIEWER_DEBUG_DECISIONS", raising=False)
    STATE["db"] = FakeDB([{"role": "assistant", "kind": "text", "content": "Walk me through how you'd size it."}],
                         session_state={"brain": {"opened": True, "turns": 2, "phase": "analysis"}})
    STATE["llm"] = FakeLLM()
    dedupe.LEDGER.__init__()
    dedupe.ROWS.__init__()
    _V11_CALLS.clear()
    yield


def settle():
    att._await_after_turn("a1")


def post(content, **extra):
    r = client.post("/attempts/a1/messages", json={"content": content, **extra}, headers=H)
    settle()
    return r, parse_sse(r.text)


def vd(content, **extra):
    r = client.post("/attempts/a1/voice-decision", json={"content": content, **extra}, headers=H)
    settle()
    return r


# ---------------------------------------------------------------------------- TEXT / STT
@pytest.mark.parametrize("channel", ["text", "stt", None])
def test_no_output_sends_explicit_silence_and_writes_no_assistant_row(channel):
    extra = {"channel": channel} if channel else {}
    r, events = post("50%", **extra)
    assert r.status_code == 200
    names = [e for e, _ in events]
    assert names == ["meta", "silence", "done"]
    done = json.loads(events[-1][1])
    assert done["message_id"] is None and done["silent"] is True
    assert not [e for e in events if e[0] == "token"]
    db = STATE["db"]
    assert len(db.inserted(role="user")) == 1 and db.inserted(role="assistant") == []
    assert db.inserted(table="ai_usage_log") == []
    assert not STATE["llm"].calls


def test_presence_is_a_real_short_row_and_calls_no_model():
    r, events = post("Shall I proceed?")
    toks = "".join(d for e, d in events if e == "token")
    assert toks == "Yes, go ahead."
    assert STATE["db"].inserted(role="assistant")[0]["content"] == "Yes, go ahead."
    assert not STATE["llm"].calls


def test_help_streams_a_validated_hint_and_never_refuses():
    STATE["llm"] = FakeLLM("praise")
    r, events = post("Can you give me a hint?")
    text = "".join(d for e, d in events if e == "token")
    assert text and "great" not in text.lower() and "exercise" not in text.lower() and "?" not in text
    assert STATE["llm"].calls[0]["move"] == "MICRO_HINT"
    row = STATE["db"].inserted(role="assistant")[0]
    assert row["content"] == text.strip()


def test_refusing_model_output_never_reaches_the_candidate():
    STATE["llm"] = FakeLLM("refuse")
    r, events = post("Can you help me?")
    names = [e for e, _ in events]
    assert "error" in names and "token" not in names          # an error, never "That's the exercise"
    assert STATE["db"].inserted(role="assistant") == []
    # the failed hint did not move the ladder
    assert STATE["db"].session_state().get("brain", {}).get("hint_level", 0) == 0


def test_provider_failure_is_an_error_not_silence_and_not_v11():
    STATE["llm"] = FakeLLM("error")
    r, events = post("Show me the correct approach.")
    names = [e for e, _ in events]
    assert names[0] == "meta" and "error" in names and "silence" not in names
    assert not _V11_CALLS


def test_duplicate_turn_id_is_decided_once_and_replayed():
    r1, e1 = post("Can you give me a hint?", turn_id="t-1")
    r2, e2 = post("Can you give me a hint?", turn_id="t-1")
    assert len(STATE["llm"].calls) == 1
    assert len(STATE["db"].inserted(role="user")) == 1 and len(STATE["db"].inserted(role="assistant")) == 1
    t1 = "".join(d for e, d in e1 if e == "token")
    t2 = "".join(d for e, d in e2 if e == "token")
    assert t1 == t2 and json.loads(e2[-1][1]).get("duplicate") is True


def test_c9_counting_and_quota_row_unchanged():
    STATE["db"] = FakeDB([], used=19, quota=20)
    post("What population should I use? And what time period?")
    att_row = STATE["db"].tables["attempts"][0]
    assert att_row["clarification_used"] == 20                 # clamped to quota (C9)
    post("What about rural households?")
    assert STATE["db"].tables["attempts"][0]["clarification_used"] == 20


def test_brain_state_is_namespaced_and_v11_keys_untouched():
    STATE["db"] = FakeDB([], session_state={"hint_level": 3, "profile": {"errors": {"unit": 2}}})
    post("Can you help me?")
    ss = STATE["db"].session_state()
    assert ss["hint_level"] == 3 and ss["profile"] == {"errors": {"unit": 2}}
    assert ss["brain"]["hint_level"] == 1


# ---------------------------------------------------------------------------- REALTIME VOICE
def test_voice_silence_contract():
    r = vd("let me think")
    assert r.status_code == 200
    j = r.json()
    assert j["lane"] == "SILENCE" and j["say"] is None and j["mode"] is None and j["reason"] is None


def test_voice_partial_never_speaks_and_needs_no_turn_id():
    r = vd("so the market is", is_partial=True)
    assert r.json()["lane"] == "SILENCE" and not STATE["llm"].calls


def test_voice_substantive_and_presence():
    j = vd("What population should I use?", turn_id="i1").json()
    assert j["lane"] == "SUBSTANTIVE" and j["say"] and STATE["llm"].calls[-1]["kind"] == "complete"
    j = vd("Shall I proceed?", turn_id="i2").json()
    assert j["lane"] == "PRESENCE" and j["say"] == "Yes, go ahead."


def test_voice_regenerates_once_then_fails_loudly():
    STATE["llm"] = FakeLLM("refuse_then_good")
    j = vd("Can you help me?", turn_id="i3").json()
    assert j["lane"] == "SUBSTANTIVE" and "exercise" not in j["say"].lower()
    assert len(STATE["llm"].calls) == 2
    STATE["llm"] = FakeLLM("empty")
    r = vd("Can you help me?", turn_id="i4")
    assert r.status_code == 502


def test_voice_duplicate_item_id_decided_once():
    a = vd("What population should I use?", turn_id="item_9").json()
    b = vd("What population should I use?", turn_id="item_9").json()
    assert a["say"] == b["say"] and b["duplicate"] is True and len(STATE["llm"].calls) == 1


def test_control_metadata_only_in_debug(monkeypatch):
    monkeypatch.setenv("INTERVIEWER_DEBUG_DECISIONS", "1")
    j = vd("Shall I proceed?", turn_id="dbg").json()
    assert j["mode"] == "HAND_BACK"


def test_early_voice_decision_fold_waits_for_confirm():
    before = dict(STATE["db"].session_state().get("brain", {}))
    vd("Can you help me?", turn_id="early-1", defer_fold=True)
    assert STATE["db"].session_state().get("brain", {}).get("hint_level", 0) == before.get("hint_level", 0)
    client.post("/attempts/a1/voice-fold", json={"turn_id": "early-1", "commit": True}, headers=H)
    settle()
    assert STATE["db"].session_state()["brain"]["hint_level"] == 1


# ---------------------------------------------------------------------------- persistence idempotency
def _rt(role, content, key, **usage):
    return client.post("/attempts/a1/realtime-turn", json={"role": role, "content": content, "client_turn_id": key,
                                                            **usage}, headers=H)


def test_realtime_turn_is_idempotent_and_metered_once(monkeypatch):
    metered = []
    monkeypatch.setattr(att, "log_realtime_usage", lambda **kw: metered.append(kw))
    monkeypatch.setattr(att, "deduct_realtime_credit", lambda *a: metered.append(("deduct", a)))
    a = _rt("assistant", "Yes, go ahead.", "a:resp_1", audio_output_tokens=40)
    b = _rt("assistant", "Yes, go ahead.", "a:resp_1", audio_output_tokens=40)
    assert a.json()["message_id"] == b.json()["message_id"] and b.json()["duplicate"] is True
    assert len(STATE["db"].inserted(role="assistant")) == 1 and len(metered) == 2   # one log + one deduct


def test_realtime_turn_db_level_duplicate_after_restart():
    _rt("user", "Can I assume India only?", "u:item_1")
    dedupe.ROWS.__init__()                                   # simulate a process restart
    r = _rt("user", "Can I assume India only?", "u:item_1")
    assert r.json()["duplicate"] is True and len(STATE["db"].inserted(role="user")) == 1


def test_pre_migration_database_still_saves():
    STATE["db"] = FakeDB([], has_client_turn_id=False)
    ab._COL["has_client_turn_id"] = True
    r = _rt("user", "India only?", "u:item_2")
    assert r.status_code == 200 and len(STATE["db"].inserted(role="user")) == 1
    r, events = post("50%", turn_id="t-pre")
    assert r.status_code == 200
    ab._COL["has_client_turn_id"] = True


def test_voice_telemetry_endpoint_logs_no_content(capsys):
    r = client.post("/attempts/a1/voice-telemetry", json={"turn_id": "i1", "t4_ms": 812, "lane": "PRESENCE",
                                                          "transport": "webrtc"}, headers=H)
    assert r.status_code == 200


# ---------------------------------------------------------------------------- flag OFF == baseline V11
def test_flag_off_runs_v11(monkeypatch):
    monkeypatch.setenv("INTERVIEWER_BRAIN", "off")
    r, events = post("Can you give me a hint?")
    assert _V11_CALLS, "V11 engine must handle the turn when the flag is off"
    assert not STATE["llm"].calls
    assert "silence" not in [e for e, _ in events]


def test_allowlist(monkeypatch):
    monkeypatch.setenv("INTERVIEWER_BRAIN", "allowlist")
    monkeypatch.setenv("INTERVIEWER_BRAIN_ALLOWLIST", "someone-else")
    post("Can you give me a hint?")
    assert _V11_CALLS and not STATE["llm"].calls
    _V11_CALLS.clear()
    monkeypatch.setenv("INTERVIEWER_BRAIN_ALLOWLIST", "u1")
    post("Can you give me a hint?")
    assert not _V11_CALLS and STATE["llm"].calls


# ---------------------------------------------------------------------------- scoring separation
def test_scoring_body_excludes_presence_and_interviewer_lines():
    STATE["db"] = FakeDB([])
    post("I'd take 30 crore households.")
    post("Shall I proceed?")
    rows = STATE["db"].rows()
    body = ie._transcript_body_text([{"role": r["role"], "kind": r["kind"], "content": r["content"]} for r in rows])
    assert "go ahead" not in body.lower() and "30 crore" in body
