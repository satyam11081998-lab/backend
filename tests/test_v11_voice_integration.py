"""
One interviewer everywhere: integration tests for the wiring around the frozen
interviewer engine (services/{session_signals,interviewer_decision,
interviewer_mode,interview_engine}.py). Engine-version agnostic: route
behaviour is tested with the engine's own control output, not its wording.

Proves, with fake providers and a fake database (no network, no keys):
  * /attempts/{id}/messages   -- V11 SILENCE saves no assistant row, emits no
                                 token, and ends with message_id null; any other
                                 empty reply still takes the normal path.
  * /attempts/{id}/voice-decision -- realtime voice turns reach the frozen V11
                                 engine; SILENCE / PRESENCE / SUBSTANTIVE come
                                 back as {lane, mode, reason, say, event}; a
                                 partial transcript never produces speech.
  * /realtime/session, /realtime-gemini/session -- the speech models get the
                                 voice-renderer instructions only (no interviewer
                                 prompt), and OpenAI auto-responses are off.
  * the removed static interviewer prompt cannot come back.

Needs the app's venv (fastapi, httpx, openai, supabase):
    python -m tests.test_v11_voice_integration
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _k, _v in {
    "OPENAI_API_KEY": "sk-test", "SUPABASE_URL": "https://example.supabase.co",
    "SUPABASE_SERVICE_ROLE_KEY": "test", "GEMINI_API_KEY": "test",
}.items():
    os.environ.setdefault(_k, _v)
os.environ["ADAPTIVE_INTERVIEWER"] = "true"

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import routes.attempts as att  # noqa: E402
import routes.realtime as rt  # noqa: E402
import routes.realtime_gemini as rtg  # noqa: E402
import services.interview_engine as ie  # noqa: E402
import services.interviewer_decision as idec  # noqa: E402
import prompts.interview_prompts as old_prompts  # noqa: E402
from prompts.voice_renderer import VOICE_RENDERER_INSTRUCTIONS  # noqa: E402

_fail: list = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond or not detail else f"   [{detail}]"))
    if not cond:
        _fail.append(name)


OLD_PHRASES = ("That's the exercise", "NO HINTS, NO SOLUTIONS", "can't provide hints", "DECLINE and turn it")


# ---------------------------------------------------------------- fakes
class _Delta:
    def __init__(self, c):
        self.content = c


class _Choice:
    def __init__(self, delta=None, content=None):
        self.delta = _Delta(delta)
        self.message = types.SimpleNamespace(content=content)


class _Chunk:
    def __init__(self, tok=None, usage=None):
        self.choices = [_Choice(delta=tok)] if tok is not None else []
        self.usage = usage
        self.id = "fake"


class FakeLLM:
    """OpenAI-compatible fake. `reply` is str or callable(messages)->str."""

    def __init__(self, reply="Focus on the cost side first."):
        self.reply, self.calls, self.raise_exc = reply, [], None
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if self.raise_exc:
            raise self.raise_exc
        text = self.reply(kw["messages"]) if callable(self.reply) else self.reply
        if kw.get("stream"):
            return iter([_Chunk(text[i:i + 4]) for i in range(0, len(text), 4)]
                        + [_Chunk(None, usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))])
        return types.SimpleNamespace(choices=[_Choice(content=text)], usage=None, id="fake")


class _Res:
    def __init__(self, data=None, count=None):
        self.data, self.count = data, count


class _Q:
    def __init__(self, db, table):
        self.db, self.table, self.op, self.payload, self.filters, self.single = db, table, "select", None, [], False

    def select(self, *a, **k): return self
    def order(self, *a, **k): return self
    def limit(self, *a, **k): return self

    def insert(self, p):
        self.op, self.payload = "insert", p
        return self

    def update(self, p):
        self.op, self.payload = "update", p
        return self

    def eq(self, k, v):
        self.filters.append((k, v))
        return self

    def maybe_single(self):
        self.single = True
        return self

    def execute(self):
        rows = self.db.tables.setdefault(self.table, [])
        if self.op == "insert":
            row = dict(self.payload, id=f"{self.table}-{len(rows) + 1}", created_at=f"t{len(rows) + 1:04d}")
            rows.append(row)
            self.db.writes.append(("insert", self.table, row))
            return _Res([row])
        hit = [r for r in rows if all(r.get(k) == v for k, v in self.filters)]
        if self.op == "update":
            for r in hit:
                r.update(self.payload)
            self.db.writes.append(("update", self.table, dict(self.payload)))
            return _Res(hit)
        return _Res(hit[0] if hit else None) if self.single else _Res(hit, count=len(hit))


class FakeDB:
    def __init__(self, history, case_type="guesstimate", status="active", used=0):
        self.writes = []
        self.tables = {
            "attempts": [{"id": "a1", "user_id": "u1", "status": status, "case_id": "c1",
                          "clarification_quota": 20, "clarification_used": used,
                          "tier_at_start": "pro", "session_state": {}}],
            "cases": [{"id": "c1", "type": case_type, "title": "Smartphones", "difficulty": "easy",
                       "content": "Estimate smartphones sold in Bangalore per year.", "is_active": True}],
            "attempt_messages": [dict(m, attempt_id="a1", id=f"m{i}", created_at=f"t{i:04d}")
                                 for i, m in enumerate(history)],
        }

    def table(self, name):
        return _Q(self, name)

    def assistant_rows(self):
        return [w[2] for w in self.writes if w[0] == "insert" and w[1] == "attempt_messages" and w[2].get("role") == "assistant"]

    def state_writes(self):
        return [w[2]["session_state"] for w in self.writes if w[0] == "update" and w[1] == "attempts" and "session_state" in w[2]]


LLM = FakeLLM()
ie.openai_client = lambda: LLM
ie.resolve_llm = lambda feature: (LLM, "fake-model", "openai")
ie.log_ai_usage = lambda **kw: None
idec.openai_client = lambda: None  # assessor off: deterministic V11 routing only

DB = FakeDB([])
att.get_supabase_client = lambda: DB
att.get_verified_user = lambda sb, auth: ("u1", {"id": "u1"}) if auth else (_ for _ in ()).throw(
    att.HTTPException(status_code=401, detail="Missing authentication token"))
att.get_verified_user_id = lambda sb, auth: "u1"
att.is_guest_user = lambda u: False
att.check_rate_limit = lambda *a, **k: None
att.assert_daily_budget = lambda *a, **k: None
att.llm_case_content = lambda case: case["content"]

app = FastAPI()
app.include_router(att.router)
client = TestClient(app)
H = {"Authorization": "Bearer t"}

LIVE = [
    {"role": "user", "kind": "voice", "content": "So the number of smartphone sellers in Bangalore can you tell me the population of Bangalore"},
    {"role": "assistant", "kind": "voice", "content": "Take the population of Bangalore as 1.3 crore."},
    {"role": "user", "kind": "voice", "content": "I think there will be approximately 1.3 crore smartphones also."},
    {"role": "assistant", "kind": "voice", "content": "How did you get there?"},
]


def settled():
    """The session_state fold runs just after the response (the next turn waits
    for it); let it land before inspecting the fake DB."""
    if hasattr(att, "_await_after_turn"):
        att._await_after_turn("a1")


def decide(text, history=LIVE, reply="Here is a nudge: split buyers by age first. What next?", partial=False, **dbkw):
    global DB
    DB = FakeDB(history, **dbkw)
    LLM.reply, LLM.calls, LLM.raise_exc = reply, [], None
    r = client.post("/attempts/a1/voice-decision", json={"content": text, "is_partial": partial}, headers=H)
    settled()
    return r, DB, list(LLM.calls)


def system_text(calls):
    return "\n".join(m["content"] for c in calls for m in c["messages"] if m["role"] == "system")


print("=" * 72)
print("1. /voice-decision: realtime voice turns are decided by V11")
print("=" * 72)
r, db, calls = decide("Can you help me here?")
d = r.json()
check("'Can you help me here?' -> 200", r.status_code == 200, r.text[:200])
check("'Can you help me here?' -> SUBSTANTIVE HINT", (d.get("lane"), d.get("mode")) == ("SUBSTANTIVE", "HINT"), d)
check("HINT line comes from V11 deep lane (one LLM call)", len(calls) == 1)
check("HINT line has no question (V11 HINT forbids one)", d.get("say") and "?" not in d["say"], d.get("say"))
check("deep-lane prompt is V11's ('YOUR MOVE THIS TURN: HINT')", "YOUR MOVE THIS TURN: HINT" in system_text(calls))
check("deep-lane prompt carries none of the old static interviewer", not any(p in system_text(calls) for p in OLD_PHRASES))
check("no attempt_messages row written by voice-decision", not [w for w in db.writes if w[1] == "attempt_messages"])
# V12: an untagged reply still folds its response function (the next turn's
# presence cool-down / repetition memory), but nothing the tag would drive moves.
check("untagged deep-lane reply: only the response function is folded (hint ladder unchanged)",
      len(db.state_writes()) == 1 and db.state_writes()[0].get("last_function") == "MICRO_HINT"
      and db.state_writes()[0].get("hint_level") == 0, db.state_writes())
r, db, calls = decide("Can you help me here?", reply="<<mode=coach; intervention=micro_hint; hint=2>>\n\nSplit buyers by age first.")
d = r.json()
check("control tag never reaches the voice line", d.get("say") == "Split buyers by age first.", d.get("say"))
check("tagged reply folds session_state like /messages (hint_level 2)",
      len(db.state_writes()) == 1 and db.state_writes()[0].get("hint_level") == 2, db.state_writes())

hist2 = LIVE + [{"role": "user", "kind": "voice", "content": "Can you help me here?"},
                {"role": "assistant", "kind": "voice", "content": "Here is a nudge: split buyers by age first."}]
r, db, calls = decide("No, I want a hint.", history=hist2)
d = r.json()
check("'No, I want a hint.' -> SUBSTANTIVE HINT", (d.get("lane"), d.get("mode")) == ("SUBSTANTIVE", "HINT"), d)

for text, mode in [("Show me the correct approach.", "DELIVER_SOLUTION"),
                   ("Tell me the answer", "DELIVER_SOLUTION"),
                   ("This is irritating, you keep asking the same question", "REPAIR"),
                   ("So 1 litre = 100 ml, then I multiply by households", "CORRECT_MATERIAL")]:
    r, db, calls = decide(text, reply="Line for this mode.")
    d = r.json()
    check(f"{text[:34]!r} -> {mode} with V11 line", d.get("mode") == mode and d.get("lane") == "SUBSTANTIVE" and d.get("say"), d)

QUIET = [{"role": "assistant", "kind": "voice", "content": "Let's begin."},
         {"role": "user", "kind": "voice", "content": "okay sure"}]
r, db, calls = decide("the market is big", history=QUIET)
d = r.json()
check("SILENCE: lane SILENCE, say null, event null", (d.get("lane"), d.get("say"), d.get("event")) == ("SILENCE", None, None), d)
check("SILENCE: zero LLM calls", calls == [])
check("SILENCE: no attempt_messages row", not [w for w in db.writes if w[1] == "attempt_messages"])

r, db, calls = decide("I'll split households into urban and rural first.")
d = r.json()
check("PRESENCE: lane PRESENCE with typed V11 event", d.get("lane") == "PRESENCE" and (d.get("event") or {}).get("event_type") == "interviewer_presence", d)
check("PRESENCE: say == the event's own text", d.get("say") == ((d.get("event") or {}).get("data") or {}).get("text"))
check("PRESENCE: zero LLM calls", calls == [])

r, db, calls = decide("Can you help me here?", partial=True)
d = r.json()
check("partial transcript -> SILENCE, nothing to say", (d.get("lane"), d.get("say")) == ("SILENCE", None), d)
check("partial transcript -> zero LLM calls, no state write", calls == [] and db.state_writes() == [])

r, db, calls = decide("I'm stuck, give me a hint", reply="")
check("empty model text -> V11's own HINT fallback line (enforce_mode)",
      r.status_code == 200 and r.json().get("say") == idec._MODE_FALLBACK["HINT"], r.text[:160])
_real_stream = att.stream_interviewer_reply


def _empty_substantive(**kw):
    kw["control_out"].update({"mode": "HINT", "tag": {}, "reason": "help_requested"})
    return iter(())


att.stream_interviewer_reply = _empty_substantive
DB = FakeDB(LIVE)
r = client.post("/attempts/a1/voice-decision", json={"content": "I'm stuck"}, headers=H)
att.stream_interviewer_reply = _real_stream
check("empty non-silence output -> 502, never treated as silence", r.status_code == 502, r.text[:120])
DB = FakeDB(LIVE)
LLM.raise_exc = RuntimeError("provider down")
r = client.post("/attempts/a1/voice-decision", json={"content": "I'm stuck, give me a hint"}, headers=H)
LLM.raise_exc = None
check("provider failure -> 502 with engine message", r.status_code == 502 and "failed" in r.text, r.text[:120])
r, db, calls = decide("Can you help me here?", status="submitted")
check("submitted attempt -> 400", r.status_code == 400)
r = client.post("/attempts/a1/voice-decision", json={"content": "hi"})
check("no auth -> 401", r.status_code == 401)
r = client.post("/attempts/a1/voice-decision", json={"content": ""}, headers=H)
check("empty content -> 422 (validation)", r.status_code == 422)

print()
print("=" * 72)
print("2. /messages: SILENCE is a control state, never an empty assistant row")
print("=" * 72)


def post_msg(text, history=LIVE, reply="Line."):
    global DB
    DB = FakeDB(history)
    LLM.reply, LLM.calls = reply, []
    r = client.post("/attempts/a1/messages", json={"content": text, "kind": "text"}, headers=H)
    settled()
    ev = []
    for raw in r.text.split("\n\n"):
        if raw.strip():
            name = next((l[7:] for l in raw.split("\n") if l.startswith("event: ")), "")
            data = next((l[6:] for l in raw.split("\n") if l.startswith("data: ")), "")
            ev.append((name, data))
    return ev, DB


# Route handling of the engine's SILENCE lane, driven by the engine's own control
# output (the engine may never choose silence on text; the route must still be right).
_real_stream_s = att.stream_interviewer_reply


def _engine_silence(**kw):
    kw["control_out"].update({"tag": {"mode": "NO_OUTPUT", "intervention": "silence"},
                              "mode": "NO_OUTPUT", "reason": "no_presence_needed"})
    yield ""   # some engine versions emit an empty chunk for silence


att.stream_interviewer_reply = _engine_silence
ev, db = post_msg("anything at all")
att.stream_interviewer_reply = _real_stream_s
check("SILENCE: SSE is meta -> (empty token) -> done", [e for e, _ in ev if e != "token"] == ["meta", "done"] and all(d == "" for e, d in ev if e == "token"), ev)
check("SILENCE: done carries message_id null", ev[-1] == ("done", '{"message_id": null}'), ev[-1:])
check("SILENCE: no assistant row persisted", db.assistant_rows() == [])
check("SILENCE: user row still persisted", any(w[2].get("role") == "user" for w in db.writes if w[0] == "insert"))
check("SILENCE: session_state still folded", len(db.state_writes()) == 1)
ev, db = post_msg("Can you help me here?", reply="Split buyers by age first.")
check("HINT: token + done with a real message_id", [e for e, _ in ev] == ["meta", "token", "done"] and "null" not in ev[-1][1], ev)
check("HINT: assistant row persisted with V11 text", [r["content"] for r in db.assistant_rows()] == ["Split buyers by age first."])
ev, db = post_msg("I'll split households into urban and rural first.")
check("PRESENCE (text): plain-text token, row persisted", [e for e, _ in ev] == ["meta", "token", "done"] and not ev[1][1].startswith("{") and len(db.assistant_rows()) == 1, ev)

# An empty reply that is NOT V11 silence must not be swallowed as silence.
_real = att.stream_interviewer_reply


def _empty_non_silence(**kw):
    kw["control_out"].update({"mode": "HINT", "tag": {}, "reason": "help_requested"})
    return iter(())


att.stream_interviewer_reply = _empty_non_silence
ev, db = post_msg("anything")
att.stream_interviewer_reply = _real
check("unexpected empty (not V11 silence) keeps the old persistence path", len(db.assistant_rows()) == 1 and ev[-1][0] == "done")

print()
print("=" * 72)
print("3. Realtime sessions: the speech model is a voice only")
print("=" * 72)


class _FakeResp:
    status_code = 200
    text = "{}"

    def json(self):
        return {"value": "ek_test", "expires_at": 1}


class _FakeAsyncClient:
    last = None

    def __init__(self, *a, **k): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False

    async def post(self, url, headers=None, json=None):
        _FakeAsyncClient.last = json
        return _FakeResp()


rt.httpx = types.SimpleNamespace(AsyncClient=_FakeAsyncClient)
rt.OPENAI_API_KEY = "sk-test"
rt.get_supabase_client = lambda: FakeDB([])
rt.get_verified_user = lambda sb, a: ("u1", {"id": "u1"})
rt.check_rate_limit = lambda *a, **k: None
rt.assert_daily_budget = lambda *a, **k: None
rt.get_ai_input_quota = lambda sb, uid: {"tier": "pro"}
rt.has_credit = lambda *a, **k: True
rt.get_balance = lambda *a, **k: {"total_remaining": 10}
rt.log_ai_usage = lambda **k: None
# These checks cover the renderer interviewer (V11/V12 decides every turn). Since
# 2026-10-02 realtime defaults to the model-led interviewer (tests/test_voice_model_led.py).
os.environ["VOICE_INTERVIEWER"] = "renderer"
out = asyncio.run(rt.create_realtime_session(rt.RealtimeSessionRequest(case_id="c1", attempt_id="a1"), authorization="Bearer t"))
os.environ.pop("VOICE_INTERVIEWER")
sess = (_FakeAsyncClient.last or {}).get("session", {})
td = sess.get("audio", {}).get("input", {}).get("turn_detection", {})
check("OpenAI session minted", out.get("client_secret") == "ek_test")
check("OpenAI instructions == voice renderer only", sess.get("instructions") == VOICE_RENDERER_INSTRUCTIONS)
check("OpenAI instructions carry no old interviewer text", not any(p in sess.get("instructions", "") for p in OLD_PHRASES))
check("OpenAI turn_detection.create_response is False", td.get("create_response") is False, td)
check("OpenAI barge-in kept (interrupt_response True)", td.get("interrupt_response") is True, td)
rt.REALTIME_TURN_MODE = "server_vad"
check("server_vad fallback also has create_response False", rt.build_turn_detection().get("create_response") is False)
rt.REALTIME_TURN_MODE = "semantic_vad"

captured = {}


class _FakeTokens:
    def create(self, config):
        captured["config"] = config
        return types.SimpleNamespace(name="auth_tokens/test")


fake_genai = types.ModuleType("google.genai")
fake_genai.Client = lambda api_key=None: types.SimpleNamespace(auth_tokens=_FakeTokens())
import google  # noqa: E402

sys.modules["google.genai"] = fake_genai
google.genai = fake_genai
rtg.GEMINI_API_KEY = "test"
rtg.get_supabase_client = lambda: FakeDB([])
rtg.get_verified_user = lambda sb, a: ("u1", {"id": "u1"})
rtg.is_guest_user = lambda u: False
rtg.check_rate_limit = lambda *a, **k: None
rtg.assert_daily_budget = lambda *a, **k: None
rtg.get_ai_input_quota = lambda sb, uid: {"tier": "pro"}
rtg.has_credit = lambda *a, **k: True
rtg.get_balance = lambda *a, **k: {"total_remaining": 10}
rtg.log_ai_usage = lambda **k: None
rtg._resolve_live_model = lambda: "fake-live"
os.environ["VOICE_INTERVIEWER"] = "renderer"  # these checks cover the renderer flow (live is the default)
_res = rtg.create_gemini_session(rtg.GeminiSessionRequest(case_id="c1", attempt_id="a1"), authorization="Bearer t")
os.environ.pop("VOICE_INTERVIEWER")
out = asyncio.run(_res) if asyncio.iscoroutine(_res) else _res  # sync handler since the speed pass
cfg = ((captured.get("config") or {}).get("live_connect_constraints") or {}).get("config") or {}
check("Gemini session minted", out.get("token") == "auth_tokens/test")
check("Gemini system_instruction == voice renderer only", cfg.get("system_instruction") == VOICE_RENDERER_INSTRUCTIONS)
check("Gemini instructions carry no old interviewer text", not any(p in str(cfg) for p in OLD_PHRASES))
check("Gemini still transcribes input + output", "input_audio_transcription" in cfg and "output_audio_transcription" in cfg)

print()
print("=" * 72)
print("5. Speed: early voice decisions change nothing unless confirmed; one turn at a time")
print("=" * 72)
if hasattr(att, "_resolve_fold"):
    import threading  # noqa: E402

    TAGGED = "<<mode=coach; intervention=micro_hint; hint=2>>\n\nSplit buyers by age first."

    def vd(text, reply=TAGGED, **extra):
        LLM.reply, LLM.raise_exc = reply, None
        r = client.post("/attempts/a1/voice-decision", json=dict({"content": text}, **extra), headers=H)
        settled()
        return r

    def fold(turn_id, commit):
        r = client.post("/attempts/a1/voice-fold", json={"turn_id": turn_id, "commit": commit}, headers=H)
        settled()
        return r

    DB = FakeDB(LIVE)
    r = vd("Can you help me here?", turn_id="T1", defer_fold=True)
    check("early decision answers normally", r.status_code == 200 and r.json().get("mode") == "HINT", r.text[:200])
    check("early decision: NO learner-state fold yet", DB.state_writes() == [])
    r = fold("T1", True)
    check("confirm -> the fold lands (hint_level 2)", r.status_code == 200 and r.json().get("found") is True
          and len(DB.state_writes()) == 1 and DB.state_writes()[0].get("hint_level") == 2, (r.text, DB.state_writes()))
    check("confirming twice changes nothing", fold("T1", True).json().get("found") is False and len(DB.state_writes()) == 1)

    DB = FakeDB(LIVE)
    vd("Can you help", turn_id="T2", defer_fold=True)
    fold("T2", False)
    check("voided early decision: never folded", DB.state_writes() == [])
    vd("Can you help me here?")
    check("...the next decision folds only itself", len(DB.state_writes()) == 1)

    DB = FakeDB(LIVE)
    vd("Can you help", turn_id="T3", defer_fold=True)
    seen = []
    _real = att.stream_interviewer_reply

    def _spy(**kw):
        seen.append(dict(kw.get("prior_state") or {}))
        return _real(**kw)

    att.stream_interviewer_reply = _spy
    vd("Can you help me here?", discard_turn_ids=["T3"])
    check("discard_turn_ids on the next decision drops the early fold first",
          seen and seen[-1] == {} and len(DB.state_writes()) == 1, (seen, DB.state_writes()))

    DB = FakeDB(LIVE)
    vd("Can you help me here?", turn_id="T4", defer_fold=True)   # client never says
    seen.clear()
    vd("No, I want a hint.")
    check("an unconfirmed, undiscarded early fold is applied before the next turn reads state",
          seen and seen[-1].get("hint_level") == 2 and len(DB.state_writes()) == 2, (seen, DB.state_writes()))

    DB = FakeDB(LIVE)
    fold("T5", False)                                               # void arrives BEFORE the decision
    vd("Can you help", turn_id="T5", defer_fold=True)
    vd("Can you help me here?")
    check("a void that overtakes its decision still wins", len(DB.state_writes()) == 1)

    r = client.post("/attempts/a1/voice-fold", json={"turn_id": "x", "commit": True})
    check("voice-fold requires auth", r.status_code == 401)

    # A typed turn (or submit) after an UNFINISHED early voice decision: those words
    # were never saved, so their fold must not land.
    DB = FakeDB(LIVE)
    vd("Can you help", turn_id="T6", defer_fold=True)
    LLM.reply = "Line."
    client.post("/attempts/a1/messages", json={"content": "typing now", "kind": "text"}, headers=H)
    settled()
    check("typed turn drops an unfinished early voice fold", all(w.get("hint_level") != 2 for w in DB.state_writes()), DB.state_writes())
    check("...and a late confirm for it changes nothing", fold("T6", True).json().get("found") is False)

    # Two overlapping turns for one attempt: the second waits for the first's fold,
    # exactly as when requests ran one at a time.
    DB = FakeDB(LIVE)
    seen.clear()
    import time as _t  # noqa: E402

    def _slow_first(messages):
        if any("SLOW" in (m.get("content") or "") for m in messages if m["role"] == "user"):
            _t.sleep(0.6)
        return TAGGED

    LLM.reply = _slow_first
    th = threading.Thread(target=lambda: client.post("/attempts/a1/voice-decision",
                                                     json={"content": "SLOW can you help me here?"}, headers=H))
    th.start()
    _t.sleep(0.15)
    client.post("/attempts/a1/voice-decision", json={"content": "No, I want a hint."}, headers=H)
    th.join()
    settled()
    check("overlapping turn N+1 read turn N's folded state (serialised per attempt)",
          len(seen) == 2 and seen[0] == {} and seen[1].get("hint_level") == 2, seen)
    check("...and the folds landed in order (N, then N+1)", len(DB.state_writes()) == 2)

    # The full turn after a voided early look does not wait for that look's model call.
    DB = FakeDB(LIVE)
    th = threading.Thread(target=lambda: client.post("/attempts/a1/voice-decision",
                                                     json={"content": "SLOW can you", "turn_id": "T7", "defer_fold": True}, headers=H))
    th.start()
    _t.sleep(0.15)
    t0 = _t.monotonic()
    r = client.post("/attempts/a1/voice-decision", json={"content": "Can you help me here?", "discard_turn_ids": ["T7"]}, headers=H)
    waited = _t.monotonic() - t0
    th.join()
    settled()
    check(f"full turn not held behind the voided early look ({waited * 1000:.0f} ms < 400 ms)", r.status_code == 200 and waited < 0.4)
    check("...and only the full turn folded", len(DB.state_writes()) == 1)
    att.stream_interviewer_reply = _real
else:
    print("(skipped: this backend has no early-decision support)")

print()
print("=" * 72)
print("4. The old static interviewer cannot come back")
print("=" * 72)
for name in ("CASE_INTERVIEWER_SYSTEM_PROMPT", "GUESSTIMATE_INTERVIEWER_SYSTEM_PROMPT",
             "INTERVIEWER_SYSTEM_PROMPT", "VOICE_INTERVIEWER_ADDENDUM", "CLARIFICATIONS_EXHAUSTED_DIRECTIVE"):
    check(f"prompts.interview_prompts.{name} is gone", not hasattr(old_prompts, name))
try:
    old_prompts.build_interviewer_messages("c", "guesstimate", [], "hi")
    check("build_interviewer_messages fails closed", False)
except old_prompts.LegacyInterviewerRemovedError:
    check("build_interviewer_messages fails closed", True)
check("scoring prompt untouched and present", hasattr(old_prompts, "CONVERSATION_SCORING_SYSTEM_PROMPT")
      and hasattr(old_prompts, "build_conversation_scoring_user_prompt"))
os.environ["ADAPTIVE_INTERVIEWER"] = "false"
ev, db = post_msg("Can you help me here?")
os.environ["ADAPTIVE_INTERVIEWER"] = "true"
check("ADAPTIVE forced off -> turn fails closed (SSE error), no old-prompt reply",
      ev and ev[-1][0] == "error" and "LegacyInterviewerRemovedError" in ev[-1][1] and db.assistant_rows() == [], ev[-1:])
os.environ.pop("ADAPTIVE_INTERVIEWER", None)
import importlib  # noqa: E402
import main  # noqa: E402,F401

check("main.py pins ADAPTIVE_INTERVIEWER=true", os.environ.get("ADAPTIVE_INTERVIEWER") == "true")
check("app still registers /messages and /voice-decision",
      {"/attempts/{attempt_id}/messages", "/attempts/{attempt_id}/voice-decision"} <= {r.path for r in main.app.routes})

print()
print(f"{len(_fail)} FAILED: {_fail}" if _fail else "ALL PASS")
sys.exit(1 if _fail else 0)
