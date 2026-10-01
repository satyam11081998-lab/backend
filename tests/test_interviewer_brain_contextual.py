"""
Contextual presence: the interviewer picks the FUNCTION, the model words it from what the
candidate actually said, the validator checks it. Replays the turns that used to get a stock
"Okay." / "Right." / "That works. Go ahead." and pins the new behaviour.

    python -m pytest tests/test_interviewer_brain_contextual.py -q
"""
from __future__ import annotations

import json
import re

import pytest

from tests.interviewer_fakes import FakeDB, FakeLLM, contextual_line, install_route_fakes, parse_sse

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import routes.attempts as att  # noqa: E402
from services.interviewer import dedupe, engine, prompting, telemetry  # noqa: E402
from services.interviewer.assessor import Assessment  # noqa: E402
from services.interviewer.types import (  # noqa: E402
    CONTEXTUAL_PRESENCE, CaseContext, Channel, Intervention, Lane, TurnInput,
)
from services.interviewer.validate import _CORRECTNESS_RE, _LEAK_RE, contextual_violations, validate_text  # noqa: E402

GUESS = CaseContext(case_type="guesstimate", content="Estimate the number of new cars sold in India per year.")
MID = {"brain": {"opened": True, "turns": 3, "phase": "analysis"}}
Q = [{"role": "assistant", "content": "Walk me through how you'd size it."}]
STRUCTURE = "I'd split households by urban and rural, then by income band, then by car ownership. That's my structure."
LONG_STEP = ("So I take 30 crore households, 10 percent can afford a car, that is 3 crore households, and with a "
             "seven year replacement cycle that gives about 43 lakh cars a year.")
STOCK = re.compile(r"^\s*(okay|ok|right|alright|got it|mm-?hm|that works\. go ahead|okay\. take it from there)[.!]?\s*$", re.I)


def plan(msg, channel=Channel.TEXT, state=MID, transcript=Q, assess=None, llm=None):
    return engine.decide_turn(turn=TurnInput(text=msg, channel=channel), case=GUESS, transcript=transcript,
                              session_state=state, assess=assess, llm=llm)


def ok_assessor(**kw):
    return Assessment(material=False)


# ------------------------------------------------------------------ replay: no stock line for substantive work
@pytest.mark.parametrize("msg,channel,assess,function", [
    (STRUCTURE, Channel.TEXT, ok_assessor, Intervention.REFLECT_PROGRESS),
    (STRUCTURE, Channel.VOICE, ok_assessor, Intervention.REFLECT_PROGRESS),
    ("Is my approach okay? I'm going households, then affordability, then replacement.", Channel.TEXT, ok_assessor,
     Intervention.ACKNOWLEDGE_AND_CONTINUE),
    (LONG_STEP, Channel.VOICE, None, Intervention.ACKNOWLEDGE_AND_CONTINUE),
])
def test_substantive_work_gets_a_contextual_line_not_a_stock_one(msg, channel, assess, function):
    p = plan(msg, channel=channel, assess=assess, llm=FakeLLM())
    assert p.decision.intervention == function and p.needs_model and p.decision.max_questions == 0
    line = engine.word_complete(p)
    assert line and not STOCK.match(line) and "?" not in line
    assert not contextual_violations(line, msg)            # it is about what they said
    assert not _LEAK_RE.search(line)


def test_short_floor_yields_and_numbers_stay_instant_and_fixed():
    # No model for these: they need no context (and they must stay fast).
    assert plan("Shall I proceed?").text == "Yes, go ahead."
    for msg in ("50%", "3", "1.4 billion / 3 is about 0.46B", "let me think"):
        assert plan(msg).decision.lane == Lane.NO_OUTPUT


def test_text_long_progress_still_stays_silent():
    # The candidate keeps the floor in text; only voice marks a long finished step.
    assert plan(LONG_STEP, channel=Channel.TEXT).decision.lane == Lane.NO_OUTPUT


# ------------------------------------------------------------------ verdicts only when something checked
def test_verdict_permission_follows_verification():
    unchecked = plan("Is my structure okay?", assess=lambda **kw: Assessment(material=False, ok=False,
                                                                             error_type="assessor_timeout"))
    assert unchecked.decision.detail["may_say_correct"] is False
    checked = plan("Is my structure okay?", assess=ok_assessor)
    assert checked.decision.detail["may_say_correct"] is True
    verified_step = plan("So 1.4 billion / 4.5 = 31 crore households, then 20 percent of them can afford a car "
                         "which gives 6 crore households to work with from here.", channel=Channel.VOICE)
    assert verified_step.decision.intervention == Intervention.ACKNOWLEDGE_AND_CONTINUE
    assert verified_step.decision.detail["may_say_correct"] is True
    assert verified_step.decision.detail["verified_claims"]


def test_unverified_correctness_claims_are_removed():
    r = validate_text("That's correct, you're on the right track. You've split it by urban and rural.",
                      max_questions=0, max_sentences=2, may_say_correct=False)
    assert "unverified_claim" in r.violations and "correct" not in r.text.lower()
    assert r.text == "You've split it by urban and rural."
    # with a verification behind it the plain statement may stay
    r2 = validate_text("That division holds. Carry on with the replacement cycle.", max_questions=0, max_sentences=2,
                       may_say_correct=True)
    assert "That division holds." in r2.text


@pytest.mark.parametrize("bad,why", [("Right.", "generic_ack"), ("Okay, go ahead.", "generic_ack"),
                                     ("Carry on with the next part.", "unreferenced")])
def test_stock_or_unrelated_beats_are_rejected(bad, why):
    assert why in contextual_violations(bad, STRUCTURE)


def test_a_stock_reply_is_regenerated_then_falls_back_without_an_error():
    llm = FakeLLM(lambda messages, mv: "Right.")
    p = plan(STRUCTURE, assess=ok_assessor, llm=llm)
    line = engine.word_complete(p)
    assert len(llm.calls) == 2                               # one regeneration, then stop
    assert line == p.decision.detail["fallback"] and p.soft_error == "EMPTY_MODEL_OUTPUT"
    assert len(p.metas) == 2                                 # both calls accounted for (cost)


def test_a_regenerated_good_line_is_used():
    seq = {"n": 0}

    def flaky(messages, mv):
        seq["n"] += 1
        return "Right." if seq["n"] == 1 else contextual_line(messages, mv)
    p = plan(STRUCTURE, assess=ok_assessor, llm=FakeLLM(flaky))
    line = engine.word_complete(p)
    assert line != p.decision.detail["fallback"] and "regenerated" in p.violations


def test_provider_failure_on_a_contextual_beat_falls_back_but_hard_moves_still_fail():
    p = plan(STRUCTURE, assess=ok_assessor, llm=FakeLLM("error"))
    assert engine.word_complete(p) == p.decision.detail["fallback"] and p.soft_error == "PROVIDER_ERROR"
    hint = plan("Can you give me a hint?", llm=FakeLLM("error"))
    with pytest.raises(Exception):
        engine.word_complete(hint)                           # a hint is never replaced by canned text


# ------------------------------------------------------------------ orient: the next part of THEIR plan
def test_stage_done_orients_by_the_candidates_own_plan():
    t = Q + [{"role": "user", "content": STRUCTURE}, {"role": "assistant", "content": "Okay. Take it from there."},
             {"role": "user", "content": "Urban households are about 11 crore."}]
    p = plan("So that's the urban side.", transcript=t, llm=FakeLLM())
    assert p.decision.intervention == Intervention.ACKNOWLEDGE_AND_ORIENT
    assert "urban and rural" in p.decision.detail["candidate_plan"]
    # they already said what is next -> nothing to add
    assert plan("Urban side is done, now rural.", transcript=t).decision.lane == Lane.NO_OUTPUT
    # no plan on record -> no orienting
    assert plan("So that's the urban side.").decision.intervention != Intervention.ACKNOWLEDGE_AND_ORIENT


def test_contextual_beats_never_come_twice_in_a_row():
    p1 = plan(LONG_STEP, channel=Channel.VOICE, llm=FakeLLM())
    st = {"brain": engine.finalize(p1, engine.word_complete(p1))}
    p2 = plan(LONG_STEP.replace("43", "44"), channel=Channel.VOICE, state=st)
    assert p2.decision.lane == Lane.NO_OUTPUT


# ------------------------------------------------------------------ the control packet
def test_model_receives_a_json_control_packet_not_a_line():
    p = plan(STRUCTURE, assess=ok_assessor)
    msgs = prompting.build_messages(p.decision, GUESS, p.state_before, Channel.TEXT, Q, STRUCTURE)
    sysmsg = msgs[0]["content"]
    raw = sysmsg.split("INTERVIEWER CONTROL PACKET:\n", 1)[1].split("\n", 1)[0]
    packet = json.loads(raw)
    assert packet["response_function"] == "REFLECT_PROGRESS"
    assert packet["permissions"] == {"questions_max": 0, "hint": False, "correction": False, "solution": False,
                                     "new_case_facts": False}
    assert packet["generation"]["must_reference_candidate_content"] is True
    assert packet["verification"]["may_say_correct"] is True
    assert "MOVE:" not in sysmsg and "fallback" not in raw            # the fallback line is never shown to the model
    for key in ("response_function", "may_say_correct", "verified_claims", "control packet"):
        assert _LEAK_RE.search(f"As the {key} says, carry on.")       # packet words never reach the candidate


def test_every_model_move_gets_a_packet_with_its_permissions():
    hint = plan("Can you give me a hint?")
    packet = prompting.control_packet(hint.decision, GUESS, hint.state_before, Channel.VOICE, "Can you give me a hint?")
    assert packet["response_function"] == "MICRO_HINT" and packet["permissions"]["hint"] is True
    assert packet["generation"]["medium"] == "spoken" and packet["verification"]["may_say_correct"] is None


# ------------------------------------------------------------------ through the real routes
S = {"db": FakeDB(), "llm": FakeLLM()}
app = FastAPI()
app.include_router(att.router)
client = TestClient(app)
H = {"Authorization": "Bearer t"}


@pytest.fixture()
def routes_on(monkeypatch):
    monkeypatch.setenv("INTERVIEWER_BRAIN", "on")
    for name in ("get_supabase_client", "get_verified_user", "get_verified_user_id", "is_guest_user",
                 "check_rate_limit", "assert_daily_budget", "llm_case_content"):
        monkeypatch.setattr(att, name, getattr(att, name))
    install_route_fakes(att, lambda: S["db"])
    monkeypatch.setattr(engine, "LLM_FACTORY", lambda plan: S["llm"])
    monkeypatch.setattr(engine, "default_assessor", lambda uid: ok_assessor)
    S["db"] = FakeDB([{"role": "assistant", "kind": "text", "content": "Walk me through how you'd size it."}],
                     session_state=MID)
    dedupe.LEDGER.__init__()
    dedupe.ROWS.__init__()
    yield


def test_route_text_structure_gets_a_contextual_line(routes_on):
    S["llm"] = FakeLLM()
    r = client.post("/attempts/a1/messages", json={"content": STRUCTURE, "channel": "text", "turn_id": "c1"}, headers=H)
    att._await_after_turn("a1")
    ev = parse_sse(r.text)
    text = "".join(d for e, d in ev if e == "token")
    assert [e for e, _ in ev][-1] == "done" and text and not STOCK.match(text)
    assert S["db"].inserted(role="assistant")[-1]["content"] == text


def test_route_provider_down_gives_the_plain_hand_back_not_an_error(routes_on, capsys, monkeypatch):
    S["llm"] = FakeLLM("error")
    monkeypatch.setattr(telemetry, "_ENABLED", True)
    r = client.post("/attempts/a1/messages", json={"content": STRUCTURE, "channel": "text", "turn_id": "c2"}, headers=H)
    att._await_after_turn("a1")
    names = [e for e, _ in parse_sse(r.text)]
    assert "error" not in names and names[-1] == "done"
    out = capsys.readouterr().out
    line = next(json.loads(m.group(1)) for m in re.finditer(r"\[interviewer\.turn\] (\{.*\})", out))
    assert line["error_type"] == "PROVIDER_ERROR:fallback" and line["interviewer_mode"] == "REFLECT_PROGRESS"
    # a hint with the provider down still fails loudly
    r = client.post("/attempts/a1/messages", json={"content": "Can you give me a hint?", "channel": "text",
                                                    "turn_id": "c3"}, headers=H)
    att._await_after_turn("a1")
    assert "error" in [e for e, _ in parse_sse(r.text)]


def test_route_voice_long_step_is_presence_with_a_contextual_line(routes_on):
    S["llm"] = FakeLLM()
    j = client.post("/attempts/a1/voice-decision", json={"content": LONG_STEP, "turn_id": "v1"}, headers=H).json()
    att._await_after_turn("a1")
    assert j["lane"] == "PRESENCE" and j["say"] and not STOCK.match(j["say"]) and "?" not in j["say"]
