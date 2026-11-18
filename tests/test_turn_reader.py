"""Turn reader (services/turn_reader.py): the model reads what a turn MEANT; the keyword flags stay as the floor.

Pinned here, with the two live keyword failures that motivated it (both reproduced on 2026-11-18):
  * "yaar dimaag kaam nahi kar raha, kahan se shuru karu" -> keywords read it as ordinary working (a MISS)
  * "just tell me if my structure is fine"               -> keywords read a demand for the answer and HIGH
                                                             frustration (a MISFIRE)
and the guarantees: off changes nothing, shadow changes nothing, a failed read falls back to the keywords,
a low-confidence read can only add, a "final answer" still needs a number, voice is never read.

No network, no keys: the model is faked.
"""
from __future__ import annotations

import os
import time
import types

for _k, _v in {"OPENAI_API_KEY": "sk-test", "SUPABASE_URL": "https://example.supabase.co",
               "SUPABASE_SERVICE_ROLE_KEY": "test"}.items():
    os.environ.setdefault(_k, _v)
os.environ["ADAPTIVE_INTERVIEWER"] = "true"

from services import interview_engine as ie  # noqa: E402
from services import turn_reader as tr  # noqa: E402
from services.session_signals import compute_signals  # noqa: E402

NOTE = {"role": "system", "content": "Case ready: X. Ask any clarifying questions before structuring."}
HIST = [NOTE, {"role": "assistant", "content": "How would you structure this?"},
        {"role": "user", "content": "I'll split by households first"},
        {"role": "assistant", "content": "Go ahead."}]
NO_AUDIT = {"assessor": lambda *a, **k: {}, "numbers_auditor": lambda *a, **k: {"issues": [], "verified": []}}

ALL_FALSE = {k: False for k in tr.FLAG_MAP}


def reading(confidence="high", concept=None, **on):
    r = dict(ALL_FALSE, **on, confidence=confidence)
    if concept:
        r["concept"] = concept
    return tr.normalize(r)


def plan(msg, *, mode, reader=None, history=HIST, channel="text"):
    ov = dict(NO_AUDIT, turn_reader_mode=mode)
    if reader is not None:
        ov["turn_reader"] = reader
    return ie._plan_adaptive_turn("Estimate air purifiers sold in Delhi.", "guesstimate", list(history), msg, False,
                                  None, {}, channel, False, prompt_overrides=ov)


# ---------------------------------------------------------------- normalize
def test_normalize_reads_json_text_and_drops_junk():
    r = tr.normalize('noise {"help": "yes", "solution": 0, "made_up": true, "confidence": "HIGH", '
                     '"concept": "  CAGR "} tail')
    assert r == {"help": True, "solution": False, "concept": "CAGR", "confidence": "high"}
    assert tr.normalize("not json") == {}
    assert tr.normalize({"confidence": "high"}) == {}  # no flag at all is not a reading
    assert tr.normalize({"help": True, "concept": "null"})["confidence"] == "low"


# ---------------------------------------------------------------- the two live failures
def test_hinglish_help_the_keywords_miss_is_read():
    msg = "yaar dimaag kaam nahi kar raha, kahan se shuru karu"
    kw = compute_signals(HIST, msg)
    assert kw["help_requested"] is False and kw["intent"] == "answering"  # the miss, as it is today
    s = compute_signals(HIST, msg, reading=reading(help=True, blank=True))
    assert s["help_requested"] and s["stuck_blank"] and s["intent"] == "asking_for_help"
    # rated exactly as the keyword path rates an English "i'm stuck, no idea where to start" here
    assert s["frustration"] == compute_signals(HIST, "i'm stuck, no idea where to start")["frustration"]
    assert "help_requested" in s["turn_reader"]["changed"]


def test_asking_whether_own_work_is_right_is_not_a_demand_for_the_answer():
    msg = "just tell me if my structure is fine"
    kw = compute_signals(HIST, msg)
    assert kw["solution_requested"] and kw["frustration"] == "high"  # the misfire, as it is today
    s = compute_signals(HIST, msg, reading=reading(check=True, asks_interviewer=True))
    assert s["solution_requested"] is False and s["asks_if_correct"] is True
    assert s["intent"] != "asking_for_solution" and s["frustration"] != "high"


def test_the_route_changes_end_to_end_when_on():
    msg = "just tell me if my structure is fine"
    off = plan(msg, mode="off")
    on = plan(msg, mode="on", reader=lambda *a: reading(check=True, asks_interviewer=True))
    assert off["signals"]["solution_requested"] and not on["signals"]["solution_requested"]
    assert off["function"] != on["function"]


# ---------------------------------------------------------------- merge rules
def test_low_confidence_can_add_but_never_remove():
    msg = "just tell me if my structure is fine"
    s = compute_signals(HIST, msg, reading=reading("low", check=True))
    assert s["solution_requested"] is True and s["asks_if_correct"] is True


def test_final_answer_still_needs_a_number():
    hist = HIST[:2]  # no numbers anywhere yet
    s = compute_signals(hist, "that's my final recommendation, go with option B", reading=reading(final=True))
    assert s["final_declared"] is False
    s2 = compute_signals(hist, "final answer: about 2 lakh units", reading=reading(final=True))
    assert s2["final_declared"] is True and s2["states_final_estimate"] is True


def test_concept_from_the_reading_fills_in_when_keywords_found_none():
    s = compute_signals(HIST, "matlab ye contribution margin kya hota hai", reading=reading(concept="contribution margin",
                                                                                           asks_interviewer=True))
    assert s["concept_asked"] == "contribution margin"


# ---------------------------------------------------------------- modes and safety
def test_on_by_default_and_off_is_the_kill_switch(monkeypatch):
    monkeypatch.delenv("INTERVIEWER_TURN_READER", raising=False)
    assert tr.mode() == "on" and tr.mode("nonsense") == "on"
    monkeypatch.setenv("INTERVIEWER_TURN_READER", "off")
    assert tr.mode() == "off"


def test_off_makes_no_call_and_changes_nothing():
    called = []
    p = plan("just tell me if my structure is fine", mode="off",
             reader=lambda *a: called.append(1) or reading(check=True))
    assert not called and p["signals"]["solution_requested"] and "turn_reader" not in p["signals"]


def test_shadow_never_changes_the_live_turn(capsys):
    done = []

    def reader(*a):
        done.append(1)
        return reading(check=True, asks_interviewer=True)

    live = plan("just tell me if my structure is fine", mode="shadow", reader=reader)
    assert live["signals"]["solution_requested"] is True  # keywords still route
    for _ in range(50):
        if done:
            break
        time.sleep(0.02)
    time.sleep(0.1)
    assert done and "[turn_reader] disagree" in capsys.readouterr().out


def test_a_failed_read_falls_back_to_the_keywords():
    def boom(*a):
        raise TimeoutError("slow")

    p = plan("just tell me if my structure is fine", mode="on", reader=boom)
    assert p["signals"]["solution_requested"] is True
    assert plan("just tell me if my structure is fine", mode="on", reader=lambda *a: {})["signals"]["solution_requested"]


def test_voice_and_bare_affirmations_are_never_read():
    called = []
    plan("okay", mode="on", reader=lambda *a: called.append(1) or {})
    plan("I'm stuck here", mode="on", reader=lambda *a: called.append(1) or {}, channel="voice")
    assert not called


# ---------------------------------------------------------------- the real call, faked
class _Resp:
    def __init__(self, content):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
        self.usage = None
        self.id = "r1"


class _Client:
    def __init__(self, content=None, exc=None):
        self.content, self.exc, self.opts, self.kwargs = content, exc, None, None
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def with_options(self, **kw):
        self.opts = kw
        return self

    def _create(self, **kw):
        self.kwargs = kw
        if self.exc:
            raise self.exc
        return _Resp(self.content)


def test_read_turn_sends_one_json_call_with_a_timeout_and_records_usage():
    cli = _Client('{"help": true, "confidence": "high"}')
    seen = []
    r = tr.read_turn(HIST, "help na", "case text", client=cli, record=lambda resp, ms, m: seen.append(m))
    assert r == {"help": True, "confidence": "high"}
    assert cli.opts == {"timeout": tr.READER_TIMEOUT_S, "max_retries": 0}
    assert cli.kwargs["response_format"] == {"type": "json_object"} and cli.kwargs["temperature"] == 0.0
    assert "THE TURN TO READ" in cli.kwargs["messages"][1]["content"] and seen == [tr.READER_MODEL_DEFAULT]


def test_read_turn_never_raises():
    assert tr.read_turn(HIST, "x", "c", client=_Client(exc=RuntimeError("down"))) == {}
    assert tr.read_turn(HIST, "x", "c", client=_Client("garbage")) == {}


def test_prompt_and_merge_cover_the_same_flags():
    for k in tr.FLAG_MAP:
        assert f"\n{k} " in tr.READER_PROMPT or f"\n{k}  " in tr.READER_PROMPT, k
