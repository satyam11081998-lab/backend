"""Short lines (services/short_lines.py): the interviewer's brief beats worded by a model, fixed lines as fallback.

Pinned: fixed mode is byte-for-byte the old behaviour; model mode ships a worded line only when it passes every
rule a fixed line satisfies (short, no question, no verdict, no number, not a repeat); voice is never worded;
every failure is the fixed line. No network, no keys.
"""
from __future__ import annotations

import os
import types

for _k, _v in {"OPENAI_API_KEY": "sk-test", "SUPABASE_URL": "https://example.supabase.co",
               "SUPABASE_SERVICE_ROLE_KEY": "test"}.items():
    os.environ.setdefault(_k, _v)
os.environ["ADAPTIVE_INTERVIEWER"] = "true"

from services import interview_engine as ie  # noqa: E402
from services import short_lines as sl  # noqa: E402
from services.interviewer_decision import _FAST_PHRASES, fast_kind  # noqa: E402

NOTE = {"role": "system", "content": "Case ready: X. Ask any clarifying questions before structuring."}
HIST = [NOTE, {"role": "assistant", "content": "How would you structure this?"},
        {"role": "user", "content": "I'll split by households first"},
        {"role": "assistant", "content": "Go ahead."}]
NO_AUDIT = {"assessor": lambda *a, **k: {}, "numbers_auditor": lambda *a, **k: {"issues": [], "verified": []}}
FIXED = {p["text"] for opts in _FAST_PHRASES.values() for p in opts}


def reply(msg, *, mode, liner=None, channel="text"):
    ov = dict(NO_AUDIT, short_line_mode=mode)
    if liner is not None:
        ov["short_liner"] = liner
    ctl = {}
    out = ie.complete_interviewer_reply("Estimate air purifiers sold in Delhi.", "guesstimate", list(HIST), msg,
                                        control_out=ctl, channel=channel, prompt_overrides=ov)
    return out, ctl


# ---------------------------------------------------------------- the line checker
def test_clean_line_accepts_a_plain_beat_and_tidies_it():
    assert sl.clean_line('"sure, take your time with the numbers"', []) == "Sure, take your time with the numbers."
    assert sl.clean_line("Interviewer: Haan, aaram se karo.", []) == "Haan, aaram se karo."


def test_clean_line_rejects_everything_a_fixed_line_never_does():
    recent = ["Go ahead."]
    for bad in ["Shall we move on?",                       # a question
                "Great, carry on.",                        # a verdict
                "That's correct, keep going.",             # certifies their work
                "Use 4 people per household.",             # a number / a hint
                "go ahead",                                # repeats the last line
                "Okay so here is what you should probably think about doing next with the market",  # too long
                ""]:
        assert sl.clean_line(bad, recent) is None, bad


# ---------------------------------------------------------------- end to end
def test_fixed_mode_is_the_old_behaviour():
    called = []
    out, ctl = reply("ek second, abhi calculate kar raha hoon...", mode="fixed",
                     liner=lambda *a: called.append(1) or "Aaram se karo.")
    assert not called and out in FIXED and ctl["tag"]["intervention"] == "fast_lane"


def test_model_mode_words_the_beat_for_this_turn():
    seen = []

    def liner(kind, sig, msg):
        seen.append(kind)
        return "Haan, aaram se calculate karo."

    out, ctl = reply("ek second, abhi calculate kar raha hoon...", mode="model", liner=liner)
    assert out == "Haan, aaram se calculate karo." and seen == ["THINK"]
    assert ctl["tag"]["intervention"] == "model_short_line" and ctl["render"] == "fast"


def test_a_failed_or_rejected_line_falls_back_to_the_fixed_one():
    def boom(*a):
        raise TimeoutError("slow")

    assert reply("ek second, abhi calculate kar raha hoon...", mode="model", liner=boom)[0] in FIXED
    assert reply("ek second, abhi calculate kar raha hoon...", mode="model", liner=lambda *a: None)[0] in FIXED


def test_voice_is_never_worded():
    called = []
    reply("one sec, calculating", mode="model", liner=lambda *a: called.append(1) or "Sure.", channel="voice")
    assert not called


def test_the_kind_matches_the_fixed_lane():
    assert fast_kind("SHORT_ACK", {"asks_for_space": True}) == "SPACE"
    assert fast_kind("HAND_BACK", {"candidate_mid_thought": True}) == "THINK"
    assert fast_kind("VALIDATE", {}) == "HAND_BACK"
    assert set(sl.PURPOSE) == set(_FAST_PHRASES)


# ---------------------------------------------------------------- the real call, faked
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
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=self.content))],
                                     usage=None, id="r")


def test_word_line_is_one_short_call_with_a_timeout():
    cli = _Client("Sure, it's all yours.")
    sig = {"recent_assistant_turns": ["How would you structure this?"]}
    assert sl.word_line("SPACE", sig, "stop asking, let me solve", client=cli) == "Sure, it's all yours."
    assert cli.opts == {"timeout": sl.SHORT_TIMEOUT_S, "max_retries": 0} and cli.kwargs["max_tokens"] == 30
    assert "stop asking, let me solve" in cli.kwargs["messages"][0]["content"]
    assert sl.word_line("SPACE", sig, "x", client=_Client(exc=RuntimeError("down"))) is None
    assert sl.word_line("UNKNOWN", sig, "x", client=cli) is None


def test_the_streamed_route_words_the_beat_too():
    ctl = {}
    ov = dict(NO_AUDIT, short_line_mode="model", short_liner=lambda *a: "Sure, it's all yours.")
    out = "".join(ie.stream_interviewer_reply("Estimate air purifiers sold in Delhi.", "guesstimate", list(HIST),
                                              "stop asking so many questions, let me solve", control_out=ctl,
                                              prompt_overrides=ov))
    assert out == "Sure, it's all yours." and ctl["tag"]["intervention"] == "model_short_line"


def test_model_by_default_and_fixed_is_the_kill_switch(monkeypatch):
    monkeypatch.delenv("INTERVIEWER_SHORT_LINES", raising=False)
    assert sl.mode() == "model"
    for off in ("fixed", "off", "FIXED"):
        monkeypatch.setenv("INTERVIEWER_SHORT_LINES", off)
        assert sl.mode() == "fixed"
