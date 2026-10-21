"""The arena must show exactly what production sends. Nothing else it reports is
worth anything if the text in the pane is not the text a candidate would receive.

v1 of the arena ran complete_interviewer_reply, which sanitises every reply. The live
text route streams instead, and in the question-allowed functions (OPEN,
TARGETED_QUESTION, PROBE, RETHINK_CUE ...) it ships the model's words with NO
sanitize_reply -- so a stacked second question reached users while the arena quietly
trimmed it away. These tests pin the arena to the production generator, lane by lane,
and pin the two things copied from routes/attempts.py (quota ladder, opening note) to
their source. The model is faked: free, deterministic, no network.
"""
from __future__ import annotations

import ast
import io
import os

import pytest

os.environ["ADAPTIVE_INTERVIEWER"] = "true"

CASE = "Spice Express sales fell 12% YoY.\n- 240 outlets\n- AOV Rs 420"
NOTE = {"role": "system", "kind": "system_note",
        "content": ("Case ready: Spice Express. On a phone, tap the menu at the top left to read the full "
                    "brief. Ask any clarifying questions before structuring.")}

# A reply that production's post-processing treats differently by lane: a praise
# opener, then two questions.
STACKED = ("<<mode=probe; intervention=targeted_question>>\n"
           "Great point, love that! What's the split between dine-in and delivery? "
           "And how many outlets are company-owned?")


class _Usage:
    prompt_tokens = 900
    completion_tokens = 40
    total_tokens = 940


class _Msg:
    def __init__(self, content):
        self.content = content


class _Resp:
    id = "fake-1"

    def __init__(self, content):
        self.choices = [type("Ch", (), {"message": _Msg(content)})()]
        self.usage = _Usage()


class _Chunk:
    def __init__(self, token=None, usage=None):
        self.choices = [] if token is None else [type("Ch", (), {"delta": _Msg(token)})()]
        self.usage = usage
        self.id = "fake-stream"


class _FakeClient:
    """Answers both the streamed and the non-streamed call the engine makes."""

    def __init__(self, content):
        self.content, self.calls = content, []
        outer = self

        class _Completions:
            @staticmethod
            def create(**kw):
                outer.calls.append(kw)
                if kw.get("stream"):
                    toks = [outer.content[i:i + 7] for i in range(0, len(outer.content), 7)]
                    return iter([_Chunk(t) for t in toks] + [_Chunk(None, _Usage())])
                return _Resp(outer.content)

        self.chat = type("Chat", (), {"completions": _Completions()})()


@pytest.fixture()
def wired(monkeypatch):
    """Production engine and arena runner, both pointed at ONE fake model."""
    from services import arena as arena_mod
    from services import interview_engine as eng

    client = _FakeClient(STACKED)
    monkeypatch.setattr(eng, "openai_client", lambda: client)
    monkeypatch.setattr(eng, "_resolve_adaptive_llm", lambda: (client, "gpt-4o-mini", "openai"))
    monkeypatch.setattr(eng, "assess_context_with_llm", lambda *a, **k: {})
    monkeypatch.setattr(arena_mod, "openai_client", lambda: client)
    prod_logged, arena_logged = [], []
    monkeypatch.setattr(eng, "log_ai_usage", lambda **kw: prod_logged.append(kw))
    monkeypatch.setattr(arena_mod, "log_ai_usage", lambda **kw: arena_logged.append(kw))
    return {"eng": eng, "arena": arena_mod, "client": client,
            "prod_logged": prod_logged, "arena_logged": arena_logged}


def _production(eng, transcript, message):
    """Exactly what routes/attempts.py post_message does with the generator."""
    ctl = {}
    text = "".join(eng.stream_interviewer_reply(
        case_content=CASE, case_type="profitability", transcript=transcript,
        new_user_message=message, user_id="user-1", clarifications_exhausted=False,
        teaching_policy=None, prior_state={}, control_out=ctl,
    )).strip()
    return text, ctl


def _arena(arena_mod, transcript, message, model_mode="openai-mini", variant_id="live"):
    return arena_mod.run_turn(
        case_content=CASE, case_type="profitability", variant_id=variant_id,
        transcript=transcript, new_user_message=message, prior_state={},
        teaching_policy="coached", model_mode=model_mode, user_id="admin-1",
    )


# One per lane production can take. The function each routes to is asserted, so if
# the signals change and a probe drifts into another lane the test says so.
LANES = [
    ("question_allowed_open", [NOTE],
     "Hi, I'd like to start by clarifying the objective. Is the goal to restore profitability or revenue?",
     "OPEN"),
    ("no_question_data_reveal",
     [NOTE, {"role": "user", "content": "I'd split revenue into price and volume"},
      {"role": "assistant", "content": "Go on."}],
     "I think the decline is mainly because of delivery commissions going up, so we should cut delivery.",
     None),
    ("presence_line",
     [NOTE, {"role": "user", "content": "profit = revenue - cost"},
      {"role": "assistant", "content": "Where would you start?"}],
     "Revenue fell 12%, so with 240 outlets at Rs 420 AOV and 300 orders a day, that's 240*420*300 = 3 crore a day.",
     None),
    ("help_hint", [NOTE, {"role": "user", "content": "I'd split revenue into price and volume, then look at delivery mix."},
                   {"role": "assistant", "content": "Where would you start?"}],
     "help na i'm not getting this at all", None),
    ("fast_lane", [NOTE, {"role": "user", "content": "I'd split revenue into price and volume, then look at delivery mix."},
                   {"role": "assistant", "content": "Where would you start?"}],
     "so the annual number is 50 lakh", None),
]


@pytest.mark.parametrize("name,transcript,message,expect_fn", LANES, ids=[l[0] for l in LANES])
def test_arena_text_is_byte_identical_to_production_on_every_lane(wired, name, transcript, message, expect_fn):
    prod_text, ctl = _production(wired["eng"], transcript, message)
    out = _arena(wired["arena"], transcript, message)
    if expect_fn:
        assert ctl.get("function") == expect_fn, f"{name} now routes to {ctl.get('function')}"
    assert out["error"] is None, out["error"]
    assert out["reply"] == prod_text, (
        f"{name}: arena shows {out['reply']!r} but production sends {prod_text!r}")
    assert out["control"]["function"] == ctl.get("function")
    assert out["control"]["render"] == ctl.get("render")


def test_a_stacked_question_is_trimmed_in_production_and_the_arena_shows_the_same(wired):
    """v1 failure: the arena hid the second question users received. V12.1 fixes the
    product instead: the streamed lane now drops the praise opener and cuts a stacked
    second question (ONE question per reply), and the arena still shows exactly what
    production sends -- the raw model text keeps both questions for the record."""
    transcript, message = LANES[0][1], LANES[0][2]
    prod_text, _ = _production(wired["eng"], transcript, message)
    out = _arena(wired["arena"], transcript, message)
    assert out["raw_reply"].count("?") == 2, "fixture no longer exercises the stacked question"
    assert prod_text.count("?") == 1
    assert "Great point" not in prod_text
    assert out["reply"] == prod_text
    assert "multiple_questions" not in out["violations"]
    assert out["enforced"] is True, "the rewrite is attributed to the guard, not the prompt"
    # The non-streamed entry point agrees with the streamed one.
    old = wired["eng"].complete_interviewer_reply(
        case_content=CASE, case_type="profitability", transcript=transcript,
        new_user_message=message, prior_state={}, control_out={})
    assert old.count("?") == 1


def test_production_logging_is_unchanged_when_no_arena_params_are_passed(wired):
    """With usage_out absent the engine must log exactly as it always has."""
    for _name, transcript, message, _fn in LANES:
        _production(wired["eng"], transcript, message)
    rows = wired["prod_logged"]
    assert rows, "production stopped logging usage"
    for row in rows:
        assert set(row) == {"user_id", "endpoint", "model", "response", "latency_ms"}
        assert row["user_id"] == "user-1"
        assert row["endpoint"] in {
            "/attempts/messages", "/attempts/messages/contextual",
            "/attempts/messages/fastlane_event", "/attempts/messages/fastlane_silence"}
    assert not wired["arena_logged"]


def test_arena_turns_never_log_as_production_traffic(wired):
    for _name, transcript, message, _fn in LANES:
        _arena(wired["arena"], transcript, message)
    assert not wired["prod_logged"], (
        "an arena turn was logged under a production endpoint: it would inflate the "
        "live interviewer's numbers in /admin/ai-usage")
    assert wired["arena_logged"], "arena spend disappeared from the usage log"
    for row in wired["arena_logged"]:
        assert row["endpoint"] == "/admin/arena/turn"
        assert row["model"] != "local"
        assert row["meta"]["arena_variant"] == "live"
        assert row["meta"]["engine_endpoint"].startswith("/attempts/messages")


def test_the_pinned_model_reaches_every_lane_that_calls_one(wired):
    """v1 pinned the deep lane only; presence lines silently used the live default
    while the arena reported the pinned model."""
    client = wired["client"]
    for _name, transcript, message, _fn in LANES:
        client.calls.clear()
        out = _arena(wired["arena"], transcript, message, model_mode="openai-4o")
        if out["model"] is None:
            assert not client.calls, "a model ran but the arena reported none"
            continue
        assert out["model"] == "gpt-4o"
        assert client.calls and all(c["model"] == "gpt-4o" for c in client.calls), \
            [c["model"] for c in client.calls]


def test_raw_text_is_reported_and_a_rewrite_is_attributed(wired):
    out = _arena(wired["arena"], LANES[1][1], LANES[1][2])  # no-question lane
    assert out["raw_reply"] and "<<" not in out["raw_reply"]
    if out["raw_reply"] != out["reply"]:
        assert out["enforced"] is True
    out_q = _arena(wired["arena"], LANES[0][1], LANES[0][2])  # streamed through the reply guard
    assert out_q["enforced"] is True, "V12.1: the stream guard trims praise and a stacked question"


# --------------------------------------------------------------- copied constants

def _attempts_source() -> str:
    return io.open("routes/attempts.py", encoding="utf-8").read()


def test_quota_ladder_matches_the_route_that_enforces_it():
    from routes import arena as ar
    tree = ast.parse(_attempts_source())
    live = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "CLARIFICATION_QUOTA" for t in node.targets):
            live = ast.literal_eval(node.value)
    assert live is not None, "CLARIFICATION_QUOTA moved out of routes/attempts.py"
    assert ar.CLARIFICATION_QUOTA == live


def test_opening_note_matches_the_row_production_seeds():
    from routes import arena as ar
    expr = ('f"Case ready: {case.get(\'title\')}. On a phone, tap the menu at the top left to read the full '
            'brief. Ask any clarifying questions before structuring."')
    src = _attempts_source()
    assert expr in src, "the system_note production seeds has changed: update routes/arena.py"
    assert '"kind": "system_note"' in src
    row = ar.opening_transcript({"title": "Spice Express"})[0]
    assert row == NOTE


@pytest.mark.parametrize("name,transcript,message,expect_fn", LANES, ids=[l[0] for l in LANES])
def test_the_frozen_main_arm_is_byte_identical_to_production(wired, name, transcript, message, expect_fn):
    """While main's prompt layer is unchanged, the FROZEN main arm -- its own
    interviewer prompt, packet rules and assessor -- must say exactly what production
    says. That is the proof that arm A is the existing interviewer, not a copy of it."""
    from prompts.variants import registry
    if registry.live_matches() != "baseline-main":
        pytest.skip("the deployed prompt layer is no longer main @ 5f66b3b")
    prod_text, _ = _production(wired["eng"], transcript, message)
    out = _arena(wired["arena"], transcript, message, variant_id="baseline-main")
    assert out["error"] is None, out["error"]
    assert out["reply"] == prod_text, f"{name}: frozen main says {out['reply']!r}, production {prod_text!r}"


# ----------------------------------------------------------------------- voice path

@pytest.mark.parametrize("name,transcript,message,expect_fn", LANES, ids=[l[0] for l in LANES])
def test_the_voice_path_says_exactly_what_the_voice_route_would(wired, name, transcript, message, expect_fn):
    """channel="voice": the engine's voice channel, then the SAME interpreter
    /voice-decision uses (services/voice_lane.py)."""
    from services.voice_lane import VoiceTurnError, interpret_voice_turn
    ctl = {}
    parts = list(wired["eng"].stream_interviewer_reply(
        case_content=CASE, case_type="profitability", transcript=transcript,
        new_user_message=message, user_id="user-1", teaching_policy=None,
        prior_state={}, control_out=ctl, channel="voice"))
    try:
        lane, say, _event = interpret_voice_turn(ctl, parts)
        route_error = None
    except VoiceTurnError as e:
        lane, say, route_error = None, None, str(e)
    out = wired["arena"].run_turn(
        case_content=CASE, case_type="profitability", variant_id="live",
        transcript=transcript, new_user_message=message, prior_state={},
        teaching_policy="coached", model_mode="openai-mini", user_id="admin-1", channel="voice")
    if route_error:
        assert out["error"] and route_error in out["error"]
        return
    assert out["error"] is None, out["error"]
    assert out["voice"]["lane"] == lane
    assert out["reply"] == (say or ""), f"{name}: arena says {out['reply']!r}, the voice route {say!r}"
    assert "{" not in out["reply"], "a fast-lane JSON event must be unpacked, not shown raw"
