"""Adversarial checks on the arena runner: isolation, fidelity and cost visibility.

The failure modes that matter here are silent ones:
  * an arena turn quietly writing to live user tables
  * the two arms sharing learner state, which voids every comparison
  * arena spend invisible to the daily kill switch
  * the arena drifting from the production turn and measuring a fiction
Each gets a test. The model is faked, so this suite is free and deterministic.
"""
from __future__ import annotations

import ast
import io
import os

import pytest

os.environ["ADAPTIVE_INTERVIEWER"] = "true"


class _Usage:
    prompt_tokens = 900
    completion_tokens = 40
    total_tokens = 940


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    id = "fake-1"

    def __init__(self, content):
        self.choices = [_Choice(content)]
        self.usage = _Usage()


class _FakeCompletions:
    def __init__(self, content, sink):
        self._content, self._sink = content, sink

    def create(self, **kwargs):
        self._sink.append(kwargs)
        if kwargs.get("stream"):
            # The arena now runs the production STREAMING path, which asks for a
            # stream on question-allowed turns.
            c = self._content
            toks = [c[i:i + 7] for i in range(0, len(c), 7)]
            chunks = [type("K", (), {"choices": [type("Ch", (), {"delta": _Msg(t)})()],
                                     "usage": None, "id": "fake-1"})() for t in toks]
            chunks.append(type("K", (), {"choices": [], "usage": _Usage(), "id": "fake-1"})())
            return iter(chunks)
        return _Resp(self._content)


class _FakeClient:
    def __init__(self, content, sink):
        self.chat = type("C", (), {"completions": _FakeCompletions(content, sink)})()


REPLY = "<<mode=coach; intervention=micro_hint; materiality=material; hint=1>>\n\nYour monthly and annual figures don't reconcile - which unit are you sizing in?"

TRANSCRIPT = [
    {"role": "assistant", "content": "Where would you start?"},
    {"role": "user", "content": "I'd split revenue into price and volume, then look at delivery mix."},
]
CASE = "Spice Express sales fell 12% YoY.\n- 240 outlets\n- AOV Rs 420"

# V11 routes most turns to a DETERMINISTIC lane that builds no prompt at all. Only a
# turn whose render is "model" exercises the variant, so the suite uses one.
MODEL_TURN = "help na i'm not getting this at all"
# V12.1: in chat a turn with a number in it is answered in its own terms (model-worded),
# so the fast lane is exercised with a mid-thought turn instead.
FAST_LANE_TURN = "hold on, let me think..."


@pytest.fixture()
def arena(monkeypatch):
    """Arena wired to a fake model: no network, no spend, deterministic reply."""
    from services import arena as arena_mod
    from services import interview_engine as eng

    sink = []
    client = _FakeClient(REPLY, sink)
    monkeypatch.setattr(arena_mod, "openai_client", lambda: client)
    monkeypatch.setattr(eng, "openai_client", lambda: client)
    monkeypatch.setattr(eng, "_resolve_adaptive_llm", lambda: (client, "gpt-4o-mini", "openai"))
    # The contextual assessor is a separate model call; stub it so the test stays
    # about the turn itself.
    monkeypatch.setattr(eng, "assess_context_with_llm", lambda *a, **k: {})
    logged = []
    monkeypatch.setattr(arena_mod, "log_ai_usage", lambda **kw: logged.append(kw))
    arena_mod._test_sink, arena_mod._test_logged = sink, logged
    return arena_mod


def _run(arena, variant_id="patched-2026-10-14", prior_state=None, policy="coached",
         message=None):
    return arena.run_turn(
        case_content=CASE, case_type="profitability", variant_id=variant_id,
        transcript=TRANSCRIPT, new_user_message=message or MODEL_TURN,
        prior_state=prior_state, teaching_policy=policy, model_mode="openai-mini",
        user_id="admin-uuid",
    )


def _code_without_docstrings(path: str) -> str:
    """Source with every docstring removed, so prose that merely NAMES a forbidden
    table cannot pass or fail a check about code."""
    tree = ast.parse(io.open(path, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(getattr(body[0], "value", None), ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def test_arena_module_never_touches_live_user_tables():
    """Static proof: the runner cannot create an attempt or persist a profile."""
    tree = ast.parse(io.open("services/arena.py", encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert not any("supabase" in m for m in imported)
    code = _code_without_docstrings("services/arena.py")
    for forbidden in ("attempt_messages", "attempts", "get_supabase_client",
                      "user_skill_profile", ".insert(", ".delete("):
        assert forbidden not in code, forbidden
    # The ONLY write the arena is allowed is its own usage log.
    assert "log_ai_usage" in code


def test_turn_runs_end_to_end_and_strips_the_control_tag(arena):
    out = _run(arena)
    assert out["error"] is None
    assert out["reply"]
    # The control tag must never reach the candidate, in the shipped OR the raw text.
    for field in ("reply", "raw_reply"):
        assert "<<" not in out[field] and "mode=" not in out[field]
    assert out["control"]["intervention"] == "micro_hint"
    assert out["control"]["materiality"] == "material"
    assert out["prompt_consulted"] is True
    assert out["metrics"]["chars"] > 0
    assert out["latency_ms"] >= 0
    assert out["model"] == "gpt-4o-mini"


def test_a_fast_lane_turn_is_flagged_as_not_consulting_the_prompt(arena):
    """The trap this flag exists to stop: V11 answers many turns deterministically,
    both arms then emit identical text, and that reads as 'the prompts are the same'."""
    out = _run(arena, message=FAST_LANE_TURN)
    assert out["prompt_consulted"] is False
    assert out["error"] is None
    # No prompt was built, so no model was called and nothing was spent.
    assert not arena._test_logged


def test_enforcement_is_attributed_to_the_enforcer_not_the_prompt(arena):
    """enforce_mode can replace a reply outright. Without raw-vs-shipped you would
    credit (or blame) the prompt variant for the enforcer's rewrite."""
    out = _run(arena)
    assert out["raw_reply"], "raw model text must be captured"
    if out["raw_reply"] != out["reply"]:
        assert out["enforced"] is True


def test_arena_spend_is_logged_so_the_kill_switch_can_see_it(arena):
    _run(arena)
    assert arena._test_logged, "arena spend was not logged: it would be invisible to the budget cap"
    row = arena._test_logged[-1]
    assert row["endpoint"] == "/admin/arena/turn"
    assert row["response"].usage.total_tokens == 940  # real counts, not an estimate
    assert row["meta"]["arena_variant"] == "patched-2026-10-14"


def test_each_arm_keeps_its_own_state_and_the_input_is_not_mutated(arena):
    state_a = {"hint_level": 2, "profile": {"marker": "arm-a"}}
    snapshot = {"hint_level": 2, "profile": {"marker": "arm-a"}}
    out_a = _run(arena, prior_state=state_a)
    # Arm A's own dict must come back untouched: a shared mutable state object would
    # let one arm drive the other's hint ladder and void the comparison.
    assert state_a == snapshot
    assert out_a["next_state"] is not state_a
    out_b = _run(arena, variant_id="baseline-main", prior_state=None)
    assert out_b["next_state"] is not out_a["next_state"]


def test_the_two_arms_receive_different_system_prompts(arena):
    """Fidelity: the variant really does reach the model, per arm."""
    arena._test_sink.clear()
    _run(arena, variant_id="baseline-main")
    _run(arena, variant_id="patched-2026-10-14")
    systems = [
        next(m["content"] for m in call["messages"] if m["role"] == "system")
        for call in arena._test_sink
    ]
    assert len(systems) >= 2
    assert systems[0] != systems[-1]
    assert "warm, human interviewer" in systems[0]
    assert "MECE's AI interviewer" in systems[-1]


def test_violations_are_reported_for_an_offending_reply(arena, monkeypatch):
    from services import interview_engine as eng
    bad = ("Great job, perfect! That isn't specified in the prompt. Start with volume. Then price. Then mix. "
           "Then outlets. Then AOV. Then margin. What's the population? And how many per household?")
    client = _FakeClient(bad, [])
    monkeypatch.setattr(eng, "_resolve_adaptive_llm", lambda: (client, "gpt-4o-mini", "openai"))
    monkeypatch.setattr(arena, "openai_client", lambda: client)
    out = _run(arena)
    # MODEL_TURN routes to a no-question function, where production ALSO runs the
    # sanitiser (enforce_mode) -- so here the gush is legitimately gone from the
    # shipped text. V12.1: production also regenerates a "not specified" refusal once
    # and then drops the refusing sentence, so it never ships either.
    assert "Great job" not in out["reply"], "the sanitiser should have removed the gush"
    assert "isn't specified" not in out["reply"], "a refusal must not ship"
    assert "too_long" in out["violations"], "violations are still reported for what ships"
    # The gush, the refusal and the double question survive in the raw model text,
    # which is why the arena shows both.
    assert "Great job" in out["raw_reply"]
    assert "isn't specified" in out["raw_reply"]
    assert out["raw_reply"].count("?") >= 2


def test_a_failing_arm_degrades_instead_of_killing_the_comparison(arena, monkeypatch):
    from services import interview_engine as eng

    class _Boom:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise RuntimeError("provider down")

    monkeypatch.setattr(eng, "_resolve_adaptive_llm", lambda: (_Boom(), "gpt-4o-mini", "openai"))
    monkeypatch.setattr(eng, "openai_client", lambda: _Boom())
    monkeypatch.setattr(arena, "openai_client", lambda: _Boom())
    out = _run(arena)
    assert out["error"] and out["reply"] == ""
    assert out["violations"] == []


def test_unknown_variant_is_refused_before_any_model_spend(arena):
    with pytest.raises(KeyError):
        _run(arena, variant_id="does-not-exist")
    assert not arena._test_logged


# ------------------------------------------------- the whole prompt layer, per arm

def test_each_arm_briefs_the_model_with_its_own_packet_rules(arena):
    """v1 built the packet rules from the deployed module for BOTH arms, so the
    branch arm never ran the branch's rules."""
    from prompts.variants import registry
    ids = ("baseline-main", "patched-2026-10-14")
    for vid in ids:
        arena._test_sink.clear()
        _run(arena, variant_id=vid)
        own = registry.get_variant(vid)["module"].PACKET_RULES
        other = registry.get_variant(ids[1] if vid == ids[0] else ids[0])["module"].PACKET_RULES
        systems = [m["content"] for call in arena._test_sink
                   for m in call["messages"] if m["role"] == "system"]
        assert any("INTERVIEWER CONTROL PACKET" in s and own in s for s in systems), vid
        assert not any(other in s for s in systems), f"{vid} was briefed with the other arm's rules"


def test_each_arm_runs_its_own_context_assessor(arena, monkeypatch):
    """The assessor feeds the response-function DECISION, not just the wording. v1 ran
    the deployed assessor for both arms."""
    from prompts.variants import registry
    from services import interview_engine as eng
    monkeypatch.setattr(eng, "needs_contextual_assessment", lambda sig: True)
    for vid in ("baseline-main", "patched-2026-10-14"):
        arena._test_sink.clear()
        _run(arena, variant_id=vid)
        calls = [c for c in arena._test_sink if c.get("response_format")]
        assert calls, f"{vid}: its assessor never ran"
        assert calls[0]["messages"][0]["content"] == registry.get_variant(vid)["module"]._ASSESSOR_PROMPT
