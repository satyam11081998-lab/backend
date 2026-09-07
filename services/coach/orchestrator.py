"""
The ORCHESTRATOR loop — the heart of the Prep Copilot's agentic AI.

The loop is the multi-agent form of an agent loop. Its "actions" are whole
SPECIALIST agents in different domains rather than low-level tools:

    while not done and steps < budget:
        decision = PLAN(goal, what_i've_learned_so_far)   # planner reasons
        if decision == 'synthesize':                       # planner decides done
            return compose(findings)
        result = DELEGATE(decision.agent, decision.args)   # run a specialist
        findings += result                                 # observe, then plan again

Why this is *agentic AI* and not a fixed pipeline: the planner chooses, at
runtime and from what it has learned, WHICH specialists to deploy, in WHAT
order, with WHAT arguments, and WHEN to stop. A candidate weak on quant aiming
at a bank and a candidate weak on synthesis aiming at product get genuinely
different teams and different plans. The control flow belongs to the planner.

The loop is model-agnostic: it takes a `planner` callable. Production injects an
OpenAI-function-calling planner (live.py); the always-on demo injects a
deterministic planner (simulation.py). Identical loop code runs both — which is
how we prove the machinery is real independent of the model.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .schemas import CoachStep, CoachResult, SpecialistResult
from .specialists import DataAccess, run_specialist, SPECIALISTS

# planner(goal, history) -> decision
#   decision: {"action": "delegate", "agent": str, "args": {...}, "rationale": str}
#          or {"action": "synthesize", "summary": str, "rationale": str}
#   history: list of {"agent": str, "args": {...}, "result": {...}}
Planner = Callable[[str, List[Dict[str, Any]]], Dict[str, Any]]


def orchestrate(
    goal: str,
    planner: Planner,
    data: DataAccess,
    *,
    mode: str = "sim",
    max_steps: int = 8,
    on_step: Optional[Callable[[CoachStep], None]] = None,
    model: Optional[str] = None,
) -> CoachResult:
    """Run the plan -> delegate -> observe -> synthesize loop to completion."""
    steps: List[CoachStep] = []
    history: List[Dict[str, Any]] = []
    agents_used: List[str] = []
    idx = 0

    def emit(step: CoachStep) -> None:
        steps.append(step)
        if on_step:
            try:
                on_step(step)
            except Exception:  # noqa: BLE001 — a UI hook must never break the loop
                pass

    for _ in range(max_steps):
        try:
            decision = planner(goal, history) or {}
        except Exception as e:  # noqa: BLE001 — a planner error becomes a graceful synthesis
            idx += 1
            summary = _fallback_summary(goal, history)
            emit(CoachStep(index=idx, phase="synthesize", agent=None,
                           rationale=f"Planner error ({type(e).__name__}); composed from what I gathered.",
                           output={"summary": summary}))
            return CoachResult(goal=goal, mode=mode, summary=summary, steps=steps,
                               agents_used=agents_used, stopped_reason="planner_error", model=model)

        action = decision.get("action")

        if action == "synthesize" or not action:
            idx += 1
            summary = (decision.get("summary") or "").strip() or _fallback_summary(goal, history)
            emit(CoachStep(index=idx, phase="synthesize", agent=None,
                           rationale=decision.get("rationale") or "I have enough to build the plan.",
                           output={"summary": summary}))
            return CoachResult(goal=goal, mode=mode, summary=summary, steps=steps,
                               agents_used=agents_used, stopped_reason="synthesized", model=model)

        # action == "delegate"
        agent = decision.get("agent")
        args = decision.get("args") or {}
        rationale = decision.get("rationale") or f"Deploying {agent}."

        idx += 1
        emit(CoachStep(index=idx, phase="plan", agent=agent, rationale=rationale))

        result: SpecialistResult = run_specialist(data, agent, args)
        agents_used.append(agent)
        history.append({"agent": agent, "args": args, "result": result.to_dict()})

        idx += 1
        emit(CoachStep(index=idx, phase="observe", agent=agent, rationale=None, output=result.to_dict()))

    # Step-budget backstop: synthesise what we have rather than returning nothing.
    idx += 1
    summary = _fallback_summary(goal, history)
    emit(CoachStep(index=idx, phase="synthesize", agent=None,
                   rationale="Reached the step budget; composing from what I gathered.",
                   output={"summary": summary}))
    return CoachResult(goal=goal, mode=mode, summary=summary, steps=steps,
                       agents_used=agents_used, stopped_reason="max_steps", model=model)


def _fallback_summary(goal: str, history: List[Dict[str, Any]]) -> str:
    """Deterministic synthesis if the planner didn't supply one — stitches each
    specialist's headline + top recommendation into a plan narrative."""
    if not history:
        return f"I couldn't gather enough to build a plan for: {goal}"
    lines = [f"Your prep plan — {goal}", ""]
    for h in history:
        r = h.get("result", {})
        lines.append(f"\u2022 [{r.get('domain', '?')}] {r.get('headline', '')}")
        for a in (r.get("recommended_actions") or [])[:1]:
            lines.append(f"    \u2192 {a}")
    lines.append("")
    lines.append(f"Specialists deployed: {', '.join(h.get('agent', '?') for h in history) or 'none'}.")
    return "\n".join(lines)


def catalog() -> List[Dict[str, str]]:
    """The specialist roster, for the coach UI / info endpoint."""
    return [{"name": name, "label": label, "domain": domain}
            for name, (_fn, label, domain) in SPECIALISTS.items()]
