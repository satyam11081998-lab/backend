"""
The ORCHESTRATOR loop — the heart of MECE's agentic AI.

This is the multi-agent version of an agent loop. Instead of calling low-level
tools, the orchestrator's "actions" are whole SPECIALIST AGENTS in different
domains. The loop is:

    while not done and steps < budget:
        decision = PLAN(mission, what_i've_learned_so_far)   # planner reasons
        if decision == 'synthesize':                         # planner decides done
            return compose(findings)
        result = DELEGATE(decision.agent, decision.args)     # run a specialist
        findings += result                                   # observe, then plan again

Why this is *agentic AI* and not a fixed pipeline: the planner chooses, at
runtime and from what it has learned, WHICH specialists to deploy, in WHAT order,
with WHAT arguments, and WHEN to stop. Two missions — "run the daily brief" vs
"cut our AI costs" — send in completely different teams. The control flow is the
planner's, not ours.

The loop is model-agnostic: it takes a `planner` callable. Production injects an
OpenAI-function-calling planner (live.py); the always-on admin demo injects a
deterministic planner (simulation.py) so it works with no key, no budget, no data.
Identical loop code runs both — which is how you prove the machinery is real
independent of the model.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .schemas import OrchestrationStep, OrchestrationResult, SpecialistResult
from .specialists import DataAccess, run_specialist, SPECIALISTS


# A planner decision. `action` is either "delegate" (then agent+args) or
# "synthesize" (then summary). The planner returns this given the mission and the
# results gathered so far.
PlannerDecision = Dict[str, Any]

# planner(mission, history) -> PlannerDecision
#   history: list of {"agent": str, "result": dict}
Planner = Callable[[str, List[Dict[str, Any]]], PlannerDecision]


def orchestrate(
    mission: str,
    planner: Planner,
    data: DataAccess,
    *,
    mode: str = "sim",
    max_steps: int = 6,
    on_step: Optional[Callable[[OrchestrationStep], None]] = None,
    model: Optional[str] = None,
) -> OrchestrationResult:
    """Run the plan -> delegate -> observe -> synthesize loop to completion."""
    steps: List[OrchestrationStep] = []
    history: List[Dict[str, Any]] = []
    agents_used: List[str] = []
    idx = 0

    def emit(step: OrchestrationStep) -> None:
        steps.append(step)
        if on_step:
            on_step(step)

    for _ in range(max_steps):
        decision = planner(mission, history)
        action = (decision or {}).get("action")

        if action == "synthesize" or not action:
            idx += 1
            summary = (decision or {}).get("summary") or _fallback_summary(mission, history)
            emit(OrchestrationStep(index=idx, phase="synthesize", agent=None,
                                   rationale=(decision or {}).get("rationale") or "I have enough to answer.",
                                   output={"summary": summary}))
            return OrchestrationResult(
                mission=mission, mode=mode, summary=summary, steps=steps,
                agents_used=agents_used, stopped_reason="synthesized", model=model,
            )

        # action == "delegate"
        agent = decision.get("agent")
        args = decision.get("args") or {}
        rationale = decision.get("rationale") or f"Deploying {agent}."

        idx += 1
        emit(OrchestrationStep(index=idx, phase="plan", agent=agent, rationale=rationale))

        result: SpecialistResult = run_specialist(data, agent, args)
        agents_used.append(agent)
        history.append({"agent": agent, "args": args, "result": result.to_dict()})

        idx += 1
        emit(OrchestrationStep(index=idx, phase="observe", agent=agent, rationale=None,
                               output=result.to_dict()))

    # Budget backstop. Synthesise whatever we have rather than returning nothing.
    idx += 1
    summary = _fallback_summary(mission, history)
    emit(OrchestrationStep(index=idx, phase="synthesize", agent=None,
                           rationale="Reached the step budget; composing from what I gathered.",
                           output={"summary": summary}))
    return OrchestrationResult(
        mission=mission, mode=mode, summary=summary, steps=steps,
        agents_used=agents_used, stopped_reason="max_steps", model=model,
    )


def _fallback_summary(mission: str, history: List[Dict[str, Any]]) -> str:
    """Deterministic synthesis if the planner didn't supply one — stitches each
    specialist's headline + top recommendation into an executive brief."""
    if not history:
        return f"No specialist produced a usable result for: {mission}"
    lines = [f"Executive brief — {mission}", ""]
    for h in history:
        r = h.get("result", {})
        lines.append(f"• [{r.get('domain','?')}] {r.get('headline','')}")
        for a in (r.get("recommended_actions") or [])[:1]:
            lines.append(f"    → {a}")
    return "\n".join(lines)


def catalog() -> List[Dict[str, str]]:
    """The specialist roster, for the admin UI / info endpoint."""
    return [{"name": name, "label": label, "domain": domain}
            for name, (_fn, label, domain) in SPECIALISTS.items()]
