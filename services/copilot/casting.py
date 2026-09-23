"""
Prep Copilot v2 - CASTING.

Turns a Pack + a chosen Scenario into the inputs the ISOLATED interview engine
runs: the case_content the interviewer works from, and a role/company emphasis
block that casts WHAT it asks (never how it adaptively behaves - that logic is the
copied engine's, tweakable in services/copilot/engine/). The engine copy accepts
these as ordinary inputs, exactly like the live engine accepts a case prompt.
"""
from __future__ import annotations

from typing import Optional

from .schemas import Pack, Scenario


def pick_scenario(pack: Pack, focus: str = "", difficulty: str = "") -> Optional[Scenario]:
    """Choose a scenario from the pack, optionally biased to a focus dimension /
    difficulty. Returns None only if the pack has no scenarios."""
    scen = pack.scenarios or []
    if not scen:
        return None
    pool = scen
    if focus:
        f = [s for s in pool if s.focus == focus]
        if f:
            pool = f
    if difficulty:
        d = [s for s in pool if (s.difficulty or "").lower() == difficulty.lower()]
        if d:
            pool = d
    return pool[0]


def to_case_content(pack: Pack, scenario: Scenario) -> str:
    """The scenario text the interviewer engine works from - the same role the
    case prompt plays in the live engine, so the copied engine needs no new wiring."""
    lines = [scenario.prompt.strip()]
    if scenario.numerical_ask:
        lines.append(f"\n**Quantitative ask:** {scenario.numerical_ask}")
    if (pack.assessment or {}).get("numericals_expected"):
        lines.append("\n(The interviewer should expect and press for the numbers this role lives on.)")
    return "\n".join(lines).strip()


def role_emphasis_block(pack: Pack, scenario: Scenario) -> str:
    """A compact instruction the copied engine can prepend to its interviewer
    system prompt to CAST the session by role/company - what to probe and value,
    grounded in the pack. Does not alter the engine's adaptive decision logic."""
    dims = ", ".join(d.label for d in pack.rubric.dimensions[:6]) or "the role's core competencies"
    fw = "; ".join(f.name for f in pack.frameworks[:6])
    tgt = pack.display_role + (f" at {pack.display_company}" if pack.display_company else "")
    parts = [
        f"ROLE-CAST: You are interviewing for {tgt}. Press on what THIS role is hired for: {dims}.",
        f"Reward correct APPLICATION of the role's real frameworks ({fw}); never reward name-drops." if fw else "",
        "This role expects numerical problem-solving - if the candidate hand-waves the math, ask them to actually compute it."
        if (pack.assessment or {}).get("numericals_expected") else "",
        f"Scenario focus: {scenario.focus}." if scenario.focus else "",
    ]
    return "\n".join(p for p in parts if p)
