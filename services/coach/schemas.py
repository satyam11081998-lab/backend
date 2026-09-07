"""
Typed structures for the Prep Copilot's agentic layer.

Kept deliberately separate from services/agentic/schemas.py so the two
orchestrators can evolve independently and a change to one can never break the
other. Every run emits a full, inspectable trace (plan -> delegate -> observe ->
synthesize) because an orchestrator you cannot audit is a liability, not a
feature — and because the trace is exactly what the UI streams so a user (or a
watching PM) can literally watch the system think.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class SpecialistResult:
    """What a single domain specialist hands back to the orchestrator."""
    domain: str
    headline: str
    findings: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    recommended_actions: List[str] = field(default_factory=list)
    ok: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CoachStep:
    """One move by the orchestrator.

    phase:
        plan       -> the orchestrator decides which specialist to deploy next
        observe    -> it records what that specialist returned
        synthesize -> it composes the final personalised plan and stops
    """
    index: int
    phase: str
    agent: Optional[str]
    rationale: Optional[str]
    output: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CoachResult:
    goal: str
    mode: str                     # "sim" | "live"
    summary: str                  # the synthesised, human-readable plan narrative
    steps: List[CoachStep] = field(default_factory=list)
    agents_used: List[str] = field(default_factory=list)
    stopped_reason: str = "synthesized"   # synthesized | max_steps
    ok: bool = True
    error: Optional[str] = None
    model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["steps"] = [s.to_dict() for s in self.steps]
        return d
