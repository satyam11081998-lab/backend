"""
Typed structures for MECE's *agentic AI* layer.

Vocabulary note (this is the distinction you asked for):
  * an "AI agent"  = ONE autonomous, tool-using entity.
  * "agentic AI"   = a SYSTEM that exhibits agency by PLANNING and ORCHESTRATING
                     several specialist agents across DIFFERENT DOMAINS, deciding
                     at runtime which to deploy, reading their results, and
                     synthesising an outcome.

This module is the second kind. The orchestrator is a planner that delegates to
domain specialists (growth analytics, content strategy, curriculum design, FinOps
cost control, deck retrieval) and composes their findings. Every run emits a full,
inspectable trace — the plan, each delegation, each observation, the synthesis —
because an orchestrator you cannot audit is a liability, not a feature.
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
class OrchestrationStep:
    """One move by the orchestrator. `phase` makes the trace legible in the UI:
        plan      -> the orchestrator reasons about what to do next
        delegate  -> it invokes a specialist agent (the ACTION)
        observe   -> it records what the specialist returned
        synthesize-> it composes the final answer and stops
    """
    index: int
    phase: str
    agent: Optional[str]
    rationale: Optional[str]
    output: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OrchestrationResult:
    mission: str
    mode: str                     # "sim" | "live"
    summary: str                  # the synthesised executive answer
    steps: List[OrchestrationStep] = field(default_factory=list)
    agents_used: List[str] = field(default_factory=list)
    stopped_reason: str = "synthesized"   # synthesized | max_steps
    ok: bool = True
    error: Optional[str] = None
    model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["steps"] = [s.to_dict() for s in self.steps]
        return d
