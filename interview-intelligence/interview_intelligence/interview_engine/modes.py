"""Mode configuration engine (spec §14–§16, §86–§88).

Interview type, depth, difficulty and duration are COMPOSABLE parameters. A mode is data
(section weights + behaviour toggles); difficulty and depth are vectors; the blueprint
builder multiplies them with the role family's section bias. No mode is a hard-coded flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

SECTION_KINDS = ["intro", "cv", "functional", "technical", "behavioral", "situational", "case", "company",
                 "motivation", "closing"]

SECTION_TITLES = {
    "intro": "Introduction", "cv": "CV & experience", "functional": "Functional knowledge",
    "technical": "Technical", "behavioral": "Behavioral", "situational": "Situational judgment",
    "case": "Case / problem solving", "company": "Company & role", "motivation": "Motivation",
    "closing": "Closing",
}


@dataclass(frozen=True)
class Mode:
    id: str
    label: str
    weights: Dict[str, float]
    pushback_bonus: float = 0.0
    probe_bonus: int = 0
    claim_focus: bool = False
    requires_history: bool = False
    description: str = ""


MODES: Dict[str, Mode] = {m.id: m for m in [
    Mode("cv_jd", "CV + JD", {"cv": 3, "functional": 2, "behavioral": 1.5, "situational": 1, "motivation": 0.6},
         description="Your experience against the role's requirements."),
    Mode("cv_deep_dive", "CV deep dive", {"cv": 6, "functional": 1, "behavioral": 1}, probe_bonus=1, claim_focus=True,
         description="Investigates your CV claims in depth."),
    Mode("hr_behavioral", "HR / behavioral", {"behavioral": 5, "situational": 2, "motivation": 1.5},
         description="Motivation, ownership, conflict, failure, teamwork, self-awareness."),
    Mode("functional", "Functional", {"functional": 6, "situational": 1, "cv": 1},
         description="Deep functional questions for the role."),
    Mode("technical", "Technical", {"technical": 6, "cv": 1.5}, description="Role-specific technical assessment."),
    Mode("situational", "Situational", {"situational": 6, "behavioral": 1}, description="Realistic job situations."),
    Mode("case", "Case / problem solving", {"case": 6, "functional": 1}, description="Role-specific cases."),
    Mode("mixed", "Mixed", {"cv": 2, "functional": 2, "technical": 1.5, "behavioral": 2, "situational": 1.5, "case": 1},
         description="A balanced interview across dimensions."),
    Mode("company_simulation", "Company + role simulation", {"company": 3, "functional": 2, "situational": 2, "motivation": 1},
         description="Uses the company, the JD and your CV together."),
    Mode("hiring_manager", "Hiring manager", {"situational": 3, "cv": 2, "functional": 2, "motivation": 1},
         description="Business judgment, ownership and prioritisation."),
    Mode("final_round", "Final round", {"cv": 1.5, "functional": 2, "technical": 1.5, "behavioral": 2, "situational": 1.5,
                                         "case": 1, "motivation": 1}, description="Broad assessment across major competencies."),
    Mode("cv_attack", "CV attack / defense", {"cv": 9}, probe_bonus=2, claim_focus=True, pushback_bonus=0.1,
         description="Every major CV claim, probed."),
    Mode("weakness_targeting", "Weakness targeting", {"functional": 2, "behavioral": 2, "situational": 2, "cv": 1,
                                                       "technical": 1, "case": 1},
         requires_history=True, description="Focuses on competencies flagged in earlier interviews."),
    Mode("technical_deep_dive", "Technical deep dive", {"technical": 8, "functional": 1}, probe_bonus=2,
         description="Very deep functional or technical assessment."),
    Mode("stress", "Stress / pressure", {"cv": 2, "functional": 2, "situational": 3, "behavioral": 1}, pushback_bonus=0.3,
         description="Professional pushback, constraints and forced choices."),
    Mode("grill", "Grill mode", {"cv": 3, "functional": 3, "situational": 2, "behavioral": 1}, probe_bonus=2,
         pushback_bonus=0.2, claim_focus=True, description="High scrutiny, high depth, low tolerance for vague answers."),
]}

DIFFICULTY_LEVELS = ["easy", "medium", "hard", "expert", "grill"]
DEPTHS = ["standard", "deep", "extreme"]


class DifficultyVector(BaseModel):
    conceptual: int = 3
    reasoning: int = 3
    ambiguity: int = 2
    technical_depth: int = 3
    follow_up_intensity: int = 2
    time_pressure: int = 2
    behavioral_depth: int = 3
    role_specificity: int = 3
    pushback_rate: float = 0.0
    guidance: int = 2

    @property
    def level(self) -> int:
        return max(1, min(5, round((self.conceptual + self.reasoning + self.technical_depth) / 3)))

    @property
    def max_probes(self) -> int:
        return {1: 1, 2: 2, 3: 2, 4: 3, 5: 4}[max(1, min(5, self.follow_up_intensity))]


_BASE = {
    "easy":   dict(conceptual=2, reasoning=2, ambiguity=1, technical_depth=2, follow_up_intensity=1, time_pressure=1,
                   behavioral_depth=2, role_specificity=3, pushback_rate=0.0, guidance=3),
    "medium": dict(conceptual=3, reasoning=3, ambiguity=2, technical_depth=3, follow_up_intensity=2, time_pressure=2,
                   behavioral_depth=3, role_specificity=3, pushback_rate=0.05, guidance=2),
    "hard":   dict(conceptual=4, reasoning=4, ambiguity=3, technical_depth=4, follow_up_intensity=3, time_pressure=3,
                   behavioral_depth=4, role_specificity=4, pushback_rate=0.15, guidance=1),
    "expert": dict(conceptual=5, reasoning=5, ambiguity=4, technical_depth=5, follow_up_intensity=3, time_pressure=3,
                   behavioral_depth=4, role_specificity=5, pushback_rate=0.2, guidance=1),
    "grill":  dict(conceptual=4, reasoning=4, ambiguity=3, technical_depth=4, follow_up_intensity=5, time_pressure=4,
                   behavioral_depth=5, role_specificity=4, pushback_rate=0.35, guidance=0),
}


def difficulty_vector(difficulty: str, depth: str, mode: Mode) -> DifficultyVector:
    v = dict(_BASE.get(difficulty, _BASE["medium"]))
    bump = {"standard": 0, "deep": 1, "extreme": 2}.get(depth, 0)
    v["follow_up_intensity"] = min(5, v["follow_up_intensity"] + bump + mode.probe_bonus)
    v["technical_depth"] = min(5, v["technical_depth"] + bump)
    v["behavioral_depth"] = min(5, v["behavioral_depth"] + (1 if bump else 0))
    v["pushback_rate"] = min(0.6, v["pushback_rate"] + mode.pushback_bonus)
    if mode.id == "stress":
        v["time_pressure"] = min(5, v["time_pressure"] + 1)
    return DifficultyVector(**v)


def persona(difficulty: str, mode: Mode) -> dict:
    if difficulty == "grill" or mode.id == "grill":
        style = ("Demanding and precise. Treats every vague statement as unfinished. Asks for baselines, numbers, "
                 "personal contribution and evidence. Professional, never hostile.")
    elif mode.id == "stress":
        style = ("Calm but sceptical. Applies pressure with constraints, forced choices and 'I'm not convinced'. "
                 "Professional, never rude.")
    elif difficulty in ("hard", "expert"):
        style = "Crisp, senior and efficient. Minimal small talk. Pushes for depth and trade-offs."
    elif difficulty == "easy":
        style = "Warm and encouraging in tone (without praising answers). Rephrases if the candidate is stuck."
    else:
        style = "Professional, neutral and attentive. Natural conversational rhythm."
    return {"style": style, "acknowledgement": "rare" if difficulty in ("hard", "expert", "grill") else "occasional"}


class InterviewConfig(BaseModel):
    """What the user chose (spec §88). Validated server-side; nothing here grants access."""
    model_config = ConfigDict(extra="ignore")
    mode: str = "mixed"
    depth: str = "standard"
    difficulty: str = "medium"
    duration_minutes: int = 45
    focus_areas: List[str] = Field(default_factory=list)
    target_competencies: List[str] = Field(default_factory=list)
    company_name: str = ""
    company_notes: str = ""
    voice: bool = False

    @field_validator("mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        if v not in MODES:
            raise ValueError(f"unknown mode '{v}'")
        return v

    @field_validator("depth")
    @classmethod
    def _depth(cls, v: str) -> str:
        if v not in DEPTHS:
            raise ValueError(f"unknown depth '{v}'")
        return v

    @field_validator("difficulty")
    @classmethod
    def _difficulty(cls, v: str) -> str:
        if v not in DIFFICULTY_LEVELS:
            raise ValueError(f"unknown difficulty '{v}'")
        return v

    @field_validator("focus_areas", "target_competencies")
    @classmethod
    def _short_list(cls, v: List[str]) -> List[str]:
        return [s.strip()[:80] for s in v if s and s.strip()][:8]

    @field_validator("company_name")
    @classmethod
    def _company(cls, v: str) -> str:
        return (v or "").strip()[:120]

    @field_validator("company_notes")
    @classmethod
    def _notes(cls, v: str) -> str:
        return (v or "").strip()[:3000]


@dataclass
class SectionPlan:
    kind: str
    budget_s: int
    weight: float
    competency_ids: List[str] = field(default_factory=list)
    n_items: int = 0


def intro_closing_budget(duration_s: int) -> tuple[int, int]:
    if duration_s <= 15 * 60:
        return 60, 60
    return 90, 150


def allocate_sections(cfg: InterviewConfig, family_bias: Dict[str, float], technical_role: bool,
                      case_friendly: bool, has_company_context: bool, avg_item_s: int) -> List[SectionPlan]:
    mode = MODES[cfg.mode]
    total = cfg.duration_minutes * 60
    intro_s, closing_s = intro_closing_budget(total)
    main = max(300, total - intro_s - closing_s)
    weights: Dict[str, float] = {}
    for kind, w in mode.weights.items():
        k = kind
        if k == "technical" and not technical_role:
            # "Technical" for a non-technical role means deep role knowledge, not coding.
            k = "functional"
        if k == "case" and not case_friendly and cfg.mode not in ("case",):
            k = "situational"
        if k == "company" and not has_company_context:
            k = "functional"
        bias = family_bias.get(k, 1.0) if cfg.mode not in ("hr_behavioral", "cv_attack", "cv_deep_dive", "case",
                                                         "technical", "technical_deep_dive", "situational",
                                                         "functional") else 1.0
        weights[k] = weights.get(k, 0.0) + w * (bias if bias > 0 else 0.0)
    weights = {k: v for k, v in weights.items() if v > 0.05}
    s = sum(weights.values()) or 1.0
    plans = [SectionPlan("intro", intro_s, 0)]
    order = [k for k in ["cv", "functional", "technical", "case", "company", "behavioral", "situational", "motivation"]
             if k in weights]
    for k in order:
        budget = int(main * weights[k] / s)
        n = max(1, round(budget / max(avg_item_s, 90)))
        plans.append(SectionPlan(k, budget, weights[k], n_items=n))
    plans.append(SectionPlan("closing", closing_s, 0))
    return plans
