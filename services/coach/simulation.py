"""
Simulation engine — the ALWAYS-WORKS demo brain for the Prep Copilot.

It runs the REAL orchestrator loop against a deterministic planner and rich
sample candidates, so the demo works with no OpenAI key, no budget left, and no
rows in the database. It is honest about what it is (mode="sim"): the planner
here is scripted, but the loop, the delegation, the specialist code and the
multi-agent structure are the same objects the live path uses. The demo proves
the machinery; live mode swaps the scripted planner for the model.

The scripted planner is still *adaptive*: it reads what the specialists returned
so far (e.g. the weakest dimension the diagnostician found, the priority the
target strategist derived) and feeds those into the next specialist's arguments
— so different candidates produce genuinely different teams, orders and plans.
That divergence is exactly what you demo on stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .specialists import DataAccess


# ---------------------------------------------------------------------------
# Sample candidates — deliberately different so the paths visibly diverge.
# ---------------------------------------------------------------------------
def _sub(structure, quant, synth, judg, crea, pres, score):
    return {"score": score, "feedback_json": {"breakdown": {
        "structure": structure, "quantitative": quant, "synthesis": synth,
        "business_judgment": judg, "creativity": crea, "presence": pres}}}


@dataclass
class SampleCandidate:
    key: str
    name: str
    goal: str
    target_company: str
    domain: str
    weekly_hours: int
    submissions: List[Dict[str, Any]] = field(default_factory=list)
    headlines: List[Dict[str, Any]] = field(default_factory=list)
    bank: List[Dict[str, Any]] = field(default_factory=list)
    decks: List[Dict[str, Any]] = field(default_factory=list)


_HEADLINES = [
    {"title": "RBI holds repo rate as inflation cools", "gd_worthiness_score": 91, "category": "macro", "source_name": "Sample Wire"},
    {"title": "Quick-commerce burns cash chasing 10-minute delivery", "gd_worthiness_score": 88, "category": "business", "source_name": "Sample Wire"},
    {"title": "India's data-centre boom strains the power grid", "gd_worthiness_score": 84, "category": "tech", "source_name": "Sample Wire"},
    {"title": "MBA hiring shifts toward AI-fluent generalists", "gd_worthiness_score": 82, "category": "jobs", "source_name": "Sample Wire"},
]

_BANK = [
    {"id": "sample-c1", "title": "Market entry: EV two-wheelers in Tier-2 India", "type": "case", "difficulty": "hard"},
    {"id": "sample-c2", "title": "Profitability turnaround: regional QSR chain", "type": "case", "difficulty": "medium"},
    {"id": "sample-g1", "title": "How many EV chargers does Bengaluru need by 2030?", "type": "guesstimate", "difficulty": "medium"},
    {"id": "sample-g2", "title": "Daily coffee cups sold in Mumbai", "type": "guesstimate", "difficulty": "easy"},
]

_DECKS = [
    {"title": "Market entry: EV two-wheelers", "competition": "IIM-A Case Cup", "result": "Winner"},
    {"title": "Profitability turnaround: QSR chain", "competition": "BCG Ace", "result": "Finalist"},
    {"title": "Pricing strategy: OTT bundling", "competition": "Bain BLC", "result": "National Finalist"},
]

SAMPLE_CANDIDATES: Dict[str, SampleCandidate] = {
    "aarav": SampleCandidate(
        key="aarav", name="Aarav (weak on quant, targeting a bank)",
        goal="Break into investment banking — fix my numbers under pressure.",
        target_company="Goldman Sachs", domain="finance / IB", weekly_hours=6,
        submissions=[_sub(22, 12, 16, 12, 7, 8, 64), _sub(23, 11, 17, 12, 7, 8, 63),
                     _sub(21, 13, 15, 11, 6, 7, 61), _sub(22, 12, 16, 12, 7, 8, 62),
                     _sub(23, 14, 17, 13, 7, 8, 66), _sub(22, 13, 16, 12, 7, 8, 65)],
        headlines=_HEADLINES, bank=_BANK, decks=_DECKS,
    ),
    "diya": SampleCandidate(
        key="diya", name="Diya (weak on synthesis, targeting MBB)",
        goal="Get into McKinsey — my answers ramble, I need to lead with the point.",
        target_company="McKinsey", domain="consulting / strategy", weekly_hours=5,
        submissions=[_sub(23, 18, 9, 11, 7, 8, 66), _sub(22, 17, 8, 10, 7, 7, 62),
                     _sub(24, 18, 10, 12, 7, 8, 68), _sub(23, 17, 9, 11, 6, 7, 64),
                     _sub(22, 18, 8, 11, 7, 8, 63), _sub(24, 19, 10, 12, 7, 8, 69)],
        headlines=_HEADLINES, bank=_BANK, decks=_DECKS,
    ),
    "kabir": SampleCandidate(
        key="kabir", name="Kabir (no graded attempts yet)",
        goal="Explore product management — not sure where I stand.",
        target_company="Product management", domain="tech / product", weekly_hours=4,
        submissions=[], headlines=_HEADLINES, bank=_BANK, decks=_DECKS,
    ),
}


class SampleUserData(DataAccess):
    """DataAccess over a single sample candidate — no DB, no key, always works."""

    def __init__(self, cand: SampleCandidate):
        self.c = cand

    def profile(self) -> Dict[str, Any]:
        return {"name": self.c.name, "goal_text": self.c.goal,
                "target_company": self.c.target_company, "domain": self.c.domain,
                "weekly_hours_target": self.c.weekly_hours}

    def my_scored_submissions(self, limit: int = 40) -> List[Dict[str, Any]]:
        return self.c.submissions[:limit]

    def attempted_case_ids(self) -> List[str]:
        return []

    def recommend_cases(self, *, case_type: str, difficulty: Optional[str], firm: Optional[str],
                        exclude_ids: List[str], limit: int = 3) -> List[Dict[str, Any]]:
        pool = [c for c in self.c.bank if c["type"] == case_type and c["id"] not in set(exclude_ids)]
        if difficulty:
            exact = [c for c in pool if c.get("difficulty") == difficulty]
            pool = exact or pool
        return pool[:limit]

    def top_headlines(self, domain: Optional[str] = None, limit: int = 6) -> List[Dict[str, Any]]:
        return self.c.headlines[:limit]

    def search_decks(self, topic: str, limit: int = 3) -> List[Dict[str, Any]]:
        return self.c.decks[:limit]


# ---------------------------------------------------------------------------
# Deterministic, adaptive planner
# ---------------------------------------------------------------------------
def _done(history: List[Dict[str, Any]], agent: str) -> bool:
    return any(h.get("agent") == agent for h in history)


def _find(history: List[Dict[str, Any]], agent: str) -> Dict[str, Any]:
    for h in history:
        if h.get("agent") == agent:
            return h.get("result", {})
    return {}


def make_sim_planner(*, target_company: str = "", domain: str = "", weekly_hours: int = 5):
    """A scripted planner that still adapts to what it observes.

    Sequence for a candidate WITH data:
        diagnose -> target strategy -> curate cases -> news angle -> exemplars ->
        roadmap -> synthesize
    A candidate with NO graded attempts takes a visibly SHORTER branch: the
    planner reasons that any diagnosis would be invented, so it deploys only the
    target strategist and stops early — different-length control flow chosen from
    the data is the essence of agency.
    """

    def planner(goal: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 1) Always diagnose first.
        if not _done(history, "diagnostician"):
            return {"action": "delegate", "agent": "diagnostician",
                    "rationale": "Before advising, measure where this candidate actually stands."}

        diag = _find(history, "diagnostician")
        weak = (diag.get("data", {}) or {}).get("weakest_dimensions") or []

        # No usable diagnosis (no attempts / no breakdown): short-circuit.
        if not diag.get("ok", False) or not weak:
            if not _done(history, "target_strategist"):
                return {"action": "delegate", "agent": "target_strategist",
                        "args": {"target_company": target_company, "domain": domain, "weakest_dimensions": []},
                        "rationale": "No graded attempts yet — I can still map what the target tests, "
                                     "but I won't invent a weakness diagnosis."}
            return {"action": "synthesize",
                    "rationale": "Without graded attempts, the honest move is to point them at the target's "
                                 "emphasis and ask them to complete a few cases, not fabricate a plan.",
                    "summary": _compose(goal, history, target_company)}

        # 2) Interpret the target against the diagnosis.
        if not _done(history, "target_strategist"):
            return {"action": "delegate", "agent": "target_strategist",
                    "args": {"target_company": target_company, "domain": domain, "weakest_dimensions": weak},
                    "rationale": f"Diagnosis says weakest is {weak[0]}. Now map that against {target_company or 'the target'}."}

        priority = (_find(history, "target_strategist").get("data", {}) or {}).get("priority_dimensions") or weak
        focus = priority[0]

        # 3) Curate real practice for the priority skill.
        if not _done(history, "case_curator"):
            ctype = "guesstimate" if focus == "quantitative" else "case"
            return {"action": "delegate", "agent": "case_curator",
                    "args": {"focus_dimension": focus, "case_type": ctype, "difficulty": "hard", "firm": target_company},
                    "rationale": f"Pull unattempted {ctype}s that stress '{focus}', plus a bespoke stretch case."}

        # 4) A current-events case angle in their domain.
        if not _done(history, "news_analyst"):
            return {"action": "delegate", "agent": "news_analyst", "args": {"domain": domain},
                    "rationale": "Add a live business-news angle so practice connects to the real world."}

        # 5) Real-world exemplars of the priority skill.
        if not _done(history, "exemplar_scout"):
            return {"action": "delegate", "agent": "exemplar_scout",
                    "args": {"focus_dimension": focus, "topic": domain},
                    "rationale": f"Show what strong '{focus}' looks like in the wild + winning decks to model."}

        # 6) Sequence it all into a path.
        if not _done(history, "roadmap_architect"):
            return {"action": "delegate", "agent": "roadmap_architect",
                    "args": {"weakest_dimensions": weak, "target": target_company, "weekly_hours": weekly_hours},
                    "rationale": "Sequence the diagnosis, curated practice, news and exemplars into a 1-week path."}

        return {"action": "synthesize", "rationale": "Diagnosis, target, practice, news, exemplars and path are ready.",
                "summary": _compose(goal, history, target_company)}

    return planner


def _compose(goal: str, history: List[Dict[str, Any]], target: str) -> str:
    lines = [f"Your personalised prep plan for {target or 'your target'}", "", f"Goal: {goal}", ""]
    for h in history:
        r = h.get("result", {})
        lines.append(f"\u2022 [{r.get('domain', '?')}] {r.get('headline', '')}")
        for a in (r.get("recommended_actions") or [])[:1]:
            lines.append(f"    \u2192 {a}")
    lines.append("")
    lines.append(f"Specialists deployed: {', '.join(h.get('agent', '?') for h in history) or 'none'}.")
    return "\n".join(lines)
