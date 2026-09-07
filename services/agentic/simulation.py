"""
Simulation engine — the ALWAYS-WORKS demo brain.

This makes the admin demo bulletproof: it runs the REAL orchestrator loop against
a deterministic planner and rich sample data, so it works with no OpenAI key, no
budget left, and no rows in the database. It's what you show people.

It is honest about what it is (mode="sim"): the planner here is scripted, but the
loop, the delegation, the specialist code, and the multi-agent structure are all
the same objects the live path uses. The demo proves the machinery; live mode
swaps the scripted planner for the model.

The scripted planner is still *adaptive*: it reads the results gathered so far
(e.g. the weakest dimension the growth analyst found) and feeds them into the next
specialist's arguments — so different missions produce genuinely different teams.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .specialists import DataAccess


# ---------------------------------------------------------------------------
# Sample data so specialists produce vivid output with no DB.
# ---------------------------------------------------------------------------
class SampleData(DataAccess):
    def recent_scored_submissions(self, limit: int = 40) -> List[Dict[str, Any]]:
        # Cohort skewed weak on synthesis — the analyst will surface that.
        base = [
            {"score": 61, "feedback_json": {"breakdown": {"structure": 21, "quantitative": 17, "synthesis": 9, "business_judgment": 10, "creativity": 6, "presence": 7}}},
            {"score": 58, "feedback_json": {"breakdown": {"structure": 20, "quantitative": 16, "synthesis": 8, "business_judgment": 10, "creativity": 6, "presence": 7}}},
            {"score": 64, "feedback_json": {"breakdown": {"structure": 22, "quantitative": 18, "synthesis": 10, "business_judgment": 11, "creativity": 6, "presence": 8}}},
            {"score": 55, "feedback_json": {"breakdown": {"structure": 19, "quantitative": 15, "synthesis": 8, "business_judgment": 9, "creativity": 6, "presence": 7}}},
            {"score": 67, "feedback_json": {"breakdown": {"structure": 23, "quantitative": 18, "synthesis": 11, "business_judgment": 11, "creativity": 7, "presence": 8}}},
        ]
        return (base * 8)[:limit]

    def top_headlines(self, limit: int = 8) -> List[Dict[str, Any]]:
        return [
            {"title": "RBI holds repo rate as inflation cools", "gd_worthiness_score": 91, "category": "macro", "source_name": "Sample Wire"},
            {"title": "Quick-commerce burns cash chasing 10-minute delivery", "gd_worthiness_score": 88, "category": "business", "source_name": "Sample Wire"},
            {"title": "India's data-centre boom strains the power grid", "gd_worthiness_score": 84, "category": "tech", "source_name": "Sample Wire"},
            {"title": "MBA hiring shifts toward AI-fluent generalists", "gd_worthiness_score": 82, "category": "jobs", "source_name": "Sample Wire"},
        ][:limit]

    def ai_spend_by_feature(self) -> Dict[str, float]:
        return {"scoring": 4.10, "interviewer": 3.30, "stt": 1.20, "tts": 0.95, "validity": 0.60, "news_classify": 0.35}

    def search_decks(self, topic: str, limit: int = 3) -> List[Dict[str, Any]]:
        return [
            {"title": "Market entry: EV two-wheelers", "competition": "IIM-A Case Cup", "result": "Winner"},
            {"title": "Profitability turnaround: QSR chain", "competition": "BCG Ace", "result": "Finalist"},
            {"title": "Pricing strategy: OTT bundling", "competition": "Bain BLC", "result": "National Finalist"},
        ][:limit]

    def create_practice_case(self, title: str, content: str, case_type: str, difficulty: str, focus: str) -> Dict[str, Any]:
        return {"id": f"sample_{focus}_{difficulty}", "title": title}


# ---------------------------------------------------------------------------
# Deterministic, adaptive planner.
# ---------------------------------------------------------------------------
def _done(history: List[Dict[str, Any]], agent: str) -> bool:
    return any(h.get("agent") == agent for h in history)


def _find(history: List[Dict[str, Any]], agent: str) -> Dict[str, Any]:
    for h in history:
        if h.get("agent") == agent:
            return h.get("result", {})
    return {}


def make_sim_planner():
    def planner(mission: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        m = (mission or "").lower()

        # ---- Mission: cut AI cost ----
        if any(k in m for k in ("cost", "spend", "cheaper", "finops", "budget", "bill")):
            if not _done(history, "cost_optimizer"):
                return {"action": "delegate", "agent": "cost_optimizer",
                        "rationale": "This is a FinOps mission — start by reading where the money goes."}
            cost = _find(history, "cost_optimizer")
            return {"action": "synthesize",
                    "rationale": "I have the spend breakdown and the safe levers.",
                    "summary": _compose(mission, history)}

        # ---- Mission: design this week's practice focus ----
        if any(k in m for k in ("week", "practice", "focus", "curriculum", "drill")):
            if not _done(history, "growth_analyst"):
                return {"action": "delegate", "agent": "growth_analyst",
                        "rationale": "First measure where candidates are weakest before designing practice."}
            weak = _find(history, "growth_analyst").get("data", {}).get("weakest_dimension", "structure")
            if not _done(history, "curriculum_designer"):
                return {"action": "delegate", "agent": "curriculum_designer",
                        "args": {"focus_dimension": weak, "case_type": "case", "difficulty": "hard"},
                        "rationale": f"Growth flagged '{weak}'. Designing a case that stresses exactly that."}
            if not _done(history, "deck_librarian"):
                return {"action": "delegate", "agent": "deck_librarian",
                        "args": {"topic": weak},
                        "rationale": f"Pull winning decks that model strong '{weak}' so candidates have exemplars."}
            return {"action": "synthesize", "rationale": "Focus, case and exemplars are ready.",
                    "summary": _compose(mission, history)}

        # ---- Default / "daily brief" / freeform: a broad platform sweep ----
        if not _done(history, "content_strategist"):
            return {"action": "delegate", "agent": "content_strategist",
                    "rationale": "Lead with today's best discussion topic."}
        if not _done(history, "growth_analyst"):
            return {"action": "delegate", "agent": "growth_analyst",
                    "rationale": "Then diagnose where the cohort is weakest."}
        weak = _find(history, "growth_analyst").get("data", {}).get("weakest_dimension", "structure")
        if not _done(history, "curriculum_designer"):
            return {"action": "delegate", "agent": "curriculum_designer",
                    "args": {"focus_dimension": weak, "case_type": "case", "difficulty": "medium"},
                    "rationale": f"Design today's targeted practice around '{weak}'."}
        return {"action": "synthesize", "rationale": "Topic, diagnosis and today's case are set.",
                "summary": _compose(mission, history)}

    return planner


def _compose(mission: str, history: List[Dict[str, Any]]) -> str:
    lines = [f"Executive brief — {mission}", ""]
    for h in history:
        r = h.get("result", {})
        lines.append(f"• [{r.get('domain','?')}] {r.get('headline','')}")
        for a in (r.get("recommended_actions") or [])[:2]:
            lines.append(f"    → {a}")
    lines.append("")
    lines.append(f"Specialists deployed: {', '.join(h.get('agent','?') for h in history) or 'none'}.")
    return "\n".join(lines)
