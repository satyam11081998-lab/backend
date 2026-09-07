"""
The domain SPECIALISTS — the multiple, different-domain capabilities the
orchestrator can deploy. This is what makes the system *agentic AI* rather than a
single agent: the orchestrator has a team of specialists and decides who to send in.

Five domains, deliberately spread wide so the breadth is obvious:

  1. growth_analyst   — LEARNING ANALYTICS. Where are candidates (or one candidate)
                        weakest right now, and is it trending?
  2. content_strategist — EDITORIAL. Which current news topic is the best Group-
                        Discussion prompt, and what's the two-sided angle?
  3. curriculum_designer — INSTRUCTIONAL DESIGN. Produce a practice case targeted at
                        a specific weakness. (An ACT capability — it makes a thing.)
  4. cost_optimizer   — FinOps / PLATFORM OPS. Read AI spend and recommend safe,
                        quality-preserving provider switches. (Operates on the
                        SYSTEM ITSELF — a genuinely different domain.)
  5. deck_librarian   — RETRIEVAL. Surface the most relevant winning decks for a
                        topic or weakness.

Every specialist:
  * goes through a DataAccess boundary (Supabase in prod, sample fixtures in the
    demo) so it can run with or without a live database;
  * NEVER raises — on missing tables/columns/empty data it returns a graceful
    `ok=False` SpecialistResult so the orchestrator (and the admin demo) can't 500.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from .schemas import SpecialistResult

DIMENSIONS = ["structure", "quantitative", "synthesis", "business_judgment", "creativity", "presence"]
DIM_MAX = {"structure": 25, "quantitative": 20, "synthesis": 20, "business_judgment": 15, "creativity": 10, "presence": 10}
_DIM_TO_TECHNIQUE = {
    "structure": "MECE issue trees",
    "quantitative": "sanity-checking every estimate a second way",
    "synthesis": "the Pyramid Principle (answer first)",
    "business_judgment": "tying recommendations to macro + industry context",
    "creativity": "hypothesis-driven problem solving",
    "presence": "calibrated, top-down delivery",
}


# ---------------------------------------------------------------------------
# Data access boundary
# ---------------------------------------------------------------------------
class DataAccess(Protocol):
    def recent_scored_submissions(self, limit: int = 40) -> List[Dict[str, Any]]: ...
    def top_headlines(self, limit: int = 8) -> List[Dict[str, Any]]: ...
    def ai_spend_by_feature(self) -> Dict[str, float]: ...
    def search_decks(self, topic: str, limit: int = 3) -> List[Dict[str, Any]]: ...
    def create_practice_case(self, title: str, content: str, case_type: str, difficulty: str, focus: str) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Specialist implementations. Each takes (data, **args) -> SpecialistResult.
# ---------------------------------------------------------------------------
def growth_analyst(data: DataAccess, **_: Any) -> SpecialistResult:
    subs = data.recent_scored_submissions(limit=40)
    if not subs:
        return SpecialistResult("growth", "No graded attempts to analyse yet.", ok=False,
                                findings=["The platform has no recent scored submissions."],
                                recommended_actions=["Seed a few attempts, then re-run."])
    totals = {d: 0.0 for d in DIMENSIONS}
    counts = {d: 0 for d in DIMENSIONS}
    scores: List[int] = []
    for s in subs:
        scores.append(int(s.get("score", 0) or 0))
        bd = ((s.get("feedback_json") or {}).get("breakdown")) or {}
        for d in DIMENSIONS:
            v = bd.get(d)
            if isinstance(v, (int, float)):
                totals[d] += float(v); counts[d] += 1
    mastery = {d: round(100 * (totals[d] / counts[d]) / DIM_MAX[d]) for d in DIMENSIONS if counts[d]}
    if not mastery:
        return SpecialistResult("growth", "Submissions exist but carry no dimension breakdown.", ok=False)
    weakest = min(mastery, key=mastery.get)
    avg = round(sum(scores) / len(scores)) if scores else 0
    trend = "flat"
    if len(scores) >= 12:
        recent = sum(scores[:6]) / 6; prev = sum(scores[6:12]) / 6
        trend = "improving" if recent > prev + 3 else "declining" if recent < prev - 3 else "flat"
    return SpecialistResult(
        domain="growth",
        headline=f"Platform-wide weakest skill is {weakest} ({mastery[weakest]}/100); avg score {avg}, trend {trend}.",
        findings=[f"{d}: {mastery[d]}/100" for d in sorted(mastery, key=mastery.get)],
        data={"weakest_dimension": weakest, "mastery": mastery, "avg_score": avg, "trend": trend, "n": len(subs)},
        recommended_actions=[f"Push practice that stresses {weakest} using {_DIM_TO_TECHNIQUE.get(weakest, 'targeted drills')}."],
    )


def content_strategist(data: DataAccess, **_: Any) -> SpecialistResult:
    heads = data.top_headlines(limit=8)
    if not heads:
        return SpecialistResult("content", "No fresh headlines available.", ok=False,
                                recommended_actions=["Run the news refresh, then re-run."])
    best = max(heads, key=lambda h: h.get("gd_worthiness_score", 0) or 0)
    title = best.get("title", "Untitled")
    return SpecialistResult(
        domain="content",
        headline=f"Best GD topic today: “{title}”.",
        findings=[f"{h.get('title','?')} (score {h.get('gd_worthiness_score','?')})" for h in heads[:5]],
        data={"topic": title, "score": best.get("gd_worthiness_score"),
              "category": best.get("category"), "source": best.get("source_name")},
        recommended_actions=[
            f"Frame both sides of “{title}” (for vs against) and seed 3 stakeholder angles.",
            "Feature it as today's GD brief on the dashboard.",
        ],
    )


def curriculum_designer(data: DataAccess, focus_dimension: str = "structure",
                        case_type: str = "case", difficulty: str = "medium", **_: Any) -> SpecialistResult:
    if focus_dimension not in DIMENSIONS:
        focus_dimension = "structure"
    technique = _DIM_TO_TECHNIQUE.get(focus_dimension, "targeted drills")
    title = f"{difficulty.title()} {case_type}: stress-test {focus_dimension}"
    content = (
        f"An Indian firm faces a decision that hinges on {focus_dimension}. "
        f"Structure the problem, work the numbers in Rs/crore, and deliver a top-down recommendation. "
        f"(Designed to force {technique}.)"
    )
    created = data.create_practice_case(title, content, case_type, difficulty, focus_dimension)
    return SpecialistResult(
        domain="curriculum",
        headline=f"Designed a {difficulty} {case_type} targeting {focus_dimension}.",
        findings=[f"Technique forced: {technique}", f"Title: {title}"],
        data={"case_id": created.get("id"), "title": title, "focus_dimension": focus_dimension,
              "case_type": case_type, "difficulty": difficulty},
        recommended_actions=[f"Assign this case to candidates weak on {focus_dimension}."],
    )


def cost_optimizer(data: DataAccess, **_: Any) -> SpecialistResult:
    spend = data.ai_spend_by_feature() or {}
    if not spend:
        return SpecialistResult("cost", "No AI spend recorded in the window.", ok=False)
    total = sum(spend.values())
    # Safe, quality-preserving levers: move cheap-tolerant features to Groq/Google;
    # NEVER scoring. Rough savings model mirrors the ~9x / ~3.75x notes in the code.
    movable = {"stt": 0.9, "interviewer": 0.85, "validity": 0.9, "news_classify": 0.9, "tts": 0.73}
    est_savings = sum(spend.get(f, 0.0) * frac for f, frac in movable.items())
    top = sorted(spend.items(), key=lambda kv: kv[1], reverse=True)[:5]
    actions = [f"Switch '{f}' to its cheaper provider (~save ${spend.get(f,0)*movable[f]:.2f})."
               for f in movable if spend.get(f, 0) > 0]
    return SpecialistResult(
        domain="cost",
        headline=f"~${est_savings:.2f} of ${total:.2f} is safely removable without touching scoring.",
        findings=[f"{f}: ${amt:.2f}" for f, amt in top],
        data={"total_usd": round(total, 2), "est_savings_usd": round(est_savings, 2), "by_feature": spend},
        recommended_actions=actions or ["Spend is already near-optimal; keep scoring on OpenAI."],
    )


def deck_librarian(data: DataAccess, topic: str = "", **_: Any) -> SpecialistResult:
    decks = data.search_decks(topic or "strategy", limit=3)
    if not decks:
        return SpecialistResult("decks", f"No decks matched “{topic}”.", ok=False)
    return SpecialistResult(
        domain="decks",
        headline=f"Found {len(decks)} winning decks relevant to “{topic}”.",
        findings=[f"{d.get('title','?')} — {d.get('competition','')} {d.get('result','')}".strip() for d in decks],
        data={"decks": decks},
        recommended_actions=["Link these from the practice page for candidates on this focus."],
    )


# name -> (callable, human label, domain)
SPECIALISTS = {
    "growth_analyst": (growth_analyst, "Growth Analyst", "Learning analytics"),
    "content_strategist": (content_strategist, "Content Strategist", "Editorial"),
    "curriculum_designer": (curriculum_designer, "Curriculum Designer", "Instructional design"),
    "cost_optimizer": (cost_optimizer, "Cost Optimizer", "Platform FinOps"),
    "deck_librarian": (deck_librarian, "Deck Librarian", "Retrieval"),
}


def run_specialist(data: DataAccess, name: str, args: Optional[Dict[str, Any]] = None) -> SpecialistResult:
    """Dispatch to a specialist by name. Unknown/failed specialists return a
    graceful result rather than raising — the orchestrator must never crash on a
    single specialist's failure."""
    entry = SPECIALISTS.get(name)
    if entry is None:
        return SpecialistResult(name or "unknown", f"No such specialist: {name}", ok=False)
    fn = entry[0]
    try:
        return fn(data, **(args or {}))
    except Exception as e:  # noqa: BLE001
        return SpecialistResult(name, f"{name} failed: {type(e).__name__}: {e}", ok=False)
