"""
The domain SPECIALISTS the Prep Copilot orchestrator can deploy for ONE
candidate. This is what makes it *agentic AI* and not one wrapped call: the
orchestrator has a team spanning six distinct domains and decides, at runtime,
who to send in, in what order, and with what arguments.

Design principle — GROUNDED SPECIALISTS, PLANNING MODEL. Every factual claim a
specialist makes is computed from real data (the user's submissions, the case
bank, the news table, the deck vault) or looked up in a small curated map. The
model's job is the ORCHESTRATION (which specialist, when, with what args) and
the final narration — not to invent facts. That is a deliberate,
defensible-to-an-expert split: the autonomy is real, the outputs don't
hallucinate.

Six domains:
  1. diagnostician     — PERFORMANCE ANALYTICS on THIS candidate. Where are they
                         weakest right now, and is it trending?
  2. target_strategist — CAREER INTELLIGENCE. Given the target company/role, what
                         does it actually test, and which of the candidate's
                         weaknesses matter most for it?
  3. case_curator      — CURATION + RETRIEVAL. Pick real, not-yet-attempted cases
                         and guesstimates from the bank matched to the priority
                         skill/firm, and draft one bespoke "stretch" case.
  4. news_analyst      — EDITORIAL / BUSINESS CONTEXT. Turn a current, GD-worthy
                         headline in their target domain into a case/GD angle.
  5. exemplar_scout    — REAL-WORLD GROUNDING. Surface tech/business-world
                         examples and winning decks that model the target skill.
  6. roadmap_architect — INSTRUCTIONAL DESIGN. Sequence everything into a
                         milestone prep path (the structured guidance the user
                         asked for).

Every specialist goes through a DataAccess boundary (Supabase in prod, sample
fixtures in the demo) and NEVER raises — on missing tables/columns/empty data it
returns a graceful ok=False result so the orchestrator (and the demo) can't 500.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from .schemas import SpecialistResult

# The 6-dimension case rubric. Guesstimates carry a subset; unknown dimensions
# are handled generically (normalised against the max observed) so a different
# rubric can never crash the analytics.
DIMENSIONS = ["structure", "quantitative", "synthesis", "business_judgment", "creativity", "presence"]
DIM_MAX = {"structure": 25, "quantitative": 20, "synthesis": 20, "business_judgment": 15, "creativity": 10, "presence": 10}

DIM_LABEL = {
    "structure": "structuring / MECE issue trees",
    "quantitative": "quant & estimation",
    "synthesis": "synthesis (answer-first)",
    "business_judgment": "business judgement",
    "creativity": "creativity / hypotheses",
    "presence": "communication & presence",
}
DIM_TECHNIQUE = {
    "structure": "MECE issue trees drawn before talking",
    "quantitative": "sanity-checking every number a second way",
    "synthesis": "the Pyramid Principle — answer first, then support",
    "business_judgment": "tying each recommendation to industry + macro context",
    "creativity": "hypothesis-driven problem solving",
    "presence": "calibrated, top-down delivery under time pressure",
}

# Grounded career-intelligence map. What each target *emphasises*, expressed as
# the rubric dimensions it leans on hardest. Kept small and honest — the point is
# to ANCHOR the model, not to pretend at precision we don't have.
FIRM_EMPHASIS: Dict[str, Dict[str, Any]] = {
    "mckinsey": {"label": "McKinsey", "dims": ["structure", "synthesis", "business_judgment"],
                 "note": "Highly structured, top-down synthesis; expect a crisp so-what early."},
    "bcg": {"label": "BCG", "dims": ["structure", "creativity", "quantitative"],
            "note": "Structured but hypothesis-led; creative framing is rewarded."},
    "bain": {"label": "Bain", "dims": ["business_judgment", "quantitative", "presence"],
             "note": "Answer-first, results-oriented; strong on the quant and the 'so what'."},
    "kearney": {"label": "Kearney", "dims": ["structure", "quantitative"],
                "note": "Operations/quant heavy; clean structure and numbers."},
    "big4": {"label": "Big 4 (Deloitte/PwC/EY/KPMG) consulting", "dims": ["structure", "business_judgment"],
             "note": "Practical, client-ready structuring and judgement."},
    "product": {"label": "Product management (tech)", "dims": ["creativity", "quantitative", "synthesis"],
                "note": "Guesstimation, product sense and prioritisation matter more than pure strategy."},
    "finance": {"label": "Finance / IB", "dims": ["quantitative", "business_judgment"],
                "note": "Numbers under pressure and commercial judgement dominate."},
    "general": {"label": "General management / other", "dims": ["structure", "synthesis", "business_judgment"],
                "note": "Balanced structuring, synthesis and judgement."},
}

# Real-world examples keyed by dimension — used by the exemplar scout to make the
# advice concrete ("here's this skill in the wild"). These are illustrative
# anchors, framed as examples, not claims about any company's internal decisions.
DIM_EXAMPLES: Dict[str, List[str]] = {
    "structure": [
        "Amazon's working-backwards PR/FAQ forces a MECE problem definition before any build.",
        "Spotify's squad model is a textbook clean decomposition of a big org into non-overlapping units.",
    ],
    "quantitative": [
        "Netflix sizing content spend vs. subscriber LTV is a classic market-sizing chain.",
        "Zomato/Swiggy unit economics (AOV × take-rate − delivery cost) is a live guesstimate.",
    ],
    "synthesis": [
        "A McKinsey exec summary states the recommendation in sentence one — that's synthesis.",
        "Jeff Bezos's one-page memos are answer-first: conclusion up top, evidence below.",
    ],
    "business_judgment": [
        "Reliance Jio's free-then-monetise entry is a judgement call on land-grab vs. margin.",
        "Tata's EV bet weighs macro fuel policy against near-term unit cost — pure judgement.",
    ],
    "creativity": [
        "Airbnb reframing 'hotels' as 'belong anywhere' opened a non-obvious market hypothesis.",
        "Dollar Shave Club attacked razors on distribution, not product — a creative angle.",
    ],
    "presence": [
        "A strong candidate leads with the answer, then walks structure — calm, top-down.",
        "Great analysts signpost ('three reasons; first…') so the listener never gets lost.",
    ],
}


# ---------------------------------------------------------------------------
# Data-access boundary (user-scoped)
# ---------------------------------------------------------------------------
class DataAccess(Protocol):
    def profile(self) -> Dict[str, Any]: ...
    def my_scored_submissions(self, limit: int = 40) -> List[Dict[str, Any]]: ...
    def attempted_case_ids(self) -> List[str]: ...
    def recommend_cases(self, *, case_type: str, difficulty: Optional[str],
                        firm: Optional[str], exclude_ids: List[str], limit: int = 3) -> List[Dict[str, Any]]: ...
    def top_headlines(self, domain: Optional[str] = None, limit: int = 6) -> List[Dict[str, Any]]: ...
    def search_decks(self, topic: str, limit: int = 3) -> List[Dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mastery(subs: List[Dict[str, Any]]) -> Dict[str, int]:
    """Per-dimension mastery 0..100 from graded submissions. Robust to any rubric:
    known dims normalise against DIM_MAX; unknown dims against the max observed."""
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    observed_max: Dict[str, float] = {}
    for s in subs:
        bd = ((s.get("feedback_json") or {}).get("breakdown")) or {}
        for k, v in bd.items():
            if isinstance(v, (int, float)):
                totals[k] = totals.get(k, 0.0) + float(v)
                counts[k] = counts.get(k, 0) + 1
                observed_max[k] = max(observed_max.get(k, 0.0), float(v))
    out: Dict[str, int] = {}
    for k in counts:
        avg = totals[k] / counts[k]
        cap = float(DIM_MAX.get(k) or observed_max.get(k) or 0)
        if cap <= 0:
            continue
        out[k] = max(0, min(100, round(100 * avg / cap)))
    return out


def _norm_firm(target_company: str, domain: str) -> str:
    t = f"{target_company} {domain}".lower()
    if "mckinsey" in t or "mbb" in t and "bcg" not in t and "bain" not in t:
        return "mckinsey"
    if "bcg" in t or "boston consulting" in t:
        return "bcg"
    if "bain" in t:
        return "bain"
    if "kearney" in t or "at kearney" in t:
        return "kearney"
    if any(k in t for k in ("deloitte", "pwc", "kpmg", "ey ", "big 4", "big four")):
        return "big4"
    if any(k in t for k in ("product", "pm ", "apm", "tech", "google", "microsoft", "amazon", "meta", "flipkart")):
        return "product"
    if any(k in t for k in ("finance", "investment bank", "ib", "goldman", "jpmorgan", "j.p. morgan", "banking")):
        return "finance"
    if any(k in t for k in ("consult", "mbb", "strategy")):
        return "mckinsey"
    return "general"


# ---------------------------------------------------------------------------
# Specialists
# ---------------------------------------------------------------------------
def diagnostician(data: DataAccess, **_: Any) -> SpecialistResult:
    subs = data.my_scored_submissions(limit=40)
    if not subs:
        return SpecialistResult("diagnostics", "No graded attempts yet to diagnose.", ok=False,
                                recommended_actions=["Complete a few scored cases, then run the coach."])
    scores = [int(s.get("score", 0) or 0) for s in subs]
    mastery = _mastery(subs)
    if not mastery:
        return SpecialistResult("diagnostics", "Attempts exist but carry no dimension breakdown.", ok=False)
    ordered = sorted(mastery, key=mastery.get)
    weakest = ordered[:2]
    avg = round(sum(scores) / len(scores)) if scores else 0
    trend = "flat"
    if len(scores) >= 6:
        recent = sum(scores[:3]) / 3
        prev = sum(scores[3:6]) / 3
        trend = "improving" if recent > prev + 3 else "declining" if recent < prev - 3 else "flat"
    weak_labels = ", ".join(DIM_LABEL.get(d, d) for d in weakest)
    return SpecialistResult(
        domain="diagnostics",
        headline=f"Across {len(subs)} graded attempts, weakest is {weak_labels}; avg {avg}/100, trend {trend}.",
        findings=[f"{DIM_LABEL.get(d, d)}: {mastery[d]}/100" for d in ordered],
        data={"weakest_dimensions": weakest, "mastery": mastery, "avg_score": avg,
              "trend": trend, "n": len(subs)},
        recommended_actions=[f"Prioritise {DIM_LABEL.get(weakest[0], weakest[0])} — "
                             f"drill {DIM_TECHNIQUE.get(weakest[0], 'targeted reps')}."],
    )


def target_strategist(data: DataAccess, target_company: str = "", domain: str = "",
                      weakest_dimensions: Optional[List[str]] = None, **_: Any) -> SpecialistResult:
    firm_key = _norm_firm(target_company, domain)
    firm = FIRM_EMPHASIS[firm_key]
    emphasised = firm["dims"]
    weak = weakest_dimensions or []
    # The priority = what the TARGET stresses AND the candidate is weak on.
    priority = [d for d in emphasised if d in weak] or emphasised[:1]
    also = [d for d in weak if d not in emphasised]
    label = firm["label"]
    tgt = target_company.strip() or label
    return SpecialistResult(
        domain="target",
        headline=f"For {tgt}, the make-or-break skills are {', '.join(DIM_LABEL.get(d, d) for d in emphasised)}.",
        findings=[firm["note"],
                  f"Your priority (target stresses it AND you're weak): {', '.join(DIM_LABEL.get(d, d) for d in priority)}."]
                 + ([f"Also shore up: {', '.join(DIM_LABEL.get(d, d) for d in also)}."] if also else []),
        data={"firm_key": firm_key, "firm_label": label, "target": tgt,
              "emphasised_dimensions": emphasised, "priority_dimensions": priority},
        recommended_actions=[f"Aim practice at {DIM_LABEL.get(priority[0], priority[0])} first — "
                             f"it's where {label}'s bar and your gap overlap."],
    )


def case_curator(data: DataAccess, focus_dimension: str = "structure", case_type: str = "case",
                 difficulty: str = "medium", firm: str = "", **_: Any) -> SpecialistResult:
    if focus_dimension not in DIMENSIONS:
        focus_dimension = "structure"
    if case_type not in ("case", "guesstimate"):
        case_type = "case"
    exclude = data.attempted_case_ids()
    recs = data.recommend_cases(case_type=case_type, difficulty=difficulty,
                                firm=firm or None, exclude_ids=exclude, limit=3)
    technique = DIM_TECHNIQUE.get(focus_dimension, "targeted reps")
    # A bespoke "stretch" case — TEXT ONLY. It is returned to the user and saved
    # to THEIR plan (coach_runs); it is deliberately NOT inserted into the shared
    # `cases` bank so it can never leak into other users' practice or the
    # leaderboard.
    stretch = {
        "title": f"Stretch {case_type}: pressure-test your {DIM_LABEL.get(focus_dimension, focus_dimension)}",
        "prompt": (f"An Indian firm faces a decision that hinges on {DIM_LABEL.get(focus_dimension, focus_dimension)}. "
                   f"Frame the problem, work the numbers in Rs/crore, and deliver a top-down recommendation. "
                   f"Constraint: you must {technique}."),
        "focus_dimension": focus_dimension, "difficulty": difficulty, "case_type": case_type,
    }
    findings = [f"{c.get('title', 'Untitled')} — {c.get('type', '')} · {c.get('difficulty', '')}"
                for c in recs] or ["No unattempted match in the bank right now — use the stretch case below."]
    return SpecialistResult(
        domain="curation",
        headline=(f"Curated {len(recs)} unattempted {case_type}(s) for {DIM_LABEL.get(focus_dimension, focus_dimension)}"
                  + (f" + 1 bespoke stretch case." if recs else " (bank had none — drafted a bespoke stretch case).")),
        findings=findings,
        data={"recommended_cases": recs, "stretch_case": stretch, "focus_dimension": focus_dimension},
        recommended_actions=([f"Start with '{recs[0].get('title')}', then attempt the stretch case."]
                             if recs else ["Attempt the bespoke stretch case below."]),
    )


def news_analyst(data: DataAccess, domain: str = "", **_: Any) -> SpecialistResult:
    heads = data.top_headlines(domain=domain or None, limit=6)
    if not heads:
        return SpecialistResult("news", "No fresh, GD-worthy headline available right now.", ok=False,
                                recommended_actions=["Refresh the news pipeline, then re-run."])
    best = max(heads, key=lambda h: h.get("gd_worthiness_score", 0) or 0)
    title = best.get("title", "Untitled")
    return SpecialistResult(
        domain="news",
        headline=f"Turn today's headline into practice: \u201c{title}\u201d.",
        findings=[f"{h.get('title', '?')} (GD-worthiness {h.get('gd_worthiness_score', '?')})" for h in heads[:4]],
        data={"topic": title, "score": best.get("gd_worthiness_score"),
              "category": best.get("category"), "source": best.get("source_name"),
              "case_angle": f"Should the key player in \u201c{title}\u201d change strategy? "
                            f"Structure both sides, size the impact, and recommend."},
        recommended_actions=[f"Do a 10-minute structured take on \u201c{title}\u201d — for/against + a sized recommendation."],
    )


def exemplar_scout(data: DataAccess, focus_dimension: str = "structure", topic: str = "", **_: Any) -> SpecialistResult:
    if focus_dimension not in DIMENSIONS:
        focus_dimension = "structure"
    examples = DIM_EXAMPLES.get(focus_dimension, [])
    decks = data.search_decks(topic or focus_dimension, limit=3)
    deck_lines = [f"{d.get('title', '?')} — {d.get('competition', '')} {d.get('result', '')}".strip() for d in decks]
    return SpecialistResult(
        domain="exemplars",
        headline=f"Real-world models of strong {DIM_LABEL.get(focus_dimension, focus_dimension)}.",
        findings=examples + (deck_lines or []),
        data={"examples": examples, "decks": decks, "focus_dimension": focus_dimension},
        recommended_actions=["Study one exemplar before your next attempt and copy its move deliberately."],
    )


def roadmap_architect(data: DataAccess, weakest_dimensions: Optional[List[str]] = None,
                      target: str = "", weekly_hours: int = 5, **_: Any) -> SpecialistResult:
    weak = (weakest_dimensions or ["structure"])[:2]
    prim = weak[0]
    sec = weak[1] if len(weak) > 1 else weak[0]
    plan = [
        {"title": "Days 1–3 · Diagnose & drill the primary gap",
         "focus": DIM_LABEL.get(prim, prim),
         "items": [f"Warm-up: 1 easy case focused on {DIM_LABEL.get(prim, prim)}.",
                   f"Technique rep: {DIM_TECHNIQUE.get(prim, 'targeted reps')} on every attempt.",
                   "Review the curated exemplar, then re-attempt and compare."]},
        {"title": "Days 4–6 · Apply under target-firm conditions",
         "focus": target or "your target",
         "items": ["Attempt the curated case matched to your target's emphasis.",
                   f"Do the news-driven case angle (10 min, timed).",
                   f"Second gap: one guesstimate stressing {DIM_LABEL.get(sec, sec)}."]},
        {"title": "Day 7 · Integrate & self-assess",
         "focus": "synthesis under time",
         "items": ["Attempt the bespoke stretch case end-to-end, timed.",
                   "Re-run the coach to see the trend move.",
                   "Book one mock with a peer if you can."]},
    ]
    return SpecialistResult(
        domain="roadmap",
        headline=f"A 1-week path to lift {DIM_LABEL.get(prim, prim)} toward {target or 'your target'}.",
        findings=[m["title"] for m in plan],
        data={"plan": plan, "weekly_hours": weekly_hours, "primary": prim, "secondary": sec},
        recommended_actions=["Follow the path top to bottom; re-run the coach weekly to re-plan on fresh data."],
    )


# name -> (callable, human label, domain)
SPECIALISTS = {
    "diagnostician": (diagnostician, "Diagnostician", "Performance analytics"),
    "target_strategist": (target_strategist, "Target Strategist", "Career intelligence"),
    "case_curator": (case_curator, "Case Curator", "Curation & retrieval"),
    "news_analyst": (news_analyst, "News Analyst", "Business context"),
    "exemplar_scout": (exemplar_scout, "Exemplar Scout", "Real-world grounding"),
    "roadmap_architect": (roadmap_architect, "Roadmap Architect", "Instructional design"),
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
