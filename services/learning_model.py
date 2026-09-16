"""
Learning-intelligence layer (Phase 4). Turns the adaptive CONVERSATION into an
adaptive LEARNING engine: a skill + misconception taxonomy, intervention-outcome
tracking, per-skill independence accumulation (scaffolding + fading), next-drill
recommendation, a minimum-assistance learning block for the prompt, and a
personalised post-case debrief.

All PURE -- no model call, no DB -- so it is unit-testable and cheap. The
interviewer tags each turn (skill=, error=, modality=, intervention=) in its
control tag; this module folds those tags + the deterministic signals into the
learner model, which persists on attempts.session_state.profile (per attempt)
and, later, on a longitudinal user_skill_profile row.

DESIGN PRINCIPLE (owner, point 16): optimise for the MINIMUM assistance that
restores INDEPENDENT reasoning -- NOT 'maximise the chance the user eventually
gets the answer'. Coached is not easy. That objective is enforced in the prompt
and reflected in the 'independence' the profile tracks.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# --- taxonomies --------------------------------------------------------------
SKILLS = {
    # sizing / guesstimate
    "segmentation", "population_decomposition", "penetration", "frequency",
    "unit_conversion", "annualization", "capacity_bridge", "sanity_check",
    "sensitivity", "assumption_defense",
    # case
    "structuring", "profit_tree", "revenue_decomposition", "cost_decomposition",
    "fixed_vs_variable", "driver_identification", "quantification",
    "prioritization", "synthesis", "business_judgment", "communication",
}

ERROR_TYPES = {
    "structural", "conceptual", "calculation", "unit", "assumption", "logic",
    "communication", "scope", "double_count", "sanity_check", "prioritization",
    "capacity_bridge",
}

# HOW help is delivered (point 9): different learners unlock differently.
MODALITIES = {
    "direct_hint", "reframe", "analogy", "concrete_example", "counterexample",
    "partial_demonstration", "decompose", "micro_hint", "demonstrate",
}

# error_type -> (human label, one-line reusable rule for the debrief)
_REMEDIES: Dict[str, tuple] = {
    "capacity_bridge": ("bridging demand to capacity",
                        "When you size outlets/stores, build the bridge: total demand ÷ capacity per outlet."),
    "unit": ("unit consistency",
             "Keep units consistent -- convert once, label every number, and re-check before you multiply."),
    "double_count": ("mutually-exclusive buckets",
                     "Check your segments don't overlap, so nothing is counted twice."),
    "sanity_check": ("sanity-checking",
                     "End every estimate with a back-of-envelope check: does the implied daily number look plausible?"),
    "structural": ("clean structure first",
                   "Lay out a MECE structure before you compute -- decide the drivers, then quantify."),
    "calculation": ("careful arithmetic",
                    "Slow the driving calculation down and recompute it once before moving on."),
    "conceptual": ("the underlying concept",
                   "Name the concept in one line (e.g. penetration = share of the base that buys) before you use it."),
    "assumption": ("defending assumptions",
                   "State each assumption and the one reason it's defensible, then move on -- don't over-justify."),
    "prioritization": ("prioritising the 80/20",
                       "Attack the biggest driver first; don't spend equal time on every branch."),
    "scope": ("holding the scope",
              "Pin the scope once (geography, period, B2B/B2C) and keep every step inside it."),
    "logic": ("logical consistency",
              "Make each step follow from the last; watch for demand-vs-supply and cause-vs-effect mix-ups."),
    "synthesis": ("top-down synthesis",
                  "Lead with the recommendation, then 2-3 supports -- don't recap chronologically."),
    "communication": ("crisp communication",
                      "Say the answer first in one line, then the why."),
}


def _norm_enum(val: Optional[str], allowed) -> Optional[str]:
    v = (val or "").strip().lower().replace(" ", "_").replace("-", "_")
    return v if v in allowed else None


def _blank_profile() -> Dict[str, Any]:
    return {"skills": {}, "errors": {}, "modalities": {}, "independence": {}}


def _independence_band(hint_level) -> str:
    h = max(0, min(5, int(hint_level or 0)))
    return {0: "independent", 1: "after_h1", 2: "after_h2", 3: "after_h3",
            4: "after_demo", 5: "needed_solution"}[h]


# --- intervention outcome (point 8) ------------------------------------------
def evaluate_intervention_outcome(prior_state: Optional[dict], signals: Dict[str, Any]) -> str:
    """Did the PRIOR turn's HELP move work? Deterministic: the learner recovered
    if this turn they are no longer stuck/repeating/asking-for-help/frustrated and
    have something down. Only judged for actual help moves (not continue/probe).
    Returns 'worked' | 'failed' | 'na'."""
    last = (prior_state or {}).get("last_intervention")
    if not last or last in ("continue", "probe", "skip", None):
        return "na"
    stuck_now = bool(
        signals.get("help_requested") or signals.get("solution_requested")
        or signals.get("looks_garbage") or signals.get("candidate_repeating")
        or signals.get("turns_without_progress", 0) >= 1
        or signals.get("frustration") in ("mild", "high")
    )
    return "failed" if stuck_now else "worked"


# --- profile accumulation (points 8, 17) -------------------------------------
def update_learning_profile(profile: Optional[dict], tag: Dict[str, str],
                            outcome: str, hint_level, signals: Dict[str, Any],
                            prior_modality: Optional[str] = None) -> Dict[str, Any]:
    """Fold ONE turn's control tag + outcome into the per-attempt learner profile."""
    p = dict(profile or _blank_profile())
    for k in ("skills", "errors", "modalities", "independence"):
        p.setdefault(k, {})

    skill = _norm_enum(tag.get("skill"), SKILLS)
    err = _norm_enum(tag.get("error"), ERROR_TYPES)
    modality = _norm_enum(prior_modality, MODALITIES) or _norm_enum(tag.get("modality") or tag.get("intervention"), MODALITIES)

    if err:
        p["errors"][err] = p["errors"].get(err, 0) + 1

    if modality and outcome in ("worked", "failed"):
        m = p["modalities"].setdefault(modality, {"worked": 0, "failed": 0})
        m[outcome] += 1

    # Independence (point 17): when a skill is RESOLVED (help landed), record the
    # rung it took. Lower rung over time = fading = real learning.
    if skill and outcome == "worked":
        band = _independence_band(hint_level)
        s = p["independence"].setdefault(skill, {})
        s[band] = s.get(band, 0) + 1
    # Independent success with no help this turn also counts.
    if skill and outcome == "na" and not signals.get("help_requested") \
            and signals.get("has_work") and int(hint_level or 0) == 0:
        s = p["independence"].setdefault(skill, {})
        s["independent"] = s.get("independent", 0) + 1

    return p


def best_modality(profile: Optional[dict]) -> Optional[str]:
    """Which help modality has landed most for THIS learner (point 8/9)."""
    mods = (profile or {}).get("modalities", {})
    best, best_net = None, 0
    for name, wl in mods.items():
        net = int(wl.get("worked", 0)) - int(wl.get("failed", 0))
        if net > best_net:
            best, best_net = name, net
    return best


def weak_skills(profile: Optional[dict], k: int = 3) -> List[str]:
    """Skills the learner has stumbled on most (by error tally)."""
    errors = (profile or {}).get("errors", {})
    return [e for e, _ in sorted(errors.items(), key=lambda kv: kv[1], reverse=True)[:k]]


# --- next-best practice (points 14, 28) --------------------------------------
def recommend_next_drill(profile: Optional[dict]) -> Optional[Dict[str, str]]:
    """Weakest area -> a concrete next-practice suggestion with a difficulty tilt.
    Difficulty is multi-dimensional: a math error keeps business complexity but
    dials down numerical load; a structural error keeps numbers, dials up structure."""
    weak = weak_skills(profile, k=1)
    if not weak:
        return None
    err = weak[0]
    label, rule = _REMEDIES.get(err, (err.replace("_", " "), "Practise this deliberately."))
    tilt = {
        "calculation": "same business complexity, lighter arithmetic",
        "unit": "a sizing drill where units change (ml->litres, month->year)",
        "structural": "a case that rewards a clean upfront structure",
        "capacity_bridge": "a store/outlet-count guesstimate (demand -> capacity)",
        "sanity_check": "a guesstimate where the final number must be cross-checked",
        "prioritization": "a broad profitability case that forces an 80/20 choice",
    }.get(err, "a case that stresses this skill")
    return {"focus_error": err, "focus_label": label, "remember": rule, "suggested_practice": tilt}


# --- prompt injection (points 8, 9, 16) --------------------------------------
def build_learning_block(profile: Optional[dict], signals: Dict[str, Any], outcome: str) -> str:
    """Compact LEARNING SIGNALS block for the interviewer prompt. Always carries the
    minimum-assistance objective; adds modality-switch guidance when help failed, and
    the learner's recurring weak spots so the coach watches for them."""
    lines = [
        "- MINIMUM ASSISTANCE: give the LEAST help that lets THEM take the next step "
        "themselves. Coached is not easy -- never hand over the answer just to get them there.",
    ]
    if outcome == "failed":
        bm = best_modality(profile)
        switch = (f"this learner has responded to a {bm.replace('_', ' ')} before"
                  if bm else "try a DIFFERENT modality: a short analogy, a concrete example, or do one step WITH them")
        lines.append(f"- your LAST help did NOT land -> do NOT repeat the same kind of hint; {switch}.")
    elif outcome == "worked":
        lines.append("- your last help LANDED -> step back and let them run independently now (fade the support).")
    weak = weak_skills(profile)
    if weak:
        lines.append("- this learner has been shaky on: " + ", ".join(w.replace("_", " ") for w in weak)
                     + " -> coach it lightly if it recurs, don't force it.")
    return "LEARNING SIGNALS:\n" + "\n".join(lines)


# --- post-case debrief (points 27, 28) ---------------------------------------
def build_debrief(profile: Optional[dict]) -> Dict[str, Any]:
    """A concise, personalised debrief from the accumulated tags. Pure; meant to
    enrich the results page beyond the generic scorer output ('MECE knows how I think')."""
    p = profile or {}
    indep = p.get("independence", {})
    errors = p.get("errors", {})

    did_well = None
    strongest = sorted(indep.items(), key=lambda kv: kv[1].get("independent", 0), reverse=True)
    if strongest and strongest[0][1].get("independent", 0) > 0:
        did_well = f"Your {strongest[0][0].replace('_', ' ')} was clean -- you drove it without help."

    watch_this = remember = None
    if errors:
        top = max(errors, key=errors.get)
        label, rule = _REMEDIES.get(top, (top.replace("_", " "), "Practise this deliberately."))
        watch_this = f"Your recurring stumble was {label}."
        remember = rule

    return {
        "did_well": did_well or "You stayed in it and worked the problem end to end.",
        "watch_this": watch_this or "No single recurring error stood out this time.",
        "remember": remember or "Lead with structure, quantify the driver, sanity-check the answer.",
        "next_drill": recommend_next_drill(profile),
    }


# --- longitudinal profile (point 28) -----------------------------------------
def merge_longitudinal_profile(existing: Optional[dict], attempt: Optional[dict]) -> Dict[str, Any]:
    """Deep-ADD two learner profiles (counts accumulate) into the user's lifetime
    skill profile. Pure. Called at submit to roll one attempt into the whole history."""
    out = _blank_profile()
    for src in (existing or {}, attempt or {}):
        for cat in ("errors", "skills"):
            for k, v in (src.get(cat) or {}).items():
                out[cat][k] = out[cat].get(k, 0) + int(v or 0)
        for name, wl in (src.get("modalities") or {}).items():
            m = out["modalities"].setdefault(name, {"worked": 0, "failed": 0})
            m["worked"] += int((wl or {}).get("worked", 0))
            m["failed"] += int((wl or {}).get("failed", 0))
        for skill, bands in (src.get("independence") or {}).items():
            b = out["independence"].setdefault(skill, {})
            for band, c in (bands or {}).items():
                b[band] = b.get(band, 0) + int(c or 0)
    return out
