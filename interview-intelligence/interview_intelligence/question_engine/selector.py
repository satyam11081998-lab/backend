"""Candidate scoring + selection (docs/G_QUESTION_ENGINE.md §3)."""

from __future__ import annotations

from typing import Dict, List, Sequence

from ..role_taxonomy.library import CuratedQuestion, library
from ..textutil import jaccard
from .schemas import PlannedQuestion, SelectionReason


def curated_candidates(section_kind: str, competency_ids: Sequence[str], family_id: str, difficulty_level: int,
                       seniority: str) -> List[PlannedQuestion]:
    lib = library()
    type_ok = {
        "cv": {"cv_deep_dive", "behavioral"}, "functional": {"functional"}, "technical": {"technical", "cv_deep_dive"},
        "behavioral": {"behavioral"}, "situational": {"situational"}, "case": {"case", "estimation"},
        "company": {"company", "motivation"}, "motivation": {"motivation"}, "personal": {"personal"},
        "awareness": {"awareness"},
    }.get(section_kind, set())
    comp_set = set(competency_ids)
    out: List[PlannedQuestion] = []
    for q in lib.bank:
        if q.type not in type_ok:
            continue
        if "*" not in q.families and family_id not in q.families:
            continue
        overlap = comp_set & set(q.competencies)
        if comp_set and not overlap:
            continue
        if abs(q.difficulty - difficulty_level) > 2:
            continue
        if q.seniority == "senior" and seniority in ("intern", "entry"):
            continue
        out.append(_from_curated(q, section_kind, sorted(overlap) or q.competencies))
    return out


def _from_curated(q: CuratedQuestion, section_kind: str, comps: List[str]) -> PlannedQuestion:
    lib = library()
    arch = lib.archetypes.get(q.archetype)
    return PlannedQuestion(
        origin="curated", curated_id=q.id, archetype_id=q.archetype, section_kind=section_kind, competency_ids=comps,
        question_type=q.type if q.type in {"behavioral", "situational", "functional", "technical", "case", "cv_deep_dive",
                                           "motivation", "company", "estimation", "personal",
                                           "awareness"} else "behavioral",  # type: ignore[arg-type]
        difficulty=q.difficulty, seniority=q.seniority, text=q.text, intent=q.intent, expected_evidence=q.evidence,
        strong_signals=q.strong, weak_signals=q.weak, red_flags=list(arch.red_flags) if arch else [],
        must_not_infer=list(arch.must_not_infer) if arch else [], probe_tree=q.probes,
        expected_duration_s=arch.expected_duration_s if arch else 210,
        selection_reason=SelectionReason(explanation="Curated question for this competency."),
    )


def score(item: PlannedQuestion, *, need: Dict[str, float], difficulty_level: int, avoid: Sequence[str],
          used_archetypes: Sequence[str] = ()) -> float:
    comp_need = max((need.get(c, 0.0) for c in item.competency_ids), default=0.0)
    req_link = 1.0 if item.selection_reason.requirement_ids else 0.0
    cv_link = 1.0 if item.selection_reason.claim_ids else 0.0
    # Built on THIS candidate's hobbies/activities or on a real recent headline: beats a generic question.
    grounded = 0.3 if (item.selection_reason.news_ids or item.selection_reason.activity_refs) else 0.0
    diff_fit = 1.0 - min(1.0, abs(item.difficulty - difficulty_level) / 3.0)
    source = {"cv_specific": 1.0, "generated": 0.9, "curated": 0.8}.get(item.origin, 0.6)
    rep = max((jaccard(item.text, a) for a in avoid), default=0.0)
    penalty = 1.0 if rep >= 0.6 else (0.5 if rep >= 0.45 else 0.0)
    # Variety: the same kind of question twice in one interview reads like a template.
    same_kind = 0.18 if item.archetype_id and item.archetype_id in used_archetypes and item.section_kind != "cv" else 0.0
    return (0.40 * comp_need + 0.25 * req_link + 0.15 * diff_fit + 0.10 * cv_link + 0.10 * source + grounded
            - penalty - same_kind)


def select(candidates: List[PlannedQuestion], n: int, *, need: Dict[str, float], difficulty_level: int,
           avoid: Sequence[str], used_archetypes: Sequence[str] = ()) -> tuple[List[PlannedQuestion], List[PlannedQuestion]]:
    """Greedy pick: best score first, re-weighting need after each pick so one competency does
    not monopolise a section. Returns (chosen, reserve)."""
    pool = list(candidates)
    chosen: List[PlannedQuestion] = []
    need = dict(need)
    taken_texts: List[str] = list(avoid)
    archetypes: List[str] = list(used_archetypes)
    while pool and len(chosen) < n:
        pool.sort(key=lambda it: score(it, need=need, difficulty_level=difficulty_level, avoid=taken_texts,
                                       used_archetypes=archetypes), reverse=True)
        best = pool.pop(0)
        if any(jaccard(best.text, c.text) >= 0.6 for c in chosen):
            continue
        chosen.append(best)
        taken_texts.append(best.text)
        if best.archetype_id:
            archetypes.append(best.archetype_id)
        for c in best.competency_ids:
            need[c] = need.get(c, 0.0) * 0.45
    reserve = [p for p in pool if not any(jaccard(p.text, c.text) >= 0.6 for c in chosen)][:6]
    return chosen, reserve
