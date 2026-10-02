"""Deterministic JD checks (spec §55): de-duplicate requirements, re-number ids, derive a
specificity score that does not depend on the model's self-assessment."""

from __future__ import annotations

import re
from typing import List

from ..textutil import jaccard
from .schemas import RoleProfile

_CONTRADICTION_PAIRS = [
    (r"\b(entry[- ]level|fresher|freshers|0\s*[-–]\s*1\s*years?)\b", r"\b([5-9]|1\d)\+?\s*years?\b",
     "Described as entry-level but also asks for several years of experience."),
    (r"\bintern(ship)?\b", r"\b(manage|lead)s?\s+(a\s+)?team\b", "Internship that also asks to lead a team."),
]


def postprocess(profile: RoleProfile, raw_text: str) -> dict:
    # 1) de-duplicate requirements (keep the stronger importance), stable ids
    kept = []
    for r in profile.requirements:
        dup = next((k for k in kept if jaccard(k.text, r.text) >= 0.8), None)
        if dup is None:
            kept.append(r)
        else:
            order = {"must": 0, "should": 1, "nice": 2}
            if order[r.importance] < order[dup.importance]:
                dup.importance = r.importance
    id_map = {}
    for i, r in enumerate(kept, start=1):
        id_map[r.id] = f"R{i}"
        r.id = f"R{i}"
    profile.requirements = kept
    for ic in profile.implied_competencies:
        ic.basis = [id_map.get(b, b) for b in ic.basis]
    for i, rs in enumerate(profile.responsibilities, start=1):
        rs.id = f"RS{i}"

    # 2) specificity from content, not from the model's own label
    concrete = sum(1 for r in kept if len(r.text.split()) >= 3)
    n_resp = len(profile.responsibilities)
    words = len(re.findall(r"\w+", raw_text or ""))
    if concrete < 4 or words < 120:
        spec = "low"
    elif concrete >= 8 and n_resp >= 4 and words >= 250:
        spec = "high"
    else:
        spec = "medium"
    profile.quality.specificity = spec

    # 3) obvious contradictions the model may have missed
    low = (raw_text or "").lower()
    notes: List[str] = list(profile.quality.contradictions)
    for a, b, msg in _CONTRADICTION_PAIRS:
        if re.search(a, low) and re.search(b, low) and msg not in notes:
            notes.append(msg)
    profile.quality.contradictions = notes

    # 4) never claim seniority certainty without a basis
    if profile.seniority.level != "unknown" and not profile.seniority.basis.strip():
        profile.seniority.confidence = "low"
    return profile.model_dump()
