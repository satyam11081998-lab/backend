"""'Why was I asked this?' (spec §46) — built from the question's recorded provenance, not
from a fresh model guess, so the explanation is exactly why it was selected."""

from __future__ import annotations

from typing import Dict, List

from ..textutil import truncate


def explain(meta: dict, *, requirements: Dict[str, str], claims: Dict[str, str], competency_names: List[str],
            company_facts: Dict[str, str]) -> str:
    sr = meta.get("selection_reason") or {}
    because: List[str] = []
    for rid in (sr.get("requirement_ids") or [])[:2]:
        if rid in requirements:
            because.append(f"the job description asks for \"{truncate(requirements[rid], 110)}\"")
    for cid in (sr.get("claim_ids") or [])[:1]:
        if cid in claims:
            because.append(f"your CV says \"{truncate(claims[cid], 110)}\"")
    for fid in (sr.get("company_fact_ids") or [])[:1]:
        if fid in company_facts:
            because.append(f"of the company context: \"{truncate(company_facts[fid], 100)}\"")
    origin = meta.get("origin")
    if not because:
        if origin == "fixed":
            because.append("it is the standard opening/closing of an interview")
        elif origin == "curated":
            because.append("it is a proven question for this competency at your level")
        else:
            because.append("the interview plan needed evidence for this competency")
    tested = ", ".join(competency_names[:3]) or "the role's core competencies"
    looking = meta.get("expected_evidence") or []
    s = f"This question was selected because {' and '.join(because)}. The interviewer was testing {tested}"
    if meta.get("intent"):
        s += f" — {meta['intent'].rstrip('.').lower()}"
    if looking:
        s += f". They were listening for: {', '.join(looking[:4])}"
    return s + "."
