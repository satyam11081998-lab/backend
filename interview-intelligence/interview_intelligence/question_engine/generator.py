"""Role-specific question generation, constrained by section specs and archetypes (spec §18)."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

from ..ai.guard import wrap_untrusted
from ..ai.prompts import QUESTION_GENERATOR
from ..ai.runner import RunContext, StructuredOutputError, run_structured
from ..interview_engine.grounding import _nums
from ..role_taxonomy.library import library
from .quality import problems
from .schemas import GeneratedQuestion, GeneratedQuestions

SECTION_ARCHETYPES: Dict[str, List[str]] = {
    "cv": ["cv_claim_verification", "quantified_impact_probe", "ownership_deep_dive", "project_deep_dive_technical"],
    "functional": ["concept_explain_apply", "applied_functional_scenario", "metric_definition"],
    "technical": ["technical_concept_depth", "debugging_walkthrough", "system_design_lite", "project_deep_dive_technical"],
    "behavioral": ["ownership_deep_dive", "conflict_behavioral", "failure_reflection", "leadership_influence",
                   "quantified_impact_probe", "learning_agility", "feedback_growth", "ambiguity_initiative",
                   "ethical_judgment", "pressure_resilience", "persuasion_with_data", "teamwork_support",
                   "customer_focus", "innovation_improvement"],
    "situational": ["stakeholder_conflict_situational", "prioritization_tradeoff", "hiring_manager_judgment"],
    "case": ["mini_case", "estimation"],
    "company": ["company_role_simulation", "motivation_role_fit"],
    "motivation": ["motivation_role_fit"],
    "personal": ["interest_deep_dive", "extracurricular_leadership", "achievement_beyond_work", "self_awareness"],
    "awareness": ["news_take", "industry_trend", "business_admire"],
}

SECTION_TYPE = {"cv": "cv_deep_dive", "functional": "functional", "technical": "technical", "behavioral": "behavioral",
                "situational": "situational", "case": "case", "company": "company", "motivation": "motivation",
                "personal": "personal", "awareness": "awareness"}


def generate_for_section(*, section_kind: str, n: int, competencies: List[dict], rubrics: Dict[str, dict],
                         role: dict, seniority: str, difficulty_level: int, technical_depth: int,
                         claims: List[dict], company_facts: List[dict], focus_areas: List[str], avoid: List[str],
                         ctx: RunContext, life: Optional[dict] = None, news: Optional[List[dict]] = None,
                         used_archetypes: Sequence[str] = ()) -> List[GeneratedQuestion]:
    lib = library()
    arch_ids = [a for a in SECTION_ARCHETYPES.get(section_kind, []) if a in lib.archetypes]
    archetypes = [{"id": a, "pattern": lib.archetypes[a].pattern, "intent": lib.archetypes[a].intent,
                   "probe_tree": lib.archetypes[a].probe_tree} for a in arch_ids]
    comp_payload = [{
        "competency_id": c["competency_id"], "name": c.get("name"), "importance": c.get("importance"),
        "sub_areas": c.get("sub_areas", []),
        "what_good_looks_like": (rubrics.get(c["competency_id"]) or {}).get("what_good_looks_like", ""),
    } for c in competencies]
    reqs = [{"id": r.get("id"), "text": r.get("text"), "importance": r.get("importance")}
            for r in role.get("requirements", [])][:25]
    body = (
        f"SECTION: {section_kind} — write {n} questions of type '{SECTION_TYPE.get(section_kind, section_kind)}'.\n"
        f"ROLE: {json.dumps(role.get('identity') or {}, ensure_ascii=False)} | SENIORITY: {seniority}\n"
        f"DIFFICULTY: {difficulty_level}/5 | TECHNICAL DEPTH: {technical_depth}/5\n"
        f"CANDIDATE-CHOSEN FOCUS AREAS: {', '.join(focus_areas) or 'none'}\n"
        f"ARCHETYPES YOU MAY INSTANTIATE:\n{json.dumps(archetypes, ensure_ascii=False)}\n"
        f"COMPETENCIES TO COVER:\n{json.dumps(comp_payload, ensure_ascii=False)}\n"
        "JD REQUIREMENTS:\n" + wrap_untrusted("jd_requirements", "jd", json.dumps(reqs, ensure_ascii=False))
    )
    if claims:
        body += "\nCV CLAIMS TO INVESTIGATE (one question per claim, highest priority first):\n" + wrap_untrusted(
            "cv_claims", "cv", json.dumps(claims, ensure_ascii=False))
    if company_facts:
        body += "\nCOMPANY CONTEXT (with provenance; JD-derived facts outrank others):\n" + wrap_untrusted(
            "company", "company", json.dumps(company_facts, ensure_ascii=False))
    if life and (life.get("activities") or life.get("interests")):
        body += ("\nTHE CANDIDATE BEYOND THE JOB (from their CV — ask about these by name; one item per question):\n"
                 + wrap_untrusted("cv_life", "cv", json.dumps(life, ensure_ascii=False)))
    elif section_kind == "personal":
        body += "\nThe CV lists no hobbies or activities: ask open questions about the person (never invent an interest)."
    if section_kind == "awareness":
        if news:
            body += ("\nRECENT NEWS (real headlines with dates; the ONLY news you may mention — quote facts only as "
                     "stated, add no numbers or details, say roughly when it happened):\n"
                     + wrap_untrusted("news", "news", json.dumps(news, ensure_ascii=False)))
        else:
            body += ("\nNo verified news is available: let the candidate choose the story or trend (\"Pick a recent "
                     "business story you've followed...\"). Never state a news fact yourself.")
    if used_archetypes:
        body += "\nARCHETYPES ALREADY USED ELSEWHERE IN THIS INTERVIEW (prefer others): " + ", ".join(used_archetypes)
    if avoid:
        body += "\nAVOID (already planned or asked before):\n" + "\n".join(f"- {a}" for a in avoid[:40])

    valid_comp = {c["competency_id"] for c in competencies}
    news_ids = {str(x.get("id")) for x in (news or [])}
    news_nums = _nums(" ".join(f"{x.get('title', '')} {x.get('summary', '')}" for x in (news or [])))

    def _validate(gq: GeneratedQuestions) -> List[str]:
        p: List[str] = []
        if len(gq.questions) < max(1, n - 1):
            p.append(f"return at least {max(1, n - 1)} questions")
        for i, q in enumerate(gq.questions):
            bad = problems(q.text)
            if bad:
                p.append(f"question {i}: {', '.join(bad)}")
            if not q.intent or len(q.expected_evidence) < 2 or len(q.probe_tree) < 2:
                p.append(f"question {i}: intent, >=2 expected_evidence and >=2 probe_tree items are required")
            if not set(q.competency_ids) & valid_comp:
                p.append(f"question {i}: competency_ids must come from COMPETENCIES TO COVER")
            if section_kind == "awareness":
                if any(n not in news_ids for n in q.news_ids):
                    p.append(f"question {i}: news_ids must come from RECENT NEWS")
                stray = _nums(q.text) - news_nums
                if stray:
                    p.append(f"question {i}: numbers {sorted(stray)} are not in the news item; state only its facts")
        # (archetype variety is asked for in the prompt and enforced softly at selection — a repair
        # round-trip for it would cost more than it is worth)
        return p[:10]

    try:
        out = run_structured(QUESTION_GENERATOR, body, GeneratedQuestions, ctx, validate=_validate,
                             sim_input={"section": section_kind, "n": n, "competencies": comp_payload,
                                        "claims": claims, "role": role.get("identity") or {},
                                        "difficulty": difficulty_level, "life": life or {}, "news": news or [],
                                        "used_archetypes": list(used_archetypes)})
    except StructuredOutputError:
        return []
    keep = []
    for q in out.questions:
        if problems(q.text):
            continue
        q.competency_ids = [c for c in q.competency_ids if c in valid_comp] or [competencies[0]["competency_id"]]
        keep.append(q)
    return keep
