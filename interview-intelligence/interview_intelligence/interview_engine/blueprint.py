"""Interview Blueprint Engine (spec §17) — the `prepare_session` job.

JD + CV + role + seniority + mode + difficulty + history  ->  competency model, frozen
rubrics, time-boxed sections, planned questions with assessment purpose, CV claims to
investigate, adaptive reserves, exit criteria, QA report.
"""

from __future__ import annotations

import json
import uuid
from typing import Dict, List, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..access import flags
from ..ai.prompts import BLUEPRINT_QA
from ..ai.runner import RunContext, StructuredOutputError, run_structured
from ..company_intelligence.service import build_profile
from ..competency_engine.model import (build_rubrics, classify_role, cv_strength_summary, map_competencies,
                                        min_evidence)
from ..competency_engine.schemas import MappedCompetency
from ..db.models import (Claim, Document, InterviewBlueprint, InterviewExchange, InterviewSession, InterviewState,
                         utcnow)
from ..documents.analysis import analyze
from ..jobs import progress
from ..jobs.queue import register
from ..question_engine.generator import generate_for_section
from ..question_engine.quality import problems as q_problems
from ..question_engine.schemas import BlueprintQA, PlannedQuestion, SelectionReason
from ..question_engine import selector as qsel
from ..question_engine.selector import curated_candidates
from ..role_taxonomy.library import library
from ..textutil import jaccard, truncate
from ..versions import versions
from .lifecycle import transition
from .modes import (BREADTH_SECTIONS, MODES, SECTION_TITLES, InterviewConfig, allocate_sections,
                    difficulty_vector, persona)
from .state import new_state

CROSS_CUTTING = ["communication", "handling_challenge"]

SECTION_PREFS: Dict[str, List[str]] = {
    "ownership": ["cv", "behavioral"], "leadership": ["behavioral", "cv"],
    "stakeholder_management": ["situational", "behavioral", "cv"], "collaboration": ["behavioral", "situational"],
    "conflict_management": ["behavioral", "situational"], "learning_reflection": ["behavioral", "cv"],
    "motivation_role_fit": ["motivation", "company", "cv", "behavioral"],
    "role_company_understanding": ["company", "motivation", "functional"],
    "situational_judgment": ["situational", "behavioral", "case"], "prioritization": ["situational", "case", "behavioral"],
    "structured_thinking": ["case", "functional", "situational", "technical"],
    "problem_solving": ["case", "technical", "functional", "situational"],
    "analytical_reasoning": ["functional", "case", "technical"], "quantitative_reasoning": ["case", "functional"],
    "commercial_judgment": ["case", "situational", "functional", "company"],
    "case_structuring": ["case", "functional"], "synthesis_recommendation": ["case", "functional"],
    "market_sizing_estimation": ["case", "functional"], "strategic_thinking": ["case", "functional", "company"],
}
FALLBACK_ORDER = ["functional", "situational", "behavioral", "cv", "case", "technical", "company", "motivation"]


def _sections_for(m: MappedCompetency, present: List[str], technical_role: bool) -> List[str]:
    lib = library()
    cid = m.competency_id
    if cid in CROSS_CUTTING:
        return []
    if cid in SECTION_PREFS:
        prefs = SECTION_PREFS[cid]
    else:
        c = lib.competency(cid) or lib.competency(m.parent or "")
        cat = c.category if c else ("technical" if technical_role else "functional")
        prefs = {"technical": ["technical", "functional", "cv"], "functional": ["functional", "case", "cv", "technical"],
                 "behavioral": ["behavioral", "cv", "situational"], "core": ["functional", "situational", "case"]
                 }.get(cat, ["functional"])
    for p in prefs:
        if p in present:
            return [p]
    for p in FALLBACK_ORDER:
        if p in present:
            return [p]
    return []


def _claim_score(c: dict, flags_map: Dict[str, List[str]], linked: set) -> float:
    s = 4 - int(c.get("verification_priority") or 2)
    s += 1 if c.get("quantified") else 0
    s += 1 if c.get("type") in ("leadership", "impact", "scale", "ownership") else 0
    s += 1.5 if c.get("id") in linked else 0
    s += 2 if any(f in ("unrealistic_metric", "scale_inconsistent_with_seniority", "ownership_unclear")
                  for f in flags_map.get(c.get("id"), [])) else 0
    s += 0.5 if c.get("vague") else 0
    return s


def _diverse_claims(claims_sorted: List[dict], k: int) -> List[dict]:
    """The top claims, spread across the CV: the best claim of each role/project first, then the
    second-best of each, and so on — so a CV section never spends every question on one job."""
    buckets: Dict[str, List[dict]] = {}
    for c in claims_sorted:
        buckets.setdefault(c.get("experience_id") or f"_{c.get('type') or 'other'}", []).append(c)
    out: List[dict] = []
    while len(out) < k and any(buckets.values()):
        for key in list(buckets):
            if buckets[key] and len(out) < k:
                out.append(buckets[key].pop(0))
    return out


PERSONAL_COMPETENCIES = ["learning_reflection", "leadership", "ownership", "collaboration", "motivation_role_fit",
                         "communication"]
AWARENESS_COMPETENCIES = ["commercial_judgment", "strategic_thinking", "role_company_understanding",
                          "synthesis_recommendation", "analytical_reasoning", "structured_thinking", "communication"]


def _breadth_competencies(kind: str, model: List[MappedCompetency]) -> List[str]:
    prefs = PERSONAL_COMPETENCIES if kind == "personal" else AWARENESS_COMPETENCIES
    have = {m.competency_id for m in model}
    picked = [c for c in prefs if c in have][:3]
    return picked or [m.competency_id for m in model][:1]


def _life(cv: dict) -> dict:
    """Hobbies, interests and activities from the CV (personal data: never leaves this interview)."""
    acts = [{"id": a.get("id", ""), "kind": a.get("kind", "other"), "text": truncate(a.get("text", ""), 200),
             "organization": a.get("organization", "")} for a in (cv.get("activities") or []) if a.get("text")][:8]
    return {"activities": acts, "interests": [truncate(str(i), 60) for i in (cv.get("interests") or []) if i][:8]}


_NEWS_STOP = {"the", "and", "for", "with", "from", "this", "that", "into", "over", "after", "amid", "says", "said",
              "will", "has", "have", "its", "are", "was", "new", "india", "indian", "company", "companies"}


def _news_words(text: str) -> set:
    import re
    return {w for w in re.findall(r"[a-z][a-z0-9&-]{2,}", (text or "").lower()) if w not in _NEWS_STOP}


# Business news only: an interview must never ask for a view on politics, elections, conflict or
# religion (FAIRNESS), so political and geopolitical stories are left out before anything is asked.
NEWS_CATEGORIES = {"", "business", "macro", "micro", "tech", "jobs", "economy", "markets", "companies", "startups"}
_POLITICAL = None


def _political(text: str) -> bool:
    import re
    global _POLITICAL
    if _POLITICAL is None:
        _POLITICAL = re.compile(
            r"\b(elections?|electoral|minister|ministers|chief minister|party|parties|bjp|congress|aap|tmc|"
            r"parliament|lok sabha|rajya sabha|assembly polls?|polls?|votes?|voting|campaign trail|war|wars|military|"
            r"army|troops|missile|terror\w*|protests?|riots?|religio\w*|caste|communal|ceasefire|sanction\w*)\b",
            re.IGNORECASE)
    return bool(_POLITICAL.search(text or ""))


def pick_news(rows: List[dict], jd: dict, fam_name: str, industry: str, *, k: int = 3) -> List[dict]:
    """The recent business headlines most relevant to this role: overlap with the role's title,
    function, industry and JD keywords first, the host's own newsworthiness score second. Political,
    geopolitical and other non-business stories are never used."""
    ident = jd.get("identity") or {}
    role_words = _news_words(" ".join([ident.get("title", ""), ident.get("function", ""), ident.get("company", ""),
                                       fam_name, industry,
                                       " ".join(k.get("term", "") for k in (jd.get("keywords") or [])[:20])]))
    scored = []
    for i, r in enumerate(rows[:60]):
        if str(r.get("category") or "").lower() not in NEWS_CATEGORIES:
            continue
        text = f"{r.get('title', '')} {r.get('summary') or r.get('description') or ''} " \
               f"{' '.join(r.get('keywords') or [])} {r.get('category', '')}"
        if _political(f"{r.get('title', '')} {r.get('summary') or r.get('description') or ''}"):
            continue
        rel = len(_news_words(text) & role_words)
        scored.append((rel, float(r.get("score") or r.get("gd_worthiness_score") or 0), -i, r))
    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    out = []
    for n, (_, _, _, r) in enumerate(scored[:k], start=1):
        out.append({"id": f"N{n}", "title": truncate(str(r.get("title", "")), 200),
                    "summary": truncate(str(r.get("summary") or r.get("description") or ""), 300),
                    "source": str(r.get("source") or r.get("source_name") or "")[:60],
                    "date": str(r.get("published_at") or "")[:10]})
    return out


def agenda_of(sections: List[dict]) -> List[dict]:
    """The running order the interviewer announces at the start (kinds and rough minutes only)."""
    return [{"kind": s["kind"], "title": s["title"], "minutes": max(1, round(s["budget_s"] / 60))}
            for s in sections if s["kind"] not in ("intro", "closing") and s.get("items")]


def _cv_fallback_question(claim: dict, comps: List[str]) -> PlannedQuestion:
    lib = library()
    arch = lib.archetypes["cv_claim_verification"]
    return PlannedQuestion(
        origin="cv_specific", archetype_id=arch.id, section_kind="cv", competency_ids=comps,
        question_type="cv_deep_dive", difficulty=3,
        text=f"Your CV says: \"{truncate(claim.get('text', ''), 160)}\". Take me through it — what was the situation, "
             f"and what exactly was your part?",
        intent="Test whether this CV claim holds up and what the candidate personally did.",
        expected_evidence=list(arch.expected_evidence), strong_signals=list(arch.strong_signals),
        weak_signals=list(arch.weak_signals), red_flags=list(arch.red_flags), must_not_infer=list(arch.must_not_infer),
        probe_tree=list(arch.probe_tree), expected_duration_s=arch.expected_duration_s,
        selection_reason=SelectionReason(claim_ids=[claim.get("id", "")],
                                         explanation="A high-priority claim on your CV."),
    )


def _from_generated(g, section_kind: str) -> PlannedQuestion:
    lib = library()
    arch = lib.archetypes.get(g.archetype_id)
    return PlannedQuestion(
        origin="cv_specific" if (section_kind == "cv" and g.claim_ids) else "generated",
        archetype_id=g.archetype_id, section_kind=section_kind, competency_ids=g.competency_ids,
        sub_competency=g.sub_competency, question_type=g.question_type, difficulty=g.difficulty, text=g.text.strip(),
        intent=g.intent, expected_evidence=g.expected_evidence, strong_signals=g.strong_signals,
        weak_signals=g.weak_signals, red_flags=g.red_flags,
        must_not_infer=g.must_not_infer or (list(arch.must_not_infer) if arch else []), probe_tree=g.probe_tree,
        expected_duration_s=arch.expected_duration_s if arch else 210,
        selection_reason=SelectionReason(requirement_ids=g.requirement_ids, claim_ids=g.claim_ids,
                                         news_ids=g.news_ids, activity_refs=g.activity_refs,
                                         explanation=g.why_this_question),
    )


def _fixed(kind: str, role_title: str, n_minutes: int) -> PlannedQuestion:
    if kind == "intro":
        return PlannedQuestion(
            origin="fixed", archetype_id="intro_walkthrough", section_kind="intro",
            competency_ids=["communication", "motivation_role_fit"], question_type="intro", difficulty=1,
            text="To start, could you briefly walk me through your background and what brings you to this role?",
            intent="Warm-up; observe structure and the career narrative.",
            expected_evidence=["coherent narrative", "link to the role", "appropriate length"],
            strong_signals=["concise, chronological, ends with why this role"],
            weak_signals=["reads the CV line by line", "no link to the role"], probe_tree=[],
            expected_duration_s=150, selection_reason=SelectionReason(explanation="Standard opening question."))
    return PlannedQuestion(
        origin="fixed", archetype_id="closing_questions", section_kind="closing", competency_ids=[],
        question_type="closing", difficulty=1, text="Before we wrap up, what questions do you have for me?",
        intent="Close the interview.", expected_duration_s=120,
        selection_reason=SelectionReason(explanation="Standard close."))


def _history_avoid(db: Session, user_id: uuid.UUID, exclude_session: uuid.UUID) -> List[str]:
    rows = db.execute(
        select(InterviewExchange.question_text)
        .join(InterviewSession, InterviewSession.id == InterviewExchange.session_id)
        .where(InterviewSession.user_id == user_id, InterviewSession.id != exclude_session)
        .order_by(InterviewExchange.started_at.desc()).limit(60)
    ).scalars().all()
    return [r for r in rows if r]


def _deterministic_qa(sections: List[dict], model: List[MappedCompetency], budget_s: int,
                      repair_log: List[dict] | None = None) -> List[dict]:
    issues: List[dict] = []
    planned = [it for s in sections for it in s["items"]]
    covered = {c for it in planned for c in it["competency_ids"]}
    unplaced = {r["competency_id"]: r.get("reason", "") for r in (repair_log or []) if r.get("action") == "not_placed"}
    for m in model:
        if m.importance in ("critical", "high") and m.competency_id not in covered and m.competency_id not in CROSS_CUTTING:
            issues.append({"issue": "coverage", "competency_id": m.competency_id, "severity": "high",
                           "detail": f"No planned question targets {m.name}.",
                           # 'no_redundant_slot' = every planned question already carries a critical/high
                           # competency on its own; the adaptive engine may still reach this one.
                           "unresolved_reason": unplaced.get(m.competency_id, "not_attempted")})
    texts = [it["text"] for it in planned]
    for i, t in enumerate(texts):
        for j in range(i):
            if jaccard(t, texts[j]) >= 0.6:
                issues.append({"issue": "duplicate", "qid": planned[i]["qid"], "severity": "medium",
                               "detail": f"Near-duplicate of {planned[j]['qid']}."})
    total = sum(s["budget_s"] for s in sections)
    if abs(total - budget_s) > 0.15 * budget_s:
        issues.append({"issue": "duration", "severity": "medium", "detail": f"Planned {total}s vs budget {budget_s}s."})
    for it in planned:
        for p in q_problems(it["text"]):
            issues.append({"issue": p, "qid": it["qid"], "severity": "high", "detail": it["text"][:120]})
    return issues


def _fresh_candidates(kind: str, m: MappedCompetency, taken: List[str], *, fam_id: str, dv, seniority: str,
                      rubrics: Dict[str, dict], jd: dict, focus: List[str], ctx: RunContext) -> List[dict]:
    """Curated questions for one competency in one section; if none, ONE generator call."""
    out = [c.model_dump() for c in curated_candidates(kind, [m.competency_id], fam_id, dv.level, seniority)]
    if not out:
        gen = generate_for_section(section_kind=kind, n=1, competencies=[m.model_dump()], rubrics=rubrics, role=jd,
                                   seniority=seniority, difficulty_level=dv.level, technical_depth=dv.technical_depth,
                                   claims=[], company_facts=[], focus_areas=focus, avoid=taken, ctx=ctx)
        out = [_from_generated(g, kind).model_dump() for g in gen]
    return [c for c in out if not q_problems(c["text"]) and m.competency_id in c["competency_ids"]]


def _repair_coverage(sections: List[dict], model: List[MappedCompetency], model_by_id: Dict[str, MappedCompetency], *,
                     avoid: List[str], technical_role: bool, claim_focus: bool, fresh,
                     max_fresh: int = 6) -> List[dict]:
    """Spec §50/§56: every critical/high competency should have at least one planned question.

    Selection optimises a weighted score per section, so with few slots an important
    competency can end up with none. For each such gap (critical first) we swap out a
    REDUNDANT item — one whose competencies are all covered by other items or are not
    critical/high — using, in order: the section's reserve, a curated question, one
    generated question. Never removes the last question for another important competency,
    never removes CV-claim questions in claim-focused modes (nor the last one elsewhere), never touches the
    personal or business-awareness sections, never changes a question's text
    or relabels its competencies. Gaps that cannot be placed are logged, not hidden."""
    log: List[dict] = []
    # The person beyond the CV and business awareness are there for breadth, not coverage: never
    # traded away for another competency question.
    body = [s for s in sections if s["kind"] not in ("intro", "closing", *BREADTH_SECTIONS)]
    if not body:
        return log
    fresh_left = [max_fresh]  # bounds extra model calls (curated lookups are free)

    def important(cid: str) -> bool:
        m = model_by_id.get(cid)
        return bool(m and m.importance in ("critical", "high") and cid not in CROSS_CUTTING)

    def counts() -> Dict[str, int]:
        cnt: Dict[str, int] = {}
        for s in sections:
            for it in s["items"]:
                for c in it["competency_ids"]:
                    cnt[c] = cnt.get(c, 0) + 1
        return cnt

    def texts(exclude: dict | None = None) -> List[str]:
        return [it["text"] for s in sections for it in s["items"] if it is not exclude] + list(avoid)

    def cv_left(sec: dict) -> int:
        return sum(1 for it in sec["items"] if it.get("origin") == "cv_specific")

    def victim(sec: dict, cnt: Dict[str, int]) -> int | None:
        best: Tuple[int, int] | None = None
        for idx, it in enumerate(sec["items"]):
            if it.get("origin") == "fixed":
                continue
            if it.get("origin") == "cv_specific" and (claim_focus or cv_left(sec) <= 1):
                continue  # questions on the CV are the interview's own; keep at least one in every mode
            if any(important(c) and cnt.get(c, 0) < 2 for c in it["competency_ids"]):
                continue  # it is the only question for some important competency
            redundancy = min((cnt.get(c, 0) for c in it["competency_ids"]), default=99)
            if best is None or redundancy > best[0]:
                best = (redundancy, idx)
        return best[1] if best else None

    gaps = [m for m in model if important(m.competency_id)]
    gaps.sort(key=lambda m: 0 if m.importance == "critical" else 1)
    for m in gaps:
        cnt = counts()
        if cnt.get(m.competency_id):
            continue
        pref = _sections_for(m, [s["kind"] for s in body], technical_role)
        order = sorted(body, key=lambda s: (0 if s["kind"] in pref else 1))
        placed = False
        for sec in order:
            vi = victim(sec, cnt)
            if vi is None:
                continue
            old = sec["items"][vi]
            taken = texts(exclude=old)
            pool = [r for r in sec["reserve"] if m.competency_id in r["competency_ids"]]
            source = "reserve"
            cand = next((r for r in pool if not any(jaccard(r["text"], t) >= 0.6 for t in taken)), None)
            if cand is None and fresh_left[0] > 0:
                fresh_left[0] -= 1
                source = "fresh"
                cand = next((c for c in fresh(sec["kind"], m, taken)
                             if not any(jaccard(c["text"], t) >= 0.6 for t in taken)), None)
            if cand is None:
                continue
            if source == "reserve":
                sec["reserve"] = [r for r in sec["reserve"] if r is not cand]
            new = dict(cand)
            new["qid"], new["section_kind"] = old["qid"], sec["kind"]
            new.setdefault("selection_reason", {})
            sec["items"][vi] = new
            sec["reserve"].insert(0, {**old, "qid": f"{old['qid']}_r"})  # still available adaptively
            if m.competency_id not in sec["competency_ids"]:
                sec["competency_ids"].append(m.competency_id)
            log.append({"competency_id": m.competency_id, "action": "replaced", "section": sec["kind"],
                        "qid": old["qid"], "removed_competencies": old["competency_ids"], "source": source})
            placed = True
            break
        if not placed:
            log.append({"competency_id": m.competency_id, "action": "not_placed", "reason": "no_redundant_slot"})
    return log


def build_blueprint(db: Session, sess: InterviewSession) -> None:
    lib = library()
    ctx = RunContext(user_id=sess.user_id, session_id=sess.id)
    cfg = InterviewConfig(**(sess.config or {}))
    mode = MODES[cfg.mode]
    key = f"prep:{sess.id}"
    progress.start(key, [
        ("inputs", "Reading your CV and the job description", 1.0),
        ("role", "Identifying the role and its level", progress.eta_for(db, "role_classifier")),
        ("match", "Matching your CV to what the role needs", progress.eta_for(db, "competency_mapper")),
        ("rubrics", "Writing a scoring guide for each skill", progress.eta_for(db, "rubric_builder")),
        ("questions", "Preparing your interviewer", progress.eta_for(db, "question_generator", calls=4)),
        ("check", "A last check for repeats and fairness", progress.eta_for(db, "blueprint_qa")),
    ])
    progress.begin(key, "inputs")

    cv_doc = db.get(Document, sess.cv_document_id)
    jd_doc = db.get(Document, sess.jd_document_id)
    cv_a = analyze(db, cv_doc)
    jd_a = analyze(db, jd_doc)
    if cv_a.status != "ready" or jd_a.status != "ready":
        bad = "CV" if cv_a.status != "ready" else "job description"
        raise RuntimeError(f"{bad} analysis is not available ({cv_a.status}/{jd_a.status})")
    cv, jd = cv_a.result, jd_a.result

    progress.begin(key, "role")
    rc = classify_role(jd, ctx)
    fam = lib.family(rc.primary_family)
    seniority = (jd.get("seniority") or {}).get("level") or "unknown"
    if seniority == "unknown":
        seniority = cv.get("seniority_estimate") or "unknown"
    title_seen = (jd.get("identity") or {}).get("title") or (jd.get("identity") or {}).get("role_name") or ""
    progress.fact(key, progress.join([title_seen, fam.name, progress.seniority_label(seniority)]), step_id="role")

    progress.begin(key, "match")
    model = map_competencies(jd, cv, fam.id, cfg.mode, seniority, cfg.target_competencies, ctx)
    fit = cv_strength_summary(model)
    # What the CV and the role have in common — never what the interview will ask (that stays the
    # interviewer's to reveal, as in a real interview).
    progress.fact(key, progress.join([
        f"{progress.plural(fit['competencies_identified'], 'skill')} this role needs",
        f"{fit['strong_in_cv']} clearly backed by your CV" if fit["strong_in_cv"] else ""]), step_id="match")

    progress.begin(key, "rubrics")
    rubrics, rubric_hash = build_rubrics(model, jd, seniority, ctx)
    progress.fact(key, progress.plural(len(rubrics), "scoring guide"), step_id="rubrics")

    company = None
    if flags.flag(db, "company_intel.enabled"):
        company = build_profile(db, sess.user_id, cfg.company_name, jd, cfg.company_notes)
    company_facts = (company.facts if company else [])[:12]

    dv = difficulty_vector(cfg.difficulty, cfg.depth, mode)
    avg_item_s = 150 + 60 * dv.max_probes
    plans = allocate_sections(cfg, fam.section_bias, rc.technical_role, bool(fam.case_types), bool(company_facts),
                              avg_item_s)
    to_write = [p for p in plans if p.kind not in ("intro", "closing")]
    progress.begin(key, "questions", eta_s=progress.eta_for(db, "question_generator", calls=len(to_write) or 1))
    present = [p.kind for p in plans if p.kind not in ("intro", "closing")]
    for m in model:
        for k in _sections_for(m, present, rc.technical_role):
            for p in plans:
                if p.kind == k:
                    p.competency_ids.append(m.competency_id)
    for p in plans:
        if p.kind in BREADTH_SECTIONS:
            p.competency_ids = _breadth_competencies(p.kind, model)
    # A section nobody needs gets the model's most important unassigned competencies, or is dropped.
    for p in plans:
        if p.kind not in ("intro", "closing") and not p.competency_ids:
            p.competency_ids = [m.competency_id for m in model if m.competency_id not in CROSS_CUTTING][:2]

    linked = {cid for m in model for cid in m.cv_claim_ids}
    flags_map = cv.get("claim_flags") or {}
    claims_sorted = sorted(cv.get("claims", []), key=lambda c: -_claim_score(c, flags_map, linked))
    cv_plan = next((p for p in plans if p.kind == "cv"), None)
    n_claims = 0
    if cv_plan:
        n_claims = cv_plan.n_items if not mode.claim_focus else max(cv_plan.n_items, min(8, len(claims_sorted)))
        cv_plan.n_items = max(cv_plan.n_items, min(n_claims, len(claims_sorted)) or cv_plan.n_items)
    claims_to_investigate = _diverse_claims(claims_sorted, max(n_claims, 3))
    life = _life(cv)
    news: List[dict] = []
    if any(p.kind == "awareness" for p in plans):
        from ..host import recent_news
        news = pick_news(recent_news(), jd, fam.name, rc.industry or "")

    model_by_id = {m.competency_id: m for m in model}
    avoid = _history_avoid(db, sess.user_id, sess.id)
    need = {m.competency_id: {"critical": 1.0, "high": 0.8, "medium": 0.55, "low": 0.3}[m.importance] for m in model}
    for t in cfg.target_competencies:
        need[t] = 1.2

    sections: List[dict] = []
    qn = 0
    role_title = (jd.get("identity") or {}).get("title") or (jd.get("identity") or {}).get("role_name") or ""
    planned_texts: List[str] = []
    planned_archetypes: List[str] = []

    def add_section(kind: str, budget: int, items: List[PlannedQuestion], reserve: List[PlannedQuestion], comps: List[str]):
        nonlocal qn
        out_items = []
        for it in items:
            qn += 1
            it.qid = f"Q{qn}"
            it.section_kind = kind
            out_items.append(it.model_dump())
            planned_texts.append(it.text)
        out_res = []
        for r in reserve:
            r.qid = f"R{qn}_{len(out_res) + 1}"
            r.section_kind = kind
            out_res.append(r.model_dump())
        sections.append({"id": kind, "kind": kind, "title": SECTION_TITLES[kind], "budget_s": budget,
                         "competency_ids": comps, "items": out_items, "reserve": out_res})

    for p in plans:
        if p.kind == "intro":
            add_section("intro", p.budget_s, [_fixed("intro", role_title, cfg.duration_minutes)], [], [])
            continue
        if p.kind == "closing":
            add_section("closing", p.budget_s, [_fixed("closing", role_title, cfg.duration_minutes)], [], [])
            continue
        comps = [model_by_id[c].model_dump() for c in p.competency_ids if c in model_by_id]
        claims_for = claims_to_investigate[:p.n_items + 1] if p.kind == "cv" else []
        progress.portion(key, to_write.index(p), len(to_write))
        generated = generate_for_section(
            section_kind=p.kind, n=p.n_items + 1, competencies=comps, rubrics=rubrics, role=jd, seniority=seniority,
            difficulty_level=dv.level, technical_depth=dv.technical_depth, claims=claims_for,
            company_facts=company_facts if p.kind in ("company", "situational") else [],
            focus_areas=cfg.focus_areas, avoid=planned_texts + avoid, ctx=ctx,
            life=life if p.kind == "personal" else None, news=news if p.kind == "awareness" else None,
            used_archetypes=planned_archetypes,
        ) if comps else []
        cands = [_from_generated(g, p.kind) for g in generated]
        if p.kind == "cv":
            covered_claims = {c for it in cands for c in it.selection_reason.claim_ids}
            for c in claims_for:
                if c.get("id") not in covered_claims:
                    comp_ids = [m.competency_id for m in model if c.get("id") in m.cv_claim_ids][:2] or ["ownership"]
                    cands.append(_cv_fallback_question(c, comp_ids))
        cands += curated_candidates(p.kind, p.competency_ids, fam.id, dv.level, seniority)
        cands = [c for c in cands if not q_problems(c.text)]
        chosen, reserve = qsel.select(cands, p.n_items, need=need, difficulty_level=dv.level, avoid=planned_texts + avoid,
                                      used_archetypes=planned_archetypes)
        planned_archetypes.extend(c.archetype_id for c in chosen if c.archetype_id)
        for c in chosen:
            for cid in c.competency_ids:
                need[cid] = need.get(cid, 0.0) * 0.45
        if chosen:
            add_section(p.kind, p.budget_s, chosen, reserve, p.competency_ids)

    repair_log = _repair_coverage(
        sections, model, model_by_id, avoid=avoid, technical_role=rc.technical_role, claim_focus=mode.claim_focus,
        fresh=lambda kind, m, taken: _fresh_candidates(kind, m, taken, fam_id=fam.id, dv=dv, seniority=seniority,
                                                       rubrics=rubrics, jd=jd, focus=cfg.focus_areas, ctx=ctx))
    total_budget = cfg.duration_minutes * 60
    det_issues = _deterministic_qa(sections, model, total_budget, repair_log)

    used_news = any((it.get("selection_reason") or {}).get("news_ids") for s in sections for it in s["items"])
    progress.fact(key, "Built around your CV, the job description" + (" and this month's business news"
                                                                       if used_news else ""), step_id="questions")
    progress.detail(key, "")
    progress.begin(key, "check")

    # LLM QA judge on the plan; replace high-severity items from the section reserve once.
    llm_qa: dict = {"passed": True, "issues": []}
    plan_view = [{"qid": it["qid"], "section": s["kind"], "competencies": it["competency_ids"],
                  "difficulty": it["difficulty"], "text": it["text"]} for s in sections for it in s["items"]
                 if s["kind"] not in ("intro", "closing")]
    try:
        qa = run_structured(BLUEPRINT_QA, f"ROLE: {json.dumps(jd.get('identity') or {}, ensure_ascii=False)} | "
                            f"SENIORITY: {seniority} | MODE: {cfg.mode} | DIFFICULTY: {cfg.difficulty}\n"
                            f"PLANNED QUESTIONS:\n{json.dumps(plan_view, ensure_ascii=False)}",
                            BlueprintQA, ctx, sim_input={"plan": plan_view})
        llm_qa = qa.model_dump()
        bad = {i.qid for i in qa.issues if i.severity == "high" and i.qid}
        progress.fact(key, "Done", step_id="check")
        for s in sections:
            for idx, it in enumerate(list(s["items"])):
                if it["qid"] in bad and s["reserve"]:
                    rep = s["reserve"].pop(0)
                    rep["qid"] = it["qid"]
                    s["items"][idx] = rep
    except StructuredOutputError:
        llm_qa = {"passed": None, "issues": [], "notes": ["QA judge unavailable; deterministic checks only."]}

    comp_view = []
    for m in model:
        comp_view.append({**m.model_dump(), "min_evidence": min_evidence(m.importance),
                          "planned_questions": sum(1 for s in sections for it in s["items"]
                                                   if m.competency_id in it["competency_ids"]),
                          "probe_if": (rubrics.get(m.competency_id) or {}).get("weak_signals", [])[:3],
                          "stop_if": f"{min_evidence(m.importance)} strong signal(s) observed"})

    blueprint = {
        "version": versions()["blueprint"],
        "config": cfg.model_dump(),
        "role": {"title": role_title, "family": fam.id, "family_name": fam.name, "sub_family": rc.sub_family,
                 "industry": rc.industry, "technical": rc.technical_role, "seniority": seniority,
                 "hybrid": rc.is_hybrid, "classification": rc.model_dump()},
        "persona": persona(cfg.difficulty, mode),
        "difficulty_vector": dv.model_dump() | {"level": dv.level, "max_probes": dv.max_probes},
        "competencies": comp_view,
        "cross_cutting": CROSS_CUTTING,
        "sections": sections,
        "claims_to_investigate": [{"claim_id": c.get("id"), "text": c.get("text"), "slots": c.get("slots", {}),
                                   "flags": flags_map.get(c.get("id"), []),
                                   "ladder": library().archetypes["cv_claim_verification"].probe_tree}
                                  for c in claims_to_investigate],
        "requirements_to_test": [{"requirement_id": r.get("id"), "text": r.get("text"), "importance": r.get("importance"),
                                  "competency_ids": [m.competency_id for m in model if r.get("id") in m.requirement_ids]}
                                 for r in jd.get("requirements", []) if r.get("importance") in ("must", "should")][:20],
        "adaptive_branches": [
            "probe when the analyzer reports gaps against the item's expected evidence and probes remain",
            "skip remaining items whose competencies already meet minimum evidence (unless critical and untested)",
            "after 2 consecutive strong answers raise difficulty one level; after 2 weak answers lower it",
            "clarify any numeric contradiction with CV or earlier answers once, neutrally",
            "a declined question marks its competency 'not tested', never 'weak'",
        ],
        "exit_criteria": {"time": "closing section starts when active time reaches budget minus closing budget",
                          "early_end": "candidate may end at any time; report confidence becomes limited",
                          "cost": "per-session AI cost cap switches to deterministic prompts and steers to closing"},
        "company_facts": company_facts,
        "news": news,
        "agenda": agenda_of(sections),
        "qa": {"deterministic": det_issues, "llm": llm_qa, "coverage_repair": repair_log},
    }
    # Re-acquire the row lock: the user may have abandoned the session while we were building.
    db.refresh(sess, with_for_update=True)
    if sess.status != "analyzing":
        return
    _record_generated(db, sections, fam.id)
    bp = InterviewBlueprint(session_id=sess.id, role_profile=jd, competency_model={"competencies": comp_view},
                            rubric=rubrics, rubric_hash=rubric_hash, blueprint=blueprint, qa_report=blueprint["qa"],
                            engine_versions=versions())
    db.add(bp)
    db.add(InterviewState(session_id=sess.id, version=0, state=new_state(blueprint)))
    for c in claims_to_investigate:
        db.add(Claim(session_id=sess.id, claim_key=str(c.get("id"))[:16], source="cv", text=c.get("text", "")[:2000],
                     claim_type=c.get("type", "other"), slots=c.get("slots", {}) or {},
                     priority=int(c.get("verification_priority") or 2)))
    summary = cv_strength_summary(model)
    summary.update({
        "role": {"title": role_title, "family": fam.name, "seniority": seniority, "industry": rc.industry},
        "jd_quality": (jd.get("quality") or {}).get("specificity", "medium"),
        "jd_warnings": (jd.get("quality") or {}).get("contradictions", [])[:3],
        "cv_warnings": [i.get("detail") for i in (cv.get("timeline_issues") or [])][:3],
        "sections": [{"kind": s["kind"], "title": s["title"], "minutes": round(s["budget_s"] / 60, 1),
                      "questions": len(s["items"])} for s in sections],
        "claims_to_investigate": len(claims_to_investigate),
        "company_context": bool(company_facts),
        # Disclosed up front (spec §20): key competencies this duration cannot plan for.
        "not_planned": [model_by_id[i["competency_id"]].name for i in det_issues
                        if i["issue"] == "coverage" and i["competency_id"] in model_by_id],
    })
    sess.pre_interview_summary = summary
    sess.role_family = fam.id
    sess.role_title = role_title[:200]
    sess.company_name = ((company.company_name if company else "") or (jd.get("identity") or {}).get("company") or "")[:200]
    sess.company_profile_id = company.id if company else None
    sess.versions = versions()
    transition(db, sess, "ready", reason="blueprint built")
    progress.finish(key)


def _record_generated(db: Session, sections: List[dict], family_id: str) -> None:
    """Grow the shared question library with GENERATED items only. Anything built from a CV
    (cv section, claim-linked) is personal data and never enters a cross-user table."""
    import hashlib
    from ..db.models import GeneratedQuestion, new_id
    from ..db.session import is_postgres
    if is_postgres():
        from sqlalchemy.dialects.postgresql import insert as dialect_insert
    else:
        from sqlalchemy.dialects.sqlite import insert as dialect_insert
    for sec in sections:
        if sec["kind"] in ("cv", "personal", "awareness", "intro", "closing"):
            continue  # personal data, or news that goes stale
        for it in sec["items"]:
            sr = it.get("selection_reason") or {}
            if it.get("origin") != "generated" or sr.get("claim_ids"):
                continue
            norm = " ".join((it.get("text") or "").lower().split())
            stmt = dialect_insert(GeneratedQuestion).values(
                id=new_id(), origin="generated", role_family=family_id, competency_ids=it.get("competency_ids", []),
                question_type=it.get("question_type", "behavioral"), difficulty=int(it.get("difficulty", 3)),
                text=it["text"], text_hash=hashlib.sha256(norm.encode("utf-8")).hexdigest(),
                meta={k: it.get(k) for k in ("archetype_id", "intent", "expected_evidence", "probe_tree")},
                quality_status="unreviewed", times_selected=1, created_at=utcnow())
            db.execute(stmt.on_conflict_do_update(index_elements=["text_hash"],
                                                  set_={"times_selected": GeneratedQuestion.times_selected + 1}))


def _on_dead(db: Session, payload: dict, err: str) -> None:
    sess = db.get(InterviewSession, uuid.UUID(payload["session_id"]))
    if sess is not None and sess.status in ("created", "uploading", "analyzing"):
        transition(db, sess, "failed", reason="We couldn't prepare this interview. Please try again.")


@register("prepare_session", on_dead=_on_dead)
def handle_prepare_session(db: Session, payload: dict) -> None:
    sid = uuid.UUID(payload["session_id"])
    sess = db.execute(select(InterviewSession).where(InterviewSession.id == sid).with_for_update()).scalar_one_or_none()
    if sess is None or sess.status not in ("created", "uploading", "analyzing"):
        return
    if db.execute(select(InterviewBlueprint).where(InterviewBlueprint.session_id == sid)).scalar_one_or_none():
        transition(db, sess, "ready", reason="blueprint exists")
        return
    if sess.status != "analyzing":
        transition(db, sess, "analyzing", reason="building blueprint")
    # Make "analyzing" visible to the client and release the row lock during the slow build.
    db.commit()
    try:
        build_blueprint(db, sess)
    except Exception:
        progress.fail(f"prep:{sid}")
        raise
