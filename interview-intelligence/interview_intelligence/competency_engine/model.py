"""Role competency model + frozen rubrics for one session (docs/F_COMPETENCY_MODEL.md)."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Dict, List, Optional, Tuple

from ..ai.guard import wrap_untrusted
from ..ai.prompts import COMPETENCY_MAPPER, ROLE_CLASSIFIER, RUBRIC_BUILDER
from ..ai.runner import RunContext, StructuredOutputError, run_structured
from ..role_taxonomy.library import Library, keyword_classify, library
from .schemas import (CompetencyMapping, MappedCompetency, RoleClassification, RubricEntry, RubricSet)

IMPORTANCE_WEIGHT = {"critical": 3.0, "high": 2.0, "medium": 1.2, "low": 0.6}
MIN_EVIDENCE = {"critical": 2, "high": 2, "medium": 1, "low": 1}

# Competencies each mode must cover regardless of the JD (spec §14).
MODE_MANDATORY: Dict[str, Dict[str, str]] = {
    "hr_behavioral": {"ownership": "high", "conflict_management": "high", "learning_reflection": "high",
                      "motivation_role_fit": "high", "collaboration": "medium", "leadership": "medium"},
    "situational": {"situational_judgment": "critical", "prioritization": "high", "stakeholder_management": "high"},
    "case": {"structured_thinking": "high", "problem_solving": "high", "quantitative_reasoning": "high",
             "commercial_judgment": "high"},
    "cv_deep_dive": {"ownership": "critical"},
    "cv_attack": {"ownership": "critical", "handling_challenge": "medium"},
    "stress": {"handling_challenge": "high"},
    "grill": {"handling_challenge": "high", "ownership": "high"},
    "hiring_manager": {"prioritization": "high", "commercial_judgment": "high", "ownership": "high"},
    "final_round": {"motivation_role_fit": "medium", "communication": "medium"},
    "company_simulation": {"role_company_understanding": "high"},
}
ALWAYS = {"communication": "medium"}


def classify_role(role: dict, ctx: RunContext) -> RoleClassification:
    lib = library()
    title = (role.get("identity") or {}).get("title") or (role.get("identity") or {}).get("role_name") or ""
    fam_list = "\n".join(f"- {f.id}: {f.name} (aliases: {', '.join(f.aliases[:6])})" for f in lib.families.values())
    ind_list = ", ".join(lib.industries)
    body = (
        f"FAMILIES:\n{fam_list}\n\nINDUSTRIES: {ind_list}\n\nROLE PROFILE (from the JD):\n"
        + wrap_untrusted("role_profile", "jd", json.dumps({
            "identity": role.get("identity"), "summary": role.get("summary"),
            "requirements": [r.get("text") for r in role.get("requirements", [])][:25],
            "responsibilities": [r.get("text") for r in role.get("responsibilities", [])][:20],
            "keywords": [k.get("term") for k in role.get("keywords", [])][:30],
        }, ensure_ascii=False))
    )
    fallback_family = keyword_classify(title, json.dumps(role.get("requirements", []))[:4000])

    def _validate(rc: RoleClassification) -> List[str]:
        p = []
        if rc.primary_family not in lib.families:
            p.append(f"primary_family must be one of the listed ids, got {rc.primary_family}")
        if rc.industry and rc.industry not in lib.industries:
            p.append(f"industry must be one of the listed ids or '', got {rc.industry}")
        return p

    try:
        rc = run_structured(ROLE_CLASSIFIER, body, RoleClassification, ctx,
                            sim_input={"title": title, "fallback": fallback_family}, validate=_validate)
    except StructuredOutputError:
        rc = RoleClassification(primary_family=fallback_family, rationale="keyword fallback (classifier unavailable)",
                                technical_role=lib.family(fallback_family).technical)
    if rc.primary_family != "other" and lib.family(rc.primary_family).technical:
        rc.technical_role = True
    return rc


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return s[:48] or "role_specific"


def _normalise_mapping(mapping: CompetencyMapping, lib: Library, family_id: str, mode: str,
                       targets: List[str]) -> List[MappedCompetency]:
    """Validate ids, namespace role-specific ones, inject family defaults and mode/always/target
    competencies, cap the list. Deterministic, so the model cannot drop a mandatory area."""
    out: Dict[str, MappedCompetency] = {}
    for m in mapping.competencies:
        cid = m.competency_id.strip()
        if cid in lib.competencies:
            m.name = m.name or lib.competencies[cid].name
        else:
            base = cid[3:] if cid.startswith("rs:") else cid
            cid = "rs:" + _slug(base or m.name)
            if not m.parent or m.parent not in lib.competencies:
                m.parent = "technical_depth" if lib.family(family_id).technical else "functional_knowledge"
            if not m.definition:
                m.definition = m.name or base.replace("_", " ")
            m.name = m.name or base.replace("_", " ").title()
        m.competency_id = cid
        if cid not in out:
            out[cid] = m

    def inject(cid: str, importance: str, why: str):
        if cid not in lib.competencies:
            return
        if cid in out:
            order = ["low", "medium", "high", "critical"]
            if order.index(importance) > order.index(out[cid].importance):
                out[cid].importance = importance  # type: ignore[assignment]
            return
        out[cid] = MappedCompetency(competency_id=cid, name=lib.competencies[cid].name, importance=importance,
                                    expected_depth=why)

    fam = lib.family(family_id)
    if not mapping.competencies:
        for cid, imp in fam.defaults.items():
            inject(cid, imp, "family default")
    for cid, imp in MODE_MANDATORY.get(mode, {}).items():
        inject(cid, imp, f"required by {mode} mode")
    for cid, imp in ALWAYS.items():
        inject(cid, imp, "assessed in every interview")
    for cid in targets:
        if cid in out:
            out[cid].importance = "critical"
        elif cid.startswith("rs:"):
            out[cid] = MappedCompetency(competency_id=cid, name=cid[3:].replace("_", " ").title(), importance="critical",
                                        parent="functional_knowledge", definition="Targeted from a previous interview")
        else:
            inject(cid, "critical", "targeted weakness from a previous interview")

    ranked = sorted(out.values(), key=lambda m: -IMPORTANCE_WEIGHT[m.importance])
    keep = ranked[:12]
    for t in targets:  # targeted weaknesses can never be cut
        if t in out and out[t] not in keep:
            keep.append(out[t])
    return keep


def map_competencies(role: dict, candidate: dict, family_id: str, mode: str, seniority: str,
                     targets: List[str], ctx: RunContext) -> List[MappedCompetency]:
    lib = library()
    fam = lib.family(family_id)
    lib_list = "\n".join(f"- {c.id}: {c.name} — {c.definition}" for c in lib.competencies.values())
    defaults = ", ".join(f"{k} ({v})" for k, v in fam.defaults.items())
    claims = [{"id": c.get("id"), "text": c.get("text"), "type": c.get("type")} for c in candidate.get("claims", [])][:30]
    body = (
        f"CANONICAL LIBRARY:\n{lib_list}\n\nROLE FAMILY: {fam.id} ({fam.name}); defaults: {defaults}\n"
        f"Example functional areas for this family (NOT exhaustive): {', '.join(fam.functional_areas)}\n"
        f"INTERVIEW MODE: {mode}\nSENIORITY: {seniority}\n"
        f"TARGETED COMPETENCIES (must be included): {', '.join(targets) or 'none'}\n\n"
        "ROLE PROFILE:\n" + wrap_untrusted("role_profile", "jd", json.dumps({
            "identity": role.get("identity"),
            "requirements": role.get("requirements", [])[:30],
            "responsibilities": role.get("responsibilities", [])[:20],
            "implied_competencies": role.get("implied_competencies", [])[:12],
            "keywords": role.get("keywords", [])[:30],
        }, ensure_ascii=False))
        + "\n\nCANDIDATE CV CLAIMS:\n" + wrap_untrusted("cv_claims", "cv", json.dumps(claims, ensure_ascii=False))
    )
    try:
        mapping = run_structured(COMPETENCY_MAPPER, body, CompetencyMapping, ctx,
                                 sim_input={"family": family_id, "role": role, "candidate": candidate})
    except StructuredOutputError:
        mapping = CompetencyMapping()
    return _normalise_mapping(mapping, lib, family_id, mode, targets)


def fallback_rubric(m: MappedCompetency, lib: Library) -> RubricEntry:
    c = lib.competency(m.competency_id) or lib.competency(m.parent or "") or lib.competency("functional_knowledge")
    assert c is not None
    good = m.expected_depth if m.expected_depth and "default" not in m.expected_depth else c.definition
    return RubricEntry(
        competency_id=m.competency_id, what_good_looks_like=good,
        strong_signals=list(c.strong) or ["specific, owned, reasoned evidence"],
        weak_signals=list(c.weak) or ["generic or unsupported statements"],
        red_flags=list(c.red), expected_structure=c.structure, must_not_infer=list(lib.default_must_not_infer),
    )


def build_rubrics(model: List[MappedCompetency], role: dict, seniority: str, ctx: RunContext) -> Tuple[Dict[str, dict], str]:
    lib = library()
    payload = [{"competency_id": m.competency_id, "name": m.name, "definition": m.definition or
                (lib.competency(m.competency_id).definition if lib.competency(m.competency_id) else ""),
                "importance": m.importance, "sub_areas": m.sub_areas, "expected_depth": m.expected_depth}
               for m in model]
    body = (f"SENIORITY: {seniority}\nROLE: {json.dumps((role.get('identity') or {}), ensure_ascii=False)}\n"
            f"COMPETENCIES:\n{json.dumps(payload, ensure_ascii=False)}")

    def _validate(rs: RubricSet) -> List[str]:
        got = {r.competency_id for r in rs.rubrics}
        missing = [m.competency_id for m in model if m.competency_id not in got]
        return [f"missing rubrics for: {', '.join(missing)}"] if missing else []

    rubrics: Dict[str, dict] = {}
    try:
        rs = run_structured(RUBRIC_BUILDER, body, RubricSet, ctx, sim_input={"model": [m.model_dump() for m in model]},
                            validate=_validate)
        by_id = {r.competency_id: r for r in rs.rubrics}
    except StructuredOutputError:
        by_id = {}
    for m in model:
        r = by_id.get(m.competency_id) or fallback_rubric(m, lib)
        if len(r.strong_signals) < 2 or len(r.weak_signals) < 2:
            fb = fallback_rubric(m, lib)
            r.strong_signals = r.strong_signals or fb.strong_signals
            r.weak_signals = r.weak_signals or fb.weak_signals
        r.must_not_infer = sorted(set(r.must_not_infer) | set(lib.default_must_not_infer))
        rubrics[m.competency_id] = r.model_dump()
    digest = hashlib.sha256(json.dumps(rubrics, sort_keys=True).encode()).hexdigest()
    return rubrics, digest


def cv_strength_summary(model: List[MappedCompetency]) -> dict:
    strong = [m for m in model if m.cv_strength == "strong"]
    weak = [m for m in model if m.cv_strength != "strong"]
    return {
        "competencies_identified": len(model),
        "strong_in_cv": len(strong),
        "need_validation": len(weak),
        "competencies": [{"id": m.competency_id, "name": m.name, "importance": m.importance,
                          "cv_strength": m.cv_strength, "requirement_ids": m.requirement_ids,
                          "claim_ids": m.cv_claim_ids} for m in model],
    }


def min_evidence(importance: str) -> int:
    return MIN_EVIDENCE.get(importance, 1)


def canonical_of(cid: str, parent: Optional[str] = None) -> str:
    return library().canonical_of(cid, parent or "")
