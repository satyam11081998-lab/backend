"""Loads and validates the versioned library (role families, industries, competencies,
archetypes, curated bank) once per process. New domains = edit the JSON + bump version."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from ..question_engine.schemas import ArchetypeSpec

DATA = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Competency:
    id: str
    name: str
    category: str
    definition: str
    sub: List[str]
    strong: List[str]
    weak: List[str]
    red: List[str]
    structure: str
    parent: Optional[str]
    must_not_infer: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class Family:
    id: str
    name: str
    aliases: List[str]
    technical: bool
    defaults: Dict[str, str]
    functional_areas: List[str]
    case_types: List[str]
    section_bias: Dict[str, float]
    report_dimensions: Dict[str, List[str]]


@dataclass(frozen=True)
class CuratedQuestion:
    id: str
    families: List[str]
    competencies: List[str]
    type: str
    difficulty: int
    seniority: str
    archetype: str
    text: str
    intent: str
    evidence: List[str]
    strong: List[str]
    weak: List[str]
    probes: List[str]


@dataclass(frozen=True)
class Library:
    versions: Dict[str, str]
    competencies: Dict[str, Competency]
    families: Dict[str, Family]
    industries: Dict[str, str]
    archetypes: Dict[str, ArchetypeSpec]
    bank: List[CuratedQuestion]
    default_must_not_infer: List[str]

    def competency(self, cid: str) -> Optional[Competency]:
        return self.competencies.get(cid)

    def family(self, fid: str) -> Family:
        return self.families.get(fid) or self.families["other"]

    def canonical_of(self, cid: str, parent: str = "") -> str:
        """Comparable id for history: canonical ids map to themselves, rs: ids to their parent."""
        if cid in self.competencies:
            return cid
        if parent and parent in self.competencies:
            return parent
        return "functional_knowledge"


def _load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def library() -> Library:
    comp = _load("competencies.json")
    fams = _load("role_families.json")
    arch = _load("question_archetypes.json")
    bank = _load("question_bank.json")
    dmi = comp.get("default_must_not_infer", [])
    competencies = {
        c["id"]: Competency(
            id=c["id"], name=c["name"], category=c["category"], definition=c["definition"],
            sub=c.get("sub", []), strong=c.get("strong", []), weak=c.get("weak", []), red=c.get("red", []),
            structure=c.get("structure", ""), parent=c.get("parent"), must_not_infer=dmi,
        ) for c in comp["competencies"]
    }
    families = {
        f["id"]: Family(
            id=f["id"], name=f["name"], aliases=f.get("aliases", []), technical=bool(f.get("technical")),
            defaults=f.get("defaults", {}), functional_areas=f.get("functional_areas", []),
            case_types=f.get("case_types", []), section_bias=f.get("section_bias", {}),
            report_dimensions=f.get("report_dimensions", {}),
        ) for f in fams["families"]
    }
    archetypes = {a["id"]: ArchetypeSpec(**a) for a in arch["archetypes"]}
    questions = [CuratedQuestion(
        id=q["id"], families=q["families"], competencies=q["competencies"], type=q["type"],
        difficulty=int(q["difficulty"]), seniority=q.get("seniority", "any"), archetype=q.get("archetype", ""),
        text=q["text"], intent=q.get("intent", ""), evidence=q.get("evidence", []), strong=q.get("strong", []),
        weak=q.get("weak", []), probes=q.get("probes", []),
    ) for q in bank["questions"]]
    return Library(
        versions={"competencies": comp["version"], "taxonomy": fams["version"], "archetypes": arch["version"],
                  "bank": bank["version"]},
        competencies=competencies, families=families,
        industries={i["id"]: i["name"] for i in fams.get("industries", [])},
        archetypes=archetypes, bank=questions, default_must_not_infer=dmi,
    )


def validate_library() -> List[str]:
    """Referential integrity of the data files. Returned as problems (tests assert empty)."""
    lib = library()
    problems: List[str] = []
    for f in lib.families.values():
        for cid in f.defaults:
            if cid not in lib.competencies:
                problems.append(f"family {f.id}: unknown default competency {cid}")
        for dim, cids in f.report_dimensions.items():
            for cid in cids:
                if cid not in lib.competencies:
                    problems.append(f"family {f.id} dimension {dim}: unknown competency {cid}")
    for c in lib.competencies.values():
        if c.parent and c.parent not in lib.competencies:
            problems.append(f"competency {c.id}: unknown parent {c.parent}")
    seen = set()
    for q in lib.bank:
        if q.id in seen:
            problems.append(f"duplicate bank id {q.id}")
        seen.add(q.id)
        for cid in q.competencies:
            if cid not in lib.competencies:
                problems.append(f"bank {q.id}: unknown competency {cid}")
        for fam in q.families:
            if fam != "*" and fam not in lib.families:
                problems.append(f"bank {q.id}: unknown family {fam}")
        if q.archetype and q.archetype not in lib.archetypes:
            problems.append(f"bank {q.id}: unknown archetype {q.archetype}")
    return problems


_WORD = re.compile(r"[a-z0-9&+]+")


def _contains(hay: str, needle: str) -> bool:
    """Whole-word/phrase match. Plain substring matching let short aliases fire inside other
    words ('it' in 'security', 'pm' in 'development', 'ops' in 'devops')."""
    return re.search(r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])", hay) is not None


_GENERIC = {"manager", "analyst", "associate", "executive", "specialist", "lead", "senior", "junior", "head", "of",
            "officer", "coordinator", "engineer", "director", "intern", "trainee"}


def _specificity(alias: str) -> float:
    words = alias.split()
    core = [w for w in words if w not in _GENERIC]
    if not core:  # e.g. 'head of', 'director': a seniority marker, the weakest signal
        return 2.0
    return 3.0 + 1.5 * (len(core) - 1)


def keyword_classify(title: str, text: str) -> str:
    """Deterministic fallback when the classifier model is unavailable.

    The title decides: each family scores its MOST specific matching alias, where generic
    role nouns ('manager', 'analyst', ...) do not count towards specificity — so
    'performance marketing' beats 'marketing manager' and 'medical representative' beats
    'medical'. The body is only a tie-breaker, and only multi-word aliases count there
    (single words such as 'it', 'brand' or 'director' are too common in prose).
    With no title match, two or more distinct multi-word body matches are required.
    Returns a family id or 'other'."""
    lib = library()
    title_l = re.sub(r"\s+", " ", (title or "").lower().replace("\u2019", "'"))
    body_l = re.sub(r"\s+", " ", (text or "").lower()[:6000].replace("\u2019", "'"))
    scored = []
    for f in lib.families.values():
        if f.id == "other":
            continue
        t_best, body_hits = 0.0, 0
        for alias in set(f.aliases):
            a = alias.lower().strip()
            if not a:
                continue
            if _contains(title_l, a):
                t_best = max(t_best, _specificity(a))
            if len(a.split()) >= 2 and _contains(body_l, a):
                body_hits += 1
        scored.append((f.id, t_best, body_hits))
    with_title = [(fid, t + 0.25 * min(b, 4)) for fid, t, b in scored if t > 0]
    if with_title:
        return max(with_title, key=lambda x: x[1])[0]  # max() keeps the first on ties (file order)
    body_only = [(fid, b) for fid, _, b in scored if b >= 2]
    return max(body_only, key=lambda x: x[1])[0] if body_only else "other"
