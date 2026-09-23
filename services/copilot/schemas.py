"""
Typed structures for the Prep Copilot v2 (role/company-aware).

A Pack is the unit of grounded domain depth for ONE (role) or (role x company):
its evaluation rubric, the real frameworks a strong hire applies, what the target
actually assesses, a bank of on-the-job scenarios, and the sources every claim is
grounded in. Packs are BUILT by the Gemini researcher, STORED + refined in the
isolated corpus tables, and CAST into practice by the isolated interview engine +
scored by the isolated GPT scorer. Nothing here touches the live pipeline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

_STOP = {"the", "a", "an", "of", "for", "at", "in", "and", "role", "company", "ltd", "limited", "inc", "plc"}


def normalize_key(text: str) -> str:
    """Stable lookup key: lowercase, alnum-only tokens, stopwords dropped, joined by _.
    'BNY Asset Management' -> 'bny_asset_management'; '' -> ''."""
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    toks = [t for t in toks if t not in _STOP]
    return "_".join(toks)[:120]


@dataclass
class RubricDimension:
    key: str
    label: str
    max: int
    what_good_looks_like: str = ""
    anchors: List[str] = field(default_factory=list)   # calibration bands, high->low
    red_flags: List[str] = field(default_factory=list) # what to penalise / name

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RubricDimension":
        return RubricDimension(
            key=str(d.get("key") or "").strip() or "dimension",
            label=str(d.get("label") or d.get("key") or "Dimension"),
            max=int(d.get("max") or 0),
            what_good_looks_like=str(d.get("what_good_looks_like") or ""),
            anchors=[str(x) for x in (d.get("anchors") or []) if isinstance(x, str)],
            red_flags=[str(x) for x in (d.get("red_flags") or []) if isinstance(x, str)],
        )


@dataclass
class Rubric:
    dimensions: List[RubricDimension] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(int(d.max) for d in self.dimensions)

    def is_valid(self) -> bool:
        return bool(self.dimensions) and 95 <= self.total <= 105

    def to_dict(self) -> Dict[str, Any]:
        return {"dimensions": [d.to_dict() for d in self.dimensions], "total": self.total}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Rubric":
        dims = [RubricDimension.from_dict(x) for x in (d.get("dimensions") or []) if isinstance(x, dict)]
        return Rubric(dimensions=dims)


@dataclass
class Framework:
    name: str
    summary: str
    why_it_matters: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Framework":
        return Framework(name=str(d.get("name") or ""), summary=str(d.get("summary") or ""),
                         why_it_matters=str(d.get("why_it_matters") or ""))


@dataclass
class Scenario:
    title: str
    prompt: str
    focus: str = ""                      # which rubric dimension it stresses most
    numerical_ask: str = ""              # explicit quant task, "" if none
    solution_outline: str = ""           # shown after the attempt, never before
    difficulty: str = "medium"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Scenario":
        return Scenario(
            title=str(d.get("title") or "Scenario"),
            prompt=str(d.get("prompt") or ""),
            focus=str(d.get("focus") or ""),
            numerical_ask=str(d.get("numerical_ask") or ""),
            solution_outline=str(d.get("solution_outline") or ""),
            difficulty=str(d.get("difficulty") or "medium"),
        )


@dataclass
class Source:
    title: str
    url: str = ""
    snippet: str = ""
    confidence: str = "medium"           # high|medium|low

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Source":
        return Source(title=str(d.get("title") or ""), url=str(d.get("url") or ""),
                      snippet=str(d.get("snippet") or "")[:400],
                      confidence=str(d.get("confidence") or "medium"))


@dataclass
class Pack:
    role_key: str
    company_key: Optional[str]           # None => role-only pack
    display_role: str
    display_company: Optional[str]
    rubric: Rubric = field(default_factory=Rubric)
    frameworks: List[Framework] = field(default_factory=list)
    assessment: Dict[str, Any] = field(default_factory=dict)  # {what_they_test:[], numericals_expected:bool, formats:[], notes:str}
    scenarios: List[Scenario] = field(default_factory=list)
    sources: List[Source] = field(default_factory=list)
    confidence: str = "low"              # overall grounding confidence: high|medium|low
    version: int = 1
    status: str = "ready"                # building|ready|failed
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role_key": self.role_key, "company_key": self.company_key,
            "display_role": self.display_role, "display_company": self.display_company,
            "rubric": self.rubric.to_dict(),
            "frameworks": [f.to_dict() for f in self.frameworks],
            "assessment": self.assessment,
            "scenarios": [s.to_dict() for s in self.scenarios],
            "sources": [s.to_dict() for s in self.sources],
            "confidence": self.confidence, "version": self.version,
            "status": self.status, "notes": self.notes,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Pack":
        return Pack(
            role_key=str(d.get("role_key") or ""),
            company_key=(d.get("company_key") or None),
            display_role=str(d.get("display_role") or ""),
            display_company=(d.get("display_company") or None),
            rubric=Rubric.from_dict(d.get("rubric") or {}),
            frameworks=[Framework.from_dict(x) for x in (d.get("frameworks") or []) if isinstance(x, dict)],
            assessment=dict(d.get("assessment") or {}),
            scenarios=[Scenario.from_dict(x) for x in (d.get("scenarios") or []) if isinstance(x, dict)],
            sources=[Source.from_dict(x) for x in (d.get("sources") or []) if isinstance(x, dict)],
            confidence=str(d.get("confidence") or "low"),
            version=int(d.get("version") or 1),
            status=str(d.get("status") or "ready"),
            notes=str(d.get("notes") or ""),
        )
