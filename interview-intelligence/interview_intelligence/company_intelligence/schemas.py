"""Company intelligence with provenance (spec §60–§61, §84–§85)."""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field

SourceType = Literal["jd_derived", "user_provided", "official", "public_report", "inferred"]
# Priority (spec §85): JD > role requirements > CV > official > public reports > general knowledge.
SOURCE_PRIORITY = {"jd_derived": 0, "user_provided": 1, "official": 2, "public_report": 3, "inferred": 4}


class CompanyFact(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = ""
    fact: str
    source_type: SourceType
    source: str = ""
    url: str = ""
    published_at: str = ""
    retrieved_at: str = ""
    confidence: Literal["high", "medium", "low"] = "medium"
    relevance: Literal["high", "medium", "low"] = "medium"
    used_for: List[str] = Field(default_factory=list)


class CompanyFacts(BaseModel):
    model_config = ConfigDict(extra="ignore")
    facts: List[CompanyFact] = Field(default_factory=list)
