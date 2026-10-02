"""Deterministic CV checks (spec §54): timeline gaps/overlaps/future dates, duplicate claims,
implausible metrics. The model extracts; these rules judge, reproducibly."""

from __future__ import annotations

import re
from datetime import date
from typing import Dict, List, Optional, Tuple

from ..textutil import jaccard
from .schemas import CandidateProfile, TimelineIssue

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_PRESENT = {"present", "current", "now", "till date", "to date", "ongoing", "today"}


def parse_month(raw: str, *, today: Optional[date] = None) -> Optional[Tuple[int, int]]:
    """'Jun 2021' / 'June 2021' / '06/2021' / '2021-06' / '2021' / 'Present' -> (year, month)."""
    today = today or date.today()
    s = (raw or "").strip().lower().replace("’", "'").replace(",", " ")
    if not s:
        return None
    if s in _PRESENT or any(p in s for p in _PRESENT):
        return today.year, today.month
    m = re.search(r"([a-z]{3,9})\.?\s*'?(\d{2,4})", s)
    if m and m.group(1)[:3] in _MONTHS:
        y = int(m.group(2))
        y = y + 2000 if y < 100 else y
        return y, _MONTHS[m.group(1)[:3]]
    m = re.search(r"\b(\d{1,2})[/\-.](\d{4})\b", s)
    if m and 1 <= int(m.group(1)) <= 12:
        return int(m.group(2)), int(m.group(1))
    m = re.search(r"\b(\d{4})[/\-.](\d{1,2})\b", s)
    if m and 1 <= int(m.group(2)) <= 12:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"\b(19[5-9]\d|20\d{2})\b", s)
    if m:
        return int(m.group(1)), 6  # year only: assume mid-year, flagged as approximate upstream
    return None


def _months(ym: Tuple[int, int]) -> int:
    return ym[0] * 12 + ym[1]


def timeline_issues(profile: CandidateProfile, *, today: Optional[date] = None) -> Tuple[List[TimelineIssue], Optional[float]]:
    today = today or date.today()
    now_m = today.year * 12 + today.month
    spans = []
    issues: List[TimelineIssue] = []
    for x in profile.experience:
        s = parse_month(x.start, today=today)
        e = parse_month(x.end, today=today) if (x.end or not x.is_current) else (today.year, today.month)
        if x.is_current and not x.end:
            e = (today.year, today.month)
        if s is None or e is None:
            if x.start or x.end:
                issues.append(TimelineIssue(type="unparseable_date", detail=f"Could not read dates for {x.title or x.organization}",
                                            refs=[x.id]))
            continue
        if _months(s) > now_m + 1 or (_months(e) > now_m + 1 and not x.is_current):
            issues.append(TimelineIssue(type="future_date", detail=f"{x.title or x.organization} has a date in the future",
                                        refs=[x.id]))
        if _months(e) < _months(s):
            issues.append(TimelineIssue(type="inconsistent_dates",
                                        detail=f"{x.title or x.organization}: end date is before start date", refs=[x.id]))
            continue
        spans.append((_months(s), _months(e), x))
    spans.sort(key=lambda t: t[0])
    for (s1, e1, x1), (s2, e2, x2) in zip(spans, spans[1:]):
        if s2 - e1 > 6:
            issues.append(TimelineIssue(type="gap", detail=f"{s2 - e1} month gap between {x1.organization or x1.title} and "
                                                           f"{x2.organization or x2.title}", refs=[x1.id, x2.id]))
        elif e1 - s2 > 2 and x1.organization.lower() != x2.organization.lower():
            issues.append(TimelineIssue(type="overlap", detail=f"{x1.organization or x1.title} and {x2.organization or x2.title} "
                                                               f"overlap by {e1 - s2} months", refs=[x1.id, x2.id]))
    # merged coverage -> years of experience (excluding overlaps)
    total = 0
    cur_s = cur_e = None
    for s, e, _ in spans:
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    if cur_s is not None:
        total += cur_e - cur_s
    years = round(total / 12.0, 1) if spans else None
    return issues, years


_UNREALISTIC = {
    "growth_pct": 1000.0,
    "cost_saving_pct": 95.0,
    "time_saving_pct": 99.0,
}


def claim_flags(profile: CandidateProfile) -> Dict[str, List[str]]:
    flags: Dict[str, List[str]] = {}
    junior = profile.seniority_estimate in ("intern", "entry")
    seen: List[Tuple[str, str]] = []
    for c in profile.claims:
        f: List[str] = []
        for slot, limit in _UNREALISTIC.items():
            v = c.slots.get(slot)
            if v is not None and v > limit:
                f.append("unrealistic_metric")
        for m in c.metrics:
            if m.unit.strip() == "%" and m.value is not None and m.value > 1000:
                f.append("unrealistic_metric")
        ts = c.slots.get("team_size") or c.slots.get("direct_reports")
        if ts is not None and ((junior and ts > 25) or ts > 5000):
            f.append("scale_inconsistent_with_seniority" if junior else "unrealistic_metric")
        for prev_id, prev_text in seen:
            if jaccard(prev_text, c.text) >= 0.6:
                f.append(f"duplicate_of:{prev_id}")
                break
        if c.vague:
            f.append("vague")
        if c.ownership_language == "team" and c.type in ("leadership", "ownership"):
            f.append("ownership_unclear")
        seen.append((c.id, c.text))
        if f:
            flags[c.id] = sorted(set(f))
    return flags


def postprocess(profile: CandidateProfile) -> dict:
    issues, years = timeline_issues(profile)
    flags = claim_flags(profile)
    # Bump verification priority for claims with plausibility flags (they are the ones to probe).
    for c in profile.claims:
        cf = flags.get(c.id, [])
        if any(x in cf for x in ("unrealistic_metric", "scale_inconsistent_with_seniority")):
            c.verification_priority = 1
            c.needs_verification = True
    if profile.total_experience_years is None and years is not None:
        profile.total_experience_years = years
        profile.experience_basis = "computed_from_dates"
    data = profile.model_dump()
    data["timeline_issues"] = [i.model_dump() for i in issues]
    data["claim_flags"] = flags
    return data
