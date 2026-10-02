"""Deterministic CV/JD checks (spec §54, §55)."""

from datetime import date

from interview_intelligence.cv_intelligence.postprocess import claim_flags, parse_month, postprocess, timeline_issues
from interview_intelligence.cv_intelligence.schemas import CandidateProfile
from interview_intelligence.jd_intelligence.postprocess import postprocess as jd_post
from interview_intelligence.jd_intelligence.schemas import RoleProfile

TODAY = date(2026, 10, 2)


def test_parse_month_formats():
    assert parse_month("Jun 2021") == (2021, 6)
    assert parse_month("June, 2021") == (2021, 6)
    assert parse_month("06/2021") == (2021, 6)
    assert parse_month("2021-06") == (2021, 6)
    assert parse_month("Sept '19") == (2019, 9)
    assert parse_month("Present", today=TODAY) == (2026, 10)
    assert parse_month("2020") == (2020, 6)
    assert parse_month("sometime") is None


def _p(exp, claims=None, sen="mid"):
    return CandidateProfile(experience=exp, claims=claims or [], seniority_estimate=sen)


def test_gap_overlap_future_and_reversed_dates():
    p = _p([
        {"id": "X1", "organization": "A", "title": "Analyst", "start": "Jan 2018", "end": "Dec 2018"},
        {"id": "X2", "organization": "B", "title": "Manager", "start": "Jan 2020", "end": "Jun 2021"},
        {"id": "X3", "organization": "C", "title": "Lead", "start": "Jan 2021", "end": "Present", "is_current": True},
        {"id": "X4", "organization": "D", "title": "Future", "start": "Jan 2030", "end": "Jan 2031"},
        {"id": "X5", "organization": "E", "title": "Backwards", "start": "Jan 2019", "end": "Jan 2017"},
    ])
    issues, years = timeline_issues(p, today=TODAY)
    kinds = {i.type for i in issues}
    assert {"gap", "overlap", "future_date", "inconsistent_dates"} <= kinds
    assert years is not None and years > 5


def test_claim_flags_duplicates_unrealistic_and_seniority():
    claims = [
        {"id": "C1", "text": "Grew revenue by 4000% in one year", "slots": {"growth_pct": 4000}, "type": "impact"},
        {"id": "C2", "text": "Led a team of 60 engineers", "slots": {"team_size": 60}, "type": "leadership"},
        {"id": "C3", "text": "Grew revenue by 4000% in a single year", "slots": {}, "type": "impact"},
        {"id": "C4", "text": "Led the team on various initiatives", "vague": True, "ownership_language": "team",
         "type": "leadership"},
    ]
    p = _p([], claims, sen="entry")
    f = claim_flags(p)
    assert "unrealistic_metric" in f["C1"]
    assert "scale_inconsistent_with_seniority" in f["C2"]
    assert any(x.startswith("duplicate_of:C1") for x in f["C3"])
    assert "vague" in f["C4"] and "ownership_unclear" in f["C4"]
    data = postprocess(p)
    assert [c for c in data["claims"] if c["id"] == "C1"][0]["verification_priority"] == 1


def test_jd_dedupe_specificity_and_contradictions():
    rp = RoleProfile(requirements=[
        {"id": "A", "text": "3+ years of experience in brand management", "importance": "should"},
        {"id": "B", "text": "3+ years experience in brand management", "importance": "must"},
        {"id": "C", "text": "Excel", "importance": "nice"},
    ], seniority={"level": "senior", "confidence": "high", "basis": ""})
    out = jd_post(rp, "Entry-level role for freshers. Requires 8 years of experience. Excel.")
    assert len(out["requirements"]) == 2
    assert out["requirements"][0]["importance"] == "must" and out["requirements"][0]["id"] == "R1"
    assert out["quality"]["specificity"] == "low"
    assert any("entry-level" in c.lower() for c in out["quality"]["contradictions"])
    assert out["seniority"]["confidence"] == "low", "no basis => no certainty"
