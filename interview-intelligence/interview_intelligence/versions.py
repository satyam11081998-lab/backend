"""Engine versions (spec §98). Bump the relevant one whenever behaviour changes.

Every blueprint, analysis, assessment and report stores the versions that produced it,
so any result can be traced back and re-run, and the regression harnesses can compare
"previous vs new" on the same inputs.
"""

SERVICE_VERSION = "0.1.0"

ENGINE_VERSIONS = {
    "cv_analysis": "cv-1",
    "jd_analysis": "jd-1",
    "role_profile": "role-1",
    "taxonomy": "tax-1",
    "competency_library": "comp-1",
    "competency_model": "cm-1",
    "rubric": "rubric-1",
    "question_engine": "qe-1",
    "blueprint": "bp-1",
    "interviewer": "iv-1",
    "decision_policy": "pol-1",
    "evidence": "ev-1",
    "evaluator": "eval-1",
    "feedback": "fb-1",
    "report": "rep-1",
}


def versions() -> dict:
    return dict(ENGINE_VERSIONS, service=SERVICE_VERSION)
