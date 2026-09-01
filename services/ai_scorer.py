"""
AI Scorer - calls OpenAI to evaluate case interview answers.

Pipeline (2026-09-01):
  1. answer_validity.screen_answer() decides if this is a genuine attempt.
     - gibberish / off_topic  -> score 0 with an explanation, NO expensive call.
     - thin                   -> full scoring, but the scorer is told not to inflate.
     - valid                  -> full scoring.
  2. The model scores against an evidence-based rubric and returns richer feedback.
  3. Code ENFORCES the marking deterministically: clamp each dimension to its max
     and recompute the total from the breakdown (never trust the model's own sum).

The model and prompt logic are isolated here so we can swap providers without
touching the rest of the codebase. All new feedback fields are ADDITIVE — the
C2 return contract (score/breakdown/strengths/improvements/summary + guesstimate
total/dimensions/backstop) is preserved.
"""

import os
import json
import time
from typing import Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

from services.ai_usage import log_ai_usage
from services.answer_validity import screen_answer

from prompts.scoring_prompt import SCORING_SYSTEM_PROMPT, build_scoring_user_prompt
from prompts.guesstimate_scoring_prompt import (
    GUESSTIMATE_SCORING_SYSTEM_PROMPT,
    build_guesstimate_user_prompt,
)
from services.guesstimate_backstop import apply_backstop, DIMENSIONS as GUESSTIMATE_DIMS

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env file")

client = OpenAI(api_key=OPENAI_API_KEY)

# Model selection - GPT-4o is reliable for structured output
SCORING_MODEL = "gpt-4o"
# Guesstimates are arithmetic-driven and the deterministic backstop catches the math.
GUESSTIMATE_SCORING_MODEL = "gpt-4o-mini"

# Dimension ceilings for the 6-dim case rubric (must total 100).
CASE_DIM_MAX = {
    "structure": 25,
    "quantitative": 20,
    "synthesis": 20,
    "business_judgment": 15,
    "creativity": 10,
    "presence": 10,
}


class AIScoringError(Exception):
    """Raised when AI scoring fails for any reason."""
    pass


# Off-topic false positives are the one way this gate could hurt a real user, so
# only HARD-reject (score 0) when relevance is genuinely near zero. Gibberish is
# unambiguous and always gates; a borderline off_topic call with real relevance is
# scored instead, just flagged 'thin' so the scorer does not inflate it.
_OFF_TOPIC_RELEVANCE_FLOOR = 25


def _is_hard_reject(validity: Dict[str, Any]) -> bool:
    verdict = validity.get("verdict")
    if verdict == "gibberish":
        return True
    if verdict == "off_topic":
        try:
            relevance = int(validity.get("relevance", 0))
        except (TypeError, ValueError):
            relevance = 0
        return relevance < _OFF_TOPIC_RELEVANCE_FLOOR
    return False


# ---------------------------------------------------------------------------
# CASE SCORING
# ---------------------------------------------------------------------------

def score_case_answer(
    case_content: str,
    case_type: str,
    user_answer: str,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Screen the answer, then (if it is a genuine attempt) score it against the
    evidence-based rubric. Gibberish/off-topic returns a 0 with an explanation and
    never reaches the expensive model call.

    Returns a feedback dict: score, breakdown(6 dims), strengths, improvements,
    summary, rubric='case', plus additive dimension_feedback / red_flags /
    model_answer / validity.
    """
    validity = screen_answer(case_content, case_type, user_answer, user_id)
    if _is_hard_reject(validity):
        return _rejection_case(case_type, validity)

    user_prompt = build_scoring_user_prompt(
        case_content=case_content,
        case_type=case_type,
        user_answer=user_answer,
        # A genuine-but-thin answer, or a borderline off_topic we chose to score,
        # must not be inflated.
        thin=validity["verdict"] in ("thin", "off_topic"),
    )

    try:
        t0 = time.time()
        response = client.chat.completions.create(
            model=SCORING_MODEL,
            messages=[
                {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=4000,  # richer output (per-dimension feedback + model answer)
            response_format={"type": "json_object"},
        )
        log_ai_usage(user_id=user_id, endpoint="/submit", model=SCORING_MODEL,
                     response=response, latency_ms=int((time.time() - t0) * 1000))
    except Exception as e:
        raise AIScoringError(f"OpenAI API call failed: {str(e)}")

    raw_content = response.choices[0].message.content
    if not raw_content:
        raise AIScoringError("OpenAI returned empty response")

    try:
        feedback = json.loads(raw_content)
    except json.JSONDecodeError as e:
        raise AIScoringError(
            f"OpenAI returned invalid JSON: {str(e)}. Raw: {raw_content[:200]}"
        )

    required_keys = {"score", "breakdown", "strengths", "improvements", "summary"}
    missing = required_keys - set(feedback.keys())
    if missing:
        raise AIScoringError(f"OpenAI response missing keys: {missing}")

    if not isinstance(feedback.get("breakdown"), dict):
        raise AIScoringError("OpenAI response 'breakdown' is not an object")
    missing_breakdown = set(CASE_DIM_MAX) - set(feedback["breakdown"].keys())
    if missing_breakdown:
        raise AIScoringError(f"OpenAI breakdown missing keys: {missing_breakdown}")

    return _enforce_case(feedback, validity)


def _enforce_case(feedback: Dict[str, Any], validity: Dict[str, Any]) -> Dict[str, Any]:
    """
    Never trust the model's own arithmetic: clamp each dimension to its ceiling and
    recompute the total from the clamped breakdown, so the number and the bars can
    never disagree. Normalise the additive fields and attach the validity verdict.
    """
    breakdown = {
        dim: _clamp_int(feedback["breakdown"].get(dim, 0), 0, CASE_DIM_MAX[dim])
        for dim in CASE_DIM_MAX
    }
    score = sum(breakdown.values())  # authoritative — sums to <= 100 by construction

    # Per-dimension feedback, defaulted and kept consistent with the final scores.
    raw_df = feedback.get("dimension_feedback")
    dimension_feedback: Dict[str, Any] = {}
    for dim in CASE_DIM_MAX:
        entry = raw_df.get(dim) if isinstance(raw_df, dict) else None
        if not isinstance(entry, dict):
            entry = {}
        dimension_feedback[dim] = {
            "score": breakdown[dim],
            "evidence": str(entry.get("evidence", "") or "")[:600],
            "gap": str(entry.get("gap", "") or "")[:600],
            "to_improve": str(entry.get("to_improve", "") or "")[:600],
        }

    return {
        "score": score,
        "breakdown": breakdown,
        "dimension_feedback": dimension_feedback,
        "strengths": _str_list(feedback.get("strengths"), limit=6),
        "improvements": _str_list(feedback.get("improvements"), limit=6),
        "red_flags": _str_list(feedback.get("red_flags"), limit=6),
        "model_answer": str(feedback.get("model_answer", "") or "")[:3000],
        "summary": str(feedback.get("summary", "") or "")[:1200],
        "rubric": "case",
        "validity": validity,
    }


# ---------------------------------------------------------------------------
# GUESSTIMATE SCORING
# ---------------------------------------------------------------------------

def score_guesstimate_answer(
    case_content: str,
    user_answer: str,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Score a GUESSTIMATE answer. Same validity gate as cases; then one gpt-4o-mini
    call returns the 5 rubric dims + a transcribed calc-chain, and the deterministic
    backstop recomputes the math, OVERRIDES the arithmetic dimension, and caps the
    total. We never trust the LLM's own arithmetic.

    Returns (C2-stable): score, breakdown(5 dims), strengths, improvements, summary,
    rubric='guesstimate', backstop{...}; plus additive red_flags / model_answer /
    validity.
    """
    validity = screen_answer(case_content, "guesstimate", user_answer, user_id)
    if _is_hard_reject(validity):
        return _rejection_guesstimate(validity)

    user_prompt = build_guesstimate_user_prompt(
        case_content=case_content,
        user_answer=user_answer,
    )

    try:
        t0 = time.time()
        response = client.chat.completions.create(
            model=GUESSTIMATE_SCORING_MODEL,
            messages=[
                {"role": "system", "content": GUESSTIMATE_SCORING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=2500,
            response_format={"type": "json_object"},
        )
        log_ai_usage(user_id=user_id, endpoint="/submit", model=GUESSTIMATE_SCORING_MODEL,
                     response=response, latency_ms=int((time.time() - t0) * 1000))
    except Exception as e:
        raise AIScoringError(f"OpenAI API call failed (guesstimate): {str(e)}")

    raw_content = response.choices[0].message.content
    if not raw_content:
        raise AIScoringError("OpenAI returned empty response (guesstimate)")

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as e:
        raise AIScoringError(
            f"OpenAI returned invalid JSON (guesstimate): {str(e)}. Raw: {raw_content[:200]}"
        )

    dims = parsed.get("dimensions")
    chain = parsed.get("calc_chain") or {"steps": [], "finalValue": 0}
    if not isinstance(dims, dict):
        raise AIScoringError("Guesstimate response missing 'dimensions'")

    llm_dims = {d: _clamp_int(dims.get(d, 3), 1, 5) for d in GUESSTIMATE_DIMS}

    # Deterministic backstop: recompute the chain, override arithmetic, cap total.
    final = apply_backstop(llm_dims, chain, band=None)

    return {
        "score": int(final["total"]),
        "breakdown": final["dimensions"],
        "strengths": _str_list(parsed.get("strengths"), limit=6),
        "improvements": _str_list(parsed.get("improvements"), limit=6),
        "red_flags": _str_list(parsed.get("red_flags"), limit=6),
        "model_answer": str(parsed.get("model_answer", "") or "")[:3000],
        "summary": str(parsed.get("summary", "") or "")[:1200],
        "rubric": "guesstimate",
        "validity": validity,
        "backstop": {
            "findings": final["backstop"]["findings"],
            "summary": final["backstop"]["summary"],
            "notChecked": final["backstop"]["notChecked"],
            "arithmeticOverridden": final["arithmeticOverridden"],
            "rawTotal": final["rawTotal"],
            "totalCapFactor": final["backstop"]["totalCapFactor"],
        },
    }


# ---------------------------------------------------------------------------
# HARD-GATE (score 0) FEEDBACK — genuine, educational, but no credit.
# ---------------------------------------------------------------------------

_REJECT_STEPS_DEFAULT = [
    "Start with 1-2 clarifying questions (scope, geography, timeframe) before solving.",
    "Lay out a MECE structure — the distinct, non-overlapping buckets you'll analyse.",
    "Quantify the key driver and show the calculation, then sanity-check the number.",
    "State a top-down recommendation first, then the 2-3 reasons that support it.",
]
_REJECT_STEPS_GUESSTIMATE = [
    "Restate what you're estimating and the units (per year? India only? new vs replacement?).",
    "Build a top-down or bottom-up tree from a sensible driver, split into MECE segments.",
    "Assume defensible per-segment numbers and show the multiplication step by step.",
    "Sanity-check the final figure against a known anchor (per-capita, a comparable market).",
]

_REJECT_MODEL_CASE = (
    "A real attempt would: (1) ask a clarifying question to bound the problem; "
    "(2) lay out a MECE framework tailored to this case; (3) prioritise the biggest driver "
    "(Pareto) and work the numbers with a sanity check; (4) generate a hypothesis or two; "
    "(5) close top-down — recommendation first, then the supporting reasons and key risk."
)
_REJECT_MODEL_GUESSTIMATE = (
    "A real attempt would: state the estimate and units; pick a driver and decompose it into "
    "MECE segments; assign defensible per-segment assumptions; multiply through, showing each "
    "step; then sanity-check the total against a known anchor."
)


def _rejection_case(case_type: str, validity: Dict[str, Any]) -> Dict[str, Any]:
    reason = validity.get("reason") or "It did not read as a genuine attempt to solve this case."
    label = "off-topic" if validity["verdict"] == "off_topic" else "not a genuine attempt"
    zero_df = {
        dim: {
            "score": 0,
            "evidence": "none",
            "gap": "No genuine attempt at this dimension was detected.",
            "to_improve": step,
        }
        for dim, step in zip(CASE_DIM_MAX.keys(), _REJECT_STEPS_DEFAULT + _REJECT_STEPS_DEFAULT)
    }
    return {
        "score": 0,
        "breakdown": {dim: 0 for dim in CASE_DIM_MAX},
        "dimension_feedback": zero_df,
        "strengths": [],
        "improvements": _REJECT_STEPS_DEFAULT,
        "red_flags": [f"Scored 0 — {label}. {reason}"],
        "model_answer": _REJECT_MODEL_CASE,
        "summary": (
            f"This wasn't scored as a case answer — {reason} "
            "A genuine attempt needs a clarifying question, a MECE structure, some quantification, "
            "and a top-down recommendation. Give it a real try and you'll get a full breakdown."
        ),
        "rubric": "case",
        "validity": validity,
    }


def _rejection_guesstimate(validity: Dict[str, Any]) -> Dict[str, Any]:
    reason = validity.get("reason") or "It did not read as a genuine estimation attempt."
    label = "off-topic" if validity["verdict"] == "off_topic" else "not a genuine attempt"
    return {
        "score": 0,
        "breakdown": {d: 1 for d in GUESSTIMATE_DIMS},  # min on the 1-5 scale
        "strengths": [],
        "improvements": _REJECT_STEPS_GUESSTIMATE,
        "red_flags": [f"Scored 0 — {label}. {reason}"],
        "model_answer": _REJECT_MODEL_GUESSTIMATE,
        "summary": (
            f"This wasn't scored as an estimation — {reason} "
            "Lay out the units, a MECE decomposition, defensible assumptions with the math shown, "
            "and a sanity check, and you'll get a full breakdown."
        ),
        "rubric": "guesstimate",
        "validity": validity,
        "backstop": {
            "findings": [],
            "summary": "Not scored — no genuine estimation attempt was detected.",
            "notChecked": True,
            "arithmeticOverridden": False,
            "rawTotal": 0,
            "totalCapFactor": 0,
        },
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _clamp_int(v: Any, lo: int, hi: int) -> int:
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        n = lo
    return max(lo, min(hi, n))


def _str_list(v: Any, limit: int = 6) -> list:
    if not isinstance(v, list):
        return []
    out = []
    for item in v:
        s = str(item).strip()
        if s:
            out.append(s[:400])
        if len(out) >= limit:
            break
    return out
