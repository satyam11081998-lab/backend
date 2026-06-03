"""
AI Scorer - calls OpenAI to evaluate case interview answers.

This service is the bridge between Consilio's submission endpoint
and the OpenAI API. It loads the scoring prompt, sends the user's
answer for evaluation, parses the response, and returns structured
feedback.

The model and prompt logic are isolated here so we can swap providers
(OpenAI -> Anthropic, etc.) without touching the rest of the codebase.
"""

import os
import json
from typing import Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

from prompts.scoring_prompt import SCORING_SYSTEM_PROMPT, build_scoring_user_prompt
from prompts.guesstimate_scoring_prompt import (
    GUESSTIMATE_SCORING_SYSTEM_PROMPT,
    build_guesstimate_user_prompt,
)
from services.guesstimate_backstop import apply_backstop, DIMENSIONS as GUESSTIMATE_DIMS

load_dotenv()

# Initialize OpenAI client with API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env file")

client = OpenAI(api_key=OPENAI_API_KEY)

# Model selection - GPT-4o is reliable for structured output
# Swap to "gpt-4o-mini" for ~10x cheaper if scoring quality acceptable
SCORING_MODEL = "gpt-4o"

# Guesstimates are arithmetic-driven and the deterministic backstop catches the math
# the model would miss, so mini + backstop is the right cost/quality trade (~10x cheaper).
GUESSTIMATE_SCORING_MODEL = "gpt-4o-mini"


class AIScoringError(Exception):
    """Raised when AI scoring fails for any reason."""
    pass


def score_case_answer(
    case_content: str,
    case_type: str,
    user_answer: str,
) -> Dict[str, Any]:
    """
    Send a case answer to OpenAI for scoring and return structured feedback.
    
    Args:
        case_content: The full case prompt the student is answering
        case_type: One of 'guesstimate', 'profitability', 'market_sizing', 'growth'
        user_answer: The student's submitted answer text
    
    Returns:
        Dict with keys: score, breakdown, strengths, improvements, summary
    
    Raises:
        AIScoringError: If OpenAI call fails or response is malformed
    """
    user_prompt = build_scoring_user_prompt(
        case_content=case_content,
        case_type=case_type,
        user_answer=user_answer,
    )

    try:
        response = client.chat.completions.create(
            model=SCORING_MODEL,
            messages=[
                {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,  # Low temperature for consistent scoring
            response_format={"type": "json_object"},  # Force JSON output
        )
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

    # Validate required keys
    required_keys = {"score", "breakdown", "strengths", "improvements", "summary"}
    missing = required_keys - set(feedback.keys())
    if missing:
        raise AIScoringError(f"OpenAI response missing keys: {missing}")

    # Validate breakdown structure
    required_breakdown = {
        "structure", "quantitative", "synthesis",
        "business_judgment", "creativity", "presence",
    }
    missing_breakdown = required_breakdown - set(feedback["breakdown"].keys())
    if missing_breakdown:
        raise AIScoringError(
            f"OpenAI breakdown missing keys: {missing_breakdown}"
        )

    # Ensure score is in valid range
    if not isinstance(feedback["score"], int) or not (0 <= feedback["score"] <= 100):
        raise AIScoringError(
            f"Invalid score value: {feedback['score']} (must be int 0-100)"
        )

    return feedback

def score_guesstimate_answer(
    case_content: str,
    user_answer: str,
) -> Dict[str, Any]:
    """
    Score a GUESSTIMATE answer: one gpt-4o-mini call returns the 5 rubric dims +
    a transcribed calc-chain; the deterministic backstop then recomputes the math,
    OVERRIDES the arithmetic dimension, and caps the total. We never trust the LLM's
    own arithmetic.

    Returns a feedback dict shaped for persistence + the results page:
        score (0-100), breakdown (5 guesstimate dims, 1..5; arithmetic = backstop-corrected),
        strengths, improvements, summary, rubric='guesstimate', backstop={...}
    Raises AIScoringError on failure.
    """
    user_prompt = build_guesstimate_user_prompt(
        case_content=case_content,
        user_answer=user_answer,
    )

    try:
        response = client.chat.completions.create(
            model=GUESSTIMATE_SCORING_MODEL,
            messages=[
                {"role": "system", "content": GUESSTIMATE_SCORING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
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

    # Coerce each dimension to an int in 1..5; default to 3 if the model omitted one.
    def _clamp_dim(v) -> int:
        try:
            n = int(round(float(v)))
        except (TypeError, ValueError):
            n = 3
        return max(1, min(5, n))

    llm_dims = {d: _clamp_dim(dims.get(d, 3)) for d in GUESSTIMATE_DIMS}

    # Deterministic backstop: recompute the chain, override arithmetic, cap total.
    # band is optional (not yet stored per-guesstimate) — magnitude guard simply off when absent.
    final = apply_backstop(llm_dims, chain, band=None)

    return {
        "score": int(final["total"]),
        "breakdown": final["dimensions"],   # 5 dims, 1..5; arithmetic is backstop-corrected
        "strengths": parsed.get("strengths", []) or [],
        "improvements": parsed.get("improvements", []) or [],
        "summary": parsed.get("summary", "") or "",
        "rubric": "guesstimate",
        "backstop": {
            "findings": final["backstop"]["findings"],
            "summary": final["backstop"]["summary"],
            "notChecked": final["backstop"]["notChecked"],
            "arithmeticOverridden": final["arithmeticOverridden"],
            "rawTotal": final["rawTotal"],
            "totalCapFactor": final["backstop"]["totalCapFactor"],
        },
    }
