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

load_dotenv()

# Initialize OpenAI client with API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env file")

client = OpenAI(api_key=OPENAI_API_KEY)

# Model selection - GPT-4o is reliable for structured output
# Swap to "gpt-4o-mini" for ~10x cheaper if scoring quality acceptable
SCORING_MODEL = "gpt-4o"


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