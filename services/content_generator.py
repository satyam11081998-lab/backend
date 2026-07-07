"""
Content generator service using GPT-4o.

Generates one fresh daily Case and one fresh daily Guesstimate.

IMPORTANT (2026-06-02 fix): both are written as rows in the REAL `cases`
table — columns: (title, type, difficulty, content, hint, is_active).
The earlier version wrote to columns that do not exist (code, sector,
root_cause, ...) and to a non-existent `guesstimates` table, so every
insert hard-failed → /cron/schedule-daily 500 → no daily content ever.

The guesstimate is stored as a normal case with type='guesstimate', so it
flows through the same /cases/[id] → submit → AI-score → leaderboard path
as every other case (nothing special needed downstream).
"""

import os
import json
import time
from typing import List, Optional, Tuple
from openai import OpenAI
from services.supabase_client import get_supabase_client
from services.ai_usage import log_ai_usage


# These MUST match lib/constants.ts on the frontend.
VALID_CASE_TYPES = {"profitability", "market_sizing", "growth", "guesstimate"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


class GeneratorError(Exception):
    pass


def _coerce_type(value: Optional[str], default: str = "profitability") -> str:
    """Normalise an AI-supplied type to a valid enum; never let a bad value reach the DB."""
    v = (value or "").strip().lower().replace(" ", "_").replace("-", "_")
    return v if v in VALID_CASE_TYPES else default


def _coerce_difficulty(value: Optional[str], default: str = "medium") -> str:
    v = (value or "").strip().lower()
    return v if v in VALID_DIFFICULTIES else default


def _clean(value: Optional[str]) -> str:
    return (value or "").strip()


SYSTEM_PROMPT = """You are an expert McKinsey/BCG/Bain interviewer creating ORIGINAL daily practice \
material for Indian MBA placement aspirants. You produce exactly ONE case study and ONE guesstimate.

Everything must be freshly invented, India-flavoured (₹ figures, Indian sectors/companies/cities), \
realistic, and self-contained. Never reuse a real published casebook scenario verbatim.

# CASE STUDY
- type: one of "profitability", "market_sizing", "growth" (NOT guesstimate).
- difficulty: one of "easy", "medium", "hard".
- title: a short, specific, candidate-facing title (no company name that exists in real life unless generic).
- scenario: the full prompt the candidate reads and solves — 3-5 sentences. Set the situation, the \
protagonist (a CXO/PE fund/founder), and the explicit decision to make. Self-contained: include the \
2-3 concrete ₹ numbers the candidate needs (revenue, margin, volume, growth rate, etc.).
- quant_ask: ONE specific quantitative thing they must compute, stated as a sentence \
(e.g. "Estimate the break-even price if fixed costs are ₹50 cr and variable cost is ₹120/unit.").
- framework_hint: ONE short line nudging the structure WITHOUT giving the answer \
(e.g. "Think Profit = Revenue − Cost; split revenue into price × volume by segment.").
- solution: a concise worked model solution (4-8 sentences) — the structure a strong candidate would \
use, the key driver, the arithmetic with the ₹ numbers, and the recommendation. This is shown to the \
candidate AFTER they submit, so it should teach how to solve, not just state the answer.

# GUESSTIMATE
- difficulty: one of "easy", "medium", "hard".
- title: a short, specific estimation question (e.g. "Estimate the number of EV two-wheelers sold in \
Pune in a year.").
- prompt: 2-4 sentences telling the candidate exactly what to estimate, to state assumptions, choose \
top-down or bottom-up, segment by a sensible driver, give a single point estimate, and sanity-check it.
- approach_hint: ONE short line on a sensible starting point WITHOUT giving the answer \
(e.g. "Start from Pune's population, funnel to households, two-wheeler ownership, EV share, replacement rate.").
- solution: a concise worked estimation (4-8 sentences) — the segmentation path, each assumption with a \
number, the multiplication chain to the final ₹/unit figure, and a one-line sanity check. Shown AFTER submit.

OUTPUT FORMAT — return ONLY a valid JSON object, no markdown, exactly this shape:
{
  "case": {
    "title": "...", "type": "...", "difficulty": "...",
    "scenario": "...", "quant_ask": "...", "framework_hint": "...", "solution": "..."
  },
  "guesstimate": {
    "title": "...", "difficulty": "...", "prompt": "...", "approach_hint": "...", "solution": "..."
  }
}
"""


def generate_daily_content(recent_themes: List[str]) -> dict:
    """Call GPT-4o to produce one case + one guesstimate as a JSON object."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise GeneratorError("OPENAI_API_KEY not set")

    # Bounded so a slow/hung OpenAI call fails FAST (→ daily_scheduler falls back to
    # existing cases) instead of tying up the Render worker until the cron times out.
    # max_retries lets the SDK ride out transient 429/5xx before giving up.
    client = OpenAI(api_key=api_key, timeout=60.0, max_retries=2)

    user_prompt = "Generate one challenging MBA-level Case Study and one Guesstimate for today.\n"
    if recent_themes:
        joined = ", ".join(t for t in recent_themes if t)
        if joined:
            user_prompt += (
                "Make them DISTINCT from these recent titles (different sector AND different "
                f"mechanic):\n{joined}\n"
            )

    try:
        t0 = time.time()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.8,
            max_tokens=2000,  # one case + one guesstimate as JSON; ceiling well above the ~950 typical
        )
        log_ai_usage(endpoint="/cron/schedule-daily", model="gpt-4o", response=response,
                     latency_ms=int((time.time() - t0) * 1000))
        raw_content = response.choices[0].message.content
    except Exception as e:
        raise GeneratorError(f"OpenAI request failed: {type(e).__name__}: {e}")

    if not raw_content or not raw_content.strip():
        raise GeneratorError("Model returned empty content")

    try:
        return json.loads(raw_content)
    except Exception as e:
        raise GeneratorError(f"Model returned non-JSON content: {type(e).__name__}: {e}")


def _compose_case_row(case: dict) -> dict:
    """Map the AI case object onto the real `cases` columns."""
    scenario = _clean(case.get("scenario"))
    quant_ask = _clean(case.get("quant_ask"))
    if not scenario:
        raise GeneratorError("Case scenario missing in AI response")

    content = scenario
    if quant_ask:
        content = f"{scenario}\n\n**Quantitative ask:** {quant_ask}"

    return {
        "title": _clean(case.get("title")) or "Daily Case",
        "type": _coerce_type(case.get("type"), default="profitability"),
        "difficulty": _coerce_difficulty(case.get("difficulty")),
        "content": content,
        "hint": _clean(case.get("framework_hint")) or None,
        "solution": _clean(case.get("solution")) or None,
        "is_active": True,
    }


def _compose_guesstimate_row(guess: dict) -> dict:
    """Map the AI guesstimate object onto the real `cases` columns (type forced to guesstimate)."""
    prompt = _clean(guess.get("prompt"))
    if not prompt:
        raise GeneratorError("Guesstimate prompt missing in AI response")

    return {
        "title": _clean(guess.get("title")) or "Daily Guesstimate",
        "type": "guesstimate",  # forced — this is what makes it an attemptable guesstimate
        "difficulty": _coerce_difficulty(guess.get("difficulty")),
        "content": prompt,
        "hint": _clean(guess.get("approach_hint")) or None,
        "solution": _clean(guess.get("solution")) or None,
        "is_active": True,
    }


def save_generated_content() -> dict:
    """
    End-to-end: generate one case + one guesstimate, insert BOTH as `cases` rows,
    return their ids.

    Returns: {"case_id": <uuid>, "guesstimate_id": <uuid>}
    """
    supabase = get_supabase_client()

    # Recent titles → anti-repeat signal for the prompt.
    try:
        recent = (
            supabase.table("cases")
            .select("title")
            .order("created_at", desc=True)
            .limit(10)
            .execute()
        )
        recent_themes = [row["title"] for row in (recent.data or []) if row.get("title")]
    except Exception:
        recent_themes = []  # anti-repeat is best-effort, never block generation on it

    content = generate_daily_content(recent_themes)
    case_data = content.get("case")
    guess_data = content.get("guesstimate")
    if not case_data or not guess_data:
        raise GeneratorError("AI response missing 'case' or 'guesstimate'")

    case_row = _compose_case_row(case_data)
    guess_row = _compose_guesstimate_row(guess_data)

    # Insert the case
    case_res = supabase.table("cases").insert(case_row).execute()
    if not case_res.data:
        raise GeneratorError("Case insert returned no row")
    case_id = case_res.data[0]["id"]

    # Insert the guesstimate (as a case)
    guess_res = supabase.table("cases").insert(guess_row).execute()
    if not guess_res.data:
        raise GeneratorError("Guesstimate insert returned no row")
    guesstimate_id = guess_res.data[0]["id"]

    return {"case_id": case_id, "guesstimate_id": guesstimate_id}
