"""
Prep Copilot — CURATED TOOLS (agentic actions, not advice).

`generate_curated_case` produces ONE case or guesstimate aimed at a single
candidate's weakest skill + target firm/role, and saves it PRIVATE to them
(is_active=false, owner_id set). Because attempting a case IS the conversational
scored interview (/cases/[id] -> ConversationalSolve -> interview engine ->
scorer), one generated case gives the candidate a live, role-targeted mock they
are actually scored on — no separate interview machinery needed.

Grounded + cost-tiered: generation routes through the `daily_content` provider
feature (gpt-4o by default, admin-toggleable to Groq) and is metered in
ai_usage_log. The row shape matches services/content_generator.py exactly, so the
whole downstream loop is inherited unchanged.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from services.ai_providers import chat_with_fallback
from services.ai_usage import log_ai_usage

from .specialists import DIMENSIONS, DIM_LABEL, DIM_TECHNIQUE

VALID_CASE_TYPES = {"profitability", "market_sizing", "growth"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def _coerce_case_type(value: Optional[str]) -> str:
    v = (value or "").strip().lower().replace(" ", "_").replace("-", "_")
    return v if v in VALID_CASE_TYPES else "profitability"


def _coerce_difficulty(value: Optional[str], default: str = "hard") -> str:
    v = (value or "").strip().lower()
    return v if v in VALID_DIFFICULTIES else default


def _clean(v: Optional[str]) -> str:
    return (v or "").strip()


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty model response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def _case_prompt(dim: str, technique: str, company: str, role: str, difficulty: str) -> str:
    tgt = company.strip() or role.strip() or "a top consulting / product role"
    return (
        f"Create ONE original case-interview scenario for an Indian MBA placement aspirant.\n"
        f"TARGET: {tgt}. ROLE/DOMAIN: {role.strip() or '(general)'}. DIFFICULTY: {difficulty}.\n"
        f"IT MUST STRESS THIS SKILL above all: {DIM_LABEL.get(dim, dim)}. Design the case so the "
        f"candidate cannot do well without {technique}.\n"
        "India-flavoured (Rs/crore, Indian sectors/cities/firms), self-contained, freshly invented "
        "(never a real published casebook). Return ONLY JSON:\n"
        '{"title":"short candidate-facing title","type":"profitability|market_sizing|growth",'
        '"difficulty":"easy|medium|hard","scenario":"3-5 sentences: situation, a CXO/PE/founder '
        'protagonist, the explicit decision, and the 2-3 concrete Rs numbers needed",'
        '"quant_ask":"one specific quantitative thing to compute, as a sentence",'
        '"framework_hint":"one line nudging structure without giving the answer",'
        '"solution":"4-8 sentence worked model solution shown after submit"}'
    )


def _guess_prompt(dim: str, technique: str, company: str, role: str, difficulty: str) -> str:
    tgt = company.strip() or role.strip() or "a top consulting / product role"
    return (
        f"Create ONE original GUESSTIMATE (market-sizing/estimation) for an Indian MBA placement "
        f"aspirant.\nTARGET: {tgt}. ROLE/DOMAIN: {role.strip() or '(general)'}. DIFFICULTY: {difficulty}.\n"
        f"Pick a subject in the target's world, and design it to stress {DIM_LABEL.get(dim, dim)} "
        f"(the candidate should have to lean on {technique}).\n"
        "India-flavoured (Rs, Indian cities/sectors), self-contained, freshly invented. Return ONLY JSON:\n"
        '{"title":"short estimation question","difficulty":"easy|medium|hard",'
        '"prompt":"2-4 sentences telling them exactly what to estimate, to state assumptions, choose '
        'top-down/bottom-up, segment sensibly, give a point estimate, and sanity-check",'
        '"approach_hint":"one line on a sensible starting point without giving the answer",'
        '"solution":"4-8 sentence worked estimation shown after submit"}'
    )


def generate_curated_case(
    supabase,
    user_id: str,
    *,
    focus_dimension: str = "structure",
    target_company: str = "",
    role: str = "",
    difficulty: str = "hard",
    kind: str = "case",
) -> Dict[str, Any]:
    """Generate ONE curated case/guesstimate, save it private to the user, and
    return {case_id, title, type, difficulty}. Raises ValueError on bad output."""
    dim = focus_dimension if focus_dimension in DIMENSIONS else "structure"
    diff = _coerce_difficulty(difficulty)
    technique = DIM_TECHNIQUE.get(dim, "targeted reps")
    is_guess = (kind or "case").lower() == "guesstimate"

    prompt = (_guess_prompt if is_guess else _case_prompt)(dim, technique, target_company, role, diff)

    t0 = time.time()
    resp, model, provider = chat_with_fallback(
        "daily_content",
        messages=[
            {"role": "system", "content": "You are an expert McKinsey/BCG/Bain interviewer who writes "
             "original, India-flavoured practice material. Output strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.8,
        max_tokens=1500,
    )
    try:
        log_ai_usage(user_id=user_id, endpoint="/coach/tool/case", model=model,
                     response=resp, latency_ms=int((time.time() - t0) * 1000),
                     meta={"provider": provider, "kind": "guesstimate" if is_guess else "case",
                           "focus_dimension": dim})
    except Exception:
        pass

    data = _extract_json(resp.choices[0].message.content)

    if is_guess:
        prompt_text = _clean(data.get("prompt"))
        if not prompt_text:
            raise ValueError("Model did not return a guesstimate prompt; try again.")
        row = {
            "title": _clean(data.get("title")) or "Curated guesstimate",
            "type": "guesstimate",
            "difficulty": _coerce_difficulty(data.get("difficulty"), diff),
            "content": prompt_text,
            "hint": _clean(data.get("approach_hint")) or None,
            "solution": _clean(data.get("solution")) or None,
        }
    else:
        scenario = _clean(data.get("scenario"))
        if not scenario:
            raise ValueError("Model did not return a case scenario; try again.")
        quant = _clean(data.get("quant_ask"))
        content = f"{scenario}\n\n**Quantitative ask:** {quant}" if quant else scenario
        row = {
            "title": _clean(data.get("title")) or "Curated case",
            "type": _coerce_case_type(data.get("type")),
            "difficulty": _coerce_difficulty(data.get("difficulty"), diff),
            "content": content,
            "hint": _clean(data.get("framework_hint")) or None,
            "solution": _clean(data.get("solution")) or None,
        }

    row.update({
        "is_active": False,          # private: never in lists / daily / leaderboard
        "owner_id": user_id,
        "generated": True,
        "generated_for": {"focus_dimension": dim, "target_company": target_company,
                          "role": role, "difficulty": diff, "model": model},
    })

    try:
        ins = supabase.table("cases").insert(row).execute()
        new = (ins.data or [None])[0]
        if not new or not new.get("id"):
            raise ValueError("Case saved but no id returned.")
        return {"case_id": new["id"], "title": row["title"], "type": row["type"],
                "difficulty": row["difficulty"]}
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Generated the case but could not save it: {type(e).__name__}: {e}")
