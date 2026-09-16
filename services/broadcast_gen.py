"""
Broadcast targeted-practice generator (admin-only).

Given a topic / company + kind (case | guesstimate) + difficulty, produce N
DISTINCT draft options in ONE model call (cheap, and the model varies them), WITHOUT
saving anything. The admin reviews the options and picks one; `save_option()` then
persists JUST that one as an UNLISTED case:

    is_active = False   -> never in daily rotation / practice lists / leaderboard
    unlisted  = True    -> BUT attemptable by direct link (broadcast recipients)
    owner_id  = <admin> , generated = True

Because attempting a case IS the conversational scored interview
(/cases/[id] -> ConversationalSolve -> interview engine -> scorer), one saved
option gives a whole college a live, scored mock from a broadcast email — the row
shape matches services/content_generator.py exactly, so the entire downstream loop
is inherited unchanged.

Grounded + cost-tiered: generation routes through the `daily_content` provider
feature (gpt-4o by default, admin-toggleable) and is metered in ai_usage_log.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from services.ai_providers import chat_with_fallback
from services.ai_usage import log_ai_usage

VALID_CASE_TYPES = {"profitability", "market_sizing", "growth"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def _clean(v: Optional[str]) -> str:
    return (v or "").strip()


def _coerce_difficulty(v: Optional[str], default: str = "medium") -> str:
    x = (v or "").strip().lower()
    return x if x in VALID_DIFFICULTIES else default


def _coerce_case_type(v: Optional[str]) -> str:
    x = (v or "").strip().lower().replace(" ", "_").replace("-", "_")
    return x if x in VALID_CASE_TYPES else "profitability"


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty model response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


_CASE_SHAPE = (
    '{"focus":"a short, correctly-spelled, presentable 2-4 word label naming the REAL brand or sector this targets (extract the actual company, FIX typos, Title Case; e.g. bluestone jhwellery -> BlueStone Jewellery). NEVER echo the user phrasing verbatim.",'
    '"options":[{'
    '"title":"short candidate-facing title (no real brand name unless generic)",'
    '"type":"profitability|market_sizing|growth",'
    '"difficulty":"easy|medium|hard",'
    '"hook":"ONE line (<=90 chars) that makes an aspirant want to try it; concrete, no hype",'
    '"scenario":"3-5 sentences: situation, a CXO/PE/founder protagonist, the explicit decision, '
    'and the 2-3 concrete Rs numbers the candidate needs",'
    '"quant_ask":"one specific quantity to compute, as a sentence",'
    '"framework_hint":"one line nudging structure without giving the answer",'
    '"solution":"4-8 sentence worked model solution, shown AFTER submit"'
    "}]}"
)

_GUESS_SHAPE = (
    '{"focus":"a short, correctly-spelled, presentable 2-4 word label naming the REAL brand or sector this targets (extract the actual company, FIX typos, Title Case; e.g. bluestone jhwellery -> BlueStone Jewellery). NEVER echo the user phrasing verbatim.",'
    '"options":[{'
    '"title":"short estimation question",'
    '"difficulty":"easy|medium|hard",'
    '"hook":"ONE line (<=90 chars) that makes an aspirant want to try it; concrete, no hype",'
    '"prompt":"2-4 sentences: exactly what to estimate, state assumptions, choose top-down/bottom-up, '
    'segment sensibly, give a point estimate, and sanity-check",'
    '"approach_hint":"one line on a sensible starting point without giving the answer",'
    '"solution":"4-8 sentence worked estimation, shown AFTER submit"'
    "}]}"
)


def generate_options(topic: str, kind: str, difficulty: str, count: int = 3) -> List[Dict[str, Any]]:
    """Return N distinct DRAFT options for `topic` (unsaved). Raises ValueError on bad output."""
    topic = _clean(topic)
    if not topic:
        raise ValueError("Enter a topic or company to generate around.")
    n = max(2, min(int(count or 3), 4))
    diff = _coerce_difficulty(difficulty, "medium")
    is_guess = (kind or "case").lower() == "guesstimate"
    what = "GUESSTIMATE (market-sizing / estimation)" if is_guess else "case-interview scenario"
    shape = _GUESS_SHAPE if is_guess else _CASE_SHAPE

    system = (
        "You are an expert McKinsey/BCG/Bain interviewer writing ORIGINAL, India-flavoured "
        "(Rs/crore, Indian sectors/cities/firms) practice material for MBA placement aspirants. "
        "Output strict JSON only."
    )
    user = (
        f'Create {n} DISTINCT {what} options an aspirant could practise, all grounded in this '
        f'topic/target: "{topic}".\nDIFFICULTY: {diff}.\n'
        "Make the options genuinely different from one another (different angle AND mechanic), each "
        "self-contained and freshly invented (never a real published casebook scenario). "
        f"Return ONLY JSON of EXACTLY this shape, with {n} entries in options:\n{shape}"
    )

    t0 = time.time()
    resp, model, provider = chat_with_fallback(
        "daily_content",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0.85,
        max_tokens=2600,
    )
    try:
        log_ai_usage(endpoint="/broadcast/generate-options", model=model, response=resp,
                     latency_ms=int((time.time() - t0) * 1000),
                     meta={"provider": provider, "kind": "guesstimate" if is_guess else "case"})
    except Exception:
        pass

    data = _extract_json(resp.choices[0].message.content)
    raw = data.get("options")
    if not isinstance(raw, list) or not raw:
        raise ValueError("The model did not return any options; try again.")

    out: List[Dict[str, Any]] = []
    for o in raw[:n]:
        if not isinstance(o, dict):
            continue
        if is_guess:
            prompt_text = _clean(o.get("prompt"))
            if not prompt_text:
                continue
            out.append({
                "kind": "guesstimate",
                "title": _clean(o.get("title")) or "Guesstimate",
                "type": "guesstimate",
                "difficulty": _coerce_difficulty(o.get("difficulty"), diff),
                "hook": _clean(o.get("hook"))[:140],
                "prompt": prompt_text,
                "approach_hint": _clean(o.get("approach_hint")) or None,
                "solution": _clean(o.get("solution")) or None,
            })
        else:
            scenario = _clean(o.get("scenario"))
            if not scenario:
                continue
            out.append({
                "kind": "case",
                "title": _clean(o.get("title")) or "Case",
                "type": _coerce_case_type(o.get("type")),
                "difficulty": _coerce_difficulty(o.get("difficulty"), diff),
                "hook": _clean(o.get("hook"))[:140],
                "scenario": scenario,
                "quant_ask": _clean(o.get("quant_ask")) or None,
                "framework_hint": _clean(o.get("framework_hint")) or None,
                "solution": _clean(o.get("solution")) or None,
            })

    if not out:
        raise ValueError("The model returned options in the wrong shape; try again.")
    focus = _clean(data.get("focus"))
    if focus:
        for _o in out:
            _o["focus"] = focus
    return out


def save_option(supabase, admin_id: str, option: Dict[str, Any], topic: str = "") -> Dict[str, Any]:
    """Persist ONE chosen option as an UNLISTED case. Returns {case_id, title, type, difficulty}."""
    if not isinstance(option, dict):
        raise ValueError("No option to save.")
    is_guess = (option.get("kind") or ("guesstimate" if option.get("type") == "guesstimate" else "case")) == "guesstimate"
    diff = _coerce_difficulty(option.get("difficulty"), "medium")

    if is_guess:
        content = _clean(option.get("prompt"))
        if not content:
            raise ValueError("The chosen guesstimate has no prompt.")
        row = {
            "title": _clean(option.get("title")) or "Targeted guesstimate",
            "type": "guesstimate",
            "difficulty": diff,
            "content": content,
            "hint": _clean(option.get("approach_hint")) or None,
            "solution": _clean(option.get("solution")) or None,
        }
    else:
        scenario = _clean(option.get("scenario"))
        if not scenario:
            raise ValueError("The chosen case has no scenario.")
        quant = _clean(option.get("quant_ask"))
        content = f"{scenario}\n\n**Quantitative ask:** {quant}" if quant else scenario
        row = {
            "title": _clean(option.get("title")) or "Targeted case",
            "type": _coerce_case_type(option.get("type")),
            "difficulty": diff,
            "content": content,
            "hint": _clean(option.get("framework_hint")) or None,
            "solution": _clean(option.get("solution")) or None,
        }

    row.update({
        "is_active": False,      # never in daily / lists / leaderboard
        "unlisted": True,        # BUT attemptable by direct link (broadcast recipients)
        "owner_id": admin_id,
        "generated": True,
        "generated_for": {"broadcast": True, "topic": _clean(topic)[:280],
                          "kind": "guesstimate" if is_guess else "case", "difficulty": diff},
    })

    try:
        ins = supabase.table("cases").insert(row).execute()
        new = (ins.data or [None])[0]
        if not new or not new.get("id"):
            raise ValueError("Saved the case but no id came back.")
        return {"case_id": new["id"], "title": row["title"], "type": row["type"], "difficulty": diff}
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Could not save the chosen option: {type(e).__name__}: {e}")
