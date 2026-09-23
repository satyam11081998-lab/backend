"""
Prep Copilot v2 - ISOLATED role/company-aware GPT scorer.

This is the copilot's OWN scorer. It is deliberately NOT the live case scorer
(services/ai_scorer.py) and never imports its scoring FUNCTIONS - the live
cases/guesstimates path stays byte-for-byte untouched. What is shared is only
provider infra (OpenAI client, usage logging), which is read-only.

GPT ONLY. Scoring is locked to OpenAI platform-wide (services/ai_providers), and
we keep that here: a role rubric changes WHAT is measured, never WHICH model
grades. The evidence-based, anti-gaming spine of the original rubric prompt is
preserved; only the dimensions, maxima, calibration anchors and red flags are
swapped in from the active Pack. So an ASM answer that name-drops 'distributor
ROI' without doing the working-capital math is penalised exactly as a consulting
answer is penalised for name-dropping MECE.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from services.ai_providers import openai_client
from services.ai_usage import log_ai_usage

from .schemas import Pack, Scenario

SCORING_MODEL = "gpt-4o"   # premium, GPT-only - matches the live case scorer's tier


def _clamp_int(v: Any, lo: int, hi: int) -> int:
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        n = lo
    return max(lo, min(hi, n))


def build_rubric_system_prompt(pack: Pack) -> str:
    """Compose the scoring system prompt from the Pack's rubric. Preserves the
    original evidence rule + anti-gaming stance; parametrises the dimensions."""
    dims = pack.rubric.dimensions
    total = pack.rubric.total or 100
    role_label = pack.display_role or "the target role"
    company_label = f" targeting {pack.display_company}" if pack.display_company else ""

    dim_lines: List[str] = []
    for i, d in enumerate(dims, 1):
        line = f"{i}. {d.label.upper()} ({d.max}) - key: {d.key}"
        if d.what_good_looks_like:
            line += f"\n   What a strong hire does: {d.what_good_looks_like}"
        if d.anchors:
            line += "\n   Calibration (score to the band the EVIDENCE supports): " + " | ".join(d.anchors[:5])
        if d.red_flags:
            line += "\n   Red flags (penalise + name): " + "; ".join(d.red_flags[:5])
        dim_lines.append(line)

    fw = ""
    if pack.frameworks:
        fw = "\n\nThe frameworks a strong candidate for this role actually applies (reward APPLICATION of these, never mere mention):\n" + \
             "\n".join(f"- {f.name}: {f.summary}" for f in pack.frameworks[:12])

    breakdown_keys = ", ".join(f'"{d.key}": <0-{d.max}>' for d in dims) or '"overall": <0-100>'
    dim_fb_keys = ",\n    ".join(
        f'"{d.key}": {{"score": <int>, "evidence": "<what they actually did - quote/paraphrase, or \'none\'>", "gap": "<what was missing or wrong>", "to_improve": "<what a top answer for THIS role does here>"}}'
        for d in dims
    )

    return f"""You are a senior, domain-expert interviewer and evaluator for {role_label}{company_label}. \
You have hired and coached for this exact role and you debrief candidates honestly, at the depth a \
category head / senior practitioner would - NOT with generic 'communication skills' feedback.

You score across exactly {len(dims)} role-specific dimensions, totalling {total} points:

{chr(10).join(dim_lines)}{fw}

EVIDENCE RULE (this is what separates you from a lenient grader):
- A dimension may only reach 'good' or 'excellent' on the strength of something the candidate ACTUALLY \
said or wrote. If you cannot quote or closely paraphrase evidence for a score, the score is too high.
- Reward APPLICATION, never mention. Naming a framework or a metric earns nothing on its own; applying \
it correctly to THIS scenario earns the points. Name-dropping without the working is a red flag.
- Do NOT reward length, confident tone, buzzwords, or padding. Penalise and name in red_flags: keyword/\
framework stuffing, a memorised framework force-fit with no scenario-specific adaptation, restating the \
prompt as if it were analysis, claiming a calculation without showing it, internal contradictions, and \
recommendations with no supporting logic.
- Be strict and honest. Most real first attempts land 40-65. Do NOT cluster everyone near 70. Reserve \
80+ for answers that would genuinely impress a hiring manager for this role.
- breakdown values MUST sum to score, each within its dimension's max.

If the answer is THIN (short/underdeveloped), do not invent strengths - score what is there.

OUTPUT - return ONLY valid JSON (no markdown, no prose) in EXACTLY this shape:
{{
  "score": <int 0-{total}, = sum of breakdown>,
  "breakdown": {{ {breakdown_keys} }},
  "dimension_feedback": {{
    {dim_fb_keys}
  }},
  "strengths": ["<specific, evidence-anchored strength>", "..."],
  "improvements": ["<specific, actionable fix tied to what they said>", "..."],
  "red_flags": ["<gaming/logic/ethics problem if any - omit or [] if none>"],
  "model_answer": "<5-8 short lines: how a strong hire for THIS role would actually work this scenario - the structure, the key numbers/mechanism, the trade-off, the recommendation. Concrete to this scenario.>",
  "summary": "<3-5 sentence honest debrief: open with the one thing they did well, name where they stand and the SINGLE biggest lever for this role, end with the one concrete thing to practise next.>"
}}"""


def build_user_prompt(pack: Pack, scenario: Scenario, transcript_text: str) -> str:
    assess = pack.assessment or {}
    num = " Numericals ARE expected for this role - if the candidate avoided the math, that is a gap." \
        if assess.get("numericals_expected") else ""
    return (
        f"ROLE: {pack.display_role}\n"
        f"COMPANY: {pack.display_company or '(role-typical, no specific company)'}\n"
        f"SCENARIO TITLE: {scenario.title}\n"
        f"SCENARIO: {scenario.prompt}\n"
        + (f"EXPLICIT QUANT ASK: {scenario.numerical_ask}\n" if scenario.numerical_ask else "")
        + f"WHAT THIS TARGET ASSESSES: {', '.join(assess.get('what_they_test') or []) or '(role-typical)'}.{num}\n\n"
        "THE CANDIDATE'S FULL ANSWER / TRANSCRIPT (grade only what is here):\n"
        f"{transcript_text.strip() or '(empty)'}\n\n"
        "Score using the role rubric above; justify every dimension with evidence from the answer; return ONLY the JSON."
    )


def score_role_answer(pack: Pack, scenario: Scenario, transcript_text: str,
                      user_id: Optional[str] = None) -> Dict[str, Any]:
    """Grade one role-practice answer against the Pack rubric. GPT only. Returns
    the normalised feedback dict (same shape family the results UI already reads).
    Raises RuntimeError only if the model call itself fails."""
    client = openai_client()
    if client is None:
        raise RuntimeError("OpenAI not configured for scoring")

    dims = pack.rubric.dimensions
    sys_prompt = build_rubric_system_prompt(pack)
    user_prompt = build_user_prompt(pack, scenario, transcript_text)

    t0 = time.time()
    resp = client.chat.completions.create(
        model=SCORING_MODEL,
        messages=[{"role": "system", "content": sys_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=0.2,
        max_tokens=1600,
        response_format={"type": "json_object"},
    )
    try:
        log_ai_usage(user_id=user_id, endpoint="/copilot/score", model=SCORING_MODEL,
                     response=resp, latency_ms=int((time.time() - t0) * 1000))
    except Exception:
        pass

    raw = resp.choices[0].message.content or "{}"
    try:
        fb = json.loads(raw)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        fb = json.loads(m.group(0)) if m else {}

    return _normalise(pack, fb)


def _normalise(pack: Pack, fb: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp each dimension to its max, force breakdown to sum to score, guarantee
    the keys the UI expects. Never trusts the model's arithmetic."""
    dims = pack.rubric.dimensions
    raw_bd = fb.get("breakdown") if isinstance(fb.get("breakdown"), dict) else {}
    breakdown: Dict[str, int] = {}
    for d in dims:
        breakdown[d.key] = _clamp_int(raw_bd.get(d.key, 0), 0, d.max)
    score = sum(breakdown.values())

    dim_fb = fb.get("dimension_feedback") if isinstance(fb.get("dimension_feedback"), dict) else {}
    clean_dim_fb: Dict[str, Any] = {}
    for d in dims:
        entry = dim_fb.get(d.key) if isinstance(dim_fb.get(d.key), dict) else {}
        clean_dim_fb[d.key] = {
            "score": breakdown[d.key],
            "evidence": str(entry.get("evidence") or "none"),
            "gap": str(entry.get("gap") or ""),
            "to_improve": str(entry.get("to_improve") or ""),
        }

    def _slist(v: Any) -> List[str]:
        return [str(x) for x in v if isinstance(x, (str, int, float))][:6] if isinstance(v, list) else []

    return {
        "score": score,
        "breakdown": breakdown,
        "rubric": {"role": pack.display_role, "company": pack.display_company,
                   "dimensions": [{"key": d.key, "label": d.label, "max": d.max} for d in dims]},
        "dimension_feedback": clean_dim_fb,
        "strengths": _slist(fb.get("strengths")),
        "improvements": _slist(fb.get("improvements")),
        "red_flags": _slist(fb.get("red_flags")),
        "model_answer": str(fb.get("model_answer") or ""),
        "summary": str(fb.get("summary") or ""),
        "confidence": pack.confidence,
    }
