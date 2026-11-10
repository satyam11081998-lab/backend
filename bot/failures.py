"""C4 Step 3 deterministic candidate failure shortlisting and Gemini ranking."""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import HTTPException
from openai import OpenAI

from bot.config import DEFAULT_GEMINI_MODEL, GEMINI_BASE_URL, MAX_FAILURES_IN_REPORT, MAX_SHORTLIST_FOR_LLM
from bot.data import WindowDataset
from bot.redact import assert_no_identifiers, build_college_terms, redact_text
from bot.state import hash_identifier, parse_iso_utc
from bot.turns import AttemptEval, TurnEval


class BudgetExhaustedError(RuntimeError):
    """Raised when services.ai_usage.assert_daily_budget() trips."""


class GeminiQuotaOrAuthError(RuntimeError):
    """Raised when Gemini rejects the request (quota/429/auth) — abort without report."""


class TokenCapExceededError(RuntimeError):
    """Raised when the per-run token budget cap is exceeded."""


_SIGNAL_PRIORITY = {
    "praise_only": 6,
    "help_not_actionable": 5,
    "repeated_question": 4,
    "stacked_questions": 4,
    "question_on_hand_back": 4,
    "substantive_no_stage_advance": 3,
    "abandoned_within_2_turns": 2,
}

_DEFAULT_SHOULD_HAVE = {
    "praise_only": "Acknowledge the candidate's specific move neutrally and advance or probe one missing branch.",
    "repeated_question": "Avoid re-asking the same question; reframe the step or give a concrete foothold.",
    "help_not_actionable": "Give one concrete directional nudge or skeleton anchored in the candidate's own work.",
    "stacked_questions": "Ask only one focused question at a time so the candidate can respond clearly.",
    "question_on_hand_back": "Return the floor without asking a follow-up question when HAND_BACK is selected.",
    "substantive_no_stage_advance": "Recognise the completed reasoning step and transition cleanly to the next stage.",
    "abandoned_within_2_turns": "Provide clearer scaffolding or check what blocked the candidate before drop-off.",
}

_COMPACT_SIGNAL_KEYS = (
    "has_work",
    "prior_work",
    "intent",
    "turn_type",
    "help_requested",
    "interviewer_repeating",
    "is_substantive_reasoning",
    "frustration",
    "repair_due",
    "teaching_policy",
)


def _in_window(ts_raw: Optional[str], start: datetime, end: datetime) -> bool:
    if not ts_raw:
        return False
    try:
        dt = parse_iso_utc(str(ts_raw))
    except Exception:  # noqa: BLE001
        return False
    return start <= dt <= end


def _compact_signals(sig: Dict[str, Any]) -> Dict[str, Any]:
    return {k: sig.get(k) for k in _COMPACT_SIGNAL_KEYS if k in sig}


def _primary_signal(flags: List[str]) -> str:
    return max(flags, key=lambda f: _SIGNAL_PRIORITY.get(f, 0))


def _is_quota_or_auth_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    code = getattr(exc, "status_code", None)
    if code in (401, 403, 429):
        return True
    tokens = ("429", "401", "403", "quota", "resource_exhausted", "rate limit", "rate_limit", "api key", "permission")
    return any(t in msg for t in tokens)


def _is_transient_gemini_error(exc: Exception) -> bool:
    if _is_quota_or_auth_error(exc):
        return False
    code = getattr(exc, "status_code", None)
    if code in (500, 502, 503, 504):
        return True
    msg = str(exc).lower()
    tokens = ("500", "502", "503", "504", "unavailable", "high demand", "overloaded", "temporarily", "timeout", "timed out")
    return any(t in msg for t in tokens)


def format_deterministic_failures(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build capped C5 failure entries deterministically without a Gemini call."""
    ordered = sorted(
        candidates,
        key=lambda item: (-int(item.get("frequency_7d", 1)), int(item.get("id", 0))),
    )
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(ordered[:MAX_FAILURES_IN_REPORT], start=1):
        out.append({
            "rank": idx,
            "signal": item["signal"],
            "attempt_hash": item["attempt_hash"],
            "case_type": item["case_type"],
            "stage": item["stage"],
            "candidate_turn": item["candidate_turn"],
            "interviewer_reply": item["interviewer_reply"],
            "signals": item["signals"],
            "response_function": item["response_function"],
            "should_have": _DEFAULT_SHOULD_HAVE.get(
                item["signal"], "Respond specifically to the candidate's turn and move the case forward."
            ),
            "frequency_7d": item["frequency_7d"],
            "abandoned_within_2_turns": item["abandoned_within_2_turns"],
        })
    return out


def build_redacted_shortlist(
    dataset: WindowDataset,
    attempt_evals: List[AttemptEval],
    bot_hash_salt: str,
    windows: Dict[str, datetime],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Flag candidate turns deterministically, count 7d frequencies, and redact PII."""
    w_from, w_to = windows["window_from"], windows["window_to"]
    r_from, r_to = windows["rolling_7d_from"], windows["rolling_7d_to"]

    # Compute 7-day signal frequencies across rolling 7d window (deterministic, 0 model calls).
    freq_7d: Counter[str] = Counter()
    for ae in attempt_evals:
        if _in_window(ae.attempt.get("created_at"), r_from, r_to):
            for t in ae.turns:
                for fl in t.flags:
                    freq_7d[fl] += 1

    # Also collect window items (either attempt created in window, or attempt analyzed in window).
    evals_in_window = [
        ae
        for ae in attempt_evals
        if _in_window(ae.attempt.get("created_at"), w_from, w_to)
        or any(
            _in_window(m.get("created_at"), w_from, w_to)
            for m in (dataset.messages_by_attempt.get(str(ae.attempt.get("id") or "")) or [])
        )
    ]

    college_terms = build_college_terms(
        dataset.colleges,
        [str(u.get("college_other")) for u in dataset.users_by_id.values() if u.get("college_other")],
    )
    all_raw_ids = [str(u.get("id")) for u in dataset.users_by_id.values() if u.get("id")] + [
        str(a.get("id")) for a in dataset.attempts if a.get("id")
    ]
    all_raw_emails = [
        str(u.get(k))
        for u in dataset.users_by_id.values()
        for k in ("email", "college_email")
        if u.get(k)
    ]
    all_raw_names = [
        str(u.get(k))
        for u in dataset.users_by_id.values()
        for k in ("full_name", "name")
        if u.get(k)
    ]
    all_raw_colleges = [str(u.get("college_other")) for u in dataset.users_by_id.values() if u.get("college_other")]

    shortlist: List[Dict[str, Any]] = []
    for ae in evals_in_window:
        aid = str(ae.attempt.get("id") or "")
        uid = str(ae.attempt.get("user_id") or "")
        user_info = dataset.users_by_id.get(uid) or {"id": uid}
        attempt_hash = hash_identifier(aid, bot_hash_salt)

        for t in ae.turns:
            if not t.flags:
                continue
            sig_name = _primary_signal(t.flags)
            freq = max(1, int(freq_7d.get(sig_name, 1)))

            red_cand = redact_text(
                t.candidate_turn,
                user_info=user_info,
                known_ids=all_raw_ids,
                known_emails=all_raw_emails,
                college_terms=college_terms,
                extra_names=all_raw_names,
            )
            red_reply = redact_text(
                t.interviewer_reply,
                user_info=user_info,
                known_ids=all_raw_ids,
                known_emails=all_raw_emails,
                college_terms=college_terms,
                extra_names=all_raw_names,
            )

            shortlist.append({
                "id": len(shortlist) + 1,
                "signal": sig_name,
                "all_signals": list(t.flags),
                "attempt_hash": attempt_hash,
                "case_type": ae.case_type,
                "stage": t.stage,
                "candidate_turn": red_cand,
                "interviewer_reply": red_reply,
                "signals": _compact_signals(t.signals),
                "response_function": t.response_function,
                "frequency_7d": freq,
                "abandoned_within_2_turns": bool(t.abandoned_within_2_turns),
                "_priority": _SIGNAL_PRIORITY.get(sig_name, 0),
            })

    # Verify zero identifiers survived across ALL candidate turns & replies before anything else.
    assert_no_identifiers(
        [x for item in shortlist for x in (item["candidate_turn"], item["interviewer_reply"])],
        raw_ids=all_raw_ids,
        raw_emails=all_raw_emails,
        raw_names=all_raw_names,
        raw_colleges=all_raw_colleges,
    )

    shortlist.sort(
        key=lambda x: (-x["frequency_7d"], -int(x["abandoned_within_2_turns"]), -x["_priority"], x["id"])
    )
    leak_ctx = {
        "raw_ids": all_raw_ids,
        "raw_emails": all_raw_emails,
        "raw_names": all_raw_names,
        "raw_colleges": all_raw_colleges,
    }
    return shortlist, leak_ctx


def _build_gemini_prompt(candidates: List[Dict[str, Any]]) -> str:
    payload = [
        {
            "id": item["id"],
            "signal": item["signal"],
            "case_type": item["case_type"],
            "stage": item["stage"],
            "frequency_7d": item["frequency_7d"],
            "abandoned_within_2_turns": item["abandoned_within_2_turns"],
            "response_function": item["response_function"],
            "signals": item["signals"],
            "candidate_turn": item["candidate_turn"][:600],
            "interviewer_reply": item["interviewer_reply"][:600],
        }
        for item in candidates
    ]
    return (
        "You are auditing a consulting case interviewer bot. Below are redacted candidate failure turns.\n"
        "Rank the top items (up to 10) by product impact, weighting `frequency_7d` and `abandoned_within_2_turns` heavily.\n"
        "For each item, write ONE concise sentence (`should_have`) describing what the interviewer should have done.\n"
        "Return ONLY raw JSON matching this schema:\n"
        '{"items": [{"id": 1, "rank": 1, "should_have": "One concise sentence."}]}\n\n'
        f"CANDIDATES:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def rank_failures_with_gemini(
    shortlist: List[Dict[str, Any]],
    leak_ctx: Dict[str, Any],
    *,
    gemini_api_key: str,
    report_only: bool = False,
    max_tokens_budget: int = 50000,
    gemini_client: Any = None,
    usage_logger: Optional[Callable[..., None]] = None,
    budget_checker: Optional[Callable[[], None]] = None,
    cost_estimator: Optional[Callable[[str, Optional[int], Optional[int]], float]] = None,
    max_retries: int = 2,
    retry_delay_sec: float = 2.0,
) -> Tuple[List[Dict[str, Any]], float, int]:
    """Rank up to 10 failures using Gemini (or deterministic fallback in report-only mode)."""
    if not shortlist:
        return [], 0.0, 0

    candidates = shortlist[:MAX_SHORTLIST_FOR_LLM]

    if report_only:
        return format_deterministic_failures(candidates), 0.0, 0

    # Lazy import so GEMINI_API_KEY/git guards run before services.ai_providers/supabase_client load.
    if budget_checker is None:
        from services.ai_usage import assert_daily_budget

        budget_checker = assert_daily_budget
    if usage_logger is None:
        from services.ai_usage import log_ai_usage

        usage_logger = log_ai_usage
    if cost_estimator is None:
        from services.ai_usage import _est_cost

        cost_estimator = _est_cost
    from services.model_json import parse_model_json

    try:
        budget_checker()
    except HTTPException as exc:
        raise BudgetExhaustedError(f"Daily AI budget switch tripped: {exc.detail}") from exc

    prompt = _build_gemini_prompt(candidates)
    # Re-verify that the outbound prompt itself leaks zero raw identifiers.
    assert_no_identifiers(
        [prompt],
        raw_ids=leak_ctx.get("raw_ids", ()),
        raw_emails=leak_ctx.get("raw_emails", ()),
        raw_names=leak_ctx.get("raw_names", ()),
        raw_colleges=leak_ctx.get("raw_colleges", ()),
    )

    estimated_prompt_tokens = max(1, len(prompt) // 4)
    if estimated_prompt_tokens >= max_tokens_budget:
        raise TokenCapExceededError(
            f"Estimated prompt tokens ({estimated_prompt_tokens}) exceeds run token cap ({max_tokens_budget})"
        )

    client = gemini_client or OpenAI(api_key=gemini_api_key, base_url=GEMINI_BASE_URL)
    model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL

    t0 = time.time()
    resp = None
    last_exc: Optional[Exception] = None
    attempts_allowed = max(1, int(max_retries) + 1)
    for attempt_idx in range(attempts_allowed):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise consulting case interviewer quality reviewer. Output strictly valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=8192,
            )
            last_exc = None
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_quota_or_auth_error(exc):
                break
            if attempt_idx + 1 < attempts_allowed and _is_transient_gemini_error(exc):
                if retry_delay_sec > 0:
                    time.sleep(retry_delay_sec * (attempt_idx + 1))
                continue
            break

    if last_exc is not None:
        latency_ms = int((time.time() - t0) * 1000)
        usage_logger(
            user_id=None,
            endpoint="/bot/daily-report",
            model=model,
            response=None,
            latency_ms=latency_ms,
            success=False,
            meta={"error": type(last_exc).__name__},
        )
        if _is_quota_or_auth_error(last_exc):
            raise GeminiQuotaOrAuthError(f"Gemini quota/auth error: {type(last_exc).__name__}: {last_exc}") from last_exc
        raise last_exc

    latency_ms = int((time.time() - t0) * 1000)
    usage_logger(
        user_id=None,
        endpoint="/bot/daily-report",
        model=model,
        response=resp,
        latency_ms=latency_ms,
        success=True,
        meta={"shortlist_count": len(candidates)},
    )

    usage = getattr(resp, "usage", None)
    pt = getattr(usage, "prompt_tokens", None)
    ct = getattr(usage, "completion_tokens", None)
    tt = getattr(usage, "total_tokens", None)
    tokens_used = int(tt if tt is not None else ((pt or 0) + (ct or 0)))
    if tokens_used > max_tokens_budget:
        raise TokenCapExceededError(f"Tokens used ({tokens_used}) exceeded run budget ({max_tokens_budget})")

    est_cost = float(cost_estimator(model, pt, ct))
    raw_content = (resp.choices[0].message.content or "").strip()
    parsed = parse_model_json(raw_content)

    by_id: Dict[int, Dict[str, Any]] = {}
    for entry in parsed.get("items") or []:
        if isinstance(entry, dict) and entry.get("id") is not None:
            try:
                by_id[int(entry["id"])] = entry
            except (TypeError, ValueError):
                continue

    ranked_rows: List[Tuple[Tuple[int, int, int], Dict[str, Any]]] = []
    for item in candidates:
        llm_info = by_id.get(item["id"]) or {}
        try:
            llm_rank = int(llm_info.get("rank", 999))
        except (TypeError, ValueError):
            llm_rank = 999
        should_have = str(llm_info.get("should_have") or "").strip()
        if not should_have:
            should_have = _DEFAULT_SHOULD_HAVE.get(
                item["signal"], "Respond directly to the candidate's reasoning and guide the next step."
            )
        sort_key = (-int(item["frequency_7d"]), llm_rank, item["id"])
        ranked_rows.append((
            sort_key,
            {
                "rank": llm_rank,
                "signal": item["signal"],
                "attempt_hash": item["attempt_hash"],
                "case_type": item["case_type"],
                "stage": item["stage"],
                "candidate_turn": item["candidate_turn"],
                "interviewer_reply": item["interviewer_reply"],
                "signals": item["signals"],
                "response_function": item["response_function"],
                "should_have": should_have,
                "frequency_7d": item["frequency_7d"],
                "abandoned_within_2_turns": item["abandoned_within_2_turns"],
            },
        ))

    ranked_rows.sort(key=lambda pair: pair[0])
    top_items = [row for _, row in ranked_rows[:MAX_FAILURES_IN_REPORT]]
    for idx, row in enumerate(top_items, start=1):
        row["rank"] = idx

    return top_items, round(est_cost, 6), tokens_used
