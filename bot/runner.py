"""Main orchestration pipeline for the read-only daily reporter."""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from bot.config import (
    CONSECUTIVE_FAILURE_REPORT_ONLY,
    EXIT_ABORT,
    EXIT_ERROR,
    EXIT_OK,
    MAX_TOKENS_PER_RUN,
    MAX_WALL_SECONDS,
    default_repo_root,
    get_gemini_api_key,
    scrub_secrets,
)
from bot.data import SupabaseUnavailableError, WindowDataset, fetch_window_data
from bot.failures import (
    BudgetExhaustedError,
    GeminiQuotaOrAuthError,
    TokenCapExceededError,
    build_redacted_shortlist,
    rank_failures_with_gemini,
)
from bot.git_guard import check_git_clean_outside_bot
from bot.metrics import compute_all_metrics
from bot.redact import RedactionLeakError, assert_no_identifiers
from bot.report import build_report_payload, write_reports
from bot.state import (
    compute_windows,
    is_same_day_noop,
    ist_date_str,
    load_state,
    parse_iso_utc,
    save_state,
)
from bot.suite import run_all_checks


class WallClockExceededError(RuntimeError):
    """Raised when the 20-minute wall-clock cap is exceeded."""


def _append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    safe_msg = scrub_secrets(message)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {safe_msg}\n")


def _check_wall_clock(start_mono: float, max_wall_sec: int) -> None:
    elapsed = time.monotonic() - start_mono
    if elapsed > max_wall_sec:
        raise WallClockExceededError(f"Wall-clock cap exceeded ({elapsed:.1f}s > {max_wall_sec}s)")


def _in_window(ts_raw: Optional[str], start: datetime, end: datetime) -> bool:
    if not ts_raw:
        return False
    try:
        dt = parse_iso_utc(str(ts_raw))
    except Exception:  # noqa: BLE001
        return False
    return start <= dt <= end


def run_daily_report(
    *,
    repo_root: Optional[Path] = None,
    days: Optional[int] = None,
    since: Optional[str] = None,
    dry_run: bool = False,
    skip_suite: bool = False,
    now_utc: Optional[datetime] = None,
    supabase_client: Any = None,
    gemini_client: Any = None,
    suite_runner: Optional[Callable[[Path], Dict[str, Any]]] = None,
    usage_logger: Optional[Callable[..., None]] = None,
    budget_checker: Optional[Callable[[], None]] = None,
    cost_estimator: Optional[Callable[[str, Optional[int], Optional[int]], float]] = None,
    before_write_hook: Optional[Callable[[], None]] = None,
    max_wall_seconds: int = MAX_WALL_SECONDS,
    max_tokens_per_run: int = MAX_TOKENS_PER_RUN,
) -> int:
    """Execute Steps 1-5 under C1-C6 hard limits. Returns process exit code."""
    root = (repo_root or default_repo_root()).resolve()
    bot_dir = root / "bot"
    state_path = bot_dir / "state.json"
    reports_dir = bot_dir / "reports"
    logs_dir = bot_dir / "logs"

    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    report_date = ist_date_str(now)
    log_path = logs_dir / f"{report_date}.log"
    start_mono = time.monotonic()

    def log(msg: str, *, echo: bool = True) -> None:
        safe = scrub_secrets(msg)
        _append_log(log_path, safe)
        if echo:
            print(safe)

    # C2: GEMINI_API_KEY must be set in environment (never fall back to another provider)
    gemini_key = get_gemini_api_key()
    if not gemini_key:
        log("[ABORT] GEMINI_API_KEY is unset; aborting without fallback to any other provider.")
        return EXIT_ABORT

    # C1 & C6: Git working tree must be clean outside bot/ at start
    git_ok, dirty_paths, git_err = check_git_clean_outside_bot(root)
    if not git_ok:
        if git_err:
            log(f"[ABORT] {git_err}")
        else:
            log(f"[ABORT] Git dirty at start outside bot/: {', '.join(dirty_paths)}")
        return EXIT_ABORT

    # Step 1: Resume state + same-day no-op + window calculation
    try:
        state = load_state(state_path)
        if not dry_run and days is None and since is None and is_same_day_noop(state, report_date):
            log(f"[NO-OP] Already ran on {report_date}; skipping duplicate run.")
            return EXIT_OK
        windows = compute_windows(state, now, days=days, since=since)
    except Exception as exc:  # noqa: BLE001
        log(f"[ABORT] Failed during resume/window setup: {type(exc).__name__}: {exc}")
        return EXIT_ABORT

    consecutive_failures = int(state.get("consecutive_failures") or 0)
    report_only = consecutive_failures >= CONSECUTIVE_FAILURE_REPORT_ONLY
    if report_only:
        log(f"[WARN] consecutive_failures={consecutive_failures} >= {CONSECUTIVE_FAILURE_REPORT_ONLY}; entering REPORT-ONLY mode.")

    # Ensure AI usage rows flush synchronously before CLI exit when using real services.ai_usage
    os.environ.setdefault("AI_USAGE_LOG_SYNC", "1")

    # C6: Check budget switch + Supabase connectivity before doing heavy work
    if budget_checker is not None:
        try:
            budget_checker()
        except Exception as exc:  # noqa: BLE001
            log(f"[ABORT] Daily AI budget check failed: {type(exc).__name__}: {exc}")
            return EXIT_ABORT
    else:
        try:
            from fastapi import HTTPException
            from services.ai_usage import assert_daily_budget

            assert_daily_budget()
        except HTTPException as exc:
            log(f"[ABORT] Daily AI budget switch tripped: {exc.detail}")
            return EXIT_ABORT
        except Exception:  # noqa: BLE001
            # Let Supabase probe below surface real DB connectivity errors.
            pass

    if supabase_client is None:
        try:
            from services.supabase_client import get_supabase_client

            supabase_client = get_supabase_client()
        except Exception as exc:  # noqa: BLE001
            log(f"[ABORT] Supabase client initialization failed: {type(exc).__name__}: {exc}")
            return EXIT_ABORT

    try:
        _check_wall_clock(start_mono, max_wall_seconds)
        dataset: WindowDataset = fetch_window_data(
            supabase_client,
            query_from=windows["query_from"],
            window_from=windows["window_from"],
            window_to=windows["window_to"],
        )
    except SupabaseUnavailableError as exc:
        log(f"[ABORT] Supabase unavailable: {exc}")
        return EXIT_ABORT
    except WallClockExceededError as exc:
        log(f"[ABORT] {exc}")
        return EXIT_ABORT

    # Pipeline Steps 2-4: any unexpected exception writes a partial report, bumps consecutive_failures, exits non-zero
    notes: List[str] = []
    if report_only:
        notes.append(
            f"REPORT-ONLY MODE: consecutive_failures={consecutive_failures} (>= 3). "
            "Gemini calls skipped; manual investigation required."
        )

    metrics: Dict[str, Any] = {}
    region_counts: Dict[str, int] = {"IN": 0, "US": 0}
    sessions_in_window = 0
    failures: List[Dict[str, Any]] = []
    cost_usd = 0.0
    suite_block: Dict[str, Any] = {"pytest_passed": 0, "pytest_failed": 0, "offline_eval_drift": []}
    step_failed = False

    leak_ctx: Dict[str, Any] = {
        "raw_ids": [str(u.get("id")) for u in dataset.users_by_id.values() if u.get("id")]
        + [str(a.get("id")) for a in dataset.attempts if a.get("id")],
        "raw_emails": [
            str(u.get(k)) for u in dataset.users_by_id.values() for k in ("email", "college_email") if u.get(k)
        ],
        "raw_names": [
            str(u.get(k)) for u in dataset.users_by_id.values() for k in ("full_name", "name") if u.get(k)
        ],
        "raw_colleges": [str(u.get("college_other")) for u in dataset.users_by_id.values() if u.get("college_other")],
    }

    try:
        from bot.turns import AttemptEval, analyze_attempt

        _check_wall_clock(start_mono, max_wall_seconds)
        # Replay attempts once deterministically
        attempt_evals: List[AttemptEval] = [
            analyze_attempt(att, dataset, reference_now=windows["window_to"]) for att in dataset.attempts
        ]
        win_evals = [
            ae
            for ae in attempt_evals
            if _in_window(ae.attempt.get("created_at"), windows["window_from"], windows["window_to"])
        ]
        sessions_in_window = len(win_evals)

        # Step 2: Metrics (no model calls)
        metrics, region_counts = compute_all_metrics(dataset, attempt_evals, windows)

        # Step 3: Deterministic failure shortlist + redaction + Gemini ranking
        _check_wall_clock(start_mono, max_wall_seconds)
        shortlist, leak_ctx = build_redacted_shortlist(dataset, attempt_evals, state, windows)
        if shortlist:
            failures, cost_usd, _tokens = rank_failures_with_gemini(
                shortlist,
                leak_ctx,
                gemini_api_key=gemini_key,
                report_only=report_only,
                max_tokens_budget=max_tokens_per_run,
                gemini_client=gemini_client,
                usage_logger=usage_logger,
                budget_checker=budget_checker,
                cost_estimator=cost_estimator,
            )

        # Step 4: Test suite + offline eval diff
        _check_wall_clock(start_mono, max_wall_seconds)
        if not skip_suite:
            runner_fn = suite_runner or run_all_checks
            suite_res = runner_fn(root)
            suite_block = suite_res.get("suite") or suite_block
            notes.extend(suite_res.get("notes") or [])
        else:
            notes.append("Suite check skipped (--skip-suite).")

    except (
        BudgetExhaustedError,
        GeminiQuotaOrAuthError,
        TokenCapExceededError,
        WallClockExceededError,
        RedactionLeakError,
    ) as abort_exc:
        log(f"[ABORT] {type(abort_exc).__name__}: {abort_exc}")
        return EXIT_ABORT
    except Exception as exc:  # noqa: BLE001
        step_failed = True
        err_note = f"Run error ({type(exc).__name__}): {scrub_secrets(str(exc))}"
        notes.append(err_note)
        log(f"[ERROR] {err_note}")

    if before_write_hook is not None:
        before_write_hook()

    # C1: Re-check git status --porcelain right before writing the report
    git_ok_end, dirty_end, git_err_end = check_git_clean_outside_bot(root)
    if not git_ok_end:
        if git_err_end:
            log(f"[ABORT] Pre-write git check failed: {git_err_end}")
        else:
            log(
                "[ABORT] Refusing to write report — path(s) outside bot/ changed mid-run: "
                + ", ".join(dirty_end)
            )
        return EXIT_ABORT

    # Assemble & verify report contains zero identifiers
    payload = build_report_payload(
        report_date=report_date,
        window_from_iso=windows["window_from"].isoformat(),
        window_to_iso=windows["window_to"].isoformat(),
        sessions_total=sessions_in_window,
        regions=region_counts,
        cost_usd=cost_usd,
        suite=suite_block,
        metrics=metrics,
        failures=failures,
        notes=notes,
        report_only=report_only,
        consecutive_failures=consecutive_failures,
    )

    try:
        texts_to_verify = list(payload["notes"])
        for item in payload["failures"]:
            texts_to_verify.extend([
                str(item.get("candidate_turn") or ""),
                str(item.get("interviewer_reply") or ""),
                str(item.get("should_have") or ""),
            ])
        assert_no_identifiers(
            texts_to_verify,
            raw_ids=leak_ctx.get("raw_ids", ()),
            raw_emails=leak_ctx.get("raw_emails", ()),
            raw_names=leak_ctx.get("raw_names", ()),
            raw_colleges=leak_ctx.get("raw_colleges", ()),
        )
    except RedactionLeakError as leak_exc:
        log(f"[ABORT] {leak_exc}")
        return EXIT_ABORT

    if dry_run:
        log(f"[DRY-RUN] Report prepared for {report_date} (cost_usd={cost_usd:.6f}, failures={len(failures)}). Not written.")
        return EXIT_ERROR if step_failed else EXIT_OK

    # Step 5: Write bot/reports/<date>.md & <date>.json and update bot/state.json
    md_path, json_path = write_reports(reports_dir, payload)
    if step_failed:
        state["consecutive_failures"] = consecutive_failures + 1
        state["last_run_at"] = now.isoformat()
        state["last_run_date"] = report_date
        state["last_run_status"] = "error"
    elif report_only:
        state["consecutive_failures"] = consecutive_failures
        state["last_run_at"] = now.isoformat()
        state["last_run_date"] = report_date
        state["last_run_status"] = "report_only"
    else:
        state["consecutive_failures"] = 0
        state["last_run_at"] = now.isoformat()
        state["last_run_date"] = report_date
        state["last_run_status"] = "success"
        state["cursor"] = windows["window_to"].isoformat()

    save_state(state_path, state)
    log(f"[DONE] Wrote {md_path.relative_to(root)} and {json_path.relative_to(root)} (cost_usd={cost_usd:.6f})")
    return EXIT_ERROR if step_failed else EXIT_OK
