"""Unit and integration tests for metrics, all 6 failure triggers, redaction, caps, and suite diff."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bot.config import EXIT_ABORT, EXIT_ERROR, EXIT_OK, default_repo_root, scrub_secrets
from bot.data import SupabaseUnavailableError
from bot.failures import rank_failures_with_gemini
from bot.redact import redact_text
from bot.runner import run_daily_report
from bot.state import compute_windows, default_state, load_state, save_state
from bot.suite import parse_pytest_counts, run_offline_eval_and_diff
from bot.tests.fakes import FakeGeminiClient, FakeSupabase, init_temp_git_repo, stub_suite_ok

NOW = datetime(2026, 11, 11, 2, 0, 0, tzinfo=timezone(timedelta(hours=5, minutes=30))).astimezone(timezone.utc)
DATE_STR = "2026-11-11"


def _ts(hours_ago: float) -> str:
    return (NOW - timedelta(hours=hours_ago)).isoformat()


def test_window_resume_first_run_and_3d_cap() -> None:
    st = default_state()
    w1 = compute_windows(st, NOW)
    assert (w1["window_to"] - w1["window_from"]) == timedelta(days=2)

    st["cursor"] = (NOW - timedelta(days=10)).isoformat()
    w2 = compute_windows(st, NOW)
    assert (w2["window_to"] - w2["window_from"]) == timedelta(days=3)

    st["cursor"] = (NOW - timedelta(days=1)).isoformat()
    w3 = compute_windows(st, NOW)
    assert (w3["window_to"] - w3["window_from"]) == timedelta(days=1)

    w4 = compute_windows(st, NOW, days=5)
    assert (w4["window_to"] - w4["window_from"]) == timedelta(days=5)


def test_redaction_preserves_case_numbers_and_generic_college_words() -> None:
    sample = (
        "In a college town with 90000 orders and 1,40,00,000 population, "
        "I am assuming 20% share. Also I'm Priya Nair from XLRI Jamshedpur, "
        "contact priya@example.org or 9876543210."
    )
    red = redact_text(sample)
    assert "90000 orders" in red
    assert "1,40,00,000" in red
    assert "college town" in red
    assert "Priya Nair" not in red
    assert "XLRI Jamshedpur" not in red
    assert "priya@example.org" not in red
    assert "9876543210" not in red
    assert "[NAME]" in red
    assert "[COLLEGE]" in red
    assert "[EMAIL]" in red
    assert "[PHONE]" in red


def test_all_six_failure_triggers_and_frequency_sorting(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    repo = init_temp_git_repo(tmp_path)

    u_in = "10000000-0000-0000-0000-000000000001"
    u_us = "10000000-0000-0000-0000-000000000002"
    c_in = "20000000-0000-0000-0000-000000000001"
    c_us = "20000000-0000-0000-0000-000000000002"

    # Prior 7d (4..6 days ago) has 3 repeated_question attempts to make frequency_7d=4 for repeated_question
    attempts = []
    messages = []
    for k in range(3):
        aid = f"30000000-0000-0000-0000-00000000000{k}"
        attempts.append({
            "id": aid, "user_id": u_in, "case_id": c_in, "status": "submitted",
            "teaching_policy": "coached", "session_state": {}, "created_at": _ts(96 + k),
        })
        messages.extend([
            {"id": f"pm_{k}_1", "attempt_id": aid, "role": "assistant", "kind": "text", "content": "What is your first step?", "created_at": _ts(95.9 + k)},
            {"id": f"pm_{k}_2", "attempt_id": aid, "role": "user", "kind": "text", "content": "Not sure.", "created_at": _ts(95.8 + k)},
            {"id": f"pm_{k}_3", "attempt_id": aid, "role": "assistant", "kind": "text", "content": "What is your first step?", "created_at": _ts(95.7 + k)},
        ])

    # Current window attempt exercising repeated_question, help_not_actionable, stacked_questions,
    # question_on_hand_back, substantive_no_stage_advance, and abandoned_within_2_turns
    aid_cur = "40000000-0000-0000-0000-000000000001"
    attempts.append({
        "id": aid_cur, "user_id": u_us, "case_id": c_us, "status": "abandoned",
        "teaching_policy": "coached",
        "session_state": {"structure": [{"id": "rev", "label": "Revenue", "value": "", "parent": ""}]},
        "created_at": _ts(6),
    })
    messages.extend([
        {"id": "cm1", "attempt_id": aid_cur, "role": "assistant", "kind": "text", "content": "How would you size this?", "created_at": _ts(5.9)},
        {"id": "cm2", "attempt_id": aid_cur, "role": "user", "kind": "text", "content": "I am stuck, please help me.", "created_at": _ts(5.8)},
        {"id": "cm3", "attempt_id": aid_cur, "role": "assistant", "kind": "text", "content": "Okay.", "created_at": _ts(5.7)},
        {"id": "cm4", "attempt_id": aid_cur, "role": "user", "kind": "text", "content": "I'll split households first", "created_at": _ts(5.6)},
        {"id": "cm5", "attempt_id": aid_cur, "role": "assistant", "kind": "text", "content": "Wait, why households?", "created_at": _ts(5.5)},
        {"id": "cm6", "attempt_id": aid_cur, "role": "user", "kind": "text", "content": "Should we do urban vs rural?", "created_at": _ts(5.4)},
        {"id": "cm7", "attempt_id": aid_cur, "role": "assistant", "kind": "text", "content": "What is urban share? And what is rural share?", "created_at": _ts(5.3)},
        {"id": "cm8", "attempt_id": aid_cur, "role": "user", "kind": "text", "content": "Urban is 30% and rural is 70%, let me think.", "created_at": _ts(5.2)},
        {"id": "cm9", "attempt_id": aid_cur, "role": "assistant", "kind": "text", "content": "How would you size this?", "created_at": _ts(5.1)},
        {
            "id": "cm10",
            "attempt_id": aid_cur,
            "role": "user",
            "kind": "text",
            "content": "First I will segment by urban and rural households, then multiply by penetration and annual replacement rate.",
            "created_at": _ts(5.0),
        },
        {"id": "cm11", "attempt_id": aid_cur, "role": "assistant", "kind": "text", "content": "Sure, continue.", "created_at": _ts(4.9)},
    ])

    sb = FakeSupabase({
        "users": [
            {"id": u_in, "name": "IN User", "email": "in@example.com", "market": "IN"},
            {"id": u_us, "name": "US User", "email": "us@example.com", "market": "US"},
        ],
        "cases": [
            {"id": c_in, "title": "IN Case", "type": "guesstimate", "market": "IN"},
            {"id": c_us, "title": "US Case", "type": "guesstimate", "market": "US"},
        ],
        "attempts": attempts,
        "attempt_messages": messages,
    })

    gemini = FakeGeminiClient(reply_json='{"items": [{"id": 1, "rank": 2, "should_have": "Avoid repeating."}]}')
    rc = run_daily_report(
        repo_root=repo,
        now_utc=NOW,
        supabase_client=sb,
        gemini_client=gemini,
        suite_runner=stub_suite_ok,
        usage_logger=lambda **kw: None,
        budget_checker=lambda: None,
        cost_estimator=lambda _m, pt, ct: 0.02,
    )
    assert rc == EXIT_OK
    rep = json.loads((repo / "bot" / "reports" / f"{DATE_STR}.json").read_text(encoding="utf-8"))
    sigs = [f["signal"] for f in rep["failures"]]
    assert "help_not_actionable" in sigs
    assert "question_on_hand_back" in sigs
    assert "stacked_questions" in sigs
    assert "repeated_question" in sigs
    assert "substantive_no_stage_advance" in sigs
    # Sorted by frequency_7d descending
    freqs = [f["frequency_7d"] for f in rep["failures"]]
    assert freqs == sorted(freqs, reverse=True)
    assert rep["window"]["regions"] == {"IN": 0, "US": 1}


def test_c6_aborts_and_step_exception_recovery(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "sk-secret-key-never-log")
    repo = init_temp_git_repo(tmp_path)

    # 1. Supabase down -> abort before writing report/state
    rc_db = run_daily_report(
        repo_root=repo,
        now_utc=NOW,
        supabase_client=FakeSupabase(fail_with=ConnectionError("db down")),
        suite_runner=stub_suite_ok,
    )
    assert rc_db == EXIT_ABORT
    assert not (repo / "bot" / "reports" / f"{DATE_STR}.json").exists()

    # 2. Budget switch tripped -> abort before writing report/state
    from fastapi import HTTPException
    def _trip_budget() -> None:
        raise HTTPException(status_code=503, detail="budget cap reached")

    rc_budget = run_daily_report(
        repo_root=repo,
        now_utc=NOW,
        supabase_client=FakeSupabase(),
        budget_checker=_trip_budget,
        suite_runner=stub_suite_ok,
    )
    assert rc_budget == EXIT_ABORT

    # 3. Step throws -> partial report written, consecutive_failures += 1, exit non-zero
    def _boom_suite(_root: Path):
        raise RuntimeError("boom with sk-secret-key-never-log inside")

    rc_err = run_daily_report(
        repo_root=repo,
        now_utc=NOW,
        supabase_client=FakeSupabase(),
        budget_checker=lambda: None,
        suite_runner=_boom_suite,
    )
    assert rc_err == EXIT_ERROR
    st = load_state(repo / "bot" / "state.json")
    assert st["consecutive_failures"] == 1
    rep_text = (repo / "bot" / "reports" / f"{DATE_STR}.json").read_text(encoding="utf-8")
    assert "sk-secret-key-never-log" not in rep_text
    assert "[SECRET_REDACTED]" in rep_text

    # 4. Gemini quota gone -> abort before writing report/state
    st2 = default_state()
    save_state(repo / "bot" / "state.json", st2)
    (repo / "bot" / "reports" / f"{DATE_STR}.json").unlink()
    (repo / "bot" / "reports" / f"{DATE_STR}.md").unlink()

    rc_quota = run_daily_report(
        repo_root=repo,
        now_utc=NOW,
        supabase_client=FakeSupabase({
            "attempts": [{"id": "a1", "user_id": "u1", "case_id": "c1", "status": "active", "created_at": _ts(2)}],
            "attempt_messages": [
                {"id": "m1", "attempt_id": "a1", "role": "user", "kind": "text", "content": "hi", "created_at": _ts(1.9)},
                {"id": "m2", "attempt_id": "a1", "role": "assistant", "kind": "text", "content": "Great job!", "created_at": _ts(1.8)},
            ],
        }),
        gemini_client=FakeGeminiClient(raise_exc=RuntimeError("Error code: 429 - quota exceeded")),
        usage_logger=lambda **kw: None,
        budget_checker=lambda: None,
        suite_runner=stub_suite_ok,
    )
    assert rc_quota == EXIT_ABORT
    assert not (repo / "bot" / "reports" / f"{DATE_STR}.json").exists()


def test_token_cap_and_secret_scrubbing() -> None:
    assert "SECRET_REDACTED" in scrub_secrets("leak AIzaSyDummySecretValue1234567890 here")
    try:
        rank_failures_with_gemini(
            [{"id": 1, "signal": "praise_only", "attempt_hash": "abc", "case_type": "case", "stage": "Clarify",
              "candidate_turn": "x " * 400, "interviewer_reply": "Great job!", "signals": {},
              "response_function": "SHORT_ACK", "frequency_7d": 1, "abandoned_within_2_turns": False}],
            {},
            gemini_api_key="test-key",
            max_tokens_budget=10,
            budget_checker=lambda: None,
            usage_logger=lambda **kw: None,
        )
        assert False, "Expected TokenCapExceededError"
    except Exception as exc:
        assert type(exc).__name__ == "TokenCapExceededError"


def test_suite_helpers_and_real_offline_eval_zero_drift() -> None:
    assert parse_pytest_counts("517 passed, 5 skipped in 29.45s") == (517, 0)
    assert parse_pytest_counts("2 failed, 515 passed, 1 error in 20s") == (515, 3)

    real_root = default_repo_root()
    drift = run_offline_eval_and_diff(real_root)
    assert drift == []
    assert (real_root / "run_daily.bat").is_file()
    assert (real_root / ".github" / "workflows" / "daily-report.yml").is_file()


def test_all_nine_metrics_regions_and_prev_7d_comparison(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    repo = init_temp_git_repo(tmp_path)

    u_in1 = "11000000-0000-0000-0000-000000000001"
    u_in2 = "11000000-0000-0000-0000-000000000002"
    u_us1 = "11000000-0000-0000-0000-000000000003"
    c_in = "22000000-0000-0000-0000-000000000001"
    c_us = "22000000-0000-0000-0000-000000000002"

    # Window is [NOW - 2d, NOW]; prev_7d is [NOW - 9d, NOW - 2d)
    sb = FakeSupabase({
        "users": [
            {"id": u_in1, "name": "A", "email": "a@example.com", "market": "IN"},
            {"id": u_in2, "name": "B", "email": "b@example.com", "market": "IN"},
            {"id": u_us1, "name": "C", "email": "c@example.com", "market": "US"},
        ],
        "cases": [
            {"id": c_in, "title": "IN Guesstimate", "type": "guesstimate", "market": "IN"},
            {"id": c_us, "title": "US Case", "type": "profitability", "market": "US"},
        ],
        "attempts": [
            # Current window IN (submitted, >=3 nodes, Validate)
            {
                "id": "att_in_1", "user_id": u_in1, "case_id": c_in, "status": "submitted",
                "session_state": {"structure": [
                    {"id": "n1", "label": "Pop", "value": "10M", "parent": ""},
                    {"id": "n2", "label": "Pct", "value": "50%", "parent": "n1"},
                    {"id": "n3", "label": "Freq", "value": "2", "parent": "n2"},
                ]},
                "created_at": _ts(10),
            },
            # Current window IN (returning user u_in1, unfinished at Clarify)
            {
                "id": "att_in_2", "user_id": u_in1, "case_id": c_in, "status": "active",
                "session_state": {},
                "created_at": _ts(8),
            },
            # Current window US (submitted, 2 nodes, not Validate)
            {
                "id": "att_us_1", "user_id": u_us1, "case_id": c_us, "status": "submitted",
                "session_state": {"structure": [
                    {"id": "r", "label": "Revenue", "value": "", "parent": ""},
                    {"id": "c", "label": "Cost", "value": "", "parent": ""},
                ]},
                "created_at": _ts(6),
            },
            # Prev 7d IN (4 days ago)
            {
                "id": "att_prev_1", "user_id": u_in2, "case_id": c_in, "status": "submitted",
                "session_state": {},
                "created_at": _ts(96),
            },
        ],
        "attempt_messages": [
            {"id": "x1", "attempt_id": "att_in_1", "role": "user", "kind": "text", "content": "Hello", "created_at": _ts(9.9)},
            {"id": "x2", "attempt_id": "att_in_1", "role": "assistant", "kind": "text", "content": "Go ahead.", "created_at": _ts(9.8)},
            {"id": "x3", "attempt_id": "att_in_1", "role": "user", "kind": "text", "content": "10M * 50% * 2 = 10M", "created_at": _ts(9.7)},
            {"id": "x4", "attempt_id": "att_in_1", "role": "assistant", "kind": "text", "content": "Check the sanity.", "created_at": _ts(9.6)},
            {"id": "y1", "attempt_id": "att_in_2", "role": "user", "kind": "text", "content": "How to start?", "created_at": _ts(7.9)},
            {"id": "y2", "attempt_id": "att_in_2", "role": "assistant", "kind": "text", "content": "Anchor on the city population first.", "created_at": _ts(7.8)},
            {"id": "z1", "attempt_id": "att_us_1", "role": "user", "kind": "text", "content": "Profit equals revenue minus cost.", "created_at": _ts(5.9)},
            {"id": "z2", "attempt_id": "att_us_1", "role": "assistant", "kind": "text", "content": "Which side moved?", "created_at": _ts(5.8)},
        ],
        "case_attempts": [
            {"id": "ca1", "user_id": u_in1, "case_id": c_in, "submission_id": "s1", "created_at": _ts(9)},
            {"id": "ca2", "user_id": u_us1, "case_id": c_us, "submission_id": "s2", "created_at": _ts(5)},
        ],
        "submissions": [
            {"id": "s1", "user_id": u_in1, "case_id": c_in, "score": 80, "feedback_json": {"rubric": "guesstimate", "breakdown": {"structure": 80}}, "created_at": _ts(9)},
            {"id": "s2", "user_id": u_us1, "case_id": c_us, "score": 60, "feedback_json": {"rubric": "case", "breakdown": {"structure": 15}}, "created_at": _ts(5)},
        ],
    })

    rc = run_daily_report(
        repo_root=repo,
        now_utc=NOW,
        supabase_client=sb,
        gemini_client=FakeGeminiClient(),
        suite_runner=stub_suite_ok,
        budget_checker=lambda: None,
    )
    assert rc == EXIT_OK
    rep = json.loads((repo / "bot" / "reports" / f"{DATE_STR}.json").read_text(encoding="utf-8"))
    m = rep["metrics"]
    assert rep["window"]["regions"] == {"IN": 2, "US": 1}
    assert m["completion_rate"]["value"] == round(2 / 3, 4)
    assert m["completion_rate"]["regions"]["IN"]["value"] == 0.5
    assert m["completion_rate"]["regions"]["US"]["value"] == 1.0
    assert m["median_depth"]["value"] == 1.0
    assert m["structure_reached"]["value"] == round(1 / 3, 4)
    assert m["self_check_rate"]["value"] == round(1 / 3, 4)
    assert m["abandon_point"]["stage"] == "Clarify"
    assert m["return_rate"]["regions"]["IN"]["value"] == 0.5  # u_in1 (2 attempts) vs u_in2 (1 attempt in 7d)
    assert m["mean_score"]["value"] == 70.0
    assert m["mean_structure_subscore"]["value"] == 70.0  # (80 + 15*4)/2 = 70.0
    assert m["praise_only_rate"]["value"] == 0.0
    assert m["repeat_rate"]["value"] == 0.0


def test_wall_clock_cap_and_us_pii_redaction(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    repo = init_temp_git_repo(tmp_path)
    rc = run_daily_report(
        repo_root=repo,
        now_utc=NOW,
        supabase_client=FakeSupabase(),
        budget_checker=lambda: None,
        suite_runner=stub_suite_ok,
        max_wall_seconds=-1,
    )
    assert rc == EXIT_ABORT

    us_txt = "Hi, I'm John Miller at Harvard Business School, call (415) 555-0199 or john@hbs.edu."
    red = redact_text(us_txt)
    assert "John Miller" not in red
    assert "Harvard Business School" not in red
    assert "555-0199" not in red
    assert "john@hbs.edu" not in red
    assert "[NAME]" in red
    assert "[COLLEGE]" in red
    assert "[PHONE]" in red
    assert "[EMAIL]" in red


def test_real_log_ai_usage_shape_and_cli_dry_run(tmp_path: Path, monkeypatch) -> None:
    from services import ai_usage

    inserted_rows = []
    class _UsageTable:
        def insert(self, row):
            inserted_rows.append(row)
            return self
        def execute(self):
            return None

    class _MockSB:
        def table(self, name):
            assert name == "ai_usage_log"
            return _UsageTable()

    monkeypatch.setattr(ai_usage, "get_supabase_client", lambda: _MockSB())
    monkeypatch.setattr(ai_usage, "_LOG_SYNC", True)
    gemini = FakeGeminiClient()
    resp = gemini._create(model="gemini-3.8-flash", messages=[])
    ai_usage.log_ai_usage(user_id=None, endpoint="/bot/daily-report", model="gemini-3.8-flash", response=resp, latency_ms=10)
    assert len(inserted_rows) == 1
    assert inserted_rows[0]["endpoint"] == "/bot/daily-report"
    assert inserted_rows[0]["total_tokens"] == 160

    # Verify --dry-run does not write report or state.json
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    repo = init_temp_git_repo(tmp_path)
    rc = run_daily_report(
        repo_root=repo,
        now_utc=NOW,
        dry_run=True,
        supabase_client=FakeSupabase(),
        suite_runner=stub_suite_ok,
        budget_checker=lambda: None,
    )
    assert rc == EXIT_OK
    assert not (repo / "bot" / "reports" / f"{DATE_STR}.json").exists()
    assert load_state(repo / "bot" / "state.json")["last_run_at"] is None

