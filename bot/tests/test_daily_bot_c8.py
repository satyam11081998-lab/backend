"""The seven required C8 verification scenarios in bot/tests/."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bot.config import EXIT_ABORT, EXIT_OK, default_repo_root
from bot.runner import run_daily_report
from bot.state import default_state, load_state, save_state
from bot.tests.fakes import FakeGeminiClient, FakeSupabase, init_temp_git_repo, stub_suite_ok

NOW = datetime(2026, 11, 11, 2, 0, 0, tzinfo=timezone(timedelta(hours=5, minutes=30))).astimezone(timezone.utc)
DATE_STR = "2026-11-11"


def _ts(hours_ago: float) -> str:
    return (NOW - timedelta(hours=hours_ago)).isoformat()


def test_1_praise_only_reply_flagged_in_step_3(tmp_path: Path, monkeypatch) -> None:
    """C8.1: A transcript with a praise-only reply -> flagged in step 3."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    repo = init_temp_git_repo(tmp_path)

    attempt_id = "aaaaaaaa-1111-2222-3333-444444444444"
    user_id = "bbbbbbbb-1111-2222-3333-444444444444"
    case_id = "cccccccc-1111-2222-3333-444444444444"

    sb = FakeSupabase({
        "users": [{"id": user_id, "name": "Candidate One", "email": "c1@example.com", "market": "IN"}],
        "cases": [{"id": case_id, "title": "Mumbai Coffee", "type": "guesstimate", "teaching_policy": "exam", "market": "IN"}],
        "attempts": [{
            "id": attempt_id,
            "user_id": user_id,
            "case_id": case_id,
            "status": "active",
            "teaching_policy": "exam",
            "session_state": {"structure": [{"id": "pop", "label": "Population", "value": "2 cr", "parent": ""}]},
            "created_at": _ts(5),
        }],
        "attempt_messages": [
            {
                "id": "m1",
                "attempt_id": attempt_id,
                "role": "user",
                "kind": "text",
                "content": "I will start with Mumbai population of 2 crore and split by income.",
                "created_at": _ts(4.9),
            },
            {
                "id": "m2",
                "attempt_id": attempt_id,
                "role": "assistant",
                "kind": "text",
                "content": "Great job! Well done, that's really impressive!",
                "created_at": _ts(4.8),
            },
        ],
    })

    gemini = FakeGeminiClient(
        reply_json=json.dumps({
            "items": [{
                "id": 1,
                "rank": 1,
                "should_have": "Acknowledge the population anchor neutrally and probe the income split.",
            }]
        })
    )
    usage_calls = []

    rc = run_daily_report(
        repo_root=repo,
        now_utc=NOW,
        supabase_client=sb,
        gemini_client=gemini,
        suite_runner=stub_suite_ok,
        usage_logger=lambda **kw: usage_calls.append(kw),
        budget_checker=lambda: None,
        cost_estimator=lambda _m, pt, ct: 0.01,
    )

    assert rc == EXIT_OK
    assert len(gemini.calls) == 1
    assert len(usage_calls) == 1

    report_json = json.loads((repo / "bot" / "reports" / f"{DATE_STR}.json").read_text(encoding="utf-8"))
    assert len(report_json["failures"]) == 1
    failure = report_json["failures"][0]
    assert failure["signal"] == "praise_only"
    assert "Great job!" in failure["interviewer_reply"]
    assert failure["should_have"] == "Acknowledge the population anchor neutrally and probe the income split."


def test_2_clean_transcript_zero_failures_succeeds(tmp_path: Path, monkeypatch) -> None:
    """C8.2: A clean transcript -> zero failures, run still succeeds."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    repo = init_temp_git_repo(tmp_path)

    attempt_id = "11111111-aaaa-bbbb-cccc-000000000001"
    user_id = "22222222-aaaa-bbbb-cccc-000000000002"
    case_id = "33333333-aaaa-bbbb-cccc-000000000003"
    sub_id = "44444444-aaaa-bbbb-cccc-000000000004"

    sb = FakeSupabase({
        "users": [{"id": user_id, "name": "Clean User", "email": "clean@example.com", "market": "IN"}],
        "cases": [{"id": case_id, "title": "Chai Cups", "type": "guesstimate", "teaching_policy": "coached", "market": "IN"}],
        "attempts": [{
            "id": attempt_id,
            "user_id": user_id,
            "case_id": case_id,
            "status": "submitted",
            "submission_id": sub_id,
            "teaching_policy": "coached",
            "session_state": {
                "structure_archetype": "consumption",
                "structure": [
                    {"id": "pop", "label": "Population", "value": "1.2 cr", "parent": ""},
                    {"id": "drinkers", "label": "Tea drinkers", "value": "60 lakh", "parent": "pop"},
                    {"id": "cups", "label": "Cups per day", "value": "2", "parent": "drinkers"},
                ],
            },
            "created_at": _ts(6),
            "submitted_at": _ts(5.5),
        }],
        "attempt_messages": [
            {
                "id": "cm1",
                "attempt_id": attempt_id,
                "role": "user",
                "kind": "text",
                "content": "Are we sizing daily cups sold in Mumbai or home consumption too?",
                "created_at": _ts(5.9),
            },
            {
                "id": "cm2",
                "attempt_id": attempt_id,
                "role": "assistant",
                "kind": "text",
                "content": "Focus on daily cups sold by street stalls and cafes across Mumbai.",
                "created_at": _ts(5.85),
            },
            {
                "id": "cm3",
                "attempt_id": attempt_id,
                "role": "user",
                "kind": "text",
                "content": "I will break it down into Mumbai population of 1.2 crore, 50% adults buying outside, at 2 cups per day.",
                "created_at": _ts(5.8),
            },
            {
                "id": "cm4",
                "attempt_id": attempt_id,
                "role": "assistant",
                "kind": "text",
                "content": "Take that into your calculation and check what total daily cups that gives.",
                "created_at": _ts(5.75),
            },
            {
                "id": "cm5",
                "attempt_id": attempt_id,
                "role": "user",
                "kind": "text",
                "content": "60 lakh times 2 gives 1.2 crore cups per day; that's roughly 1 cup per resident.",
                "created_at": _ts(5.7),
            },
            {
                "id": "cm6",
                "attempt_id": attempt_id,
                "role": "assistant",
                "kind": "text",
                "content": "That per-resident check is clear. Tap Submit when you are ready for your debrief.",
                "created_at": _ts(5.6),
            },
        ],
        "case_attempts": [{
            "id": "ca1",
            "user_id": user_id,
            "case_id": case_id,
            "submission_id": sub_id,
            "attempt_number": 1,
            "is_first_attempt": True,
            "created_at": _ts(5.5),
        }],
        "submissions": [{
            "id": sub_id,
            "user_id": user_id,
            "case_id": case_id,
            "attempt_id": attempt_id,
            "score": 84,
            "feedback_json": {"rubric": "guesstimate", "breakdown": {"structure": 85}},
            "created_at": _ts(5.5),
        }],
    })

    gemini = FakeGeminiClient()
    rc = run_daily_report(
        repo_root=repo,
        now_utc=NOW,
        supabase_client=sb,
        gemini_client=gemini,
        suite_runner=stub_suite_ok,
        budget_checker=lambda: None,
    )

    assert rc == EXIT_OK
    assert len(gemini.calls) == 0  # Zero failures -> zero unnecessary model calls
    report_json = json.loads((repo / "bot" / "reports" / f"{DATE_STR}.json").read_text(encoding="utf-8"))
    assert report_json["failures"] == []
    assert report_json["metrics"]["completion_rate"]["value"] == 1.0
    assert report_json["metrics"]["self_check_rate"]["value"] == 1.0
    assert (repo / "bot" / "reports" / f"{DATE_STR}.md").is_file()


def test_3_touching_services_ai_scorer_mid_run_aborts_nothing_committed(tmp_path: Path, monkeypatch) -> None:
    """C8.3: Touching services/ai_scorer.py mid-run -> run aborts, nothing committed."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    repo = init_temp_git_repo(tmp_path)

    commits_before = subprocess.check_output(
        ["git", "rev-list", "--count", "HEAD"], cwd=str(repo), text=True
    ).strip()

    sb = FakeSupabase()
    gemini = FakeGeminiClient()

    def _touch_scorer_mid_run() -> None:
        scorer_file = repo / "services" / "ai_scorer.py"
        scorer_file.write_text(scorer_file.read_text(encoding="utf-8") + "# touched mid-run\n", encoding="utf-8")

    rc = run_daily_report(
        repo_root=repo,
        now_utc=NOW,
        supabase_client=sb,
        gemini_client=gemini,
        suite_runner=stub_suite_ok,
        budget_checker=lambda: None,
        before_write_hook=_touch_scorer_mid_run,
    )

    assert rc == EXIT_ABORT
    assert not (repo / "bot" / "reports" / f"{DATE_STR}.json").exists()
    assert not (repo / "bot" / "reports" / f"{DATE_STR}.md").exists()

    log_text = (repo / "bot" / "logs" / f"{DATE_STR}.log").read_text(encoding="utf-8")
    assert "services/ai_scorer.py" in log_text

    commits_after = subprocess.check_output(
        ["git", "rev-list", "--count", "HEAD"], cwd=str(repo), text=True
    ).strip()
    assert commits_before == commits_after


def test_4_pii_email_phone_college_redacted_never_in_report_or_llm(tmp_path: Path, monkeypatch) -> None:
    """C8.4: Transcript containing email, phone number, college name -> none appear in report or LLM call."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    repo = init_temp_git_repo(tmp_path)

    attempt_id = "dddddddd-1111-2222-3333-444444444444"
    user_id = "eeeeeeee-1111-2222-3333-444444444444"
    case_id = "ffffffff-1111-2222-3333-444444444444"

    sb = FakeSupabase({
        "users": [{
            "id": user_id,
            "name": "Rahul Sharma",
            "full_name": "Rahul Sharma",
            "email": "rahul.sharma@iima.ac.in",
            "market": "IN",
        }],
        "cases": [{"id": case_id, "title": "Metro Sizing", "type": "guesstimate", "teaching_policy": "exam", "market": "IN"}],
        "colleges": [{"id": "col1", "name": "Indian Institute of Management Ahmedabad", "short_name": "IIM Ahmedabad"}],
        "attempts": [{
            "id": attempt_id,
            "user_id": user_id,
            "case_id": case_id,
            "status": "active",
            "teaching_policy": "exam",
            "session_state": {},
            "created_at": _ts(3),
        }],
        "attempt_messages": [
            {
                "id": "pm1",
                "attempt_id": attempt_id,
                "role": "user",
                "kind": "text",
                "content": (
                    "Hi, I'm Rahul Sharma from IIM Ahmedabad. "
                    "My email is rahul.sharma@iima.ac.in, phone +91 98765 43210, "
                    f"user_id {user_id}. Let's estimate 20 lakh passengers."
                ),
                "created_at": _ts(2.9),
            },
            {
                "id": "pm2",
                "attempt_id": attempt_id,
                "role": "assistant",
                "kind": "text",
                "content": "Great job, Rahul Sharma! That's really impressive!",
                "created_at": _ts(2.8),
            },
        ],
    })

    gemini = FakeGeminiClient(
        reply_json=json.dumps({
            "items": [{
                "id": 1,
                "rank": 1,
                "should_have": "Skip praise and ask how the passenger estimate breaks down.",
            }]
        })
    )

    rc = run_daily_report(
        repo_root=repo,
        now_utc=NOW,
        supabase_client=sb,
        gemini_client=gemini,
        suite_runner=stub_suite_ok,
        usage_logger=lambda **kw: None,
        budget_checker=lambda: None,
        cost_estimator=lambda _m, pt, ct: 0.0,
    )

    assert rc == EXIT_OK
    assert len(gemini.calls) == 1
    llm_prompt_blob = json.dumps(gemini.calls[0]["messages"])
    json_report = (repo / "bot" / "reports" / f"{DATE_STR}.json").read_text(encoding="utf-8")
    md_report = (repo / "bot" / "reports" / f"{DATE_STR}.md").read_text(encoding="utf-8")

    forbidden = [
        "rahul.sharma@iima.ac.in",
        "98765",
        "43210",
        "IIM Ahmedabad",
        "Rahul Sharma",
        user_id,
        attempt_id,
    ]
    for artifact in (llm_prompt_blob, json_report, md_report):
        for secret_pii in forbidden:
            assert secret_pii not in artifact, f"PII {secret_pii!r} leaked into output!"

    assert "[EMAIL]" in json_report
    assert "[PHONE]" in json_report
    assert "[COLLEGE]" in json_report
    assert "[NAME]" in json_report

    # id->hash map stays ONLY in bot/state.json
    st = load_state(repo / "bot" / "state.json")
    assert attempt_id in st["id_hash_map"]


def test_5_gemini_api_key_unset_aborts_no_other_provider_called(tmp_path: Path, monkeypatch) -> None:
    """C8.5: GEMINI_API_KEY unset -> abort, no other provider called."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-never-be-called")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_should_never_be_called")

    repo = init_temp_git_repo(tmp_path)
    sb = FakeSupabase()

    # Guard against any provider client construction
    called_other = []
    import openai
    monkeypatch.setattr(openai, "OpenAI", lambda *a, **k: called_other.append((a, k)))

    rc = run_daily_report(
        repo_root=repo,
        now_utc=NOW,
        supabase_client=sb,
        suite_runner=stub_suite_ok,
    )

    assert rc == EXIT_ABORT
    assert called_other == []
    assert sb.queries == []
    assert not (repo / "bot" / "reports" / f"{DATE_STR}.json").exists()

    # Also verify via real CLI subprocess with BOT_NO_DOTENV=1 and empty GEMINI_API_KEY
    env = dict(os.environ)
    env["GEMINI_API_KEY"] = ""
    env["BOT_NO_DOTENV"] = "1"
    env["PYTHONPATH"] = str(default_repo_root())
    proc = subprocess.run(
        [sys.executable, "-m", "bot", "--repo-root", str(repo), "--skip-suite"],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert proc.returncode == EXIT_ABORT
    assert "GEMINI_API_KEY is unset" in proc.stdout


def test_6_second_run_same_date_is_noop(tmp_path: Path, monkeypatch) -> None:
    """C8.6: Second run on the same date -> no-op."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    repo = init_temp_git_repo(tmp_path)

    sb = FakeSupabase()
    gemini = FakeGeminiClient()

    rc1 = run_daily_report(
        repo_root=repo,
        now_utc=NOW,
        supabase_client=sb,
        gemini_client=gemini,
        suite_runner=stub_suite_ok,
        budget_checker=lambda: None,
    )
    assert rc1 == EXIT_OK
    report_before = (repo / "bot" / "reports" / f"{DATE_STR}.json").read_text(encoding="utf-8")
    queries_after_first = len(sb.queries)

    # Second run later on the same IST date
    later_same_day = NOW + timedelta(hours=6)
    rc2 = run_daily_report(
        repo_root=repo,
        now_utc=later_same_day,
        supabase_client=sb,
        gemini_client=gemini,
        suite_runner=stub_suite_ok,
        budget_checker=lambda: None,
    )
    assert rc2 == EXIT_OK
    assert len(sb.queries) == queries_after_first  # No DB queries on no-op
    assert len(gemini.calls) == 0
    assert (repo / "bot" / "reports" / f"{DATE_STR}.json").read_text(encoding="utf-8") == report_before
    log_text = (repo / "bot" / "logs" / f"{DATE_STR}.log").read_text(encoding="utf-8")
    assert "[NO-OP]" in log_text


def test_7_consecutive_failures_3_is_report_only(tmp_path: Path, monkeypatch) -> None:
    """C8.7: consecutive_failures = 3 -> report-only."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    repo = init_temp_git_repo(tmp_path)

    st = default_state()
    st["consecutive_failures"] = 3
    st["last_run_date"] = "2026-11-10"
    st["last_run_at"] = (NOW - timedelta(days=1)).isoformat()
    st["cursor"] = (NOW - timedelta(days=1)).isoformat()
    save_state(repo / "bot" / "state.json", st)

    attempt_id = "77777777-1111-2222-3333-444444444444"
    user_id = "88888888-1111-2222-3333-444444444444"
    case_id = "99999999-1111-2222-3333-444444444444"

    sb = FakeSupabase({
        "users": [{"id": user_id, "name": "Test User", "email": "u@example.com", "market": "US"}],
        "cases": [{"id": case_id, "title": "US Market", "type": "guesstimate", "teaching_policy": "exam", "market": "US"}],
        "attempts": [{
            "id": attempt_id,
            "user_id": user_id,
            "case_id": case_id,
            "status": "active",
            "teaching_policy": "exam",
            "session_state": {},
            "created_at": _ts(4),
        }],
        "attempt_messages": [
            {"id": "m1", "attempt_id": attempt_id, "role": "user", "kind": "text", "content": "Maybe 10M users.", "created_at": _ts(3.9)},
            {"id": "m2", "attempt_id": attempt_id, "role": "assistant", "kind": "text", "content": "Great job!", "created_at": _ts(3.8)},
        ],
    })

    gemini = FakeGeminiClient()

    rc = run_daily_report(
        repo_root=repo,
        now_utc=NOW,
        supabase_client=sb,
        gemini_client=gemini,
        suite_runner=stub_suite_ok,
        budget_checker=lambda: None,
    )

    assert rc == EXIT_OK
    assert len(gemini.calls) == 0  # Report-only mode makes ZERO Gemini calls

    md_text = (repo / "bot" / "reports" / f"{DATE_STR}.md").read_text(encoding="utf-8")
    assert md_text.lstrip().startswith("> [!CAUTION]")
    assert "REPORT-ONLY MODE ACTIVE (consecutive_failures = 3)" in md_text

    json_payload = json.loads((repo / "bot" / "reports" / f"{DATE_STR}.json").read_text(encoding="utf-8"))
    assert json_payload["report_only"] is True
    assert json_payload["cost_usd"] == 0.0
    assert "REPORT-ONLY MODE" in json_payload["notes"][0]

    # consecutive_failures remains >= 3 until manually cleared
    new_state = load_state(repo / "bot" / "state.json")
    assert new_state["consecutive_failures"] == 3
