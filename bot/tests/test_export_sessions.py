"""Tests for read-only session exporter (bot/export.py + --export-sessions CLI)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bot import cli
from bot.config import EXIT_ABORT, EXIT_OK, UUID_RE, default_repo_root
from bot.export import run_session_export
from bot.state import load_state
from bot.tests.fakes import FakeSupabase, init_temp_git_repo

NOW = datetime(2026, 11, 11, 4, 30, 0, tzinfo=timezone(timedelta(hours=5, minutes=30))).astimezone(timezone.utc)
DATE_STR = "2026-11-11"


def _ts(hours_ago: float) -> str:
    return (NOW - timedelta(hours=hours_ago)).isoformat()


def test_export_sessions_sections_funnel_order_redaction_and_zero_gemini(
    tmp_path: Path, monkeypatch
) -> None:
    """Export produces Sections 1-3 cleanly with zero Gemini calls, full opening prompts, and zero PII."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-never-call")
    monkeypatch.setenv("BOT_HASH_SALT", "export-salt-xyz")
    monkeypatch.setenv("SUPABASE_BOT_KEY", "bot-ro-key")

    called_llm = []
    import openai
    monkeypatch.setattr(openai, "OpenAI", lambda *a, **k: called_llm.append((a, k)))

    repo = init_temp_git_repo(tmp_path)
    state_before = (repo / "bot" / "state.json").read_text(encoding="utf-8")

    # Pre-existing dirty file outside bot/ should still succeed
    (repo / "HANDOFF_CLAUDE.txt").write_text("preexisting dirty\n", encoding="utf-8")
    subprocess.run(["git", "add", "HANDOFF_CLAUDE.txt"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "track handoff"], cwd=str(repo), check=True, capture_output=True)
    (repo / "HANDOFF_CLAUDE.txt").write_text("modified before run\n", encoding="utf-8")

    u_old = "10000000-0000-0000-0000-000000000001"  # signed up 20d ago (outside 14d window)
    u_zero = "10000000-0000-0000-0000-000000000002"  # signed up inside 14d window, opened case, 0 user msgs
    u_pii = "10000000-0000-0000-0000-000000000003"  # signed up inside 14d window, abandoned mid-case

    c_guess = "20000000-0000-0000-0000-000000000001"
    c_prof = "20000000-0000-0000-0000-000000000002"

    a_sub = "30000000-0000-0000-0000-000000000001"   # oldest (5d ago), IN, exam, submitted
    a_aban = "30000000-0000-0000-0000-000000000002"  # middle (2d ago), US, coached, abandoned (idle active)
    a_zero = "30000000-0000-0000-0000-000000000003"  # newest (3h ago), IN, coached, abandoned (only opening msg)
    sub_id = "40000000-0000-0000-0000-000000000001"
    pasted_uuid = "55555555-6666-7777-8888-999999999999"

    sb = FakeSupabase({
        "users": [
            {
                "id": u_old, "name": "Old User", "email": "old@example.com", "market": "IN",
                "created_at": _ts(24 * 20),
            },
            {
                "id": u_zero, "name": "Zero Msg", "email": "zero@example.com", "market": "IN",
                "created_at": _ts(48),
            },
            {
                "id": u_pii, "name": "Aarav Mehta", "full_name": "Aarav Mehta",
                "email": "aarav.mehta@isb.edu", "college_email": "aarav@iimb.ac.in",
                "phone": "9811122233", "college_other": "Ashoka College of Commerce",
                "market": "US", "created_at": _ts(72),
            },
        ],
        "colleges": [
            {"id": "col1", "name": "Indian School of Business", "short_name": "ISB"},
        ],
        "cases": [
            {"id": c_guess, "title": "Metro Cups", "type": "guesstimate", "teaching_policy": "exam", "market": "IN"},
            {"id": c_prof, "title": "Margin Drop", "type": "profitability", "teaching_policy": "coached", "market": "US"},
        ],
        "attempts": [
            {
                "id": a_sub, "user_id": u_old, "case_id": c_guess, "status": "submitted",
                "submission_id": sub_id, "teaching_policy": "exam",
                "session_state": {"structure": [
                    {"id": "n1", "label": "Pop", "value": "10M", "parent": ""},
                    {"id": "n2", "label": "Rate", "value": "2", "parent": "n1"},
                ]},
                "created_at": _ts(120),
                "submitted_at": _ts(119.5),
            },
            {
                "id": a_aban, "user_id": u_pii, "case_id": c_prof, "status": "active",
                "teaching_policy": "coached",
                "session_state": {"structure": [
                    {"id": "rev", "label": "Revenue", "value": "", "parent": ""},
                    {"id": "cost", "label": "Cost", "value": "", "parent": ""},
                ]},
                "created_at": _ts(48),
            },
            {
                "id": a_zero, "user_id": u_zero, "case_id": c_guess, "status": "active",
                "teaching_policy": "coached",
                "session_state": {},
                "created_at": _ts(3),
            },
        ],
        "attempt_messages": [
            # a_sub
            {"id": "s1", "attempt_id": a_sub, "role": "assistant", "kind": "text", "content": "Estimate daily metro cups sold in Delhi.", "created_at": _ts(120)},
            {"id": "s2", "attempt_id": a_sub, "role": "user", "kind": "text", "content": "Start with Delhi populashun 2 crore.", "created_at": _ts(119.9)},
            {"id": "s3", "attempt_id": a_sub, "role": "assistant", "kind": "text", "content": "Go ahead with penetration.", "created_at": _ts(119.8)},
            # a_aban (contains PII + typos)
            {"id": "ab1", "attempt_id": a_aban, "role": "assistant", "kind": "text", "content": "Profitability dropped 15%. Walk me through your tree.", "created_at": _ts(48)},
            {
                "id": "ab2", "attempt_id": a_aban, "role": "user", "kind": "text",
                "content": (
                    f"Hi I am Aarav Mehta from Ashoka College of Commerce and Indian School of Business. "
                    f"My email aarav.mehta@isb.edu phone 9811122233 id {u_pii} ref {pasted_uuid}. "
                    "I think reveenu dipped becuz pricing went down."
                ),
                "created_at": _ts(47.9),
            },
            {"id": "ab3", "attempt_id": a_aban, "role": "assistant", "kind": "text", "content": "Which product line saw the steepest margin decline?", "created_at": _ts(47.8)},
            # a_zero (real DB uses role="system" for opening prompt, zero user messages)
            {"id": "z1", "attempt_id": a_zero, "role": "system", "kind": "text", "content": "Estimate daily metro cups sold in Delhi.", "created_at": _ts(3)},
        ],
        "submissions": [
            {"id": sub_id, "user_id": u_old, "case_id": c_guess, "attempt_id": a_sub, "score": 84, "created_at": _ts(119.5)},
        ],
    })

    rc = run_session_export(
        repo_root=repo,
        now_utc=NOW,
        days=14,
        supabase_client=sb,
    )
    assert rc == EXIT_OK
    assert called_llm == []
    assert (repo / "bot" / "state.json").read_text(encoding="utf-8") == state_before

    export_path = repo / "bot" / "exports" / f"sessions_{DATE_STR}.md"
    assert export_path.is_file()
    md = export_path.read_text(encoding="utf-8")

    # Section ordering
    s1_idx = md.index("## Section 1")
    s2_idx = md.index("## Section 2")
    s3_idx = md.index("## Section 3")
    assert s1_idx < s2_idx < s3_idx

    # Funnel counts
    assert "- users who signed up: 2" in md
    assert "- users who opened a case but sent zero messages: 1" in md
    assert "- attempts started: 3" in md
    assert "- attempts with >=1 candidate message: 2" in md
    assert "  - Clarify: 3" in md
    assert "  - Decompose: 2" in md
    assert "  - Conclude: 1" in md
    assert "- attempts submitted: 1" in md
    assert "- attempts by region: IN=2, US=1" in md
    assert "- attempts by teaching_policy: coached=2, exam=1" in md

    # Newest-first ordering in Section 2: a_zero -> a_aban -> a_sub
    salt = "export-salt-xyz"
    h_zero = hashlib.sha256((salt + a_zero).encode("utf-8")).hexdigest()[:16]
    h_aban = hashlib.sha256((salt + a_aban).encode("utf-8")).hexdigest()[:16]
    h_sub = hashlib.sha256((salt + a_sub).encode("utf-8")).hexdigest()[:16]

    pos_zero = md.index(f"### attempt {h_zero} · guesstimate · IN · coached · abandoned")
    pos_aban = md.index(f"### attempt {h_aban} · profitability · US · coached · abandoned")
    pos_sub = md.index(f"### attempt {h_sub} · guesstimate · IN · exam · submitted")
    assert pos_zero < pos_aban < pos_sub

    # Full opening message preserved even when repeated + typos kept + scores shown
    assert md.count("Estimate daily metro cups sold in Delhi.") >= 2
    assert "populashun" in md
    assert "reveenu dipped becuz" in md
    assert "0 candidate messages sent" in md
    assert "score: 84" in md
    assert "score: -" in md

    # Section 3 contains abandoned attempts only (last assistant message / opening prompt grouped by stage)
    sec3 = md[s3_idx:]
    assert h_zero in sec3
    assert h_aban in sec3
    assert h_sub not in sec3
    assert "Which product line saw the steepest margin decline?" in sec3
    assert "[opening system prompt] Estimate daily metro cups sold in Delhi." in sec3

    # Strict zero-PII & zero-UUID guarantee across entire export file + bot logs
    forbidden = [
        u_old, u_zero, u_pii, c_guess, c_prof, a_sub, a_aban, a_zero, sub_id, pasted_uuid,
        "Aarav", "Mehta", "aarav.mehta@isb.edu", "9811122233",
        "Ashoka College of Commerce", "Indian School of Business",
    ]
    for f in forbidden:
        assert f not in md, f"Forbidden PII {f!r} leaked into session export!"
    assert UUID_RE.search(md) is None
    assert "[EMAIL]" in md
    assert "[PHONE]" in md
    assert "[COLLEGE]" in md
    assert "[NAME]" in md


def test_export_malformed_rows_zero_message_empty_git_guard_and_cli(
    tmp_path: Path, monkeypatch
) -> None:
    """Bad rows skipped & counted at top; truly empty 0-message transcript labeled; mid-run edit aborts; CLI works."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("BOT_HASH_SALT", "cli-export-salt")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_BOT_KEY", "bot-ro-key")

    # 1. Malformed rows + completely empty 0-message attempt
    repo1 = init_temp_git_repo(tmp_path / "r1")
    sb = FakeSupabase({
        "users": [
            {"id": "u1", "name": "Ok User", "market": "IN", "created_at": _ts(5)},
            {"id": "u_bad", "name": "Bad User", "created_at": "bad-date"},
        ],
        "cases": [{"id": "c1", "type": "guesstimate", "market": "IN"}],
        "attempts": [
            {"id": "att_empty", "user_id": "u1", "case_id": "c1", "status": "abandoned", "created_at": _ts(4)},
            {"id": "", "user_id": "u1", "case_id": "c1", "status": "active", "created_at": _ts(3.5)},
        ],
        "attempt_messages": [
            {"id": "m_bad", "attempt_id": "att_empty", "role": "user", "content": "hi", "created_at": "bad-ts"},
        ],
        "submissions": [
            {"id": "sub_bad", "user_id": "u1", "case_id": "c1", "attempt_id": "att_empty", "score": "N/A", "created_at": _ts(3.5)},
        ],
    })
    rc1 = run_session_export(repo_root=repo1, now_utc=NOW, days=14, supabase_client=sb)
    assert rc1 == EXIT_OK
    md1 = (repo1 / "bot" / "exports" / f"sessions_{DATE_STR}.md").read_text(encoding="utf-8")
    assert "Skipped unparseable rows: 3" in md1
    assert "(empty transcript — 0 messages sent)" in md1
    assert "(no assistant message)" in md1

    # 2. Git guard aborts if path outside bot/ changes during export
    repo2 = init_temp_git_repo(tmp_path / "r2")
    rc2 = run_session_export(
        repo_root=repo2,
        now_utc=NOW,
        days=14,
        supabase_client=FakeSupabase(),
        before_write_hook=lambda: (repo2 / "services" / "ai_scorer.py").write_text("# modified\n", encoding="utf-8"),
    )
    assert rc2 == EXIT_ABORT
    assert not (repo2 / "bot" / "exports" / f"sessions_{DATE_STR}.md").exists()

    # 3. CLI flag --export-sessions --days 14 dispatch (no GEMINI_API_KEY)
    repo3 = init_temp_git_repo(tmp_path / "r3")
    import bot.export as export_mod
    monkeypatch.setattr(export_mod, "build_supabase_client", lambda _u, _k: FakeSupabase())
    rc_cli = cli.main(["--repo-root", str(repo3), "--no-dotenv", "--export-sessions", "--days", "14"])
    assert rc_cli == EXIT_OK
    exports = list((repo3 / "bot" / "exports").glob("sessions_*.md"))
    assert len(exports) == 1

    # 4. BOT_HASH_SALT unset via CLI subprocess aborts cleanly
    env = dict(os.environ)
    env["BOT_HASH_SALT"] = ""
    env["GEMINI_API_KEY"] = ""
    env["BOT_NO_DOTENV"] = "1"
    env["PYTHONPATH"] = str(default_repo_root())
    proc = subprocess.run(
        [sys.executable, "-m", "bot", "--repo-root", str(repo3), "--export-sessions", "--days", "14"],
        cwd=str(repo3),
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert proc.returncode == EXIT_ABORT
    assert "BOT_HASH_SALT is unset" in proc.stdout
