"""CLI entry point for the read-only daily reporter bot."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="MECE backend read-only daily reporter")
    ap.add_argument("--repo-root", type=str, default=None, help="Override repository root path")
    ap.add_argument("--days", type=int, default=None, help="Lookback window in days (manual override)")
    ap.add_argument("--since", type=str, default=None, help="ISO-8601 start timestamp (manual override)")
    ap.add_argument("--dry-run", action="store_true", help="Write preview reports without updating state.json")
    ap.add_argument("--export-sessions", action="store_true", help="Export redacted session transcripts (no Gemini calls)")
    ap.add_argument("--skip-suite", action="store_true", help="Skip pytest + offline eval checks")
    ap.add_argument("--require-bot-key", action="store_true", help="Fail if SUPABASE_BOT_KEY is unset (no service-role fallback)")
    ap.add_argument("--no-dotenv", action="store_true", help="Do not load .env from disk")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if not args.no_dotenv and os.getenv("BOT_NO_DOTENV", "").strip() != "1":
        load_dotenv(override=False)
    repo_root = Path(args.repo_root) if args.repo_root else None
    req_bot_key = True if args.require_bot_key else None
    if args.export_sessions:
        from bot.export import run_session_export

        return run_session_export(
            repo_root=repo_root,
            days=args.days,
            since=args.since,
            require_bot_key=req_bot_key,
        )

    from bot.runner import run_daily_report

    return run_daily_report(
        repo_root=repo_root,
        days=args.days,
        since=args.since,
        dry_run=args.dry_run,
        skip_suite=args.skip_suite,
        require_bot_key=req_bot_key,
    )


if __name__ == "__main__":
    sys.exit(main())
