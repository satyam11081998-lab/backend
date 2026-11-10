"""Configuration, caps, and secret hygiene helpers for the daily reporter bot."""
from __future__ import annotations

import os
import re
from datetime import timedelta, timezone
from pathlib import Path
from typing import Iterable

IST = timezone(timedelta(hours=5, minutes=30))

FIRST_RUN_DAYS = 2
MAX_WINDOW_DAYS = 3
ROLLING_7D_DAYS = 7
MAX_FAILURES_IN_REPORT = 10
MAX_SHORTLIST_FOR_LLM = 25
CONSECUTIVE_FAILURE_REPORT_ONLY = 3

MAX_WALL_SECONDS = int(os.getenv("BOT_MAX_WALL_SECONDS", str(20 * 60)))
MAX_TOKENS_PER_RUN = int(os.getenv("BOT_MAX_TOKENS_PER_RUN", "50000"))

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_ABORT = 2

_SECRET_ENV_NAMES = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "GROQ_API_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
    "CRON_SECRET",
    "TELEGRAM_BOT_TOKEN",
)
_KEY_PATTERNS = (
    re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}\b"),
    re.compile(r"\bsk-(?:proj-|live-|test-)?[0-9A-Za-z\-_]{16,}\b"),
    re.compile(r"\bgsk_[0-9A-Za-z\-_]{16,}\b"),
    re.compile(r"\beyJ[0-9A-Za-z\-_]+\.[0-9A-Za-z\-_]+\.[0-9A-Za-z\-_]+\b"),
)


def default_repo_root() -> Path:
    """Return the repository root directory (BOT_REPO_ROOT or parent of bot/)."""
    env_root = (os.environ.get("BOT_REPO_ROOT") or "").strip()
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parent.parent


def get_gemini_api_key() -> str:
    """Read GEMINI_API_KEY strictly from os.environ (never falling back to other keys)."""
    return (os.environ.get("GEMINI_API_KEY") or "").strip()


def scrub_secrets(text: str, extra_secrets: Iterable[str] = ()) -> str:
    """Remove any configured secret values or API-key shapes from logs/reports."""
    out = text or ""
    candidates = [os.environ.get(name, "").strip() for name in _SECRET_ENV_NAMES]
    candidates.extend((s or "").strip() for s in extra_secrets)
    for secret in sorted({c for c in candidates if len(c) >= 6}, key=len, reverse=True):
        out = out.replace(secret, "[SECRET_REDACTED]")
    for pat in _KEY_PATTERNS:
        out = pat.sub("[SECRET_REDACTED]", out)
    return out
