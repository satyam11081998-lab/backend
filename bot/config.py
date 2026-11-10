"""Configuration, caps, credential resolution, and secret/UUID hygiene helpers."""
from __future__ import annotations

import os
import re
from datetime import timedelta, timezone
from pathlib import Path
from typing import Iterable, Tuple

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

SERVICE_ROLE_WARN = (
    "[WARN] running with the service role key — create a read-only role before enabling the workflow"
)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_ABORT = 2

_SECRET_ENV_NAMES = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "GROQ_API_KEY",
    "SUPABASE_BOT_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
    "BOT_HASH_SALT",
    "CRON_SECRET",
    "TELEGRAM_BOT_TOKEN",
)
_KEY_PATTERNS = (
    re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}\b"),
    re.compile(r"\bsk-(?:proj-|live-|test-)?[0-9A-Za-z\-_]{16,}\b"),
    re.compile(r"\bgsk_[0-9A-Za-z\-_]{16,}\b"),
    re.compile(r"\beyJ[0-9A-Za-z\-_]+\.[0-9A-Za-z\-_]+\.[0-9A-Za-z\-_]+\b"),
)
UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)


def default_repo_root() -> Path:
    """Return the repository root directory (BOT_REPO_ROOT or parent of bot/)."""
    env_root = (os.environ.get("BOT_REPO_ROOT") or "").strip()
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parent.parent


def get_gemini_api_key() -> str:
    """Read GEMINI_API_KEY strictly from os.environ (never falling back to other keys)."""
    return (os.environ.get("GEMINI_API_KEY") or "").strip()


def get_bot_hash_salt() -> str:
    """Read BOT_HASH_SALT from os.environ."""
    return (os.environ.get("BOT_HASH_SALT") or "").strip()


def get_supabase_credentials() -> Tuple[str, str, bool]:
    """Return (url, key, used_service_role), preferring SUPABASE_BOT_KEY over service role."""
    url = (os.environ.get("SUPABASE_URL") or "").strip()
    bot_key = (os.environ.get("SUPABASE_BOT_KEY") or "").strip()
    if bot_key:
        return url, bot_key, False
    svc_key = (os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    return url, svc_key, True


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


def scrub_uuids(text: str) -> str:
    """Ensure no raw UUID ever survives in bot state, logs, or reports."""
    return UUID_RE.sub("[UUID_REDACTED]", text or "")
