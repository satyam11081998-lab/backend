"""Service configuration.

Everything comes from environment variables owned by THIS service (prefix `II_`).
Nothing here reads the existing MECE backend's env names, on purpose: the two services
must be deployable, rotatable and revocable independently.

Runtime-tunable flags and limits live in the `system_config` table (see
`access/flags.py`); the values here are only their defaults.
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name, default) or "").strip()


def _bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


def _csv(name: str) -> List[str]:
    return [p.strip() for p in _env(name).split(",") if p.strip()]


def normalize_pem(raw: str) -> str:
    """Accept a PEM given raw, with literal '\\n' escapes, or base64-encoded."""
    raw = (raw or "").strip().strip('"').strip("'")
    if not raw:
        return ""
    if "BEGIN" not in raw:
        try:
            raw = base64.b64decode(raw).decode("utf-8")
        except Exception:
            return ""
    return raw.replace("\\n", "\n").strip() + "\n"


NON_PRODUCTION_ENVS = frozenset({"dev", "test", "qa"})


@dataclass(frozen=True)
class Settings:
    # Fail safe: anything that is not explicitly dev/test/qa is treated as production (no
    # simulated AI, no ephemeral encryption key, no /docs). A forgotten II_ENV cannot
    # quietly put fake scores in front of users.
    env: str = "prod"
    database_url: str = "sqlite+pysqlite:///:memory:"
    db_schema: str = "interview_intel"

    # Entitlement assertion (MECE -> II). Public keys only; II can never mint tokens.
    assertion_public_keys: Dict[str, str] = field(default_factory=dict)
    assertion_issuer: str = "mece-app"
    assertion_audience: str = "mece-interview-intelligence"
    assertion_max_lifetime_s: int = 900
    assertion_leeway_s: int = 30

    admin_emails: List[str] = field(default_factory=list)
    trust_mece_admin: bool = True
    bootstrap_test_emails: List[str] = field(default_factory=list)

    encryption_key: str = ""  # Fernet key (urlsafe base64, 32 bytes)

    # Limits (defaults; overridable in system_config)
    max_active_sessions: int = 2
    max_sessions_per_day: int = 5
    max_upload_mb: int = 5
    session_cost_cap_usd: float = 1.50
    daily_budget_usd: float = 25.0

    # AI
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    groq_api_key: str = ""
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    model_routes_json: str = ""
    model_fast: str = "gpt-4o-mini"
    model_strong: str = "gpt-4o"
    provider_fast: str = "openai"
    provider_strong: str = "openai"
    # Gemini text model for the light stages (shared with the backend's GEMINI_MODEL: Google
    # retires names without notice, one rename should fix every caller).
    gemini_model: str = "gemini-3.6-flash"
    ai_timeout_s: float = 45.0
    # Live voice (OpenAI Realtime). The model only VOICES the lines II decides.
    realtime_model: str = "gpt-realtime-2.1"
    realtime_transcribe_model: str = "gpt-4o-mini-transcribe"
    realtime_eagerness: str = "low"  # semantic VAD: low = waits through thinking pauses

    # Drive
    gdrive_root_folder_id: str = ""
    gdrive_refresh_token: str = ""
    gdrive_client_id: str = ""
    gdrive_client_secret: str = ""
    gdrive_sa_json_b64: str = ""
    drive_folder_salt: str = ""
    purge_blob_after_drive_sync: bool = False

    # Runtime
    worker_threads: int = 2
    db_pool_size: int = 5
    cors_origins: List[str] = field(default_factory=list)
    cors_origin_regex: str = r"https://[a-z0-9-]+-consilioo\.vercel\.app"
    log_level: str = "INFO"

    @property
    def is_production(self) -> bool:
        return self.env not in NON_PRODUCTION_ENVS

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def drive_configured(self) -> bool:
        has_oauth = bool(self.gdrive_refresh_token and self.gdrive_client_id and self.gdrive_client_secret)
        return bool(self.gdrive_root_folder_id) and (has_oauth or bool(self.gdrive_sa_json_b64))


def _load_public_keys() -> Dict[str, str]:
    keys: Dict[str, str] = {}
    multi = _env("II_ASSERTION_PUBLIC_KEYS")
    if multi:
        try:
            data = json.loads(multi)
            for kid, pem in (data or {}).items():
                norm = normalize_pem(str(pem))
                if norm:
                    keys[str(kid)] = norm
        except json.JSONDecodeError:
            pass
    single = normalize_pem(_env("II_ASSERTION_PUBLIC_KEY"))
    if single:
        keys.setdefault(_env("II_ASSERTION_KID", "k1") or "k1", single)
    return keys


def load_settings() -> Settings:
    url = _env("II_DATABASE_URL", "sqlite+pysqlite:///:memory:")
    # Supabase hands out postgres:// URLs; SQLAlchemy+psycopg3 needs the driver named.
    url = re.sub(r"^postgres(ql)?://", "postgresql+psycopg://", url)
    origins = _csv("II_CORS_ORIGINS") or ["http://localhost:3000", "https://mece.in", "https://www.mece.in"]
    return Settings(
        env=(_env("II_ENV", "prod") or "prod").lower(),
        database_url=url,
        db_schema=_env("II_DB_SCHEMA", "interview_intel") or "interview_intel",
        assertion_public_keys=_load_public_keys(),
        assertion_max_lifetime_s=_int("II_ASSERTION_MAX_LIFETIME_S", 900),
        admin_emails=[e.lower() for e in _csv("II_ADMIN_EMAILS")],
        trust_mece_admin=_bool("II_TRUST_MECE_ADMIN", True),
        bootstrap_test_emails=[e.lower() for e in _csv("II_BOOTSTRAP_TEST_EMAILS")],
        encryption_key=_env("II_ENCRYPTION_KEY"),
        max_active_sessions=_int("II_MAX_ACTIVE_SESSIONS", 2),
        max_sessions_per_day=_int("II_MAX_SESSIONS_PER_DAY", 5),
        max_upload_mb=_int("II_MAX_UPLOAD_MB", 5),
        session_cost_cap_usd=_float("II_SESSION_COST_CAP_USD", 1.50),
        daily_budget_usd=_float("II_DAILY_BUDGET_USD", 25.0),
        openai_api_key=_env("II_OPENAI_API_KEY") or _env("OPENAI_API_KEY"),
        openai_base_url=_env("II_OPENAI_BASE_URL", "https://api.openai.com/v1"),
        groq_api_key=_env("II_GROQ_API_KEY") or _env("GROQ_API_KEY"),
        gemini_api_key=_env("II_GEMINI_API_KEY") or _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY"),
        anthropic_api_key=_env("II_ANTHROPIC_API_KEY") or _env("ANTHROPIC_API_KEY"),
        model_routes_json=_env("II_MODEL_ROUTES"),
        model_fast=_env("II_MODEL_FAST", "gpt-4o-mini"),
        model_strong=_env("II_MODEL_STRONG", "gpt-4o"),
        provider_fast=_env("II_PROVIDER_FAST", "openai"),
        provider_strong=_env("II_PROVIDER_STRONG", "openai"),
        gemini_model=_env("II_GEMINI_MODEL") or _env("GEMINI_MODEL") or "gemini-3.6-flash",
        ai_timeout_s=_float("II_AI_TIMEOUT_S", 45.0),
        realtime_model=_env("II_REALTIME_MODEL") or _env("REALTIME_MODEL") or "gpt-realtime-2.1",
        realtime_transcribe_model=_env("II_REALTIME_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe") or "gpt-4o-mini-transcribe",
        realtime_eagerness=(_env("II_REALTIME_EAGERNESS", "low") or "low").lower(),
        gdrive_root_folder_id=_env("II_GDRIVE_ROOT_FOLDER_ID"),
        gdrive_refresh_token=_env("II_GDRIVE_REFRESH_TOKEN") or _env("GOOGLE_DRIVE_REFRESH_TOKEN"),
        gdrive_client_id=_env("II_GDRIVE_CLIENT_ID") or _env("GOOGLE_DRIVE_CLIENT_ID"),
        gdrive_client_secret=_env("II_GDRIVE_CLIENT_SECRET") or _env("GOOGLE_DRIVE_CLIENT_SECRET"),
        gdrive_sa_json_b64=_env("II_GDRIVE_SA_JSON"),
        drive_folder_salt=_env("II_DRIVE_FOLDER_SALT"),
        purge_blob_after_drive_sync=_bool("II_PURGE_BLOB_AFTER_DRIVE_SYNC", False),
        worker_threads=_int("II_WORKER_THREADS", 2),
        db_pool_size=_int("II_DB_POOL_SIZE", 5),
        cors_origins=origins,
        cors_origin_regex=_env("II_CORS_ORIGIN_REGEX", r"https://[a-z0-9-]+-consilioo\.vercel\.app"),
        log_level=_env("II_LOG_LEVEL", "INFO"),
    )


_override: Optional[Settings] = None


@lru_cache(maxsize=1)
def _cached() -> Settings:
    return load_settings()


def get_settings() -> Settings:
    return _override if _override is not None else _cached()


def set_settings_for_tests(settings: Optional[Settings]) -> None:
    global _override
    _override = settings
