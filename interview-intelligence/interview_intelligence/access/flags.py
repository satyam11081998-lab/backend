"""Feature flags and runtime limits (spec §90), stored in `system_config`.

Env (config.py) supplies defaults; an admin override in the DB wins; values are cached
for 30 s per process so the hot path does not read the table on every request.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db.models import SystemConfig, utcnow

CACHE_TTL_S = 30.0


def defaults() -> Dict[str, Any]:
    s = get_settings()
    return {
        "ii.enabled": True,
        "ii.enabled_for_pro": False,  # launch flag: off => only admins + test users
        "admin.test_access": True,
        "voice.enabled": False,
        "company_intel.enabled": True,
        "company_intel.web_research": False,
        "technical.advanced_mode": True,
        "technical.coding_exercises": False,  # needs an isolated code sandbox (not built)
        "ocr.enabled": False,
        "drive.export_reports": True,
        "limits.max_active_sessions": s.max_active_sessions,
        "limits.max_sessions_per_day": s.max_sessions_per_day,
        "limits.max_sessions_per_day_test": max(s.max_sessions_per_day, 10),
        "limits.allowed_durations": [15, 30, 45, 60],
        "limits.session_cost_cap_usd": s.session_cost_cap_usd,
        "limits.daily_budget_usd": s.daily_budget_usd,
        "limits.max_upload_mb": s.max_upload_mb,
    }


# Types an admin may set, so a typo cannot turn a number into a string.
_TYPES = {
    k: (bool if isinstance(v, bool) else type(v))
    for k, v in {
        "ii.enabled": True, "ii.enabled_for_pro": False, "admin.test_access": True,
        "voice.enabled": False, "company_intel.enabled": True, "company_intel.web_research": False,
        "technical.advanced_mode": True, "technical.coding_exercises": False, "ocr.enabled": False,
        "drive.export_reports": True, "limits.max_active_sessions": 2, "limits.max_sessions_per_day": 5,
        "limits.max_sessions_per_day_test": 10, "limits.allowed_durations": [45],
        "limits.session_cost_cap_usd": 1.0, "limits.daily_budget_usd": 1.0, "limits.max_upload_mb": 5,
    }.items()
}

_cache: Dict[str, Any] = {"ts": 0.0, "data": {}}
_lock = threading.Lock()


def invalidate() -> None:
    with _lock:
        _cache["ts"] = 0.0


def _load_overrides(db: Session) -> Dict[str, Any]:
    now = time.time()
    with _lock:
        if _cache["ts"] and now - _cache["ts"] < CACHE_TTL_S:
            return _cache["data"]
    rows = db.execute(select(SystemConfig)).scalars().all()
    data = {r.key: (r.value or {}).get("v") for r in rows if isinstance(r.value, dict)}
    with _lock:
        _cache["data"] = data
        _cache["ts"] = now
    return data


def all_flags(db: Session) -> Dict[str, Any]:
    merged = defaults()
    for k, v in _load_overrides(db).items():
        if k in merged and v is not None:
            merged[k] = v
    return merged


def flag(db: Session, key: str) -> Any:
    return all_flags(db).get(key)


def coerce(key: str, value: Any) -> Any:
    if key not in _TYPES:
        raise KeyError(key)
    t = _TYPES[key]
    if t is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if t is int:
        v = int(value)
        if key == "limits.max_active_sessions" and not (1 <= v <= 10):
            raise ValueError("max_active_sessions must be between 1 and 10")
        if v < 0:
            raise ValueError("must be >= 0")
        return v
    if t is float:
        v = float(value)
        if v < 0:
            raise ValueError("must be >= 0")
        return v
    if t is list:
        if not isinstance(value, list) or not all(isinstance(x, int) and 5 <= x <= 90 for x in value):
            raise ValueError("expected a list of minutes between 5 and 90")
        return sorted(set(value))
    return value


def set_flag(db: Session, key: str, value: Any, *, actor: str) -> Any:
    v = coerce(key, value)
    row = db.get(SystemConfig, key)
    if row is None:
        db.add(SystemConfig(key=key, value={"v": v}, updated_by=actor, updated_at=utcnow()))
    else:
        row.value = {"v": v}
        row.updated_by = actor
        row.updated_at = utcnow()
    db.flush()
    invalidate()
    return v
