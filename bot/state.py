"""Persistent state, window calculation, and id->hash mapping in bot/state.json."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from bot.config import FIRST_RUN_DAYS, IST, MAX_WINDOW_DAYS, ROLLING_7D_DAYS


def default_state() -> Dict[str, Any]:
    """Return a clean initial bot state."""
    return {
        "last_run_at": None,
        "last_run_date": None,
        "last_run_status": None,
        "cursor": None,
        "consecutive_failures": 0,
        "hash_salt": None,
        "id_hash_map": {},
    }


def load_state(state_path: Path) -> Dict[str, Any]:
    """Load bot/state.json, returning clean defaults if absent."""
    base = default_state()
    if not state_path.is_file():
        return base
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid state file format in {state_path}")
    base.update(raw)
    if not isinstance(base.get("id_hash_map"), dict):
        base["id_hash_map"] = {}
    if base.get("consecutive_failures") is None:
        base["consecutive_failures"] = 0
    return base


def save_state(state_path: Path, state: Dict[str, Any]) -> None:
    """Atomically write bot/state.json."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp_path, state_path)


def parse_iso_utc(ts: str) -> datetime:
    """Parse an ISO timestamp string into timezone-aware UTC datetime."""
    val = (ts or "").strip()
    if val.endswith("Z"):
        val = val[:-1] + "+00:00"
    dt = datetime.fromisoformat(val)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def ist_date_str(dt: datetime) -> str:
    """Format a datetime as YYYY-MM-DD in IST."""
    aware = dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    return aware.astimezone(IST).date().isoformat()


def is_same_day_noop(state: Dict[str, Any], target_date: str) -> bool:
    """Return True if a run for `target_date` already completed today."""
    if state.get("last_run_date") == target_date:
        return True
    last_at = state.get("last_run_at")
    if not last_at:
        return False
    try:
        return ist_date_str(parse_iso_utc(last_at)) == target_date
    except Exception:  # noqa: BLE001
        return False


def compute_windows(
    state: Dict[str, Any],
    now_utc: datetime,
    *,
    days: Optional[int] = None,
    since: Optional[str] = None,
) -> Dict[str, datetime]:
    """Compute current window, prior 7d comparison window, and rolling 7d window.

    Normal resume: sessions since cursor, capped at 3 days; first run takes 2 days.
    """
    now = now_utc if now_utc.tzinfo is not None else now_utc.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)

    if days is not None:
        if days <= 0:
            raise ValueError("--days must be positive")
        w_from = now - timedelta(days=days)
    elif since is not None:
        w_from = parse_iso_utc(since)
        if w_from > now:
            raise ValueError("--since cannot be in the future")
    elif state.get("cursor"):
        cursor_dt = parse_iso_utc(state["cursor"])
        cap_floor = now - timedelta(days=MAX_WINDOW_DAYS)
        w_from = max(cursor_dt, cap_floor)
        if w_from > now:
            w_from = cap_floor
    else:
        w_from = now - timedelta(days=FIRST_RUN_DAYS)

    prev_7d_to = w_from
    prev_7d_from = w_from - timedelta(days=ROLLING_7D_DAYS)
    rolling_7d_to = now
    rolling_7d_from = now - timedelta(days=ROLLING_7D_DAYS)
    query_from = min(prev_7d_from, rolling_7d_from, w_from)

    return {
        "window_from": w_from,
        "window_to": now,
        "prev_7d_from": prev_7d_from,
        "prev_7d_to": prev_7d_to,
        "rolling_7d_from": rolling_7d_from,
        "rolling_7d_to": rolling_7d_to,
        "query_from": query_from,
    }


def hash_identifier(state: Dict[str, Any], raw_id: str) -> str:
    """Return a stable short hash for `raw_id` and store the id->hash map ONLY in state."""
    rid = str(raw_id or "").strip()
    if not rid:
        return ""
    id_map = state.setdefault("id_hash_map", {})
    if rid in id_map:
        return id_map[rid]
    salt = state.get("hash_salt")
    if not salt:
        salt = secrets.token_hex(16)
        state["hash_salt"] = salt
    digest = hashlib.sha256(f"{salt}:{rid}".encode("utf-8")).hexdigest()[:12]
    id_map[rid] = digest
    return digest
