"""
Feature flag for the unified interviewer brain.

INTERVIEWER_BRAIN
  off        (default) every route runs the baseline MECE Interviewer V11 engine,
             unchanged. The removed static "no hints" prompt stays removed and
             fail-closed in both states.
  on         every attempt uses the unified brain.
  allowlist  only users listed in INTERVIEWER_BRAIN_ALLOWLIST (comma-separated
             user ids and/or emails) use it; everyone else stays on V11.

Read on every call so a Render env change applies on the next process restart
without any code change.
"""
from __future__ import annotations

import os
from typing import Optional


def brain_mode() -> str:
    v = (os.getenv("INTERVIEWER_BRAIN") or "off").strip().lower()
    if v in ("1", "true", "yes", "on", "unified"):
        return "on"
    if v in ("allowlist", "allow", "beta"):
        return "allowlist"
    return "off"


def _allowlist() -> set:
    raw = os.getenv("INTERVIEWER_BRAIN_ALLOWLIST") or ""
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def brain_enabled_for(user_id: Optional[str], email: Optional[str] = None) -> bool:
    mode = brain_mode()
    if mode == "on":
        return True
    if mode == "allowlist":
        allowed = _allowlist()
        return bool((user_id and user_id.lower() in allowed) or (email and email.strip().lower() in allowed))
    return False


def debug_decisions() -> bool:
    """Return mode/reason to clients only when explicitly debugging (they are internal control data)."""
    return (os.getenv("INTERVIEWER_DEBUG_DECISIONS") or "").strip().lower() in ("1", "true", "yes", "on")
