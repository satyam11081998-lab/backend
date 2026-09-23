"""
Prep Copilot v2 - the SELF-IMPROVING CORPUS.

A repository of grounded Packs (role, and role x company) that keeps improving:
- first request for a (role[,company]) triggers a Gemini-grounded build and stores it;
- later requests reuse the stored pack instantly;
- a stale (older than TTL) or low-confidence pack is rebuilt on next request and
  supersedes the old one (version bump), so the corpus sharpens over time;
- every build writes a research-log row for provenance / auditing.

Persistence is best-effort: if the DB write fails, the caller still gets a usable
Pack (research keeps the copilot working even when the corpus can't save). Reads
and writes use the service-scoped Supabase client the route passes in; these
tables are service-role-only (RLS on, no policy) and never touched by the client.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .schemas import Pack, normalize_key
from .research import research_pack

PACK_TTL_DAYS = int(os.getenv("COPILOT_PACK_TTL_DAYS", "30") or "30")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _db_company_key(company_key: Optional[str]) -> str:
    # store '' for role-only packs so UNIQUE(role_key, company_key) dedupes cleanly
    return company_key or ""


def get_pack(supabase, role_key: str, company_key: Optional[str]) -> Optional[Pack]:
    try:
        r = (supabase.table("copilot_packs")
             .select("pack, version, updated_at, confidence, status")
             .eq("role_key", role_key).eq("company_key", _db_company_key(company_key))
             .maybe_single().execute())
        row = r.data or None
    except Exception:
        return None
    if not row or not row.get("pack"):
        return None
    try:
        pack = Pack.from_dict(row["pack"])
        pack.version = int(row.get("version") or pack.version)
        return pack
    except Exception:
        return None


def _is_stale(supabase, role_key: str, company_key: Optional[str]) -> bool:
    try:
        r = (supabase.table("copilot_packs").select("updated_at, confidence")
             .eq("role_key", role_key).eq("company_key", _db_company_key(company_key))
             .maybe_single().execute())
        row = r.data or {}
    except Exception:
        return False
    ts = row.get("updated_at")
    if not ts:
        return True
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        age_days = (_now() - dt).total_seconds() / 86400.0
    except Exception:
        return False
    # low-confidence packs are refreshed sooner (a quarter of the TTL)
    conf = str(row.get("confidence") or "low")
    ttl = PACK_TTL_DAYS if conf in ("high", "medium") else max(3, PACK_TTL_DAYS // 4)
    return age_days > ttl


def save_pack(supabase, pack: Pack, *, bump: bool = False) -> Optional[str]:
    """Upsert a pack by (role_key, company_key). Best-effort; returns row id or None."""
    ck = _db_company_key(pack.company_key)
    now_iso = _now().isoformat()
    payload: Dict[str, Any] = {
        "role_key": pack.role_key, "company_key": ck,
        "display_role": pack.display_role, "display_company": pack.display_company,
        "pack": pack.to_dict(), "confidence": pack.confidence, "status": pack.status,
        "updated_at": now_iso,
    }
    try:
        existing = (supabase.table("copilot_packs").select("id, version")
                    .eq("role_key", pack.role_key).eq("company_key", ck).maybe_single().execute())
        row = existing.data or None
    except Exception:
        row = None
    try:
        if row and row.get("id"):
            payload["version"] = int(row.get("version") or 1) + (1 if bump else 0)
            payload["pack"]["version"] = payload["version"]
            supabase.table("copilot_packs").update(payload).eq("id", row["id"]).execute()
            return row["id"]
        payload["version"] = 1
        payload["created_at"] = now_iso
        ins = supabase.table("copilot_packs").insert(payload).execute()
        return ((ins.data or [{}])[0] or {}).get("id")
    except Exception:
        return None


def _log_research(supabase, pack: Pack, user_id: Optional[str], note: str) -> None:
    try:
        supabase.table("copilot_research_log").insert({
            "role_key": pack.role_key, "company_key": _db_company_key(pack.company_key),
            "user_id": user_id, "ok": pack.status == "ready", "confidence": pack.confidence,
            "n_sources": len(pack.sources), "note": note,
        }).execute()
    except Exception:
        pass


def get_or_build(supabase, role: str, company: str = "", *, force_refresh: bool = False,
                 user_id: Optional[str] = None) -> Pack:
    """Resolve a Pack for (role[,company]): reuse a fresh stored pack, else build a
    grounded one, store it, and return it. Self-improving on staleness / low
    confidence. Never raises."""
    role_key = normalize_key(role)
    company_key = normalize_key(company) or None

    if not force_refresh:
        cached = get_pack(supabase, role_key, company_key)
        if cached and cached.status == "ready" and not _is_stale(supabase, role_key, company_key):
            return cached

    fresh = research_pack(role, company)          # never raises
    save_pack(supabase, fresh, bump=force_refresh)  # best-effort persistence
    _log_research(supabase, fresh, user_id, "refresh" if force_refresh else "build")

    # if the fresh build failed but we had an older cached pack, prefer the cache
    if fresh.status != "ready":
        cached = get_pack(supabase, role_key, company_key)
        if cached and cached.status == "ready":
            return cached
    return fresh
