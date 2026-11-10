"""Read-only Supabase queries for attempts, messages, submissions, users, cases, and colleges."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

from services.markets import case_market, content_market_of


class SupabaseUnavailableError(RuntimeError):
    """Raised when Supabase cannot be reached or fails connectivity checks."""


@dataclass
class WindowDataset:
    """All records needed to compute metrics and failure candidates."""

    attempts: List[Dict[str, Any]] = field(default_factory=list)
    messages_by_attempt: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    case_attempts: List[Dict[str, Any]] = field(default_factory=list)
    submissions: List[Dict[str, Any]] = field(default_factory=list)
    users_by_id: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cases_by_id: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    colleges: List[Dict[str, Any]] = field(default_factory=list)

    def attempt_region(self, attempt: Dict[str, Any]) -> str:
        user = self.users_by_id.get(str(attempt.get("user_id") or ""))
        if user and user.get("market"):
            return content_market_of(user.get("market"))
        case = self.cases_by_id.get(str(attempt.get("case_id") or ""))
        if case and case.get("market"):
            return case_market(case)
        return content_market_of((user or {}).get("market"))

    def record_region(self, row: Dict[str, Any]) -> str:
        user = self.users_by_id.get(str(row.get("user_id") or ""))
        if user and user.get("market"):
            return content_market_of(user.get("market"))
        case = self.cases_by_id.get(str(row.get("case_id") or ""))
        if case and case.get("market"):
            return case_market(case)
        return "IN"


def _fetch_all(make_query: Callable[[], Any], page_size: int = 1000) -> List[Dict[str, Any]]:
    """Page through a PostgREST query using .range() when available."""
    out: List[Dict[str, Any]] = []
    offset = 0
    while True:
        q = make_query()
        has_range = hasattr(q, "range")
        if has_range:
            q = q.range(offset, offset + page_size - 1)
        res = q.execute()
        rows = list(getattr(res, "data", None) or [])
        out.extend(rows)
        if not has_range or len(rows) < page_size:
            break
        offset += page_size
    return out


def _fetch_in_chunks(
    supabase: Any,
    table: str,
    col: str,
    ids: Iterable[str],
    select_cols: str = "*",
    order_col: Optional[str] = None,
    chunk_size: int = 100,
) -> List[Dict[str, Any]]:
    unique_ids = sorted({str(i) for i in ids if i})
    if not unique_ids:
        return []
    out: List[Dict[str, Any]] = []
    for i in range(0, len(unique_ids), chunk_size):
        chunk = unique_ids[i : i + chunk_size]

        def _make(c: List[str] = chunk) -> Any:
            q = supabase.table(table).select(select_cols).in_(col, c)
            if order_col and hasattr(q, "order"):
                q = q.order(order_col, desc=False)
            return q

        out.extend(_fetch_all(_make))
    return out


def probe_supabase(supabase: Any) -> None:
    """Verify Supabase is reachable before performing work."""
    try:
        supabase.table("attempts").select("id").limit(1).execute()
    except Exception as exc:  # noqa: BLE001
        raise SupabaseUnavailableError(f"Supabase health probe failed: {type(exc).__name__}: {exc}") from exc


def fetch_window_data(
    supabase: Any,
    *,
    query_from: datetime,
    window_from: datetime,
    window_to: datetime,
) -> WindowDataset:
    """Read attempts, attempt_messages, case_attempts, submissions, users, cases, colleges."""
    probe_supabase(supabase)

    q_from_iso = query_from.isoformat()
    w_from_iso = window_from.isoformat()
    w_to_iso = window_to.isoformat()

    try:
        attempts = _fetch_all(
            lambda: (
                supabase.table("attempts")
                .select("*")
                .gte("created_at", q_from_iso)
                .lte("created_at", w_to_iso)
                .order("created_at", desc=False)
            )
        )
        attempt_ids: Set[str] = {str(a["id"]) for a in attempts if a.get("id")}

        # Also catch older attempts whose messages landed inside the current window.
        recent_msgs = _fetch_all(
            lambda: (
                supabase.table("attempt_messages")
                .select("attempt_id")
                .gte("created_at", w_from_iso)
                .lte("created_at", w_to_iso)
            )
        )
        missing_aids = {
            str(m["attempt_id"])
            for m in recent_msgs
            if m.get("attempt_id") and str(m["attempt_id"]) not in attempt_ids
        }
        if missing_aids:
            extra_attempts = _fetch_in_chunks(supabase, "attempts", "id", missing_aids, select_cols="*")
            for a in extra_attempts:
                aid = str(a.get("id") or "")
                if aid and aid not in attempt_ids:
                    attempt_ids.add(aid)
                    attempts.append(a)

        all_messages = _fetch_in_chunks(
            supabase,
            "attempt_messages",
            "attempt_id",
            attempt_ids,
            select_cols="id, attempt_id, role, kind, content, is_clarification, created_at",
            order_col="created_at",
        )
        messages_by_attempt: Dict[str, List[Dict[str, Any]]] = {}
        for m in all_messages:
            aid = str(m.get("attempt_id") or "")
            if aid:
                messages_by_attempt.setdefault(aid, []).append(m)
        for msgs in messages_by_attempt.values():
            msgs.sort(key=lambda r: (str(r.get("created_at") or ""), str(r.get("id") or "")))

        case_attempts = _fetch_all(
            lambda: (
                supabase.table("case_attempts")
                .select("*")
                .gte("created_at", q_from_iso)
                .lte("created_at", w_to_iso)
            )
        )
        submissions = _fetch_all(
            lambda: (
                supabase.table("submissions")
                .select("*")
                .gte("created_at", q_from_iso)
                .lte("created_at", w_to_iso)
            )
        )

        user_ids: Set[str] = set()
        case_ids: Set[str] = set()
        for coll in (attempts, case_attempts, submissions):
            for row in coll:
                if row.get("user_id"):
                    user_ids.add(str(row["user_id"]))
                if row.get("case_id"):
                    case_ids.add(str(row["case_id"]))

        users_rows = _fetch_in_chunks(supabase, "users", "id", user_ids, select_cols="*")
        cases_rows = _fetch_in_chunks(supabase, "cases", "id", case_ids, select_cols="*")
        users_by_id = {str(u["id"]): u for u in users_rows if u.get("id")}
        cases_by_id = {str(c["id"]): c for c in cases_rows if c.get("id")}
    except SupabaseUnavailableError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SupabaseUnavailableError(f"Supabase read failed: {type(exc).__name__}: {exc}") from exc

    colleges: List[Dict[str, Any]] = []
    try:
        colleges = _fetch_all(lambda: supabase.table("colleges").select("id, name, short_name, slug"))
    except Exception:  # noqa: BLE001
        colleges = []

    return WindowDataset(
        attempts=attempts,
        messages_by_attempt=messages_by_attempt,
        case_attempts=case_attempts,
        submissions=submissions,
        users_by_id=users_by_id,
        cases_by_id=cases_by_id,
        colleges=colleges,
    )
