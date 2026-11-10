"""Read-only Supabase queries with defensive row validation and skip tracking."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

from services.markets import case_market, content_market_of

from bot.state import parse_iso_utc


class SupabaseUnavailableError(RuntimeError):
    """Raised when Supabase cannot be reached or fails connectivity checks."""


def build_supabase_client(url: str, key: str) -> Any:
    """Construct a Supabase client from explicit (url, key) without requiring service role."""
    if not url or not key:
        raise SupabaseUnavailableError("Missing SUPABASE_URL or Supabase API key")
    try:
        import httpx
        from supabase import create_client
        from supabase.lib.client_options import SyncClientOptions

        http = httpx.Client(
            http2=False,
            follow_redirects=True,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        return create_client(url, key, options=SyncClientOptions(httpx_client=http))
    except Exception:  # noqa: BLE001
        from supabase import create_client

        return create_client(url, key)


def normalize_session_state(raw_state: Any) -> Dict[str, Any]:
    """Coerce legacy, null, or malformed session_state into a safe dict."""
    if isinstance(raw_state, str):
        try:
            raw_state = json.loads(raw_state)
        except Exception:  # noqa: BLE001
            return {}
    if not isinstance(raw_state, dict):
        return {}
    out = dict(raw_state)
    raw_struct = out.get("structure")
    if raw_struct is not None:
        if isinstance(raw_struct, list):
            clean_nodes: List[Dict[str, Any]] = []
            for idx, n in enumerate(raw_struct):
                if not isinstance(n, dict):
                    continue
                nid = str(n.get("id") or n.get("label") or f"node-{idx}").strip()
                clean_nodes.append({
                    "id": nid,
                    "label": str(n.get("label") or nid),
                    "value": str(n.get("value") or "") if n.get("value") is not None else "",
                    "parent": str(n.get("parent") or ""),
                    "kind": str(n.get("kind") or "stated"),
                })
            out["structure"] = clean_nodes
        else:
            out["structure"] = []
    return out


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
    skipped_counts: Dict[str, int] = field(default_factory=dict)
    skipped_reasons: List[str] = field(default_factory=list)

    def note_skip(self, table: str, reason: str) -> None:
        self.skipped_counts[table] = self.skipped_counts.get(table, 0) + 1
        if len(self.skipped_reasons) < 25 and reason not in self.skipped_reasons:
            self.skipped_reasons.append(f"{table}: {reason}")

    def total_skipped(self) -> int:
        return sum(self.skipped_counts.values())

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


def _valid_ts_row(row: Any, table: str, ds: WindowDataset) -> bool:
    if not isinstance(row, dict):
        ds.note_skip(table, "non-dict row")
        return False
    if not row.get("id"):
        ds.note_skip(table, "missing id")
        return False
    ts = row.get("created_at")
    if ts is not None:
        try:
            parse_iso_utc(str(ts))
        except Exception:  # noqa: BLE001
            ds.note_skip(table, "unparseable created_at")
            return False
    elif table in ("attempts", "case_attempts", "submissions"):
        ds.note_skip(table, "missing created_at")
        return False
    return True


def _valid_submission_row(row: Any, ds: WindowDataset) -> bool:
    if not _valid_ts_row(row, "submissions", ds):
        return False
    raw_score = row.get("score")
    if raw_score is not None:
        if isinstance(raw_score, bool):
            ds.note_skip("submissions", "unparseable score")
            return False
        try:
            float(raw_score)
        except (TypeError, ValueError):
            ds.note_skip("submissions", "unparseable score")
            return False
    fj = row.get("feedback_json")
    if isinstance(fj, str) and fj.strip():
        try:
            json.loads(fj)
        except Exception:  # noqa: BLE001
            ds.note_skip("submissions", "unparseable feedback_json")
            return False
    return True


fetch_all = _fetch_all
fetch_in_chunks = _fetch_in_chunks
valid_ts_row = _valid_ts_row
valid_submission_row = _valid_submission_row


def fetch_window_data(
    supabase: Any,
    *,
    query_from: datetime,
    window_from: datetime,
    window_to: datetime,
) -> WindowDataset:
    """Read attempts, attempt_messages, case_attempts, submissions, users, cases, colleges."""
    probe_supabase(supabase)
    ds = WindowDataset()

    q_from_iso = query_from.isoformat()
    w_from_iso = window_from.isoformat()
    w_to_iso = window_to.isoformat()

    try:
        raw_attempts = _fetch_all(
            lambda: (
                supabase.table("attempts")
                .select("*")
                .gte("created_at", q_from_iso)
                .lte("created_at", w_to_iso)
                .order("created_at", desc=False)
            )
        )
        attempts: List[Dict[str, Any]] = []
        attempt_ids: Set[str] = set()
        for a in raw_attempts:
            if not _valid_ts_row(a, "attempts", ds):
                continue
            aid = str(a["id"])
            clean_a = dict(a)
            clean_a["session_state"] = normalize_session_state(a.get("session_state"))
            attempts.append(clean_a)
            attempt_ids.add(aid)

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
            if isinstance(m, dict) and m.get("attempt_id") and str(m["attempt_id"]) not in attempt_ids
        }
        if missing_aids:
            extra_attempts = _fetch_in_chunks(supabase, "attempts", "id", missing_aids, select_cols="*")
            for a in extra_attempts:
                if not _valid_ts_row(a, "attempts", ds):
                    continue
                aid = str(a["id"])
                if aid not in attempt_ids:
                    clean_a = dict(a)
                    clean_a["session_state"] = normalize_session_state(a.get("session_state"))
                    attempt_ids.add(aid)
                    attempts.append(clean_a)

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
            if not isinstance(m, dict) or not m.get("attempt_id"):
                ds.note_skip("attempt_messages", "missing attempt_id or non-dict")
                continue
            if m.get("created_at"):
                try:
                    parse_iso_utc(str(m["created_at"]))
                except Exception:  # noqa: BLE001
                    ds.note_skip("attempt_messages", "unparseable created_at")
                    continue
            aid = str(m["attempt_id"])
            messages_by_attempt.setdefault(aid, []).append(m)
        for msgs in messages_by_attempt.values():
            msgs.sort(key=lambda r: (str(r.get("created_at") or ""), str(r.get("id") or "")))

        raw_ca = _fetch_all(
            lambda: (
                supabase.table("case_attempts")
                .select("*")
                .gte("created_at", q_from_iso)
                .lte("created_at", w_to_iso)
            )
        )
        case_attempts = [r for r in raw_ca if _valid_ts_row(r, "case_attempts", ds)]

        raw_subs = _fetch_all(
            lambda: (
                supabase.table("submissions")
                .select("*")
                .gte("created_at", q_from_iso)
                .lte("created_at", w_to_iso)
            )
        )
        submissions = [r for r in raw_subs if _valid_submission_row(r, ds)]

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
        users_by_id = {str(u["id"]): u for u in users_rows if isinstance(u, dict) and u.get("id")}
        cases_by_id = {str(c["id"]): c for c in cases_rows if isinstance(c, dict) and c.get("id")}
    except SupabaseUnavailableError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SupabaseUnavailableError(f"Supabase read failed: {type(exc).__name__}: {exc}") from exc

    # Best-effort fetch of colleges purely to augment PII redaction terms.
    colleges: List[Dict[str, Any]] = []
    try:
        colleges = _fetch_all(lambda: supabase.table("colleges").select("id, name, short_name, slug"))
    except Exception:  # noqa: BLE001
        colleges = []

    ds.attempts = attempts
    ds.messages_by_attempt = messages_by_attempt
    ds.case_attempts = case_attempts
    ds.submissions = submissions
    ds.users_by_id = users_by_id
    ds.cases_by_id = cases_by_id
    ds.colleges = colleges
    return ds
