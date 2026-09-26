"""
Markets — India vs International (US + Europe). Backend twin of lib/market.ts.

VOCABULARY (must match the frontend exactly)
    users.market   'IN' | 'US' | 'EU'   (NULL = not stamped yet → India)
    cases.market   'IN' | 'US'          (the bank a case belongs to)
    content market 'IN' | 'US'          (EU accounts practise the US bank)

An account practises ONLY its own content market's bank. This module is the
authoritative backend check for that (assert_market_access) plus the per-market
"today" and daily-pair lookups the tier gate and the submit path need.

DEPLOY-ORDER SAFETY: migration 0070 adds users.market, cases.market and
market_daily_schedule. Until it has run, every read here degrades to India —
which is exactly what every pre-0070 row is — so deploying this before the
migration changes nothing for anyone.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Set

from fastapi import HTTPException

IST = timezone(timedelta(hours=5, minutes=30))

try:  # zoneinfo needs a tz database; tzdata (requirements.txt) guarantees one.
    from zoneinfo import ZoneInfo
    _US_EASTERN = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover — last-resort fixed offset (EST)
    _US_EASTERN = timezone(timedelta(hours=-5))

VALID_MARKETS = {"IN", "US", "EU"}


def normalize_market(value) -> str:
    v = (value or "").strip().upper() if isinstance(value, str) else ""
    return v if v in VALID_MARKETS else "IN"


def content_market_of(value) -> str:
    return "US" if normalize_market(value) in ("US", "EU") else "IN"


def case_market(case: Optional[dict]) -> str:
    """The bank a case row belongs to. Absent / NULL / pre-0070 → India."""
    v = ((case or {}).get("market") or "IN")
    return "US" if str(v).upper() == "US" else "IN"


def _missing_market_column(exc: Exception) -> bool:
    s = str(exc).lower()
    return "market" in s and ("does not exist" in s or "schema cache" in s or "42703" in s or "pgrst204" in s)


def user_market(supabase, user_id: str) -> str:
    """The account's market ('IN' | 'US' | 'EU').

    NULL (not stamped yet) and a missing column (pre-0070) both read as 'IN'.
    A genuine outage re-raises: the caller is a gate, and a gate that silently
    guesses is worse than one that fails loudly.
    """
    try:
        r = supabase.table("users").select("market").eq("id", user_id).maybe_single().execute()
        data = (r.data or {}) if r else {}
        return normalize_market(data.get("market"))
    except Exception as exc:  # noqa: BLE001 — narrowed below
        if _missing_market_column(exc):
            return "IN"
        raise


def user_content_market(supabase, user_id: str) -> str:
    return content_market_of(user_market(supabase, user_id))


# ── Calendar days per market ────────────────────────────────────────────────

def market_now(content: str) -> datetime:
    return datetime.now(_US_EASTERN if content == "US" else IST)


def market_today(content: str) -> str:
    """YYYY-MM-DD 'today' for a content market (IST for India, US Eastern for US)."""
    return market_now(content).date().isoformat()


def market_day_start_iso(content: str, date_str: Optional[str] = None) -> str:
    """ISO instant of local midnight for the market's day (DST-correct for US)."""
    tz = _US_EASTERN if content == "US" else IST
    d = datetime.fromisoformat(date_str) if date_str else market_now(content)
    start = datetime(d.year, d.month, d.day, tzinfo=tz)
    return start.isoformat()


# ── The daily pair per market ───────────────────────────────────────────────

def intl_daily_ids(supabase, market: str = "US", date_str: Optional[str] = None, exact: bool = True) -> Set[str]:
    """Case ids of the international daily pair.

    exact=True  → only the row for `date_str` (default: today in the market).
    exact=False → the most recent row on/before that date (the frontend's
                  fallback when the cron has not run yet).
    Never raises: a missing table (pre-0070) or any read error → empty set.
    """
    day = date_str or market_today("US")
    try:
        q = supabase.table("market_daily_schedule").select("case_id, guesstimate_id, scheduled_date").eq("market", market)
        if exact:
            q = q.eq("scheduled_date", day)
        else:
            q = q.lte("scheduled_date", day).order("scheduled_date", desc=True)
        res = q.limit(1).execute()
        row = (res.data or [None])[0] if res and res.data else None
    except Exception:
        return set()
    ids: Set[str] = set()
    if row:
        if row.get("case_id"):
            ids.add(row["case_id"])
        if row.get("guesstimate_id"):
            ids.add(row["guesstimate_id"])
    return ids


# ── The gate ────────────────────────────────────────────────────────────────

def _is_admin(supabase, user_id: str) -> bool:
    """users.is_admin (a guarded column). Any read error → False (fail closed)."""
    try:
        r = supabase.table("users").select("is_admin").eq("id", user_id).maybe_single().execute()
        data = (r.data or {}) if r else {}
        return bool(data.get("is_admin"))
    except Exception:  # noqa: BLE001
        return False


def assert_market_access(supabase, user_id: str, case: dict) -> str:
    """Raise 403 unless the case belongs to the user's content market.

    Returns the user's content market so callers do not read it twice.
    The owner of a private (copilot-generated) case may always open it.
    """
    content = user_content_market(supabase, user_id)
    if case.get("owner_id") and case.get("owner_id") == user_id:
        return content
    if case_market(case) != content:
        # Admins may open either bank (the admin "view as US" preview). Read
        # only on this rare mismatch path, so normal attempts pay nothing. The
        # case is then gated by ITS market's rules (US quota / US daily).
        if _is_admin(supabase, user_id):
            return case_market(case)
        raise HTTPException(
            status_code=403,
            detail="This case isn't available in your region. Open Practice to see your case bank.",
        )
    return content


# ── Interviewer / scorer register ───────────────────────────────────────────

US_MARKET_NOTE = (
    "\n\n[Interviewer-only context, never shown to or quoted to the candidate] "
    "MARKET: UNITED STATES. The candidate is recruiting for US consulting, finance and "
    "strategy roles (MBB, Tier-2 firms, Big 4 strategy arms, corporate strategy; MBA summer "
    "internships and full-time offers). This OVERRIDES any Indian-register guidance above: "
    "speak natural American English; quote money in US dollars ($ with thousand / million / "
    "billion, never Rs, lakh or crore); use US geography, US companies and US data anchors "
    "(about 335 million people, about 130 million households, 50 states); use miles, gallons "
    "and degrees Fahrenheit where natural. Any feedback, scoring commentary or worked figures "
    "must use the same US conventions."
)


def llm_case_content(case: Optional[dict]) -> str:
    """The case text the MODELS see (interviewer, scorer).

    India cases: the stored content, byte-for-byte (unchanged behaviour).
    US cases: the same content plus an interviewer-only market note, so every
    model path — plain, adaptive, realtime, scoring — switches to the US
    register without touching a single system prompt shared with India.
    The candidate's screen renders cases.content directly, never this string.
    """
    content = (case or {}).get("content") or ""
    if case_market(case) == "US":
        return content + US_MARKET_NOTE
    return content
