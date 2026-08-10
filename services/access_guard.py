"""
Tier / quota guard for case + guesstimate submissions and for gated features.

Policy (decided 2026-06-04, see PROJECT_BRAIN §9.38):
- pro            -> unlimited everything
- daily case/gst -> attemptable by ALL tiers (re-attempt rules still apply)
- free           -> ONLY today's daily case + daily guesstimate, 0 re-attempts,
                    GD briefs LOCKED
- lite           -> daily pair + up to 2 extra cases AND 2 extra guesstimates per
                    IST day, unlimited re-attempts (re-attempts don't consume +2),
                    GD briefs unlocked

Quota resets at IST midnight. Buckets: 'guesstimate' vs 'case' (non-guesstimate).
This is the AUTHORITATIVE gate — the frontend mirror (lib/access.ts) is UX only.
"""

from datetime import datetime, timedelta, timezone
from fastapi import HTTPException

IST_OFFSET = timezone(timedelta(hours=5, minutes=30))
LITE_DAILY_EXTRA = {"case": 2, "guesstimate": 2}
_TIER_RANK = {"free": 0, "lite": 1, "pro": 2}


def _today_ist() -> str:
    return datetime.now(IST_OFFSET).date().isoformat()


def _effective_tier_from_row(row: dict) -> str:
    tier = (row or {}).get("subscription_tier") or "free"
    if tier == "free":
        return "free"
    expires = (row or {}).get("subscription_expires_at")
    if expires:
        try:
            exp = datetime.fromisoformat(str(expires).replace("Z", "+00:00"))
            if exp < datetime.now(timezone.utc):
                return "free"
        except Exception:
            pass
    return tier


def effective_tier_and_guest(supabase, user_id: str) -> tuple[str, bool]:
    """Tier + guest flag from ONE users read.

    DEPLOY-ORDER SAFETY: `is_guest` does not exist until migration 0045 has been
    run. Selecting a missing column makes PostgREST return 400 and supabase-py
    raise — and this function sits under `assert_can_attempt`, so an unguarded
    select would turn "backend deployed a few minutes before the migration" into
    "nobody can start a case". The fallback re-reads without the column and
    reports is_guest=False, which is exactly the pre-0045 behaviour.

    Deliberately NOT a bare `except Exception` around the whole body: a genuine
    outage must still surface. Only the missing-column case degrades.
    """
    try:
        u = supabase.table("users").select(
            "subscription_tier, subscription_expires_at, is_guest"
        ).eq("id", user_id).maybe_single().execute()
        data = (u.data or {}) if u else {}
        return _effective_tier_from_row(data), bool(data.get("is_guest"))
    except Exception as exc:  # noqa: BLE001 — narrowed by the re-raise below
        if "is_guest" not in str(exc):
            raise
        u = supabase.table("users").select(
            "subscription_tier, subscription_expires_at"
        ).eq("id", user_id).maybe_single().execute()
        data = (u.data or {}) if u else {}
        return _effective_tier_from_row(data), False


def effective_tier(supabase, user_id: str) -> str:
    tier, _ = effective_tier_and_guest(supabase, user_id)
    return tier


def assert_can_submit(supabase, user_id: str) -> None:
    """Raise 403 if an anonymous guest tries to submit for scoring.

    GUEST MODE (0045). This is THE wall. A guest works the whole case for free —
    clarifications, structure, arithmetic, recommendation — and is asked for an
    account at the one moment it buys them something: the score.

    It must be enforced here and not only in the UI. `submitAttempt` is a plain
    authenticated call and an anonymous JWT is a perfectly valid one, so a guest
    who skips the dialog would otherwise get scoring — the most expensive call
    in the product — for free and unbounded.

    Conversion keeps the SAME auth.users row, so by the time this passes, the
    attempt being submitted is already theirs. Nothing is re-parented.
    """
    _, is_guest = effective_tier_and_guest(supabase, user_id)
    if is_guest:
        raise HTTPException(
            status_code=403,
            detail="Create a free account to see your score. Your answer is saved.",
        )


def assert_tier_at_least(supabase, user_id: str, minimum: str) -> None:
    """Raise 403 if the user's effective tier is below `minimum`."""
    tier = effective_tier(supabase, user_id)
    if _TIER_RANK.get(tier, 0) < _TIER_RANK.get(minimum, 0):
        raise HTTPException(
            status_code=403,
            detail=f"This feature requires the {minimum.capitalize()} plan. Upgrade to unlock it.",
        )


def assert_can_attempt(supabase, user_id: str, case: dict) -> None:
    """Raise HTTPException(403) if this user may NOT attempt this case right now."""
    case_id = case["id"]
    case_type = case.get("type", "")
    bucket = "guesstimate" if case_type == "guesstimate" else "case"
    today = _today_ist()

    # ONE users read for both facts. Do not re-query for is_guest further down:
    # this function is on the hot path of every attempt start, and the guest
    # check was originally written as a second effective_tier_and_guest() call,
    # which silently doubled the round-trip on the non-daily branch.
    tier, is_guest = effective_tier_and_guest(supabase, user_id)
    if tier == "pro":
        return

    # today's daily schedule (case_id + guesstimate case id)
    sched = supabase.table("daily_schedule").select(
        "case_id, guesstimate_code"
    ).eq("scheduled_date", today).limit(1).execute()
    srow = (sched.data or [None])[0] if sched and sched.data else None
    daily_ids = set()
    if srow:
        if srow.get("case_id"):
            daily_ids.add(srow["case_id"])
        if srow.get("guesstimate_code"):
            daily_ids.add(srow["guesstimate_code"])
    is_daily = case_id in daily_ids

    # first attempt vs re-attempt
    prior = supabase.table("case_attempts").select("id").eq(
        "user_id", user_id
    ).eq("case_id", case_id).limit(1).execute()
    is_first_attempt = not (prior and prior.data)

    # Guests get today's daily pair and nothing else. A guest is a free-tier
    # user whose one-time non-daily extras are zero — the daily branch below
    # is reached unchanged, so no separate guest quota exists anywhere.
    if not is_daily and is_guest:
        raise HTTPException(
            status_code=403,
            detail="Sign up free to practise beyond today's case and guesstimate.",
        )

    if is_daily:
        # GUEST MODE (0045): unlimited re-attempts at the daily pair. The wall
        # is at SUBMIT (see assert_can_submit), so a guest has never been
        # scored and the free-tier one-attempt rule has nothing to protect.
        # Blocking here would strand a half-finished conversation instead.
        if is_guest:
            return
        if tier == "free" and not is_first_attempt:
            raise HTTPException(
                status_code=403,
                detail="Free tier allows one attempt per case. Upgrade to Lite for unlimited re-attempts.",
            )
        return  # the daily pair is allowed for everyone

    # ---- non-daily case/guesstimate ----
    if tier == "free":
        rows = supabase.table("case_attempts").select(
            "case_id, counted_for_daily"
        ).eq("user_id", user_id).eq("is_first_attempt", True).execute()
        
        candidate_ids = [
            r["case_id"] for r in ((rows.data or []) if rows else [])
            if not r.get("counted_for_daily") and r.get("case_id") not in daily_ids
        ]
        used = 0
        if candidate_ids:
            types = supabase.table("cases").select("id, type").in_("id", candidate_ids).execute()
            for t in ((types.data or []) if types else []):
                tb = "guesstimate" if t.get("type") == "guesstimate" else "case"
                if tb == bucket:
                    used += 1

        # LinkedIn follow perk (2026-07): a one-time claim permanently raises
        # the free bank by +1 case and +1 guesstimate. Mirrors lib/access.ts
        # and LINKEDIN_FOLLOW_PERK in lib/constants.ts — keep in sync.
        perk = supabase.table("users").select(
            "linkedin_follow_claimed_at"
        ).eq("id", user_id).maybe_single().execute()
        claimed = bool((((perk.data or {}) if perk else {})).get("linkedin_follow_claimed_at"))
        cap = 1 + (1 if claimed else 0)

        if used >= cap:
            label = "guesstimate" if bucket == "guesstimate" else "case"
            hint = (
                "Upgrade to Lite to practise the full bank."
                if claimed
                else "Follow MECE on LinkedIn to unlock one more, or upgrade to Lite for the full bank."
            )
            raise HTTPException(
                status_code=403,
                detail=f"You've used your free bank {label}s. {hint}",
            )
        return

    # tier == "lite"
    if not is_first_attempt:
        return  # unlimited re-attempts, don't consume the +2

    start_iso = f"{today}T00:00:00+05:30"
    rows = supabase.table("case_attempts").select(
        "case_id, counted_for_daily, created_at"
    ).eq("user_id", user_id).eq("is_first_attempt", True).gte(
        "created_at", start_iso
    ).execute()
    candidate_ids = [
        r["case_id"] for r in ((rows.data or []) if rows else [])
        if not r.get("counted_for_daily") and r.get("case_id") not in daily_ids
    ]
    used = 0
    if candidate_ids:
        types = supabase.table("cases").select("id, type").in_("id", candidate_ids).execute()
        for t in ((types.data or []) if types else []):
            tb = "guesstimate" if t.get("type") == "guesstimate" else "case"
            if tb == bucket:
                used += 1

    if used >= LITE_DAILY_EXTRA[bucket]:
        label = "guesstimates" if bucket == "guesstimate" else "cases"
        raise HTTPException(
            status_code=403,
            detail=f"Lite tier allows 2 extra {label} per day beyond the daily ones. "
                   f"Upgrade to Pro for unlimited practice.",
        )
    return
