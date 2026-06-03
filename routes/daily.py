"""
Daily content routes.

GET /daily/today           — today's daily case + guesstimate + star headline
GET /daily/leaderboard     — top scorers on today's daily case
"""

from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from services.supabase_client import get_supabase_client

router = APIRouter(prefix="/daily", tags=["daily"])

IST_OFFSET = timezone(timedelta(hours=5, minutes=30))


def today_ist_date() -> str:
    """Returns today's date in IST as YYYY-MM-DD."""
    return datetime.now(IST_OFFSET).date().isoformat()


class TodayCaseInfo(BaseModel):
    id: str
    title: str
    type: str
    difficulty: str


class TodayHeadlineInfo(BaseModel):
    id: str
    title: str
    source_name: str
    thumbnail_url: Optional[str]


class TodayResponse(BaseModel):
    date: str  # YYYY-MM-DD in IST
    case: Optional[TodayCaseInfo]
    # The daily guesstimate is a real `cases` row (type='guesstimate'); this is the
    # attemptable object the frontend links to. `guesstimate_code` is kept for
    # back-compat and now carries the same id.
    guesstimate: Optional[TodayCaseInfo] = None
    guesstimate_code: Optional[str]
    guesstimate_title: Optional[str] = None
    brief: Optional[TodayHeadlineInfo]
    attempted_by_current_user: bool = False  # filled by frontend or auth-aware version later


class DailyLeaderboardEntry(BaseModel):
    user_id: str
    name: Optional[str]
    avatar_url: Optional[str]
    score: int
    submission_id: str
    submitted_at: str
    rank: int


class DailyLeaderboardResponse(BaseModel):
    date: str
    case_id: Optional[str]
    case_title: Optional[str]
    entries: List[DailyLeaderboardEntry]
    total_attempts: int


@router.get("/today", response_model=TodayResponse)
async def get_today() -> TodayResponse:
    """
    Return today's daily content. NEVER 404s — if no schedule, returns empty fields.
    """
    supabase = get_supabase_client()
    today = today_ist_date()
    
    # Fetch today's schedule
    try:
        sched_res = supabase.table("daily_schedule") \
            .select("*") \
            .eq("scheduled_date", today) \
            .limit(1) \
            .execute()
    except Exception as e:
        raise HTTPException(500, f"Schedule fetch failed: {e}")
    
    sched_row = (sched_res.data or [None])[0] if sched_res and sched_res.data else None
    
    case_info = None
    if sched_row and sched_row.get("case_id"):
        try:
            case_res = supabase.table("cases") \
                .select("id, title, type, difficulty") \
                .eq("id", sched_row["case_id"]) \
                .limit(1) \
                .execute()
            case_row = (case_res.data or [None])[0] if case_res and case_res.data else None
            if case_row:
                case_info = TodayCaseInfo(**case_row)
        except Exception:
            pass  # fool-proof: missing case → just null, no crash
    
    # Star headline of the day (used as today's brief)
    brief_info = None
    try:
        star_res = supabase.table("news_headlines") \
            .select("id, title, source_name, thumbnail_url") \
            .eq("is_star", True) \
            .order("published_at", desc=True) \
            .limit(1) \
            .execute()
        star_row = (star_res.data or [None])[0] if star_res and star_res.data else None
        if star_row:
            brief_info = TodayHeadlineInfo(**star_row)
    except Exception:
        pass
    
    # The daily guesstimate is a real `cases` row; its id lives in guesstimate_code.
    guess_obj = None
    guess_title = None
    guess_code = sched_row.get("guesstimate_code") if sched_row else None
    if guess_code:
        try:
            guess_res = supabase.table("cases") \
                .select("id, title, type, difficulty") \
                .eq("id", guess_code) \
                .limit(1) \
                .execute()
            guess_row = (guess_res.data or [None])[0] if guess_res and guess_res.data else None
            if guess_row:
                guess_obj = TodayCaseInfo(**guess_row)
                guess_title = guess_row.get("title")
        except Exception:
            pass  # fool-proof: unresolved guesstimate → just null, no crash

    return TodayResponse(
        date=today,
        case=case_info,
        guesstimate=guess_obj,
        guesstimate_code=guess_code,
        guesstimate_title=guess_title,
        brief=brief_info,
    )


@router.get("/leaderboard", response_model=DailyLeaderboardResponse)
async def get_daily_leaderboard() -> DailyLeaderboardResponse:
    """
    Return top scorers on today's daily case.
    Only first attempts (counted_for_daily=true) are included.
    """
    supabase = get_supabase_client()
    today = today_ist_date()
    
    # Fetch today's scheduled case
    try:
        sched_res = supabase.table("daily_schedule") \
            .select("case_id") \
            .eq("scheduled_date", today) \
            .limit(1) \
            .execute()
    except Exception as e:
        raise HTTPException(500, f"Schedule fetch failed: {e}")
    
    sched_row = (sched_res.data or [None])[0] if sched_res and sched_res.data else None
    
    if not sched_row or not sched_row.get("case_id"):
        return DailyLeaderboardResponse(
            date=today, case_id=None, case_title=None, entries=[], total_attempts=0,
        )
    
    case_id = sched_row["case_id"]
    
    # Fetch case title
    case_res = supabase.table("cases") \
        .select("title") \
        .eq("id", case_id) \
        .limit(1) \
        .execute()
    case_row = (case_res.data or [None])[0] if case_res and case_res.data else None
    case_title = case_row.get("title") if case_row else None
    
    # Fetch daily attempts (joined with submission scores + user profile)
    # case_attempts.counted_for_daily=true ∧ daily_date=today → these are eligible.
    # We join with submissions to get the score.
    try:
        attempts_res = supabase.table("case_attempts") \
            .select("user_id, submission_id, created_at, submissions(score), users(name, avatar_url)") \
            .eq("daily_date", today) \
            .eq("counted_for_daily", True) \
            .execute()
    except Exception as e:
        # Foreign-table syntax can fail in some Supabase setups. Fallback to manual join.
        attempts_res = supabase.table("case_attempts") \
            .select("user_id, submission_id, created_at") \
            .eq("daily_date", today) \
            .eq("counted_for_daily", True) \
            .execute()
    
    rows = attempts_res.data or []
    
    # Enrich each row: if join didn't populate, fetch score + user manually
    entries = []
    for row in rows:
        user_id = row["user_id"]
        submission_id = row["submission_id"]
        submitted_at = row["created_at"]
        
        # Score
        score = None
        if "submissions" in row and row["submissions"]:
            score = row["submissions"].get("score")
        if score is None:
            sub_res = supabase.table("submissions") \
                .select("score") \
                .eq("id", submission_id) \
                .limit(1) \
                .execute()
            sub_row = (sub_res.data or [None])[0] if sub_res and sub_res.data else None
            score = (sub_row or {}).get("score", 0)
        
        # User profile
        name = None
        avatar_url = None
        if "users" in row and row["users"]:
            name = row["users"].get("name")
            avatar_url = row["users"].get("avatar_url")
        if name is None:
            u_res = supabase.table("users") \
                .select("name, avatar_url") \
                .eq("id", user_id) \
                .limit(1) \
                .execute()
            u_row = (u_res.data or [None])[0] if u_res and u_res.data else None
            if u_row:
                name = u_row.get("name")
                avatar_url = u_row.get("avatar_url")
        
        entries.append({
            "user_id": user_id,
            "name": name,
            "avatar_url": avatar_url,
            "score": int(score or 0),
            "submission_id": submission_id,
            "submitted_at": submitted_at,
        })
    
    # Sort by score descending, then by submitted_at ascending (tiebreaker: who finished first)
    entries.sort(key=lambda e: (-e["score"], e["submitted_at"]))
    
    # Assign ranks
    ranked = []
    for idx, e in enumerate(entries[:20]):  # cap at top 20 for response size
        ranked.append(DailyLeaderboardEntry(
            user_id=e["user_id"],
            name=e["name"],
            avatar_url=e["avatar_url"],
            score=e["score"],
            submission_id=e["submission_id"],
            submitted_at=e["submitted_at"],
            rank=idx + 1,
        ))
    
    return DailyLeaderboardResponse(
        date=today,
        case_id=case_id,
        case_title=case_title,
        entries=ranked,
        total_attempts=len(entries),
    )
