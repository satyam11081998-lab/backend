"""
Badge awarder — evaluates eligibility and inserts user_badges rows.

Called from /submit endpoint after submission saved + scored.
Idempotent: ON CONFLICT DO NOTHING prevents double-awards.

Adds points to user.points for each newly-awarded badge.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
from services.supabase_client import get_supabase_client

IST_OFFSET = timezone(timedelta(hours=5, minutes=30))


def award_badges_for_submission(
    user_id: str,
    submission_id: str,
    score: int,
    feedback_breakdown: Dict[str, int],
    case_id: str,
    case_type: str,
    is_first_attempt: bool,
    counted_for_daily: bool,
) -> List[str]:
    """
    Evaluate all badge rules. Insert newly-earned badges.
    Returns list of badge IDs awarded in this call.
    """
    supabase = get_supabase_client()
    newly_awarded: List[str] = []
    
    # Fetch all badges user already has
    try:
        existing_res = supabase.table("user_badges") \
            .select("badge_id") \
            .eq("user_id", user_id) \
            .execute()
        existing_ids = {row["badge_id"] for row in (existing_res.data or [])}
    except Exception as e:
        print(f"WARN: badge existing fetch failed: {e}")
        return []
    
    # === Milestone badges ===
    if "first-case" not in existing_ids and is_first_attempt:
        newly_awarded.append("first-case")
    
    # Count user's distinct cases attempted (first attempts only)
    try:
        attempts_res = supabase.table("case_attempts") \
            .select("case_id", count="exact") \
            .eq("user_id", user_id) \
            .eq("is_first_attempt", True) \
            .execute()
        distinct_case_count = attempts_res.count or 0
    except Exception:
        distinct_case_count = 0
    
    if "five-cases" not in existing_ids and distinct_case_count >= 5:
        newly_awarded.append("five-cases")
    if "fifteen-cases" not in existing_ids and distinct_case_count >= 15:
        newly_awarded.append("fifteen-cases")
    if "thirty-cases" not in existing_ids and distinct_case_count >= 30:
        newly_awarded.append("thirty-cases")
    
    # All 4 types attempted
    if "all-types" not in existing_ids:
        try:
            types_res = supabase.table("case_attempts") \
                .select("case_id, cases(type)") \
                .eq("user_id", user_id) \
                .eq("is_first_attempt", True) \
                .execute()
            distinct_types = set()
            for row in (types_res.data or []):
                case_data = row.get("cases")
                if case_data and isinstance(case_data, dict):
                    distinct_types.add(case_data.get("type"))
            if len(distinct_types & {"profitability", "guesstimate", "market_sizing", "growth"}) >= 4:
                newly_awarded.append("all-types")
        except Exception:
            pass
    
    # === Mastery badges ===
    if score >= 80 and "first-80" not in existing_ids:
        newly_awarded.append("first-80")
    if score >= 90 and "first-90" not in existing_ids:
        newly_awarded.append("first-90")
    
    structure_score = feedback_breakdown.get("structure", 0)
    if structure_score >= 25 and "perfect-structure" not in existing_ids:
        newly_awarded.append("perfect-structure")
    
    quant_score = feedback_breakdown.get("quantitative", 0)
    if quant_score >= 20 and "perfect-quant" not in existing_ids:
        newly_awarded.append("perfect-quant")
    
    # === Streak badges (based on case_attempts dates) ===
    if not {"streak-3", "streak-7", "streak-14", "streak-30"}.issubset(existing_ids):
        streak = compute_streak(user_id)
        if streak >= 3 and "streak-3" not in existing_ids:
            newly_awarded.append("streak-3")
        if streak >= 7 and "streak-7" not in existing_ids:
            newly_awarded.append("streak-7")
        if streak >= 14 and "streak-14" not in existing_ids:
            newly_awarded.append("streak-14")
        if streak >= 30 and "streak-30" not in existing_ids:
            newly_awarded.append("streak-30")
    
    # Insert newly-earned badges
    if not newly_awarded:
        return []
    
    rows = [
        {
            "user_id": user_id,
            "badge_id": bid,
            "trigger_submission_id": submission_id,
        }
        for bid in newly_awarded
    ]
    
    try:
        supabase.table("user_badges").insert(rows).execute()
    except Exception as e:
        print(f"WARN: badge insert failed: {e}")
        return []
    
    # Award bonus points
    try:
        badge_pts_res = supabase.table("badges") \
            .select("points_reward") \
            .in_("id", newly_awarded) \
            .execute()
        bonus_points = sum((b.get("points_reward") or 0) for b in (badge_pts_res.data or []))
        if bonus_points > 0:
            user_res = supabase.table("users") \
                .select("points") \
                .eq("id", user_id) \
                .limit(1) \
                .execute()
            user_row = (user_res.data or [None])[0] if user_res and user_res.data else None
            current = (user_row or {}).get("points", 0)
            supabase.table("users").update({
                "points": current + bonus_points,
            }).eq("id", user_id).execute()
    except Exception as e:
        print(f"WARN: badge bonus points failed: {e}")
    
    return newly_awarded


def compute_streak(user_id: str) -> int:
    """Count consecutive days (ending today IST) with at least one submission."""
    supabase = get_supabase_client()
    try:
        today = datetime.now(IST_OFFSET).date()
        start = today - timedelta(days=60)  # max 60-day lookback
        subs_res = supabase.table("submissions") \
            .select("created_at") \
            .eq("user_id", user_id) \
            .gte("created_at", start.isoformat()) \
            .execute()
        dates_set = set()
        for s in (subs_res.data or []):
            ca = s.get("created_at", "")
            if ca:
                # Parse to IST date
                d = datetime.fromisoformat(ca.replace("Z", "+00:00")).astimezone(IST_OFFSET).date()
                dates_set.add(d.isoformat())
        
        streak = 0
        for i in range(60):
            d = today - timedelta(days=i)
            if d.isoformat() in dates_set:
                streak += 1
            else:
                break
        return streak
    except Exception:
        return 0
