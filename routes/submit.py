"""
Submission endpoint - receives case answers from the frontend,
scores them using OpenAI, saves to Supabase, and returns feedback.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from services.supabase_client import get_supabase_client
from services.ai_scorer import score_case_answer, score_guesstimate_answer, AIScoringError
from services.badge_awarder import award_badges_for_submission


router = APIRouter()


class SubmissionRequest(BaseModel):
    """Data the frontend sends when a user submits a case answer."""
    user_id: str = Field(..., description="Supabase user ID")
    case_id: str = Field(..., description="Case being answered")
    answer_text: str = Field(..., min_length=50, description="User's answer (min 50 chars)")


class SubmissionResponse(BaseModel):
    """What the backend returns to the frontend after scoring.

    `breakdown` is a flexible {dimension: score} map: 6 dims for cases, 5 for
    guesstimates (see `rubric`). `backstop` is present only for guesstimates.
    """
    submission_id: str
    score: int = Field(..., ge=0, le=100)
    breakdown: Dict[str, int]
    strengths: List[str]
    improvements: List[str]
    summary: str
    rubric: str = "case"
    backstop: Optional[Dict[str, Any]] = None


@router.post("/submit", response_model=SubmissionResponse)
async def submit_answer(submission: SubmissionRequest) -> SubmissionResponse:
    """
    Receive a case submission, score it with AI, save to Supabase, return feedback.
    """
    supabase = get_supabase_client()

    # Step 1: Fetch the case content (AI needs the case prompt for context)
    try:
        case_result = supabase.table("cases").select("*").eq(
    "id", submission.case_id
).maybe_single().execute()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch case: {str(e)}"
        )

    if not case_result.data:
        raise HTTPException(
            status_code=404,
            detail=f"Case not found: {submission.case_id}"
        )

    case = case_result.data
    case_content = case["content"]
    case_type = case["type"]

    # Step 2: Score the answer using OpenAI.
    # Guesstimates use the dedicated 5-dim rubric + deterministic arithmetic backstop
    # (gpt-4o-mini); all other types use the standard 6-dim scorer (gpt-4o).
    try:
        if case_type == "guesstimate":
            feedback = score_guesstimate_answer(
                case_content=case_content,
                user_answer=submission.answer_text,
            )
        else:
            feedback = score_case_answer(
                case_content=case_content,
                case_type=case_type,
                user_answer=submission.answer_text,
            )
    except AIScoringError as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI scoring failed: {str(e)}"
        )

    # Step 3: Save submission to Supabase
    try:
        result = supabase.table("submissions").insert({
            "user_id": submission.user_id,
            "case_id": submission.case_id,
            "answer_text": submission.answer_text,
            "score": feedback["score"],
            "feedback_json": feedback,
        }).execute()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save submission: {str(e)}"
        )

    if not result.data or len(result.data) == 0:
        raise HTTPException(
            status_code=500,
            detail="Supabase returned empty result"
        )

    saved_submission = result.data[0]

    # Step 3.5: Record this attempt in case_attempts
    # - First-time attempts count toward leaderboards + earn badges
    # - Subsequent attempts (Lite/Pro re-attempts) are tracked but don't update leaderboard
    
    from datetime import datetime, timedelta, timezone
    IST_OFFSET = timezone(timedelta(hours=5, minutes=30))
    today_ist = datetime.now(IST_OFFSET).date().isoformat()
    
    # Check if this user has attempted this case before
    try:
        prior_res = supabase.table("case_attempts") \
            .select("id, attempt_number") \
            .eq("user_id", submission.user_id) \
            .eq("case_id", submission.case_id) \
            .order("attempt_number", desc=True) \
            .limit(1) \
            .execute()
        prior_row = (prior_res.data or [None])[0] if prior_res and prior_res.data else None
    except Exception as e:
        print(f"WARN: case_attempts lookup failed: {e}")
        prior_row = None
    
    is_first_attempt = prior_row is None
    attempt_number = 1 if is_first_attempt else (prior_row.get("attempt_number", 0) + 1)
    
    # Check if today's daily case matches this case
    counted_for_daily = False
    daily_date_val = None
    if is_first_attempt:
        try:
            sched_res = supabase.table("daily_schedule") \
                .select("case_id") \
                .eq("scheduled_date", today_ist) \
                .limit(1) \
                .execute()
            sched_row = (sched_res.data or [None])[0] if sched_res and sched_res.data else None
            if sched_row and sched_row.get("case_id") == submission.case_id:
                counted_for_daily = True
                daily_date_val = today_ist
        except Exception as e:
            print(f"WARN: daily schedule check failed: {e}")
    
    try:
        supabase.table("case_attempts").insert({
            "user_id": submission.user_id,
            "case_id": submission.case_id,
            "submission_id": saved_submission["id"],
            "attempt_number": attempt_number,
            "is_first_attempt": is_first_attempt,
            "counted_for_daily": counted_for_daily,
            "daily_date": daily_date_val,
        }).execute()
    except Exception as e:
        # Don't fail the submission — attempt tracking is bonus
        print(f"WARN: case_attempts insert failed: {e}")
    
    # IMPORTANT: only first attempts contribute to points
    if not is_first_attempt:
        # Skip the points update logic below
        # Return early with the saved submission
        return SubmissionResponse(
            submission_id=saved_submission["id"],
            score=feedback["score"],
            breakdown=feedback["breakdown"],
            strengths=feedback["strengths"],
            improvements=feedback["improvements"],
            summary=feedback["summary"],
            rubric=feedback.get("rubric", "case"),
            backstop=feedback.get("backstop"),
        )

    # Step 4: Update user's points (add the score to their cumulative total)
    try:
        # Fetch current points
        user_result = supabase.table("users").select("points").eq(
            "id", submission.user_id
        ).maybe_single().execute()

        current_points = (user_result.data or {}).get("points", 0)
        new_points = current_points + feedback["score"]

        supabase.table("users").update({
            "points": new_points
        }).eq("id", submission.user_id).execute()

        print(f"Updated user {submission.user_id} points: {current_points} -> {new_points}")
    except Exception as e:
        # Log clearly but don't fail the submission - score is already saved
        print(f"ERROR: Failed to update user points for {submission.user_id}: {type(e).__name__}: {str(e)}")

    # Step 5: Award any newly-earned badges
    try:
        await_badges = award_badges_for_submission(
            user_id=submission.user_id,
            submission_id=saved_submission["id"],
            score=feedback["score"],
            feedback_breakdown=feedback["breakdown"],
            case_id=submission.case_id,
            case_type=case_type,
            is_first_attempt=is_first_attempt,
            counted_for_daily=counted_for_daily,
        )
        if await_badges:
            print(f"Awarded badges to {submission.user_id}: {await_badges}")
    except Exception as e:
        print(f"WARN: badge awarding failed for {submission.user_id}: {e}")

    return SubmissionResponse(
        submission_id=saved_submission["id"],
        score=feedback["score"],
        breakdown=feedback["breakdown"],
        strengths=feedback["strengths"],
        improvements=feedback["improvements"],
        summary=feedback["summary"],
        rubric=feedback.get("rubric", "case"),
        backstop=feedback.get("backstop"),
    )