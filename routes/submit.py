"""
Submission endpoint - receives case answers from the frontend,
scores them using OpenAI, saves to Supabase, and returns feedback.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
from services.supabase_client import get_supabase_client
from services.ai_scorer import score_case_answer, AIScoringError


router = APIRouter()


class SubmissionRequest(BaseModel):
    """Data the frontend sends when a user submits a case answer."""
    user_id: str = Field(..., description="Supabase user ID")
    case_id: str = Field(..., description="Case being answered")
    answer_text: str = Field(..., min_length=50, description="User's answer (min 50 chars)")


class FeedbackBreakdown(BaseModel):
    """Score breakdown across 6 dimensions."""
    structure: int = Field(..., ge=0, le=25)
    quantitative: int = Field(..., ge=0, le=20)
    synthesis: int = Field(..., ge=0, le=20)
    business_judgment: int = Field(..., ge=0, le=15)
    creativity: int = Field(..., ge=0, le=10)
    presence: int = Field(..., ge=0, le=10)


class SubmissionResponse(BaseModel):
    """What the backend returns to the frontend after scoring."""
    submission_id: str
    score: int = Field(..., ge=0, le=100)
    breakdown: FeedbackBreakdown
    strengths: List[str]
    improvements: List[str]
    summary: str


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

    # Step 2: Score the answer using OpenAI
    try:
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

    return SubmissionResponse(
        submission_id=saved_submission["id"],
        score=feedback["score"],
        breakdown=FeedbackBreakdown(**feedback["breakdown"]),
        strengths=feedback["strengths"],
        improvements=feedback["improvements"],
        summary=feedback["summary"],
    )