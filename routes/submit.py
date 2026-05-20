"""
Submission endpoint - receives case answers from the frontend,
scores them (currently with a dummy score), saves to Supabase,
and returns feedback.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
from services.supabase_client import get_supabase_client


router = APIRouter()


class SubmissionRequest(BaseModel):
    """Data the frontend sends when a user submits a case answer."""
    user_id: str = Field(..., description="Supabase user ID")
    case_id: str = Field(..., description="Case being answered")
    answer_text: str = Field(..., min_length=50, description="User's answer (min 50 chars)")


class FeedbackBreakdown(BaseModel):
    """Score breakdown across 5 dimensions, each out of 20."""
    structure: int = Field(..., ge=0, le=20)
    logic: int = Field(..., ge=0, le=20)
    data_usage: int = Field(..., ge=0, le=20)
    communication: int = Field(..., ge=0, le=20)
    creativity: int = Field(..., ge=0, le=20)


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
    Receive a case submission, save it to Supabase, and return feedback.
    Currently uses DUMMY scoring - will be replaced with OpenAI in Chunk C.
    """
    # Generate dummy feedback (Chunk C will replace this with real OpenAI call)
    dummy_breakdown = {
        "structure": 16,
        "logic": 14,
        "data_usage": 12,
        "communication": 15,
        "creativity": 15,
    }
    dummy_score = sum(dummy_breakdown.values())  # 72
    dummy_feedback = {
        "breakdown": dummy_breakdown,
        "strengths": [
            "Clear opening framework",
            "Logical segmentation of problem",
        ],
        "improvements": [
            "Quantification missing - no numbers used",
            "Did not consider cost side of equation",
        ],
        "summary": (
            "Solid structural approach but stayed qualitative throughout. "
            "Adding specific numbers and considering both revenue and cost "
            "dimensions would strengthen the answer significantly."
        ),
    }

    # Save submission to Supabase and get the real UUID back
    supabase = get_supabase_client()
    try:
        result = supabase.table("submissions").insert({
            "user_id": submission.user_id,
            "case_id": submission.case_id,
            "answer_text": submission.answer_text,
            "score": dummy_score,
            "feedback_json": dummy_feedback,
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

    return SubmissionResponse(
        submission_id=saved_submission["id"],
        score=dummy_score,
        breakdown=FeedbackBreakdown(**dummy_breakdown),
        strengths=dummy_feedback["strengths"],
        improvements=dummy_feedback["improvements"],
        summary=dummy_feedback["summary"],
    )