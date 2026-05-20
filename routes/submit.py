"""
Submission endpoint - receives case answers from the frontend,
scores them (currently with a dummy score), and returns feedback.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List


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
    Receive a case submission and return AI-generated feedback.
    Currently returns a DUMMY score - will be replaced with OpenAI in Chunk C.
    """
    # TODO Chunk C: Replace this dummy logic with real OpenAI call
    dummy_response = SubmissionResponse(
        submission_id="dummy-submission-id-12345",
        score=72,
        breakdown=FeedbackBreakdown(
            structure=16,
            logic=14,
            data_usage=12,
            communication=15,
            creativity=15,
        ),
        strengths=[
            "Clear opening framework",
            "Logical segmentation of problem",
        ],
        improvements=[
            "Quantification missing - no numbers used",
            "Did not consider cost side of equation",
        ],
        summary="Solid structural approach but stayed qualitative throughout. Adding specific numbers and considering both revenue and cost dimensions would strengthen the answer significantly.",
    )

    return dummy_response