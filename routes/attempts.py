"""
Conversational case-interview routes.

Replaces the old POST /submit single-answer flow with a session-based
workspace. Endpoints:

  POST   /attempts                     -> start a session (gates by tier/quota)
  GET    /attempts/{id}                -> fetch case + messages
  POST   /attempts/{id}/messages       -> append user msg, stream AI reply (SSE)
  POST   /attempts/{id}/uploads        -> attach an image / document to the thread
  POST   /attempts/{id}/submit         -> finalize, score the transcript, save

All endpoints derive user_id from the verified Supabase JWT — never trust
client-supplied ids. The service-role Supabase client bypasses RLS.
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Header, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from services.supabase_client import get_supabase_client
from services.auth import get_verified_user_id
from services.access_guard import assert_can_attempt, effective_tier
from services.rate_limit import check_rate_limit
from services.limits import MESSAGE_MAX_CHARS, RECOMMENDATION_MAX_CHARS
from services.interview_engine import (
    stream_interviewer_reply,
    complete_interviewer_reply,
    score_conversation,
    count_clarifications,
    InterviewEngineError,
)
from services.badge_awarder import award_badges_for_submission

router = APIRouter(prefix="/attempts", tags=["attempts"])


# -----------------------------------------------------------------------------
# Tier -> clarification quota.
# Free/Lite = 5, Pro = 15. The spec says "Lite Users: Max 5 / Pro Users: Max 15".
# Free users only attempt the daily anyway; the same 5-cap reads as fair.
# -----------------------------------------------------------------------------

# Tier -> clarification (AI hint) quota. Free = 0 to match the pricing page
# ("no AI hints") and TIER_LIMITS.free.maxHintQuestions. Free users can still
# post structure and get scored on the daily case; they just can't spend AI
# clarification questions. Lite = 5, Pro = 15.
CLARIFICATION_QUOTA = {"free": 0, "lite": 5, "pro": 15}

# Soft cap on total messages per attempt — prevents runaway sessions.
MAX_MESSAGES_PER_ATTEMPT = 200

# Upload caps (matching schema notes).
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_DOC_BYTES = 16 * 1024 * 1024
ALLOWED_MIME_PREFIXES = ("image/",)
ALLOWED_MIME_EXACT = (
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
)


# =============================================================================
# Pydantic schemas
# =============================================================================

class StartAttemptRequest(BaseModel):
    case_id: str


class AttemptSummary(BaseModel):
    attempt_id: str
    case_id: str
    tier: str
    clarification_quota: int
    clarification_used: int
    clarification_remaining: int
    status: str


class MessageOut(BaseModel):
    id: str
    role: str
    kind: str
    content: Optional[str]
    file_id: Optional[str]
    is_clarification: bool
    created_at: str


class AttemptDetail(BaseModel):
    attempt: AttemptSummary
    case: Dict[str, Any]
    messages: List[MessageOut]


class PostMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=MESSAGE_MAX_CHARS)
    kind: str = Field("text", description="text | voice | image | file")


class SubmitRequest(BaseModel):
    final_recommendation: str = Field(..., min_length=20, max_length=RECOMMENDATION_MAX_CHARS)


class SubmitResponse(BaseModel):
    submission_id: str
    attempt_id: str
    score: int
    breakdown: Dict[str, int]
    strengths: List[str]
    improvements: List[str]
    summary: str
    rubric: str = "case"


# =============================================================================
# Helpers
# =============================================================================

def _load_attempt(supabase, attempt_id: str, user_id: str) -> dict:
    row = (
        supabase.table("attempts")
        .select("*")
        .eq("id", attempt_id)
        .maybe_single()
        .execute()
    )
    if not row.data:
        raise HTTPException(status_code=404, detail="Attempt not found")
    if row.data["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Not your attempt")
    return row.data


def _load_case(supabase, case_id: str) -> dict:
    row = supabase.table("cases").select("*").eq("id", case_id).maybe_single().execute()
    if not row.data:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")
    if row.data.get("is_active") is False:
        raise HTTPException(status_code=404, detail="This case is no longer available.")
    return row.data


def _fetch_transcript(supabase, attempt_id: str) -> List[Dict[str, str]]:
    rows = (
        supabase.table("attempt_messages")
        .select("role, kind, content, created_at")
        .eq("attempt_id", attempt_id)
        .order("created_at", desc=False)
        .execute()
    )
    return [
        {"role": r["role"], "kind": r["kind"], "content": r.get("content") or ""}
        for r in (rows.data or [])
        if r.get("content")
    ]


# =============================================================================
# POST /attempts  — start a session
# =============================================================================

@router.post("", response_model=AttemptSummary)
async def start_attempt(
    body: StartAttemptRequest,
    authorization: Optional[str] = Header(default=None),
) -> AttemptSummary:
    supabase = get_supabase_client()
    user_id = get_verified_user_id(supabase, authorization)
    check_rate_limit(f"attempts:start:{user_id}", max_calls=20, window_seconds=60)

    case = _load_case(supabase, body.case_id)
    # Tier/quota gate — same logic as the legacy /submit.
    assert_can_attempt(supabase, user_id, case)

    tier = effective_tier(supabase, user_id)
    quota = CLARIFICATION_QUOTA.get(tier, 5)

    # Resume any active attempt for this user+case rather than spawning a new one,
    # so a refresh doesn't lose state.
    existing = (
        supabase.table("attempts")
        .select("*")
        .eq("user_id", user_id)
        .eq("case_id", body.case_id)
        .eq("status", "active")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if existing.data:
        a = existing.data[0]
        return AttemptSummary(
            attempt_id=a["id"],
            case_id=a["case_id"],
            tier=a["tier_at_start"],
            clarification_quota=a["clarification_quota"],
            clarification_used=a["clarification_used"],
            clarification_remaining=max(0, a["clarification_quota"] - a["clarification_used"]),
            status=a["status"],
        )

    inserted = (
        supabase.table("attempts")
        .insert(
            {
                "user_id": user_id,
                "case_id": body.case_id,
                "tier_at_start": tier,
                "clarification_quota": quota,
                "clarification_used": 0,
                "status": "active",
            }
        )
        .execute()
    )
    if not inserted.data:
        raise HTTPException(status_code=500, detail="Failed to create attempt")

    attempt_id = inserted.data[0]["id"]

    # Seed the conversation with the case prompt as a system note,
    # so the client can render the same "interviewer just briefed me" feel.
    supabase.table("attempt_messages").insert(
        {
            "attempt_id": attempt_id,
            "role": "system",
            "kind": "system_note",
            "content": f"Case ready: {case.get('title')}. Ask any clarifying questions before structuring.",
            "is_clarification": False,
        }
    ).execute()

    return AttemptSummary(
        attempt_id=attempt_id,
        case_id=body.case_id,
        tier=tier,
        clarification_quota=quota,
        clarification_used=0,
        clarification_remaining=quota,
        status="active",
    )


# =============================================================================
# GET /attempts/{id}  — full snapshot
# =============================================================================

@router.get("/{attempt_id}", response_model=AttemptDetail)
async def get_attempt(
    attempt_id: str,
    authorization: Optional[str] = Header(default=None),
) -> AttemptDetail:
    supabase = get_supabase_client()
    user_id = get_verified_user_id(supabase, authorization)
    attempt = _load_attempt(supabase, attempt_id, user_id)
    case = _load_case(supabase, attempt["case_id"])

    msg_rows = (
        supabase.table("attempt_messages")
        .select("*")
        .eq("attempt_id", attempt_id)
        .order("created_at", desc=False)
        .execute()
    )
    messages = [
        MessageOut(
            id=m["id"],
            role=m["role"],
            kind=m["kind"],
            content=m.get("content"),
            file_id=m.get("file_id"),
            is_clarification=bool(m.get("is_clarification")),
            created_at=m["created_at"],
        )
        for m in (msg_rows.data or [])
    ]

    return AttemptDetail(
        attempt=AttemptSummary(
            attempt_id=attempt["id"],
            case_id=attempt["case_id"],
            tier=attempt["tier_at_start"],
            clarification_quota=attempt["clarification_quota"],
            clarification_used=attempt["clarification_used"],
            clarification_remaining=max(0, attempt["clarification_quota"] - attempt["clarification_used"]),
            status=attempt["status"],
        ),
        case={
            "id": case["id"],
            "title": case["title"],
            "type": case["type"],
            "difficulty": case["difficulty"],
            "content": case["content"],
            "hint": case.get("hint"),
        },
        messages=messages,
    )


# =============================================================================
# POST /attempts/{id}/messages  — append user turn + stream interviewer reply
# =============================================================================

@router.post("/{attempt_id}/messages")
async def post_message(
    attempt_id: str,
    body: PostMessageRequest,
    authorization: Optional[str] = Header(default=None),
):
    supabase = get_supabase_client()
    user_id = get_verified_user_id(supabase, authorization)
    check_rate_limit(f"attempts:msg:{user_id}", max_calls=60, window_seconds=60)

    attempt = _load_attempt(supabase, attempt_id, user_id)
    if attempt["status"] != "active":
        raise HTTPException(status_code=400, detail="Attempt already submitted")

    # Soft cap on total messages.
    count_res = (
        supabase.table("attempt_messages")
        .select("id", count="exact")
        .eq("attempt_id", attempt_id)
        .execute()
    )
    total = getattr(count_res, "count", None) or len(count_res.data or [])
    if total >= MAX_MESSAGES_PER_ATTEMPT:
        raise HTTPException(status_code=400, detail="Message limit reached for this attempt")

    case = _load_case(supabase, attempt["case_id"])
    transcript = _fetch_transcript(supabase, attempt_id)

    # Does this turn consume clarification quota?
    clar_count = count_clarifications(body.content)
    remaining = attempt["clarification_quota"] - attempt["clarification_used"]
    quota_exhausted = remaining <= 0

    # Insert the user message first so the transcript persists even if the
    # AI call later fails.
    user_row = (
        supabase.table("attempt_messages")
        .insert(
            {
                "attempt_id": attempt_id,
                "role": "user",
                "kind": body.kind if body.kind in ("text", "voice", "image", "file") else "text",
                "content": body.content,
                "is_clarification": (clar_count > 0) and not quota_exhausted,
            }
        )
        .execute()
    )
    user_msg = user_row.data[0]

    # If quota is exhausted AND they asked a question, do not invoke AI.
    # Spec: "Further clarification questions should be disabled" but the
    # workspace stays open — the user can still post notes/calcs.
    if clar_count > 0 and quota_exhausted:
        return {
            "user_message": user_msg,
            "assistant_message": None,
            "clarification_remaining": 0,
            "quota_exhausted": True,
            "reason": "Clarification quota exhausted — keep building your notes; the conversation will be evaluated when you submit.",
        }

    # Decrement quota if this counted.
    if clar_count > 0:
        new_used = attempt["clarification_used"] + clar_count
        supabase.table("attempts").update({"clarification_used": new_used}).eq("id", attempt_id).execute()
        remaining -= clar_count

    # ---------- Stream assistant reply ----------
    def event_stream():
        chunks: List[str] = []
        try:
            yield f"event: meta\ndata: {{\"clarification_remaining\": {max(0, remaining)}, \"is_clarification\": {str(clar_count > 0).lower()}}}\n\n"
            for token in stream_interviewer_reply(
                case_content=case["content"],
                case_type=case["type"],
                transcript=transcript,
                new_user_message=body.content,
            ):
                chunks.append(token)
                # SSE data lines must not contain literal newlines — escape them.
                safe = token.replace("\\", "\\\\").replace("\n", "\\n")
                yield f"event: token\ndata: {safe}\n\n"
            final_text = "".join(chunks).strip()
            # Persist the assistant turn.
            saved = (
                supabase.table("attempt_messages")
                .insert(
                    {
                        "attempt_id": attempt_id,
                        "role": "assistant",
                        "kind": "text",
                        "content": final_text,
                        "is_clarification": False,
                    }
                )
                .execute()
            )
            msg_id = saved.data[0]["id"] if saved.data else None
            yield f"event: done\ndata: {{\"message_id\": \"{msg_id}\"}}\n\n"
        except InterviewEngineError as e:
            yield f"event: error\ndata: {str(e)[:200]}\n\n"
        except Exception as e:  # noqa: BLE001
            yield f"event: error\ndata: {type(e).__name__}: {str(e)[:200]}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# =============================================================================
# POST /attempts/{id}/uploads  — image/doc attachment
# =============================================================================

@router.post("/{attempt_id}/uploads")
async def upload_file(
    attempt_id: str,
    file: UploadFile = File(...),
    caption: Optional[str] = Form(default=None),
    authorization: Optional[str] = Header(default=None),
):
    supabase = get_supabase_client()
    user_id = get_verified_user_id(supabase, authorization)
    check_rate_limit(f"attempts:upload:{user_id}", max_calls=30, window_seconds=60)

    attempt = _load_attempt(supabase, attempt_id, user_id)
    if attempt["status"] != "active":
        raise HTTPException(status_code=400, detail="Attempt already submitted")

    mime = (file.content_type or "").lower()
    is_image = mime.startswith("image/")
    is_doc = mime in ALLOWED_MIME_EXACT
    if not (is_image or is_doc):
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {mime}")

    body = await file.read()
    size = len(body)
    cap = MAX_IMAGE_BYTES if is_image else MAX_DOC_BYTES
    if size == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if size > cap:
        raise HTTPException(status_code=413, detail="File too large")

    # Upload to Supabase Storage bucket `attempt_uploads`.
    ext = (file.filename or "").split(".")[-1].lower()
    safe_ext = ext if ext.isalnum() and len(ext) <= 6 else "bin"
    import uuid as _uuid
    object_path = f"{user_id}/{attempt_id}/{_uuid.uuid4().hex}.{safe_ext}"
    try:
        supabase.storage.from_("attempt_uploads").upload(
            path=object_path,
            file=body,
            file_options={"content-type": mime, "upsert": "false"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage upload failed: {e}")

    file_row = (
        supabase.table("attempt_files")
        .insert(
            {
                "attempt_id": attempt_id,
                "storage_path": object_path,
                "mime_type": mime,
                "file_name": file.filename or object_path.split("/")[-1],
                "size_bytes": size,
            }
        )
        .execute()
    )
    file_id = file_row.data[0]["id"]

    # Insert a message referencing the file. `content` carries an optional
    # caption — the scorer reads this when it weighs the upload.
    kind = "image" if is_image else "file"
    msg = (
        supabase.table("attempt_messages")
        .insert(
            {
                "attempt_id": attempt_id,
                "role": "user",
                "kind": kind,
                "content": caption or f"[uploaded {file.filename}]",
                "file_id": file_id,
                "is_clarification": False,
            }
        )
        .execute()
    )

    # Best-effort signed URL for the frontend to render the upload.
    signed_url = None
    try:
        signed = supabase.storage.from_("attempt_uploads").create_signed_url(object_path, 60 * 60)
        signed_url = signed.get("signedURL") or signed.get("signed_url")
    except Exception:
        signed_url = None

    return {
        "message": msg.data[0],
        "file": {
            "id": file_id,
            "storage_path": object_path,
            "mime_type": mime,
            "file_name": file.filename,
            "size_bytes": size,
            "signed_url": signed_url,
        },
    }


# =============================================================================
# POST /attempts/{id}/submit  — finalize + score
# =============================================================================

@router.post("/{attempt_id}/submit", response_model=SubmitResponse)
async def submit_attempt(
    attempt_id: str,
    body: SubmitRequest,
    authorization: Optional[str] = Header(default=None),
) -> SubmitResponse:
    supabase = get_supabase_client()
    user_id = get_verified_user_id(supabase, authorization)
    check_rate_limit(f"attempts:submit:{user_id}", max_calls=10, window_seconds=60)

    attempt = _load_attempt(supabase, attempt_id, user_id)
    if attempt["status"] != "active":
        raise HTTPException(status_code=400, detail="Attempt already submitted")

    case = _load_case(supabase, attempt["case_id"])
    transcript = _fetch_transcript(supabase, attempt_id)
    if len(transcript) == 0:
        raise HTTPException(status_code=400, detail="No conversation to submit")

    # Persist the final recommendation as the closing message.
    supabase.table("attempt_messages").insert(
        {
            "attempt_id": attempt_id,
            "role": "user",
            "kind": "recommendation",
            "content": body.final_recommendation,
            "is_clarification": False,
        }
    ).execute()

    # Re-fetch with the recommendation included.
    transcript = _fetch_transcript(supabase, attempt_id)

    # Score.
    try:
        feedback = score_conversation(
            case_content=case["content"],
            case_type=case["type"],
            transcript=transcript,
            final_recommendation=body.final_recommendation,
        )
    except InterviewEngineError as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")

    # Build a flat answer_text from the transcript so the legacy
    # `submissions.answer_text` column stays populated and the existing
    # /results page can show "what the user submitted".
    flat_lines = []
    for t in transcript:
        role = t["role"].upper()
        flat_lines.append(f"[{role}] {t['content']}")
    flat_lines.append("")
    flat_lines.append(f"[FINAL RECOMMENDATION] {body.final_recommendation}")
    answer_text = "\n".join(flat_lines)

    sub_res = (
        supabase.table("submissions")
        .insert(
            {
                "user_id": user_id,
                "case_id": attempt["case_id"],
                "answer_text": answer_text,
                "score": feedback["score"],
                "feedback_json": feedback,
            }
        )
        .execute()
    )
    submission_id = sub_res.data[0]["id"]

    # Mark the attempt submitted.
    supabase.table("attempts").update(
        {
            "status": "submitted",
            "submission_id": submission_id,
            "final_recommendation": body.final_recommendation,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
    ).eq("id", attempt_id).execute()

    # ---- Mirror case_attempts + points/badges logic from the legacy /submit ----
    from datetime import timedelta, timezone as _tz
    IST = _tz(timedelta(hours=5, minutes=30))
    today_ist = datetime.now(IST).date().isoformat()

    prior_res = (
        supabase.table("case_attempts")
        .select("id, attempt_number")
        .eq("user_id", user_id)
        .eq("case_id", attempt["case_id"])
        .order("attempt_number", desc=True)
        .limit(1)
        .execute()
    )
    prior_row = (prior_res.data or [None])[0]
    is_first_attempt = prior_row is None
    attempt_number = 1 if is_first_attempt else (prior_row.get("attempt_number", 0) + 1)

    counted_for_daily = False
    daily_date_val = None
    if is_first_attempt:
        try:
            sched = (
                supabase.table("daily_schedule")
                .select("case_id, guesstimate_code")
                .eq("scheduled_date", today_ist)
                .limit(1)
                .execute()
            )
            srow = (sched.data or [None])[0]
            daily_ids = set()
            if srow:
                if srow.get("case_id"):
                    daily_ids.add(srow["case_id"])
                if srow.get("guesstimate_code"):
                    daily_ids.add(srow["guesstimate_code"])
            if attempt["case_id"] in daily_ids:
                counted_for_daily = True
                daily_date_val = today_ist
        except Exception as e:
            print(f"WARN: daily schedule check failed: {e}")

    try:
        supabase.table("case_attempts").insert(
            {
                "user_id": user_id,
                "case_id": attempt["case_id"],
                "submission_id": submission_id,
                "attempt_number": attempt_number,
                "is_first_attempt": is_first_attempt,
                "counted_for_daily": counted_for_daily,
                "daily_date": daily_date_val,
            }
        ).execute()
    except Exception as e:
        print(f"WARN: case_attempts insert failed: {e}")

    if is_first_attempt:
        try:
            ur = supabase.table("users").select("points").eq("id", user_id).maybe_single().execute()
            current_points = (ur.data or {}).get("points", 0)
            supabase.table("users").update({"points": current_points + feedback["score"]}).eq("id", user_id).execute()
        except Exception as e:
            print(f"ERROR: points update failed: {e}")
        try:
            badges = award_badges_for_submission(
                user_id=user_id,
                submission_id=submission_id,
                score=feedback["score"],
                feedback_breakdown=feedback["breakdown"],
                case_id=attempt["case_id"],
                case_type=case["type"],
                is_first_attempt=is_first_attempt,
                counted_for_daily=counted_for_daily,
            )
            if badges:
                print(f"Awarded badges to {user_id}: {badges}")
        except Exception as e:
            print(f"WARN: badge awarding failed: {e}")

    return SubmitResponse(
        submission_id=submission_id,
        attempt_id=attempt_id,
        score=feedback["score"],
        breakdown=feedback["breakdown"],
        strengths=feedback["strengths"],
        improvements=feedback["improvements"],
        summary=feedback["summary"],
        rubric=feedback.get("rubric", "case"),
    )
