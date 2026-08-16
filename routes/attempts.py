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
from services.auth import get_verified_user_id, get_verified_user, is_guest_user
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
from services.ai_usage import assert_daily_budget, log_realtime_usage

router = APIRouter(prefix="/attempts", tags=["attempts"])


# -----------------------------------------------------------------------------
# Tier -> clarification (AI hint) quota, PER ATTEMPT.
#
# 2026-08-01: free was 0, which made the free experience broken rather than
# limited — count_clarifications() fires on any '?' , so a free user's very
# first question (or even a structure containing a question mark) hit the
# exhausted branch, got NO interviewer reply at all, and only a toast saying
# "Clarification quota used up" before they had asked anything. Free tier's
# limit is CASE ACCESS (daily pair + 1 lifetime extra, enforced in
# services/access_guard.py), not conversation quality: once a free user is
# inside a case they are entitled to a real interview.
#
# Ladder is monotonic. MUST stay in sync with TIER_LIMITS.maxHintQuestions in
# the frontend's lib/tier.ts and with the pricing-page copy.
# -----------------------------------------------------------------------------
CLARIFICATION_QUOTA = {"free": 7, "lite": 12, "pro": 20}

# Soft cap on total messages per attempt — prevents runaway sessions.
MAX_MESSAGES_PER_ATTEMPT = 200

# GUEST MODE (0045). A guest reaches the interviewer with no email, no payment
# method and nothing to rate-limit against except a cookie they can clear — and
# every turn is a real LLM call. 200 is a sane ceiling for an account we can
# trace; for an anonymous session it is an open tab on the AI bill.
#
# 40 is deliberately generous against real use: a full 15-minute case runs
# 15-20 user turns, so a genuine candidate never sees this. It exists to bound
# the worst case, not to shape the good one. Raise it if real transcripts start
# hitting it; do not remove it.
GUEST_MAX_MESSAGES_PER_ATTEMPT = 40

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


class RealtimeTurnRequest(BaseModel):
    """One turn reported by a realtime (WebRTC) voice session.

    Realtime runs the conversation at the far end, so the model has ALREADY
    replied by the time we hear about it — unlike /messages, this endpoint must
    not call the interviewer. Its only job is to land the same
    `attempt_messages` rows the typed path produces, so scoring reads one
    format regardless of transport.
    """
    role: str = Field(..., description="user | assistant")
    content: str = Field(..., min_length=1, max_length=MESSAGE_MAX_CHARS)
    # Audio-token usage from the client's `response.done`. Optional because a
    # user turn carries none; when present it is what makes realtime spend
    # visible to the daily-budget kill switch.
    audio_input_tokens: Optional[int] = None
    audio_output_tokens: Optional[int] = None


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
    # One token read for both the id and the guest flag — get_verified_user_id
    # would repeat this round-trip, and this is the hottest path in the app.
    user_id, user_obj = get_verified_user(supabase, authorization)
    is_guest = is_guest_user(user_obj)

    # Guests get a tighter turn rate. A human types; a script does not wait.
    check_rate_limit(
        f"attempts:msg:{user_id}",
        max_calls=20 if is_guest else 60,
        window_seconds=60,
    )

    attempt = _load_attempt(supabase, attempt_id, user_id)
    if attempt["status"] != "active":
        raise HTTPException(status_code=400, detail="Attempt already submitted")

    assert_daily_budget()  # global spend backstop before any interviewer-turn spend

    # Soft cap on total messages.
    count_res = (
        supabase.table("attempt_messages")
        .select("id", count="exact")
        .eq("attempt_id", attempt_id)
        .execute()
    )
    total = getattr(count_res, "count", None) or len(count_res.data or [])
    cap = GUEST_MAX_MESSAGES_PER_ATTEMPT if is_guest else MAX_MESSAGES_PER_ATTEMPT
    if total >= cap:
        # Phrased as an invitation rather than a wall: a guest who genuinely
        # reached 40 turns is deeply engaged, and this is the best possible
        # moment to ask for the account.
        raise HTTPException(
            status_code=400,
            detail=(
                "You've reached the practice limit for this session. Create a free account to keep going."
                if is_guest
                else "Message limit reached for this attempt"
            ),
        )

    case = _load_case(supabase, attempt["case_id"])
    transcript = _fetch_transcript(supabase, attempt_id)

    # Does this turn consume clarification quota?
    clar_count = count_clarifications(body.content, body.kind)
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

    # Clarifications are exhausted for this turn?  We used to return early here
    # with NO assistant reply at all — the user's message just hung in the
    # thread forever, unanswered, with a toast that vanished on refresh. That
    # read as a broken product, not a paywall (2026-08-01 fix).
    #
    # Now the interviewer ALWAYS replies. When the quota is spent it is told
    # not to answer clarifications — it acknowledges and pushes the candidate
    # to state an assumption and keep going, which is what a real interviewer
    # does anyway. The turn costs one AI call but the session never dead-ends.
    clarifications_spent = clar_count > 0 and quota_exhausted

    # Decrement quota if this counted. Clamp to the quota: count_clarifications
    # counts every '?', so a single packed turn could previously push
    # clarification_used past clarification_quota and drive `remaining`
    # negative (masked by max(0, ...) on the way out, but wrong in the DB).
    if clar_count > 0 and not quota_exhausted:
        new_used = min(attempt["clarification_quota"], attempt["clarification_used"] + clar_count)
        supabase.table("attempts").update({"clarification_used": new_used}).eq("id", attempt_id).execute()
        remaining = attempt["clarification_quota"] - new_used

    # ---------- Stream assistant reply ----------
    def event_stream():
        chunks: List[str] = []
        try:
            yield (
                f"event: meta\ndata: {{"
                f"\"clarification_remaining\": {max(0, remaining)}, "
                f"\"is_clarification\": {str(clar_count > 0 and not quota_exhausted).lower()}, "
                f"\"clarifications_spent\": {str(clarifications_spent).lower()}"
                f"}}\n\n"
            )
            for token in stream_interviewer_reply(
                case_content=case["content"],
                case_type=case["type"],
                transcript=transcript,
                new_user_message=body.content,
                user_id=user_id,
                clarifications_exhausted=clarifications_spent,
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

@router.post("/{attempt_id}/realtime-turn")
async def post_realtime_turn(
    attempt_id: str,
    body: RealtimeTurnRequest,
    authorization: Optional[str] = Header(default=None),
):
    """
    Persist one turn from a realtime voice session, and meter its cost.

    Deliberately NOT a variant of post_message: that function's whole body is
    "call the interviewer and stream the reply", which realtime has already
    done. What must NOT diverge is the row that lands in `attempt_messages` —
    same table, same `kind='voice'`, so a spoken attempt is scored on the same
    document as a typed one.
    """
    supabase = get_supabase_client()
    user_id, user_obj = get_verified_user(supabase, authorization)
    if is_guest_user(user_obj):
        raise HTTPException(status_code=403, detail="Create an account to use voice interview mode.")

    # Two turns per exchange, and the far end can be quick — looser than
    # /messages, still bounded.
    check_rate_limit(f"attempts:rt:{user_id}", max_calls=120, window_seconds=60)

    attempt = _load_attempt(supabase, attempt_id, user_id)
    if attempt["status"] != "active":
        raise HTTPException(status_code=400, detail="Attempt already submitted")

    # TRUST BOUNDARY. In realtime the transcript originates in the BROWSER, so
    # a crafted request can post any text as either role — including inventing
    # the interviewer's side. Accepted deliberately (owner decision): the only
    # thing a candidate gains is a fake score on their own practice attempt.
    # It is NOT acceptable anywhere money or another user's data is involved,
    # so this endpoint must never grow beyond writing attempt_messages.
    #
    # Deliberately NOT calling assert_daily_budget() here: the audio has already
    # been spent at the far end, and refusing the write would lose the
    # transcript while keeping the cost. Spend is metered below instead.
    role = body.role if body.role in ("user", "assistant") else "user"

    count_res = (
        supabase.table("attempt_messages")
        .select("id", count="exact")
        .eq("attempt_id", attempt_id)
        .execute()
    )
    total = getattr(count_res, "count", None) or len(count_res.data or [])
    if total >= MAX_MESSAGES_PER_ATTEMPT:
        raise HTTPException(status_code=400, detail="Message limit reached for this attempt")

    # C9 v2 still applies to spoken turns. We cannot refuse mid-stream the way
    # post_message does — the far end has already answered — so this records
    # consumption rather than gating it. See the C9 open item in the realtime
    # handoff before changing this.
    clar_count = 0
    if role == "user":
        clar_count = count_clarifications(body.content, "voice")
        if clar_count > 0:
            new_used = min(attempt["clarification_quota"], attempt["clarification_used"] + clar_count)
            supabase.table("attempts").update({"clarification_used": new_used}).eq("id", attempt_id).execute()

    saved = (
        supabase.table("attempt_messages")
        .insert(
            {
                "attempt_id": attempt_id,
                "role": role,
                "kind": "voice",
                "content": body.content,
                "is_clarification": clar_count > 0,
            }
        )
        .execute()
    )

    # Meter it. gpt-realtime bills audio per token: input 1 tok/100ms, output
    # 1 tok/50ms. Booking a real cost here is the ONLY thing that makes voice
    # spend visible to spend_today_usd(), which is what assert_daily_budget()
    # reads. If this books zero, the global kill switch is blind to the most
    # expensive thing the product does.
    if body.audio_input_tokens or body.audio_output_tokens:
        log_realtime_usage(
            user_id=user_id,
            input_tokens=body.audio_input_tokens or 0,
            output_tokens=body.audio_output_tokens or 0,
            meta={"attempt_id": attempt_id, "role": role},
        )

    return {"message_id": saved.data[0]["id"] if saved.data else None}


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
            user_id=user_id,
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
