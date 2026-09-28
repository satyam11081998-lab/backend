"""
Conversational case-interview routes.

Replaces the old POST /submit single-answer flow with a session-based
workspace. Endpoints:

  POST   /attempts                     -> start a session (gates by tier/quota)
  GET    /attempts/{id}                -> fetch case + messages
  POST   /attempts/{id}/messages       -> append user msg, stream AI reply (SSE)
  POST   /attempts/{id}/voice-decision -> V11 decides one realtime voice turn (JSON)
  POST   /attempts/{id}/uploads        -> attach an image / document to the thread
  POST   /attempts/{id}/submit         -> finalize, score the transcript, save

All endpoints derive user_id from the verified Supabase JWT — never trust
client-supplied ids. The service-role Supabase client bypasses RLS.
"""

import json
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Callable, Tuple
from fastapi import APIRouter, HTTPException, Header, UploadFile, File, Form, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from services.supabase_client import get_supabase_client
from services.auth import get_verified_user_id, get_verified_user, is_guest_user
from services.access_guard import assert_can_attempt, effective_tier
from services.markets import case_market, intl_daily_ids, market_today, llm_case_content
from services.rate_limit import check_rate_limit
from services.limits import MESSAGE_MAX_CHARS, RECOMMENDATION_MAX_CHARS
from services.interview_engine import (
    stream_interviewer_reply,
    complete_interviewer_reply,
    score_conversation,
    count_clarifications,
    InterviewEngineError,
)
from services.session_signals import compute_signals
from services.interviewer_decision import update_session_state, detect_violations
from services.badge_awarder import award_badges_for_submission
from services.case_figures import pop_figures, bank_figures
from services.ai_usage import assert_daily_budget, log_realtime_usage
from services.realtime_credits import deduct as deduct_realtime_credit
from services.keyed_lock import keyed_lock

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


class VoiceDecisionRequest(BaseModel):
    """One FINAL candidate turn from a realtime voice session, for V11 to decide.

    The realtime transports (OpenAI Realtime, Gemini Live) carry audio only; they
    do not decide what the interviewer says. The browser sends each final
    candidate transcript here, the frozen V11 engine decides, and the browser has
    the speech model say exactly `say` (or nothing, for SILENCE). Persistence and
    credit metering stay on /realtime-turn, unchanged.
    """
    content: str = Field(..., min_length=1, max_length=MESSAGE_MAX_CHARS)
    # A still-being-spoken transcript. V11 answers SILENCE for it without calling
    # any model; accepted so a client can never trigger speech mid-utterance.
    is_partial: bool = False
    # SPEED (optional; older clients send none of these). A client may ask V11
    # EARLY -- while the speech model is still finishing its own discarded
    # reply -- and only use the answer if the candidate's words do not change
    # afterwards. Such a decision is `defer_fold`: its session_state fold waits
    # for POST /voice-fold {commit: true}, or is dropped by {commit: false} /
    # `discard_turn_ids`, so a discarded early decision leaves no trace and the
    # learner state is exactly what one decision on the final words would give.
    # An uncommitted, undiscarded fold is applied before the attempt's next turn
    # (the pre-existing behaviour: every decided turn folds).
    turn_id: Optional[str] = Field(default=None, max_length=64)
    defer_fold: bool = False
    discard_turn_ids: List[str] = Field(default_factory=list, max_length=8)


class VoiceFoldRequest(BaseModel):
    turn_id: str = Field(..., min_length=1, max_length=64)
    commit: bool


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
    # No minimum length: the "final recommendation" is no longer typed into a
    # separate box — the client submits the candidate's last conversational turn
    # (which can be short) and the scorer reads the WHOLE transcript. Empty is
    # tolerated too; submit_attempt still rejects a genuinely empty conversation
    # (len(transcript) == 0) below, which is the real guard.
    final_recommendation: str = Field(default="", max_length=RECOMMENDATION_MAX_CHARS)


class SubmitResponse(BaseModel):
    submission_id: str
    attempt_id: str
    score: int
    breakdown: Dict[str, int]
    strengths: List[str]
    improvements: List[str]
    summary: str
    rubric: str = "case"
    # Additive (2026-09-01) — richer, evidence-based feedback + gibberish gate.
    # All optional; the results page reads the full feedback_json from the DB, so
    # these are for parity and any direct response reader. C2 unchanged.
    backstop: Optional[Dict[str, Any]] = None
    dimension_feedback: Optional[Dict[str, Any]] = None
    red_flags: Optional[List[str]] = None
    model_answer: Optional[str] = None
    validity: Optional[Dict[str, Any]] = None


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


# Case rows are read on every turn but change only when an admin edits a case.
# Cached per process for CASE_CACHE_SECONDS (default 120; 0 disables). The
# availability / owner checks below still run on every call.
def _secs_env(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


_CASE_TTL = _secs_env("CASE_CACHE_SECONDS", 120.0)
_case_cache: Dict[str, Tuple[float, dict]] = {}
_case_lock = threading.Lock()


def _case_row(supabase, case_id: str):
    if _CASE_TTL > 0:
        with _case_lock:
            hit = _case_cache.get(case_id)
        if hit is not None and time.monotonic() - hit[0] < _CASE_TTL:
            return dict(hit[1])
    row = supabase.table("cases").select("*").eq("id", case_id).maybe_single().execute()
    data = row.data
    if data and _CASE_TTL > 0:
        with _case_lock:
            if len(_case_cache) > 2000:
                _case_cache.clear()
            _case_cache[case_id] = (time.monotonic(), dict(data))
    return data


def _load_case(supabase, case_id: str, user_id=None) -> dict:
    data = _case_row(supabase, case_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")
    if data.get("is_active") is False and not data.get("unlisted"):
        # Copilot-generated PRIVATE cases (is_active=false, owner_id set) are
        # attemptable by their OWNER — that is the whole point of the curated
        # tool. Everyone else still gets a 404 for a retired/private case.
        # UNLISTED broadcast cases (unlisted=true) are attemptable by ANYONE with the
        # link (broadcast recipients); they skip this owner-gate. Deploy-safe via .get().
        if not (user_id and data.get("owner_id") == user_id):
            raise HTTPException(status_code=404, detail="This case is no longer available.")
    return data


def _transcript_rows(rows) -> List[Dict[str, str]]:
    return [
        {"role": r["role"], "kind": r["kind"], "content": r.get("content") or ""}
        for r in (rows.data or [])
        if r.get("content")
    ]


def _fetch_transcript(supabase, attempt_id: str) -> List[Dict[str, str]]:
    rows = (
        supabase.table("attempt_messages")
        .select("role, kind, content, created_at")
        .eq("attempt_id", attempt_id)
        .order("created_at", desc=False)
        .execute()
    )
    return _transcript_rows(rows)


def _fetch_transcript_and_count(supabase, attempt_id: str) -> Tuple[List[Dict[str, str]], int]:
    """The transcript AND the message-cap count in ONE round trip.

    Same rows and order as _fetch_transcript; the count is PostgREST's exact
    count of every attempt_messages row for the attempt (Content-Range), i.e.
    exactly what the separate select("id", count="exact") returned."""
    rows = (
        supabase.table("attempt_messages")
        .select("role, kind, content, created_at", count="exact")
        .eq("attempt_id", attempt_id)
        .order("created_at", desc=False)
        .execute()
    )
    total = getattr(rows, "count", None) or len(rows.data or [])
    return _transcript_rows(rows), total


# -----------------------------------------------------------------------------
# SPEED (2026-09-28). The interviewer engine is untouched; everything here is
# about the database work AROUND it. The backend is in Oregon and Supabase in
# Tokyo (~100 ms per round trip), and a turn used to make 8-11 of them one after
# another before and after the model call.
#
#   * independent reads run concurrently (_gather);
#   * work whose result the reply does not need -- folding V11's control tag
#     into session_state -- runs after the response, in a per-attempt queue
#     (_after_turn); the next request for the same attempt waits for it
#     (_await_after_turn), so the next turn always reads the updated state.
#     The queue is in-process: correct for the single uvicorn worker we run.
# -----------------------------------------------------------------------------
_IO_POOL = ThreadPoolExecutor(max_workers=16, thread_name_prefix="attempt-io")
_AFTER_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="attempt-after")
_after_lock = threading.Lock()
_after_by_attempt: Dict[str, Future] = {}
_AFTER_WAIT_S = 10.0
_TIMING_LOG = os.getenv("LOG_TURN_TIMING", "1").strip().lower() not in ("0", "false", "no", "off")


def _gather(*fns: Callable[[], Any]) -> List[Tuple[bool, Any]]:
    """Run independent callables concurrently. Returns [(ok, value_or_exception)]
    in argument order, so callers can raise errors in the ORIGINAL order."""
    futs = [_IO_POOL.submit(fn) for fn in fns]
    out: List[Tuple[bool, Any]] = []
    for f in futs:
        try:
            out.append((True, f.result()))
        except BaseException as e:  # noqa: BLE001 -- re-raised by _unwrap
            out.append((False, e))
    return out


def _unwrap(res: Tuple[bool, Any]):
    ok, val = res
    if not ok:
        raise val
    return val


def _after_turn(attempt_id: str, fn: Callable, *args) -> None:
    """Run fn(*args) after this response, strictly after any earlier after-turn
    work for the same attempt. Never raises into the request."""
    def job(prev: Optional[Future]):
        if prev is not None:
            try:
                prev.result(timeout=_AFTER_WAIT_S)
            except Exception:
                pass
        try:
            fn(*args)
        except Exception as e:  # noqa: BLE001
            print(f"[attempts] after-turn work failed: {type(e).__name__}: {e}")

    with _after_lock:
        prev = _after_by_attempt.get(attempt_id)
        fut = _AFTER_POOL.submit(job, prev)
        _after_by_attempt[attempt_id] = fut

    def _drop(f: Future, aid=attempt_id):
        with _after_lock:
            if _after_by_attempt.get(aid) is f:
                _after_by_attempt.pop(aid, None)
    fut.add_done_callback(_drop)


def _await_after_turn(attempt_id: str) -> None:
    """Block until the previous turn's after-turn work for this attempt is done
    (normally long finished: it takes ~0.1 s and a candidate takes seconds)."""
    with _after_lock:
        fut = _after_by_attempt.get(attempt_id)
    if fut is not None:
        try:
            fut.result(timeout=_AFTER_WAIT_S)
        except Exception:
            pass


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


# Deferred folds of EARLY (speculative) voice decisions -- see VoiceDecisionRequest.
# Every schedule of a fold happens INSIDE _deferred_lock, so a concurrent
# _settle_deferred can never miss a fold that was taken out but not yet queued.
_DEFER_TTL_S = 600.0
_deferred_lock = threading.Lock()
_deferred: Dict[str, "OrderedDict[str, Tuple[str, tuple, float]]"] = {}
_discarded: Dict[str, Dict[str, float]] = {}


# One turn at a time per attempt. A decision waits until every earlier decision
# for the same attempt has finished and handed on its fold -- the order the old
# one-request-at-a-time event loop gave -- EXCEPT an early decision the client has
# already voided: its fold is dropped whatever it decides, so nothing waits for
# its model call. Other attempts never wait on each other.
class _TurnSlot:
    __slots__ = ("done", "spec_id", "prev")

    def __init__(self, spec_id: Optional[str], prev: "Optional[_TurnSlot]"):
        self.done = threading.Event()
        self.spec_id = spec_id
        self.prev = prev


_turn_lock = threading.Lock()
_turn_tail: Dict[str, _TurnSlot] = {}
_TURN_WAIT_S = 30.0


def _enter_turn(attempt_id: str, spec_id: Optional[str] = None, voided=()) -> _TurnSlot:
    voided = set(voided or ())
    with _deferred_lock:
        voided |= set(_discarded.get(attempt_id, {}))
    with _turn_lock:
        prev = _turn_tail.get(attempt_id)
        if prev is not None and prev.done.is_set():
            prev = None
        slot = _TurnSlot(spec_id, prev)
        _turn_tail[attempt_id] = slot
    deadline = time.monotonic() + _TURN_WAIT_S
    p = prev
    while p is not None and not p.done.is_set():
        if p.spec_id is not None and p.spec_id in voided:
            p = p.prev          # skip the voided early decision, keep order behind it
            continue
        if not p.done.wait(max(0.0, deadline - time.monotonic())):
            print(f"[attempts] turn for {attempt_id} waited {_TURN_WAIT_S}s for the previous one; proceeding")
        break                   # a finished slot had itself waited for everything before it
    return slot


def _exit_turn(attempt_id: str, slot: _TurnSlot) -> None:
    slot.prev = None
    slot.done.set()
    with _turn_lock:
        if _turn_tail.get(attempt_id) is slot:
            _turn_tail.pop(attempt_id, None)


def _prune_locked(now: float) -> None:
    for aid in list(_discarded):
        d = _discarded[aid]
        for tid in [t for t, ts in d.items() if now - ts > _DEFER_TTL_S]:
            d.pop(tid, None)
        if not d:
            _discarded.pop(aid, None)
    # An early decision nobody confirmed for 10 minutes belongs to an abandoned
    # session: drop it rather than keep its transcript in memory forever.
    for aid in list(_deferred):
        entries = _deferred[aid]
        for tid in [t for t, e in entries.items() if now - e[2] > _DEFER_TTL_S]:
            entries.pop(tid, None)
        if not entries:
            _deferred.pop(aid, None)


def _defer_fold(attempt_id: str, user_id: str, turn_id: str, args: tuple) -> None:
    now = time.monotonic()
    with _deferred_lock:
        if turn_id in _discarded.get(attempt_id, {}):
            return  # the client already dropped this early decision
        _deferred.setdefault(attempt_id, OrderedDict())[turn_id] = (user_id, args, now)
        if len(_discarded) + len(_deferred) > 200:
            _prune_locked(now)


def _resolve_fold(attempt_id: str, user_id: str, turn_id: str, commit: bool) -> bool:
    """Commit or drop one deferred fold. Only its owner can. True if one existed."""
    with _deferred_lock:
        entries = _deferred.get(attempt_id)
        ent = entries.get(turn_id) if entries else None
        if ent is not None and ent[0] != user_id:
            return False
        if ent is not None:
            entries.pop(turn_id, None)
            if not entries:
                _deferred.pop(attempt_id, None)
        if not commit:
            _discarded.setdefault(attempt_id, {})[turn_id] = time.monotonic()
        elif ent is not None:
            _after_turn(attempt_id, _fold_session_state, *ent[1])
    return ent is not None


def _settle_deferred(attempt_id: str, user_id: str, discard_ids=(), apply_pending: bool = True) -> None:
    """Before a turn reads session_state: drop the early decisions the client
    discarded, then either apply every other pending one in order (the next
    VOICE turn: the client's confirm is normally already here, this is the
    safety net) or drop them (typed turn / submit: a pending early decision
    there is a voice turn the candidate never finished -- its words were never
    saved, so its fold must not land). Then wait for the applied folds."""
    for tid in discard_ids or ():
        _resolve_fold(attempt_id, user_id, tid, False)
    with _deferred_lock:
        entries = _deferred.get(attempt_id)
        if entries:
            # Only the attempt owner's own requests can touch its pending folds.
            for tid in [t for t, e in entries.items() if e[0] == user_id]:
                args = entries.pop(tid)[1]
                if apply_pending:
                    _after_turn(attempt_id, _fold_session_state, *args)
                else:
                    _discarded.setdefault(attempt_id, {})[tid] = time.monotonic()
            if not entries:
                _deferred.pop(attempt_id, None)
    _await_after_turn(attempt_id)


def _is_v11_silence(ctl: Dict[str, Any], chunks: List[str]) -> bool:
    """True only for V11's deliberate SILENCE lane.

    The frozen engine marks it in control_out as mode NO_OUTPUT with a
    `silence` intervention and yields nothing. Anything else that comes back
    empty is NOT silence and must not be swallowed as if it were.
    """
    tag = ctl.get("tag") or {}
    return (
        ctl.get("mode") == "NO_OUTPUT"
        and tag.get("intervention") == "silence"
        and not any((c or "").strip() for c in chunks)
    )


def _fold_session_state(
    supabase, attempt_id: str, transcript, user_text: str,
    teaching_policy, session_state, ctl: Dict[str, Any], reply_text: str,
) -> None:
    """Phase 2: fold V11's control tag into persisted session_state, and record
    any deterministic guardrail violation for evals/metrics. Wrapped: a
    pre-migration DB (no session_state column) degrades to no-op, never a 500.
    Shared by /messages and /voice-decision so both paths keep ONE learner state.
    """
    tag = ctl.get("tag") or {}
    if not tag:
        return
    try:
        _sig = compute_signals(transcript, user_text, teaching_policy or "coached", session_state)
        _new_state = update_session_state(session_state, tag, _sig)
        supabase.table("attempts").update({"session_state": _new_state}).eq("id", attempt_id).execute()
        _viol = detect_violations(reply_text, teaching_policy or "coached", tag)
        if _viol:
            print(f"[interviewer] guardrail_violation {_viol} attempt={attempt_id} tag={tag}")
    except Exception as _e:  # noqa: BLE001
        print(f"[interviewer] session_state update skipped: {type(_e).__name__}: {_e}")


# =============================================================================
# POST /attempts  — start a session
# =============================================================================

@router.post("", response_model=AttemptSummary)
def start_attempt(
    body: StartAttemptRequest,
    authorization: Optional[str] = Header(default=None),
) -> AttemptSummary:
    supabase = get_supabase_client()
    user_id = get_verified_user_id(supabase, authorization)
    check_rate_limit(f"attempts:start:{user_id}", max_calls=20, window_seconds=60)

    case = _load_case(supabase, body.case_id, user_id)
    # Tier/quota gate — same logic as the legacy /submit.
    assert_can_attempt(supabase, user_id, case)

    tier = effective_tier(supabase, user_id)
    quota = CLARIFICATION_QUOTA.get(tier, 5)

    # Check-then-insert: one at a time per user+case (a double click must not
    # create two active attempts now that requests run concurrently).
    with keyed_lock(f"start:{user_id}:{body.case_id}"):
        return _start_or_resume(supabase, user_id, body, case, tier, quota)


def _start_or_resume(supabase, user_id: str, body: StartAttemptRequest, case: dict,
                     tier: str, quota: int) -> AttemptSummary:
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
def get_attempt(
    attempt_id: str,
    authorization: Optional[str] = Header(default=None),
) -> AttemptDetail:
    supabase = get_supabase_client()
    user_id = get_verified_user_id(supabase, authorization)
    _await_after_turn(attempt_id)
    # The attempt (ownership check) and its messages are independent reads; the
    # messages are only used once the ownership check below has passed.
    r_attempt, r_msgs = _gather(
        lambda: _load_attempt(supabase, attempt_id, user_id),
        lambda: (
            supabase.table("attempt_messages")
            .select("*")
            .eq("attempt_id", attempt_id)
            .order("created_at", desc=False)
            .execute()
        ),
    )
    attempt = _unwrap(r_attempt)
    case = _load_case(supabase, attempt["case_id"], user_id)
    msg_rows = _unwrap(r_msgs)
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
def post_message(
    attempt_id: str,
    body: PostMessageRequest,
    authorization: Optional[str] = Header(default=None),
):
    t_req = time.perf_counter()
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

    # The previous turn's session_state fold must land before this turn reads it
    # (and an unfinished early voice decision must not).
    slot = _enter_turn(attempt_id)
    try:
        _settle_deferred(attempt_id, user_id, apply_pending=False)
    finally:
        _exit_turn(attempt_id, slot)

    # Independent reads, concurrently. Errors are raised below in the ORIGINAL
    # order (attempt 404/403 -> submitted 400 -> budget 503 -> message cap 400).
    r_attempt, r_budget, r_transcript = _gather(
        lambda: _load_attempt(supabase, attempt_id, user_id),
        assert_daily_budget,  # global spend backstop before any interviewer-turn spend
        lambda: _fetch_transcript_and_count(supabase, attempt_id),
    )
    attempt = _unwrap(r_attempt)
    if attempt["status"] != "active":
        raise HTTPException(status_code=400, detail="Attempt already submitted")

    _unwrap(r_budget)

    # Soft cap on total messages.
    transcript, total = _unwrap(r_transcript)
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

    case = _load_case(supabase, attempt["case_id"], user_id)

    # Adaptive interviewer (Phase 2): persisted learner state + per-case teaching policy.
    # Both are select("*")-safe -- an absent column just yields None, handled below.
    session_state = attempt.get("session_state") or {}
    teaching_policy = case.get("teaching_policy") or None

    # Does this turn consume clarification quota?
    clar_count = count_clarifications(body.content, body.kind)
    remaining = attempt["clarification_quota"] - attempt["clarification_used"]
    quota_exhausted = remaining <= 0

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
    new_used = None
    if clar_count > 0 and not quota_exhausted:
        new_used = min(attempt["clarification_quota"], attempt["clarification_used"] + clar_count)
        remaining = attempt["clarification_quota"] - new_used

    user_row = {
        "attempt_id": attempt_id,
        "role": "user",
        "kind": body.kind if body.kind in ("text", "voice", "image", "file") else "text",
        "content": body.content,
        "is_clarification": (clar_count > 0) and not quota_exhausted,
    }

    # Insert the user message first so the transcript persists even if the
    # AI call later fails (and a failed save is still a plain 500, before any
    # model spend -- unchanged).
    supabase.table("attempt_messages").insert(user_row).execute()
    if new_used is not None:
        supabase.table("attempts").update({"clarification_used": new_used}).eq("id", attempt_id).execute()
    pre_ms = _ms(t_req)

    # ---------- Stream assistant reply ----------
    def event_stream():
        chunks: List[str] = []
        ctl: Dict[str, Any] = {}
        t_stream = time.perf_counter()
        first_ms = None
        try:
            yield (
                f"event: meta\ndata: {{"
                f"\"clarification_remaining\": {max(0, remaining)}, "
                f"\"is_clarification\": {str(clar_count > 0 and not quota_exhausted).lower()}, "
                f"\"clarifications_spent\": {str(clarifications_spent).lower()}"
                f"}}\n\n"
            )
            for token in stream_interviewer_reply(
                case_content=llm_case_content(case),
                case_type=case["type"],
                transcript=transcript,
                new_user_message=body.content,
                user_id=user_id,
                clarifications_exhausted=clarifications_spent,
                teaching_policy=teaching_policy,
                prior_state=session_state,
                control_out=ctl,
            ):
                if first_ms is None and token:
                    first_ms = _ms(t_stream)
                chunks.append(token)
                # SSE data lines must not contain literal newlines — escape them.
                safe = token.replace("\\", "\\\\").replace("\n", "\\n")
                yield f"event: token\ndata: {safe}\n\n"
            final_text = "".join(chunks).strip()
            # V11 SILENCE is an internal control decision, not a reply: no assistant
            # row, no bubble, nothing for TTS. Recognised explicitly from V11's own
            # control output -- an unexpected empty reply still takes the old path.
            silent = _is_v11_silence(ctl, chunks)
            msg_id = None
            if not silent:
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
            # Folding V11's tag into session_state is for the NEXT turn: done after
            # this response, ordered before the next request for this attempt.
            _after_turn(
                attempt_id, _fold_session_state,
                supabase, attempt_id, transcript, body.content,
                teaching_policy, session_state, ctl, final_text,
            )
            if silent:
                yield "event: done\ndata: {\"message_id\": null}\n\n"
            else:
                yield f"event: done\ndata: {{\"message_id\": \"{msg_id}\"}}\n\n"
            if _TIMING_LOG:
                print(f"[timing] /messages attempt={attempt_id} pre_ms={pre_ms} "
                      f"first_token_ms={first_ms} reply_ms={_ms(t_stream)} total_ms={_ms(t_req)} "
                      f"mode={ctl.get('mode')} silent={silent}")
        except InterviewEngineError as e:
            yield f"event: error\ndata: {str(e)[:200]}\n\n"
        except Exception as e:  # noqa: BLE001
            yield f"event: error\ndata: {type(e).__name__}: {str(e)[:200]}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Server-Timing": f"pre;dur={pre_ms}"},
    )


# =============================================================================
# POST /attempts/{id}/voice-decision  — V11 decides a realtime voice turn
# =============================================================================

@router.post("/{attempt_id}/voice-decision")
def voice_decision(
    attempt_id: str,
    body: VoiceDecisionRequest,
    response: Response,
    authorization: Optional[str] = Header(default=None),
):
    """
    Run the frozen V11 interviewer on one final candidate transcript from a
    realtime voice session and return its decision:

        {"lane": "SILENCE" | "PRESENCE" | "SUBSTANTIVE",
         "mode": <V11 mode>, "reason": <V11 reason>,
         "say": <exact line to speak, or null>,
         "event": <V11 interviewer_presence event, or null>}

    Same engine call as /messages (stream_interviewer_reply), so typed, standard
    voice and realtime voice share ONE interviewer brain and ONE learner state.
    Only the channel differs ("voice"), which changes how V11 renders a presence
    beat, not what it decides. Nothing is written to attempt_messages here: the
    client lands the turns through /realtime-turn exactly as before, and for
    SILENCE there is no assistant turn to land.
    """
    t_req = time.perf_counter()
    supabase = get_supabase_client()
    user_id, user_obj = get_verified_user(supabase, authorization)
    is_guest = is_guest_user(user_obj)
    check_rate_limit(
        f"attempts:vd:{user_id}",
        max_calls=20 if is_guest else 60,
        window_seconds=60,
    )

    # One decision at a time per attempt (see _enter_turn), from reading
    # session_state to handing its fold on, so an overlapping turn can never read
    # a stale learner state or overwrite a newer fold.
    speculative = bool(body.defer_fold and body.turn_id)
    slot = _enter_turn(attempt_id, body.turn_id if speculative else None, body.discard_turn_ids)
    try:
        return _voice_decision(attempt_id, body, response, supabase, user_id, is_guest, t_req)
    finally:
        _exit_turn(attempt_id, slot)


def _voice_decision(attempt_id: str, body: VoiceDecisionRequest, response: Response,
                    supabase, user_id: str, is_guest: bool, t_req: float):
    # Earlier folds must land before this turn reads session_state: drop the early
    # decisions the client discarded, apply the rest, wait for them.
    _settle_deferred(attempt_id, user_id, body.discard_turn_ids)

    # Independent reads, concurrently; errors raised in the original order.
    r_attempt, r_budget, r_transcript = _gather(
        lambda: _load_attempt(supabase, attempt_id, user_id),
        assert_daily_budget,  # same global spend backstop as /messages
        lambda: _fetch_transcript_and_count(supabase, attempt_id),
    )
    attempt = _unwrap(r_attempt)
    if attempt["status"] != "active":
        raise HTTPException(status_code=400, detail="Attempt already submitted")

    _unwrap(r_budget)

    transcript, total = _unwrap(r_transcript)
    cap = GUEST_MAX_MESSAGES_PER_ATTEMPT if is_guest else MAX_MESSAGES_PER_ATTEMPT
    if total >= cap:
        raise HTTPException(status_code=400, detail="Message limit reached for this attempt")

    case = _load_case(supabase, attempt["case_id"], user_id)
    session_state = attempt.get("session_state") or {}
    teaching_policy = case.get("teaching_policy") or None

    # Read-only mirror of /messages: tell V11 when this turn's clarification
    # cannot be answered. Consumption is still recorded by /realtime-turn when
    # the client lands the user turn, exactly as before.
    clar_count = count_clarifications(body.content, "voice")
    remaining = attempt["clarification_quota"] - attempt["clarification_used"]
    clarifications_spent = clar_count > 0 and remaining <= 0

    ctl: Dict[str, Any] = {}
    pre_ms = _ms(t_req)
    t_engine = time.perf_counter()
    try:
        parts = list(stream_interviewer_reply(
            case_content=llm_case_content(case),
            case_type=case["type"],
            transcript=transcript,
            new_user_message=body.content,
            user_id=user_id,
            clarifications_exhausted=clarifications_spent,
            teaching_policy=teaching_policy,
            prior_state=session_state,
            control_out=ctl,
            channel="voice",
            is_voice_partial=body.is_partial,
        ))
    except InterviewEngineError as e:
        raise HTTPException(status_code=502, detail=str(e)[:200])
    engine_ms = _ms(t_engine)

    tag = ctl.get("tag") or {}
    event = None
    if _is_v11_silence(ctl, parts):
        lane, say = "SILENCE", None
    elif tag.get("intervention") == "fast_lane":
        lane = "PRESENCE"
        raw = "".join(parts).strip()
        try:
            event = json.loads(raw)
            say = str(((event or {}).get("data") or {}).get("text") or "").strip() or None
        except (TypeError, ValueError):
            event, say = None, raw or None
        if not say:
            raise HTTPException(status_code=502, detail="The interviewer presence event had no line.")
    else:
        lane = "SUBSTANTIVE"
        say = "".join(parts).strip()
        if not say:
            # Not V11 silence (checked above) -- an empty substantive reply is an
            # error, never something to quietly treat as "say nothing".
            raise HTTPException(status_code=502, detail="The interviewer produced no content for this turn.")

    if not body.is_partial:
        fold_args = (supabase, attempt_id, transcript, body.content,
                     teaching_policy, session_state, ctl, say or "")
        if body.defer_fold and body.turn_id:
            # Early decision: folded only once the client confirms the words did
            # not change (or before the next turn, if it never says).
            _defer_fold(attempt_id, user_id, body.turn_id, fold_args)
        else:
            # For the NEXT turn only: folded after this response is sent, ordered
            # before the next request for this attempt reads session_state.
            _after_turn(attempt_id, _fold_session_state, *fold_args)

    total_ms = _ms(t_req)
    response.headers["Server-Timing"] = f"pre;dur={pre_ms}, engine;dur={engine_ms}, total;dur={total_ms}"
    if _TIMING_LOG:
        print(f"[timing] /voice-decision attempt={attempt_id} pre_ms={pre_ms} engine_ms={engine_ms} "
              f"total_ms={total_ms} lane={lane} mode={ctl.get('mode')}")

    return {
        "lane": lane,
        "mode": ctl.get("mode"),
        "reason": ctl.get("reason"),
        "say": say,
        "event": event,
    }


@router.post("/{attempt_id}/voice-fold")
def voice_fold(
    attempt_id: str,
    body: VoiceFoldRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Commit (the candidate's words did not change) or drop (they did) the
    session_state fold of an EARLY voice decision -- see VoiceDecisionRequest.
    Off the critical path: the client sends it in the background."""
    supabase = get_supabase_client()
    user_id, _ = get_verified_user(supabase, authorization)
    check_rate_limit(f"attempts:vf:{user_id}", max_calls=120, window_seconds=60)
    _load_attempt(supabase, attempt_id, user_id)  # 404 / 403 exactly like the other routes
    found = _resolve_fold(attempt_id, user_id, body.turn_id, body.commit)
    return {"ok": True, "found": found}


# =============================================================================
# POST /attempts/{id}/uploads  — image/doc attachment
# =============================================================================

@router.post("/{attempt_id}/realtime-turn")
def post_realtime_turn(
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
    # Guests (anonymous auth) may report voice turns too, so their transcript
    # actually saves and can be scored after they convert. Cost is bounded by the
    # credit trial + the daily kill-switch.

    # Two turns per exchange, and the far end can be quick — looser than
    # /messages, still bounded.
    check_rate_limit(f"attempts:rt:{user_id}", max_calls=120, window_seconds=60)

    # clarification_used is read-modify-written below: one save at a time per
    # attempt (the old one-at-a-time event loop gave this for free).
    with keyed_lock(f"attempt-rt:{attempt_id}"):
        return _realtime_turn(attempt_id, body, supabase, user_id)


def _realtime_turn(attempt_id: str, body: RealtimeTurnRequest, supabase, user_id: str):
    # The attempt and the message count are independent reads -- run them together.
    r_attempt, r_count = _gather(
        lambda: _load_attempt(supabase, attempt_id, user_id),
        lambda: (
            supabase.table("attempt_messages")
            .select("id", count="exact")
            .eq("attempt_id", attempt_id)
            .execute()
        ),
    )
    attempt = _unwrap(r_attempt)
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

    count_res = _unwrap(r_count)
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
        # Burn real-time credit for this turn. Rates match log_realtime_usage:
        # input 1 tok/100ms (600/min), output 1 tok/50ms (1200/min).
        turn_minutes = (body.audio_input_tokens or 0) / 600.0 + (body.audio_output_tokens or 0) / 1200.0
        deduct_realtime_credit(supabase, user_id, turn_minutes)

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
def submit_attempt(
    attempt_id: str,
    body: SubmitRequest,
    authorization: Optional[str] = Header(default=None),
) -> SubmitResponse:
    supabase = get_supabase_client()
    user_id = get_verified_user_id(supabase, authorization)
    check_rate_limit(f"attempts:submit:{user_id}", max_calls=10, window_seconds=60)

    # Scoring reads session_state: let the last turn's fold land first (and drop
    # an early voice decision for words the candidate never finished).
    slot = _enter_turn(attempt_id)
    try:
        _settle_deferred(attempt_id, user_id, apply_pending=False)
    finally:
        _exit_turn(attempt_id, slot)
    attempt = _load_attempt(supabase, attempt_id, user_id)
    if attempt["status"] != "active":
        raise HTTPException(status_code=400, detail="Attempt already submitted")

    case = _load_case(supabase, attempt["case_id"], user_id)
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
            case_content=llm_case_content(case),
            case_type=case["type"],
            transcript=transcript,
            final_recommendation=body.final_recommendation,
            user_id=user_id,
            case_id=attempt["case_id"],
        )
    except InterviewEngineError as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")

    # Phase 4: personalised debrief + longitudinal skill profile. The per-attempt
    # learner profile was accumulated on session_state during the session. Fully
    # best-effort: a pre-migration DB or any error degrades to no debrief, never a 500.
    try:
        from services.learning_model import build_debrief, merge_longitudinal_profile
        _attempt_profile = (attempt.get("session_state") or {}).get("profile") or {}
        feedback["learning_debrief"] = build_debrief(_attempt_profile)
        try:
            _r = supabase.table("user_skill_profile").select("profile").eq("user_id", user_id).maybe_single().execute()
            _existing = ((_r.data or {}) or {}).get("profile") or {}
        except Exception:
            _existing = {}
        _merged = merge_longitudinal_profile(_existing, _attempt_profile)
        supabase.table("user_skill_profile").upsert(
            {"user_id": user_id, "profile": _merged, "updated_at": datetime.now(timezone.utc).isoformat()},
            on_conflict="user_id",
        ).execute()
    except Exception as _e:  # noqa: BLE001
        print(f"WARN: learning debrief/profile skipped: {_e}")

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

    # ── Split the PAYWALLED figures out of the submission ────────────────
    # `visuals` (the worked profit bridge / 2x2 / driver tree) are a Pro
    # feature. They must NOT live in feedback_json, because that row belongs to
    # the user: submissions_select_own (0006) lets them read their own row
    # straight from PostgREST with their browser JWT, and three server pages
    # already pass whole feedback_json blobs into client components. Anything
    # left in here is effectively public to its owner.
    #
    # So they are popped BEFORE the insert and written to case_figures
    # (migration 0069), which has RLS on and no policy — service role only.
    _figures = pop_figures(feedback)

    # The back-link to the conversation this score came from (migration 0068).
    # Without it nothing can get from a scored submission to its transcript,
    # which is why admin analytics reported "conversation not found": the
    # forward link attempts.submission_id is set further down, but every
    # consumer starts from the submission, not the attempt.
    _sub_row = {
        "user_id": user_id,
        "case_id": attempt["case_id"],
        "attempt_id": attempt_id,
        "answer_text": answer_text,
        "score": feedback["score"],
        "feedback_json": feedback,
    }
    try:
        sub_res = supabase.table("submissions").insert(_sub_row).execute()
    except Exception as _e:  # noqa: BLE001
        # Pre-0068 database: the column does not exist yet. Scoring a real
        # attempt must never fail because a migration is outstanding, so drop
        # the new column and insert exactly as before. attempts.submission_id
        # below still records the forward link, and 0068's backfill recovers
        # this row's attempt_id the moment the migration is run.
        if "attempt_id" not in str(_e):
            raise
        print(f"WARN: submissions.attempt_id not present (run migration 0068): {_e}")
        _sub_row.pop("attempt_id", None)
        sub_res = supabase.table("submissions").insert(_sub_row).execute()
    submission_id = sub_res.data[0]["id"]

    # Bank the case's figures — one row per CASE, best-scoring answer wins.
    # Best-effort inside the helper; never fails a scored submit.
    bank_figures(supabase, attempt["case_id"], _figures, submission_id, feedback["score"])

    # Silent self-improvement: if this scored session clears the bar, bank an
    # anonymised exemplar for THIS case so future scoring calibrates against real
    # strong answers. Fully non-blocking — never affects the response or the score.
    try:
        from services.exemplar_bank import maybe_capture_exemplar
        maybe_capture_exemplar(
            case_id=attempt["case_id"],
            submission_id=submission_id,
            case_content=llm_case_content(case),
            case_type=case["type"],
            feedback=feedback,
            user_id=user_id,
        )
    except Exception as _e:
        print(f"WARN: exemplar capture skipped: {_e}")

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
    if is_first_attempt and case_market(case) == "US":
        # International daily pair (0070): its own table and US Eastern day.
        # The most recent pair on/before today, matching the access gate — so a
        # daily attempt never burns a free user's one-time bank credit.
        us_today = market_today("US")
        if attempt["case_id"] in intl_daily_ids(supabase, "US", us_today, exact=False):
            counted_for_daily = True
            daily_date_val = us_today
    elif is_first_attempt:
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
        backstop=feedback.get("backstop"),
        dimension_feedback=feedback.get("dimension_feedback"),
        red_flags=feedback.get("red_flags"),
        model_answer=feedback.get("model_answer"),
        validity=feedback.get("validity"),
    )
