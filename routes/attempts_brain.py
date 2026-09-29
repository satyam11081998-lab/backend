"""
Unified-interviewer-brain branches of the attempt routes (INTERVIEWER_BRAIN on /
allow-listed user). routes/attempts.py calls into here; with the flag off none of
this runs and the baseline V11 code path is untouched.

Channel adapters only: auth, reads, C9 counting, persistence, SSE/JSON framing.
Every interviewer decision comes from services.interviewer.engine.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

import routes.attempts as A
from services.interviewer import engine, telemetry
from services.interviewer.dedupe import LEDGER, ROWS
from services.interviewer.flags import brain_enabled_for, debug_decisions
from services.interviewer.types import CaseContext, Channel, InterviewerError, Lane, TurnInput


def _email_of(user_obj: Any) -> Optional[str]:
    if user_obj is None:
        return None
    if isinstance(user_obj, dict):
        return user_obj.get("email")
    return getattr(user_obj, "email", None)


def enabled(user_id: str, user_obj: Any) -> bool:
    return brain_enabled_for(user_id, _email_of(user_obj))


def _case_ctx(case: dict) -> CaseContext:
    return CaseContext(case_type=case.get("type") or "", content=A.llm_case_content(case),
                       teaching_policy=(case.get("teaching_policy") or "coached"), title=case.get("title") or "")


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


def _token(text: str) -> str:
    return _sse("token", text.replace("\\", "\\\\").replace("\n", "\\n"))


# ---------------------------------------------------------------------------
# Idempotent message inserts (client_turn_id, migration 0071). Degrades to a
# plain insert on a pre-migration database, like the rest of this codebase.
# ---------------------------------------------------------------------------
_COL = {"has_client_turn_id": True, "checked_at": 0.0}
_COL_RECHECK_S = 600.0
_COL_LOCK = threading.Lock()


def _column_available() -> bool:
    with _COL_LOCK:
        if _COL["has_client_turn_id"]:
            return True
        if time.monotonic() - _COL["checked_at"] > _COL_RECHECK_S:
            _COL["has_client_turn_id"] = True  # try again; the migration may have been run since
            return True
        return False


def _mark_column_missing() -> None:
    with _COL_LOCK:
        _COL["has_client_turn_id"] = False
        _COL["checked_at"] = time.monotonic()


def insert_message(supabase, row: Dict[str, Any], client_turn_id: Optional[str]) -> Tuple[Optional[str], bool]:
    """Insert one attempt_messages row. Returns (message_id, inserted). A duplicate
    client_turn_id returns the existing row's id with inserted=False."""
    attempt_id = row.get("attempt_id")
    use_col = bool(client_turn_id) and _column_available()
    payload = dict(row, client_turn_id=client_turn_id) if use_col else dict(row)
    try:
        res = supabase.table("attempt_messages").insert(payload).execute()
        return ((res.data or [{}])[0].get("id") if res.data else None), True
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if use_col and "client_turn_id" in msg and ("column" in msg.lower() or "PGRST204" in msg or "schema cache" in msg):
            _mark_column_missing()
            res = supabase.table("attempt_messages").insert(dict(row)).execute()
            return ((res.data or [{}])[0].get("id") if res.data else None), True
        if use_col and ("23505" in msg or "duplicate key" in msg.lower()):
            existing = (supabase.table("attempt_messages").select("id").eq("attempt_id", attempt_id)
                        .eq("client_turn_id", client_turn_id).limit(1).execute())
            rows = existing.data or []
            return (rows[0].get("id") if rows else None), False
        raise


def _persist_state(supabase, attempt_id: str, session_state: Dict[str, Any], brain_state: Dict[str, Any]) -> None:
    try:
        merged = engine.merge_state(session_state, brain_state)
        supabase.table("attempts").update({"session_state": merged}).eq("id", attempt_id).execute()
    except Exception as e:  # noqa: BLE001 - pre-migration DB (no session_state) degrades to no-op
        print(f"[interviewer] brain state persist skipped: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# TEXT / STT: POST /attempts/{id}/messages
# ---------------------------------------------------------------------------
def post_message(attempt_id: str, body, supabase, user_id: str, is_guest: bool, t_req: float):
    channel = Channel.STT if (getattr(body, "channel", None) or "").lower() == "stt" else Channel.TEXT
    turn_id = getattr(body, "turn_id", None) or None

    is_new, entry = LEDGER.begin(attempt_id, turn_id)
    if not is_new:
        entry = LEDGER.wait(entry)
        return StreamingResponse(_replay(entry.result), media_type="text/event-stream")

    try:
        return _post_message(attempt_id, body, supabase, user_id, is_guest, t_req, channel, turn_id, entry)
    except BaseException as e:
        LEDGER.fail(entry, error=type(e).__name__)
        raise


def _replay(result: Dict[str, Any]):
    """A duplicate turn: replay what the first request produced. No decision, no rows."""
    yield _sse("meta", json.dumps(result.get("meta") or {}))
    if result.get("error"):
        yield _sse("error", "This turn could not be completed. Please try again.")
        return
    if result.get("lane") == Lane.NO_OUTPUT.value:
        yield _sse("silence", "{}")
        yield _sse("done", json.dumps({"message_id": None, "silent": True, "duplicate": True}))
        return
    if result.get("text"):
        yield _token(result["text"])
    yield _sse("done", json.dumps({"message_id": result.get("message_id"), "duplicate": True}))


def _post_message(attempt_id, body, supabase, user_id, is_guest, t_req, channel, turn_id, entry):
    slot = A._enter_turn(attempt_id)
    try:
        A._settle_deferred(attempt_id, user_id, apply_pending=False)
    finally:
        A._exit_turn(attempt_id, slot)

    r_attempt, r_budget, r_transcript = A._gather(
        lambda: A._load_attempt(supabase, attempt_id, user_id),
        A.assert_daily_budget,
        lambda: A._fetch_transcript_and_count(supabase, attempt_id),
    )
    attempt = A._unwrap(r_attempt)
    if attempt["status"] != "active":
        raise HTTPException(status_code=400, detail="Attempt already submitted")
    A._unwrap(r_budget)
    transcript, total = A._unwrap(r_transcript)
    cap = A.GUEST_MAX_MESSAGES_PER_ATTEMPT if is_guest else A.MAX_MESSAGES_PER_ATTEMPT
    if total >= cap:
        raise HTTPException(status_code=400, detail=(
            "You've reached the practice limit for this session. Create a free account to keep going."
            if is_guest else "Message limit reached for this attempt"))
    case = A._load_case(supabase, attempt["case_id"], user_id)
    session_state = attempt.get("session_state") or {}

    # C9 - unchanged: same counter, same clamp, same quota write, same row flag.
    clar_count = A.count_clarifications(body.content, body.kind)
    remaining = attempt["clarification_quota"] - attempt["clarification_used"]
    quota_exhausted = remaining <= 0
    clarifications_spent = clar_count > 0 and quota_exhausted
    new_used = None
    if clar_count > 0 and not quota_exhausted:
        new_used = min(attempt["clarification_quota"], attempt["clarification_used"] + clar_count)
        remaining = attempt["clarification_quota"] - new_used

    user_row = {
        "attempt_id": attempt_id, "role": "user",
        "kind": body.kind if body.kind in ("text", "voice", "image", "file") else "text",
        "content": body.content, "is_clarification": (clar_count > 0) and not quota_exhausted,
    }
    _, user_inserted = insert_message(supabase, user_row, f"m:{turn_id}" if turn_id else None)
    # A retry of a turn whose first attempt failed: the row already exists, so C9 was already counted.
    if new_used is not None and user_inserted:
        supabase.table("attempts").update({"clarification_used": new_used}).eq("id", attempt_id).execute()

    turn = TurnInput(text=body.content, channel=channel, turn_id=turn_id, clarifications_exhausted=clarifications_spent)
    with A.keyed_lock(f"brain:{attempt_id}"):
        plan = engine.decide_turn(turn=turn, case=_case_ctx(case), transcript=transcript, session_state=session_state,
                                  attempt_id=attempt_id, user_id=user_id,
                                  assess=engine.default_assessor(user_id))
        # Decision-time state lands before the next turn for this attempt reads it.
        A._after_turn(attempt_id, _persist_state, supabase, attempt_id, session_state, plan.state_after.to_dict())
    pre_ms = A._ms(t_req)
    meta = {"clarification_remaining": max(0, remaining), "is_clarification": bool(clar_count > 0 and not quota_exhausted),
            "clarifications_spent": bool(clarifications_spent)}

    def stream():
        yield _sse("meta", json.dumps(meta))
        started_ms = None
        text_parts: List[str] = []
        try:
            if plan.lane == Lane.NO_OUTPUT:
                state = engine.finalize(plan, None)
                A._after_turn(attempt_id, _persist_state, supabase, attempt_id, session_state, state)
                LEDGER.finish(entry, lane=plan.lane.value, meta=meta)
                yield _sse("silence", "{}")
                yield _sse("done", json.dumps({"message_id": None, "silent": True}))
                _timing(attempt_id, pre_ms, None, t_req, plan)
                return
            for chunk in engine.word_stream(plan):
                if started_ms is None:
                    started_ms = telemetry.now_ms()
                text_parts.append(chunk)
                yield _token(chunk)
            final_text = "".join(text_parts).strip()
            msg_id, _ = insert_message(supabase, {"attempt_id": attempt_id, "role": "assistant", "kind": "text",
                                                  "content": final_text, "is_clarification": False},
                                       f"a:{turn_id}" if turn_id else None)
            state = engine.finalize(plan, final_text, response_start_ms=started_ms)
            A._after_turn(attempt_id, _persist_state, supabase, attempt_id, session_state, state)
            LEDGER.finish(entry, lane=plan.lane.value, text=final_text, message_id=msg_id, meta=meta)
            yield _sse("done", json.dumps({"message_id": msg_id}))
            _timing(attempt_id, pre_ms, started_ms, t_req, plan)
        except InterviewerError as e:
            # The line never reached the candidate: restore the pre-turn state (queued after the
            # decision-time write, so it wins) - a failed hint must not move the ladder.
            A._after_turn(attempt_id, _persist_state, supabase, attempt_id, session_state,
                          engine.finalize(plan, None, error=e))
            LEDGER.fail(entry, error=e.error_type, meta=meta)
            yield _sse("error", "The interviewer could not respond to that. Please try again.")
        except Exception as e:  # noqa: BLE001
            A._after_turn(attempt_id, _persist_state, supabase, attempt_id, session_state,
                          engine.finalize(plan, None, error=e))
            LEDGER.fail(entry, error=type(e).__name__, meta=meta)
            yield _sse("error", f"{type(e).__name__}: {str(e)[:160]}")

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Server-Timing": f"pre;dur={pre_ms}, decide;dur={plan.decision_ms}"})


def _timing(attempt_id: str, pre_ms: int, started_ms: Optional[int], t_req: float, plan) -> None:
    if A._TIMING_LOG:
        print(f"[timing] /messages(brain) attempt={attempt_id} pre_ms={pre_ms} decide_ms={plan.decision_ms} "
              f"total_ms={A._ms(t_req)} lane={plan.lane.value} mode={plan.decision.intervention.value}")


# ---------------------------------------------------------------------------
# REALTIME VOICE: POST /attempts/{id}/voice-decision
# ---------------------------------------------------------------------------
def voice_decision(attempt_id: str, body, response, supabase, user_id: str, is_guest: bool, t_req: float):
    turn_id = body.turn_id or None
    if body.is_partial:
        # A still-being-spoken transcript never produces speech and never touches state.
        return _voice_payload("SILENCE", None, None, turn_id)

    is_new, entry = LEDGER.begin(attempt_id, turn_id)
    if not is_new:
        entry = LEDGER.wait(entry)
        if entry.status == "done":
            r = entry.result
            return _voice_payload(r.get("lane_public"), r.get("say"), r.get("decision"), turn_id, duplicate=True)
        raise HTTPException(status_code=409, detail="This turn is already being handled.")
    try:
        out = _voice_decision(attempt_id, body, response, supabase, user_id, is_guest, t_req, turn_id)
        LEDGER.finish(entry, **out.pop("_ledger"))
        return out
    except BaseException as e:
        LEDGER.fail(entry, error=type(e).__name__)
        raise


def _voice_payload(lane: Optional[str], say: Optional[str], decision: Optional[Dict[str, Any]], turn_id: Optional[str],
                   duplicate: bool = False) -> Dict[str, Any]:
    dbg = debug_decisions()
    return {
        "lane": lane or "SILENCE",
        "mode": (decision or {}).get("intervention") if dbg else None,
        "reason": (decision or {}).get("reason") if dbg else None,
        "say": say,
        "event": None,
        "turn_id": turn_id,
        "duplicate": duplicate or None,
    }


def _voice_decision(attempt_id, body, response, supabase, user_id, is_guest, t_req, turn_id):
    A._settle_deferred(attempt_id, user_id, body.discard_turn_ids)
    r_attempt, r_budget, r_transcript = A._gather(
        lambda: A._load_attempt(supabase, attempt_id, user_id),
        A.assert_daily_budget,
        lambda: A._fetch_transcript_and_count(supabase, attempt_id),
    )
    attempt = A._unwrap(r_attempt)
    if attempt["status"] != "active":
        raise HTTPException(status_code=400, detail="Attempt already submitted")
    A._unwrap(r_budget)
    transcript, total = A._unwrap(r_transcript)
    cap = A.GUEST_MAX_MESSAGES_PER_ATTEMPT if is_guest else A.MAX_MESSAGES_PER_ATTEMPT
    if total >= cap:
        raise HTTPException(status_code=400, detail="Message limit reached for this attempt")
    case = A._load_case(supabase, attempt["case_id"], user_id)
    session_state = attempt.get("session_state") or {}

    # Read-only mirror of C9, exactly like the V11 path (consumption recorded by /realtime-turn).
    clar_count = A.count_clarifications(body.content, "voice")
    remaining = attempt["clarification_quota"] - attempt["clarification_used"]
    clarifications_spent = clar_count > 0 and remaining <= 0

    turn = TurnInput(text=body.content, channel=Channel.VOICE, turn_id=turn_id,
                     clarifications_exhausted=clarifications_spent)
    pre_ms = A._ms(t_req)
    plan = engine.decide_turn(turn=turn, case=_case_ctx(case), transcript=transcript, session_state=session_state,
                              attempt_id=attempt_id, user_id=user_id, session_id=getattr(body, "session_id", None),
                              assess=engine.default_assessor(user_id))
    say: Optional[str] = None
    if plan.lane != Lane.NO_OUTPUT:
        try:
            say = engine.word_complete(plan)
        except InterviewerError as e:
            engine.finalize(plan, None, error=e)
            raise HTTPException(status_code=502, detail="The interviewer could not respond to that turn.")
        if not (say or "").strip():
            e = InterviewerError("empty interviewer line")
            engine.finalize(plan, None, error=e)
            raise HTTPException(status_code=502, detail="The interviewer produced no content for this turn.")
    state = engine.finalize(plan, say, response_start_ms=telemetry.now_ms() if say else None)
    fold_args = (supabase, attempt_id, session_state, state)
    if body.defer_fold and turn_id:
        A._defer_fold(attempt_id, user_id, turn_id, fold_args, fn=_persist_state)
    else:
        A._after_turn(attempt_id, _persist_state, *fold_args)

    lane_public = {"NO_OUTPUT": "SILENCE"}.get(plan.lane.value, plan.lane.value)
    total_ms = A._ms(t_req)
    response.headers["Server-Timing"] = (f"pre;dur={pre_ms}, decide;dur={plan.decision_ms}, "
                                         f"total;dur={total_ms}")
    if A._TIMING_LOG:
        print(f"[timing] /voice-decision(brain) attempt={attempt_id} pre_ms={pre_ms} decide_ms={plan.decision_ms} "
              f"total_ms={total_ms} lane={lane_public} mode={plan.decision.intervention.value}")
    decision_public = {"intervention": plan.decision.intervention.value, "reason": plan.decision.reason}
    out = _voice_payload(lane_public, say, decision_public, turn_id)
    out["_ledger"] = {"lane_public": lane_public, "say": say, "decision": decision_public}
    return out


# ---------------------------------------------------------------------------
# Client timing reports (T0..T4, interruption) -> telemetry only, no DB.
# ---------------------------------------------------------------------------
def voice_telemetry(attempt_id: str, report: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    report = dict(report or {})
    report["attempt_id"] = attempt_id
    telemetry.emit_timing(report)
    return {"ok": True}
