"""
Model-led realtime voice: the routes the browser calls ALONGSIDE the live call.

  POST /attempts/{id}/voice-coach  after each saved candidate turn: returns the
                                   refreshed instructions (playbook + case + live
                                   coach notes) when the notes changed.
  POST /attempts/{id}/voice-tool   the speech model called a tool (get_hint,
                                   answer_request): run it on the server.

Neither route is in the path of a spoken reply: the speech model answers the
candidate by itself, and these steer its NEXT turns or hand it what only the
server may hand out. Both are owner-checked like every attempt route, rate
limited, and do nothing unless the model-led interviewer is on for this user
(VOICE_INTERVIEWER / VOICE_INTERVIEWER_ALLOWLIST).
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Union

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from services.supabase_client import get_supabase_client
from services.auth import get_verified_user
from services.rate_limit import check_rate_limit
from services.ai_usage import assert_daily_budget
from services.keyed_lock import keyed_lock
from services.markets import llm_case_content
from services import voice_coach as vc
from prompts.voice_interviewer_playbook import build_voice_interviewer_instructions

router = APIRouter()


def _attempt_ctx(supabase, attempt_id: str, user_id: str):
    # Same owner / availability checks as /messages and /voice-decision.
    from routes.attempts import _load_attempt, _load_case, _fetch_transcript
    attempt = _load_attempt(supabase, attempt_id, user_id)            # 404 / 403
    if attempt.get("status") != "active":
        raise HTTPException(status_code=400, detail="Attempt already submitted")
    case = _load_case(supabase, attempt["case_id"], user_id)
    transcript = _fetch_transcript(supabase, attempt_id)
    return attempt, case, transcript


def _require_model_led(user_id: str, user_obj) -> None:
    if vc.voice_interviewer_mode(user_id, getattr(user_obj, "email", None)) != "model_led":
        raise HTTPException(status_code=409, detail="The model-led voice interviewer is not enabled.")


def _save_state(supabase, attempt_id: str, session_state: Dict[str, Any]) -> None:
    try:
        supabase.table("attempts").update({"session_state": session_state}).eq("id", attempt_id).execute()
    except Exception as e:  # noqa: BLE001 -- a pre-migration DB must not 500 a live call
        print(f"[voice_coach] session_state not saved: {type(e).__name__}: {e}")


@router.post("/attempts/{attempt_id}/voice-coach")
def voice_coach(attempt_id: str, authorization: Optional[str] = Header(default=None)):
    supabase = get_supabase_client()
    user_id, user_obj = get_verified_user(supabase, authorization)
    _require_model_led(user_id, user_obj)
    check_rate_limit(f"attempts:vcoach:{user_id}", max_calls=90, window_seconds=60)
    with keyed_lock(f"voice-state:{attempt_id}"):
        attempt, case, transcript = _attempt_ctx(supabase, attempt_id, user_id)
        session_state = attempt.get("session_state") or {}
        notes, _sig = vc.coach_notes(transcript, attempt, session_state,
                                     (case.get("teaching_policy") or "coached"))
        v = vc.voice_state(session_state)
        changed = notes != list(v.get("coach_notes") or [])
        if changed:
            v["coach_notes"] = notes
            _save_state(supabase, attempt_id, vc.with_voice_state(session_state, v))
    instructions = build_voice_interviewer_instructions(llm_case_content(case), case.get("type") or "", notes)
    return {"changed": changed, "notes": notes, "instructions": instructions if changed else None}


class VoiceToolRequest(BaseModel):
    name: str = Field(..., max_length=64)
    # The realtime API delivers arguments as a JSON string; accept an object too.
    arguments: Optional[Union[str, Dict[str, Any]]] = None


@router.post("/attempts/{attempt_id}/voice-tool")
def voice_tool(attempt_id: str, body: VoiceToolRequest, authorization: Optional[str] = Header(default=None)):
    supabase = get_supabase_client()
    user_id, user_obj = get_verified_user(supabase, authorization)
    _require_model_led(user_id, user_obj)
    check_rate_limit(f"attempts:vtool:{user_id}", max_calls=20, window_seconds=60)
    assert_daily_budget()  # the hint ladder makes a small model call
    args: Dict[str, Any] = {}
    if isinstance(body.arguments, dict):
        args = body.arguments
    elif isinstance(body.arguments, str) and body.arguments.strip():
        try:
            parsed = json.loads(body.arguments)
            args = parsed if isinstance(parsed, dict) else {}
        except ValueError:
            args = {}

    with keyed_lock(f"voice-state:{attempt_id}"):
        attempt, case, transcript = _attempt_ctx(supabase, attempt_id, user_id)
        session_state = attempt.get("session_state") or {}
        content = llm_case_content(case)
        if body.name == "get_hint":
            output, v = vc.hint(case, content, transcript, session_state,
                                reason=str(args.get("reason") or ""),
                                where_stuck=str(args.get("where_stuck") or "")[:200], user_id=user_id)
            answer_given = False
        elif body.name == "answer_request":
            output, v, answer_given = vc.answer_request(case, content, transcript, session_state, user_id=user_id)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown tool: {body.name}")
        _save_state(supabase, attempt_id, vc.with_voice_state(session_state, v))
    print(f"[voice_coach] tool={body.name} attempt={attempt_id} hint_level={v.get('hint_level')} "
          f"answer_requests={v.get('answer_requests')} answer_given={answer_given}")
    return {"output": output, "answer_given": answer_given, "answer_allowed": bool(v.get("answer_revealed"))}
