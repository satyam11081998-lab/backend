"""
Gemini Live real-time voice — the cheap, low-latency speech-to-speech interviewer.

Same shape as routes/realtime.py (the OpenAI one) but for Google's Gemini Live:
mints a short-lived EPHEMERAL token server-side (the real GEMINI_API_KEY never
reaches the browser), pins the model + interviewer instructions into the token's
constraints, and hands the browser a constrained WebSocket URL to stream audio
directly to Google (lowest latency).

Credit-gated exactly like OpenAI realtime. Deduction is by elapsed conversation
SECONDS (Gemini Live is priced per-minute of audio), reported by the client to
/realtime-gemini/usage as it runs and on close.

NON-DEFAULT: reached only when the admin sets voice_mode = "gemini". Until then
nothing here runs, so it cannot disturb the pipeline or OpenAI realtime modes.
"""

import os
import time
import datetime
from typing import Optional

import httpx
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from services.supabase_client import get_supabase_client
from services.auth import get_verified_user, is_guest_user
from services.rate_limit import check_rate_limit
from services.ai_usage import assert_daily_budget, get_ai_input_quota, log_ai_usage
from services.realtime_credits import has_credit, get_balance, deduct as deduct_credit
from prompts.interview_prompts import build_interviewer_messages

load_dotenv()

router = APIRouter()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Model names move fast — keep it env-overridable so a bump is a config change.
GEMINI_LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-2.0-flash-live-001")
GEMINI_LIVE_VOICE = os.getenv("GEMINI_LIVE_VOICE", "Puck")
GEMINI_LIVE_PER_MIN = float(os.getenv("GEMINI_LIVE_PER_MIN", "0.04"))  # ~$/min, guardrail estimate

AUTH_TOKEN_URL = "https://generativelanguage.googleapis.com/v1beta/auth_tokens"
WS_BASE = ("wss://generativelanguage.googleapis.com/ws/"
           "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent")


class GeminiSessionRequest(BaseModel):
    case_id: str
    attempt_id: Optional[str] = None


@router.post("/session")
async def create_gemini_session(
    body: GeminiSessionRequest,
    authorization: Optional[str] = Header(default=None),
):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="Gemini voice is not configured on the server.")
    supabase = get_supabase_client()
    uid, user_obj = get_verified_user(supabase, authorization)
    if is_guest_user(user_obj):
        raise HTTPException(status_code=403, detail="Create an account to use voice interview mode.")
    check_rate_limit(f"gemini_rt:{uid}", max_calls=6, window_seconds=60)
    assert_daily_budget()

    quota = get_ai_input_quota(supabase, uid)
    if quota["tier"] != "pro":
        raise HTTPException(status_code=403, detail="Voice interview is a Pro feature.")
    if not has_credit(supabase, uid, quota["tier"]):
        raise HTTPException(
            status_code=402,
            detail="You're out of real-time interview minutes. Buy a minute pack, or use the standard voice mode — it's unlimited on Pro.",
        )

    case = supabase.table("cases").select("id, title, type, content").eq("id", body.case_id).limit(1).execute()
    if not case.data:
        raise HTTPException(status_code=404, detail="Case not found")
    case_row = case.data[0]

    messages = build_interviewer_messages(
        case_content=case_row["content"], case_type=case_row["type"],
        transcript=[], new_user_message="", clarifications_exhausted=False,
    )
    system_turns = [m["content"] for m in messages if m.get("role") == "system" and m.get("content")]
    instructions = "\n\n".join(system_turns).strip()
    if not instructions:
        raise HTTPException(status_code=500, detail="Could not build interviewer instructions.")

    model = f"models/{GEMINI_LIVE_MODEL}"
    live_config = {
        "responseModalities": ["AUDIO"],
        "systemInstruction": {"parts": [{"text": instructions}]},
        "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": GEMINI_LIVE_VOICE}}},
        "inputAudioTranscription": {},
        "outputAudioTranscription": {},
    }
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    # A plain ephemeral token (no constraints — the REST auth_tokens resource does
    # not accept liveConnectConstraints). The browser sends the full setup (model,
    # voice, instructions) on the unconstrained Bidi endpoint below.
    token_body = {
        "uses": 2,
        "expireTime": (now + datetime.timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "newSessionExpireTime": (now + datetime.timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    try:
        t0 = time.time()
        async with httpx.AsyncClient(timeout=20.0) as client:
            res = await client.post(
                AUTH_TOKEN_URL,
                json=token_body,
                headers={"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY},
            )
        if res.status_code >= 400:
            print(f"[gemini-rt] auth_tokens failed {res.status_code}: {res.text[:600]}")
            # Surface Google's reason to the client (admin-only feature, still in
            # bring-up) so the exact cause is visible without digging Render logs.
            raise HTTPException(status_code=502, detail=f"Gemini token error {res.status_code}: {res.text[:300]}")
        data = res.json()
        # The token value is under `name` (top-level or nested under `token`).
        token_name = data.get("name") or (data.get("token") or {}).get("name")
        if not token_name:
            print(f"[gemini-rt] no token in response: {str(data)[:300]}")
            raise HTTPException(status_code=502, detail="Voice session token missing from response.")

        log_ai_usage(
            user_id=uid, endpoint="/realtime-gemini/session", model=GEMINI_LIVE_MODEL,
            audio_minutes=0, latency_ms=int((time.time() - t0) * 1000), success=True,
            meta={"case_id": body.case_id, "attempt_id": body.attempt_id},
        )
        return {
            "token": token_name,
            "ws_url": f"{WS_BASE}?access_token={token_name}",
            "model": model,
            "voice": GEMINI_LIVE_VOICE,
            "instructions": instructions,
            "credits": get_balance(supabase, uid, quota["tier"]),
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[gemini-rt] session error: {e}")
        raise HTTPException(status_code=500, detail=f"Could not start the voice session: {e}")


class GeminiUsageRequest(BaseModel):
    seconds: float


@router.post("/usage")
async def gemini_usage(
    body: GeminiUsageRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Deduct real-time credit for elapsed Gemini Live seconds + book its cost.
    The client posts INCREMENTAL seconds (since its last report) periodically and
    on close, so a single report is clamped to 10 minutes as an abuse guard."""
    supabase = get_supabase_client()
    uid, user_obj = get_verified_user(supabase, authorization)
    if is_guest_user(user_obj):
        return {"ok": True}
    secs = max(0.0, min(float(body.seconds or 0), 600.0))
    minutes = secs / 60.0
    tier = get_ai_input_quota(supabase, uid)["tier"]
    if minutes > 0:
        log_ai_usage(
            user_id=uid, endpoint="/realtime-gemini", model=GEMINI_LIVE_MODEL,
            audio_minutes=minutes, success=True, meta={"src": "gemini_live"},
        )
        deduct_credit(supabase, uid, minutes)
    return {"credits": get_balance(supabase, uid, tier)}
