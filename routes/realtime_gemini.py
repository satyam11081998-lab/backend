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
           "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContentConstrained")


_MODEL_CACHE = {"model": None, "ts": 0.0}
_MODEL_TTL = 3600.0


def _list_live_models() -> list:
    """Model IDs (bare) on this key that support the Live API (bidiGenerateContent).
    Runs on the server, which can reach Google even when nothing else can."""
    try:
        with httpx.Client(timeout=15.0) as c:
            r = c.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}&pageSize=1000")
        if r.status_code >= 400:
            print(f"[gemini-rt] models list {r.status_code}: {r.text[:200]}")
            return []
        ms = r.json().get("models", [])
        return [m["name"].split("/")[-1] for m in ms
                if "bidiGenerateContent" in (m.get("supportedGenerationMethods") or [])]
    except Exception as e:
        print(f"[gemini-rt] models list failed: {e}")
        return []


def _resolve_live_model() -> str:
    """Pick a valid Live model for this key. Honors GEMINI_LIVE_MODEL when it is
    set AND actually available; otherwise auto-selects (native-audio > flash-live
    > flash). Cached for an hour. Never raises."""
    import time as _t
    now = _t.time()
    if _MODEL_CACHE["model"] and (now - _MODEL_CACHE["ts"]) < _MODEL_TTL:
        return _MODEL_CACHE["model"]
    live = _list_live_models()
    env = os.getenv("GEMINI_LIVE_MODEL", "").strip()
    chosen = None
    if env and (env in live or not live):
        chosen = env
    if not chosen and live:
        for pref in ("native-audio", "flash-live", "-live-", "flash", "live"):
            hit = next((n for n in live if pref in n), None)
            if hit:
                chosen = hit
                break
        if not chosen:
            chosen = live[0]
    if not chosen:
        chosen = env or "gemini-2.5-flash-native-audio-preview-12-2025"
    _MODEL_CACHE["model"] = chosen
    _MODEL_CACHE["ts"] = now
    print(f"[gemini-rt] resolved live model: {chosen} ({len(live)} live models available)")
    return chosen


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

    now = datetime.datetime.now(tz=datetime.timezone.utc)
    # Ephemeral token WITH constraints, minted via the google-genai SDK. Ephemeral
    # tokens are only accepted on the CONSTRAINED endpoint, and the raw-REST field
    # names differ from the SDK's — so the SDK is the reliable path. Constraints pin
    # the model, voice, modality, interviewer instructions and transcription, so the
    # browser sends only a minimal setup and none of it is client-tamperable.
    model_id = _resolve_live_model()
    constraints_config = {
        "response_modalities": ["AUDIO"],
        "system_instruction": instructions,
        "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": GEMINI_LIVE_VOICE}}},
        "input_audio_transcription": {},
        "output_audio_transcription": {},
    }
    try:
        t0 = time.time()
        from google import genai  # lazy import: keeps its cost off every other route
        gclient = genai.Client(api_key=GEMINI_API_KEY)

        def _mint(cfg):
            return gclient.auth_tokens.create(config={
                "uses": 2,
                "expire_time": now + datetime.timedelta(minutes=30),
                "new_session_expire_time": now + datetime.timedelta(minutes=2),
                "live_connect_constraints": {"model": model_id, "config": cfg},
            })

        # Tighter turn-taking: shorten how long Gemini waits after the candidate
        # stops speaking before it replies (default is long -> the "5-6s" lag).
        # If the field is ever rejected, fall back to the plain config so voice
        # never breaks over a latency tweak.
        tuned = dict(constraints_config)
        tuned["realtime_input_config"] = {"automatic_activity_detection": {"silence_duration_ms": 500}}
        try:
            tok = _mint(tuned)
        except Exception as e:
            print(f"[gemini-rt] VAD-tuned config rejected ({e}); minting plain config")
            tok = _mint(constraints_config)

        token_name = getattr(tok, "name", None)
        if not token_name:
            raise HTTPException(status_code=502, detail="Voice session token missing.")
        log_ai_usage(
            user_id=uid, endpoint="/realtime-gemini/session", model=GEMINI_LIVE_MODEL,
            audio_minutes=0, latency_ms=int((time.time() - t0) * 1000), success=True,
            meta={"case_id": body.case_id, "attempt_id": body.attempt_id},
        )
        return {
            "token": token_name,
            "ws_url": f"{WS_BASE}?access_token={token_name}",
            "model": f"models/{model_id}",
            "voice": GEMINI_LIVE_VOICE,
            "instructions": instructions,
            # Exact setup the browser should send. Constraints supply the config, so
            # this stays minimal; kept here so it is tunable without a UI redeploy.
            "setup": {"model": f"models/{model_id}"},
            "credits": get_balance(supabase, uid, quota["tier"]),
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[gemini-rt] session error: {e}")
        raise HTTPException(status_code=502, detail=f"Gemini token error: {str(e)[:300]}")


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
