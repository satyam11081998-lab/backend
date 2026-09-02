import os
import time
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Header, HTTPException
from openai import OpenAI
from dotenv import load_dotenv

from services.supabase_client import get_supabase_client
from services.auth import get_verified_user, is_guest_user
from services.rate_limit import check_rate_limit
from services.ai_usage import (
    assert_voice_quota,
    assert_daily_budget,
    get_ai_input_quota,
    log_ai_usage,
)

load_dotenv()

# Bounded client: a hung Whisper call fails fast instead of tying up the worker.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=60.0, max_retries=1)

# Optional Groq Whisper — the SAME Whisper model, ~9x cheaper ($0.04/hr vs $0.36/hr)
# and faster, exposed on an OpenAI-compatible endpoint so it is a drop-in. If
# GROQ_API_KEY is unset we transparently stay on OpenAI; if a Groq call fails at
# request time we fall back to OpenAI, so the cheaper provider can never break voice.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
_groq_client = (
    OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1", timeout=60.0, max_retries=1)
    if GROQ_API_KEY else None
)
_STT_PROMPT = (
    "Consulting case interview answer. Expected terms: EBITDA, CAGR, profitability, "
    "revenues, fixed costs, variable costs, market size, competitors."
)


def _run_stt(filename: str, file_bytes: bytes):
    """Transcribe via Groq when configured, else OpenAI. On any Groq error, fall
    back to OpenAI so a cheaper provider can never break voice input.
    Returns (transcription, model_name_used)."""
    if _groq_client is not None:
        try:
            tr = _groq_client.audio.transcriptions.create(
                model=GROQ_STT_MODEL, file=(filename, file_bytes),
                response_format="verbose_json", prompt=_STT_PROMPT,
            )
            return tr, GROQ_STT_MODEL
        except Exception as e:
            print(f"[transcribe] Groq STT failed ({e}); falling back to OpenAI whisper-1")
    tr = client.audio.transcriptions.create(
        model="whisper-1", file=(filename, file_bytes),
        response_format="verbose_json", prompt=_STT_PROMPT,
    )
    return tr, "whisper-1"


router = APIRouter()

# ~6 MB ≈ 5-6 min of webm/opus — comfortably covers a spoken case answer while
# blocking someone from streaming huge files to burn Whisper minutes.
MAX_AUDIO_BYTES = 6 * 1024 * 1024


@router.post("")
async def transcribe_audio(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(default=None),
):
    """
    Transcribe a short audio clip (from the client's MediaRecorder) via Whisper.

    Guarded: requires a valid Supabase JWT, is rate-limited, is bounded by the
    caller's per-tier daily voice-minute quota, and is size-capped. Every call is
    logged to ai_usage_log with its billed minutes.
    """
    supabase = get_supabase_client()
    uid, user_obj = get_verified_user(supabase, authorization)          # 401 if missing/invalid
    if is_guest_user(user_obj):
        raise HTTPException(status_code=403, detail="Create an account to use voice input.")
    check_rate_limit(f"transcribe:{uid}", max_calls=12, window_seconds=60)
    assert_daily_budget()                                        # 503 if global cap hit
    assert_voice_quota(supabase, uid)                            # 429 if user out of minutes

    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")
        if len(file_bytes) > MAX_AUDIO_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Audio too long — please keep voice input under ~5 minutes.",
            )

        filename = file.filename if file.filename else "audio.webm"
        if not filename.endswith((".webm", ".mp4", ".mp3", ".wav", ".m4a", ".ogg")):
            filename = "audio.webm"  # default for MediaRecorder

        t0 = time.time()
        # verbose_json returns the exact `duration` (seconds) so we bill real minutes,
        # not a byte-size estimate. Routes to Groq when configured (see _run_stt).
        transcription, stt_model = _run_stt(filename, file_bytes)
        latency_ms = int((time.time() - t0) * 1000)

        duration_s = getattr(transcription, "duration", None)
        minutes = (float(duration_s) / 60.0) if duration_s else (len(file_bytes) / (1024 * 1024))
        text = getattr(transcription, "text", "") or ""

        log_ai_usage(
            user_id=uid, endpoint="/transcribe", model=stt_model,
            audio_minutes=minutes, latency_ms=latency_ms, success=True,
            meta={"bytes": len(file_bytes), "provider": "groq" if stt_model != "whisper-1" else "openai"},
        )

        return {"text": text, "quota": get_ai_input_quota(supabase, uid)}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to transcribe audio: {str(e)}")
