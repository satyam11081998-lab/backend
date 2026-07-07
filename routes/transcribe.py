import os
import time
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Header, HTTPException
from openai import OpenAI
from dotenv import load_dotenv

from services.supabase_client import get_supabase_client
from services.auth import get_verified_user_id
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
    uid = get_verified_user_id(supabase, authorization)          # 401 if missing/invalid
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
        # not a byte-size estimate.
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=(filename, file_bytes),
            response_format="verbose_json",
            prompt="Consulting case interview answer. Expected terms: EBITDA, CAGR, profitability, revenues, fixed costs, variable costs, market size, competitors.",
        )
        latency_ms = int((time.time() - t0) * 1000)

        duration_s = getattr(transcription, "duration", None)
        minutes = (float(duration_s) / 60.0) if duration_s else (len(file_bytes) / (1024 * 1024))
        text = getattr(transcription, "text", "") or ""

        log_ai_usage(
            user_id=uid, endpoint="/transcribe", model="whisper-1",
            audio_minutes=minutes, latency_ms=latency_ms, success=True,
            meta={"bytes": len(file_bytes)},
        )

        return {"text": text, "quota": get_ai_input_quota(supabase, uid)}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to transcribe audio: {str(e)}")
