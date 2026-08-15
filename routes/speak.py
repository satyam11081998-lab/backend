import os
import time
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

from services.supabase_client import get_supabase_client
from services.auth import get_verified_user, is_guest_user
from services.rate_limit import check_rate_limit
from services.ai_usage import (
    assert_daily_budget,
    get_ai_input_quota,
    log_ai_usage,
    TTS_CHARS_PER_MIN,
)

load_dotenv()

# Bounded client: a hung TTS call fails fast instead of tying up the worker,
# same posture as /transcribe.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=60.0, max_retries=1)

router = APIRouter()

# Named constants so swapping the voice is a one-line change. The voice is a
# product decision (it IS the interviewer, to a candidate wearing headphones) —
# audition it on a real interviewer reply before hardening.
TTS_MODEL = os.getenv("TTS_MODEL", "tts-1")
TTS_VOICE = os.getenv("TTS_VOICE", "alloy")
TTS_FORMAT = "mp3"

# The interviewer speaks in 1-3 sentences by prompt contract, and the client
# sends ONE sentence per call. 1200 chars is generous headroom; anything larger
# is not an interviewer turn, it is someone using us as a free TTS endpoint.
MAX_TEXT_CHARS = 1200


class SpeakRequest(BaseModel):
    text: str


@router.post("")
async def speak(
    body: SpeakRequest,
    authorization: Optional[str] = Header(default=None),
):
    """
    Speak one sentence of interviewer text. Returns audio/mpeg bytes.

    Guarded exactly like /transcribe, plus a Pro gate: requires a valid Supabase
    JWT, blocks guests, is Pro-only, is rate-limited, is bounded by the global
    daily budget AND the caller's per-day TTS-minute quota, and is size-capped.
    Every call is logged to ai_usage_log with its billed minutes.

    The Pro gate here is the REAL one. lib/tier.ts carries a mirror flag for the
    UI, but a client flag is a suggestion — this is the enforcement.
    """
    supabase = get_supabase_client()
    uid, user_obj = get_verified_user(supabase, authorization)          # 401 if missing/invalid
    if is_guest_user(user_obj):
        raise HTTPException(status_code=403, detail="Create an account to use voice interview mode.")

    # One interviewer reply is 1-3 sentences = 1-3 calls. 40/min leaves room for
    # a fast exchange without letting a script stream a novel through us.
    check_rate_limit(f"speak:{uid}", max_calls=40, window_seconds=60)
    assert_daily_budget()                                        # 503 if global cap hit

    # ONE quota snapshot serves the tier gate, the daily meter and the response
    # header. Doing it the obvious way — effective_tier() + assert_tts_quota() +
    # a second get_ai_input_quota() for the header — costs NINE Supabase
    # round-trips per sentence, and an interviewer reply is 1-3 sentences. That
    # is 20-30 round-trips per turn on the one path where latency is the whole
    # product. This is four.
    quota = get_ai_input_quota(supabase, uid)
    if quota["tier"] != "pro":
        raise HTTPException(
            status_code=403,
            detail="Voice interview mode is a Pro feature. Upgrade to talk through a case out loud.",
        )
    speak_quota = quota["speak"]
    if speak_quota["remaining_min"] <= 0:
        raise HTTPException(
            status_code=429,
            detail=f"Daily voice-interview limit reached ({speak_quota['limit_min']} min). "
                   f"Resets at midnight IST — you can carry on in the chat.",
        )

    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Nothing to speak")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Text too long to speak (max {MAX_TEXT_CHARS} characters).",
        )

    try:
        t0 = time.time()
        speech = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
            response_format=TTS_FORMAT,
        )
        audio_bytes = speech.read() if hasattr(speech, "read") else speech.content
        latency_ms = int((time.time() - t0) * 1000)

        # The TTS response carries no duration, so bill on characters. This is an
        # estimate for guardrails, not billing truth — same contract as the rest
        # of ai_usage.
        minutes = len(text) / TTS_CHARS_PER_MIN

        log_ai_usage(
            user_id=uid, endpoint="/speak", model=TTS_MODEL,
            audio_minutes=minutes, latency_ms=latency_ms, success=True,
            meta={"chars": len(text), "voice": TTS_VOICE, "bytes": len(audio_bytes)},
        )

        # Deliberately NO quota header. A custom response header is invisible to
        # the browser unless CORSMiddleware sets `expose_headers`, which main.py
        # does not — so an X-Speak-Remaining-Min would have been dead weight that
        # merely looked like a feature. The client already refreshes the full
        # quota (including the `speak` block) from every /transcribe response,
        # which happens once per spoken turn anyway.
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={"Cache-Control": "no-store"},
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error synthesizing speech: {e}")
        log_ai_usage(
            user_id=uid, endpoint="/speak", model=TTS_MODEL,
            audio_minutes=0, success=False, meta={"chars": len(text), "error": str(e)[:200]},
        )
        raise HTTPException(status_code=500, detail=f"Failed to synthesize speech: {str(e)}")
