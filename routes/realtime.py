import os
import time
from typing import Optional

import httpx
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from services.supabase_client import get_supabase_client
from services.auth import get_verified_user, is_guest_user
from services.rate_limit import check_rate_limit
from services.ai_usage import assert_daily_budget, get_ai_input_quota, log_ai_usage
from services.realtime_credits import has_credit, get_balance
from prompts.voice_renderer import VOICE_RENDERER_INSTRUCTIONS

load_dotenv()

router = APIRouter()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLIENT_SECRETS_URL = "https://api.openai.com/v1/realtime/client_secrets"

# The realtime family moves fast — keep this in one env-overridable place so a
# model bump is a config change, not a redeploy.
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-realtime-2.1")
REALTIME_VOICE = os.getenv("REALTIME_VOICE", "alloy")

# Kill switch, mirroring AI_TTS_MIN_PRO for the pipeline. Set to "0" to disable
# realtime voice for everyone WITHOUT a redeploy.
REALTIME_ENABLED = os.getenv("REALTIME_ENABLED", "1") != "0"

# Session ceiling, matching MAX_SESSION_MS in the client.
#
# HONEST SCOPE: this value is ADVISORY. It is returned to the client, which
# enforces it. Nothing here stops a tampered client from holding a peer
# connection open longer — once the SDP handshake completes, the audio flows
# browser-to-OpenAI and this process is not in the path to cut it off.
#
# What actually bounds a hostile client is the ephemeral token's own expiry
# (below) and `assert_daily_budget()` catching the spend after the fact. If a
# hard server-side cap is ever needed, it has to come from a short token TTL,
# not from this constant.
MAX_SESSION_SECONDS = int(os.getenv("REALTIME_MAX_SESSION_SECONDS", "600"))

# Per-IP daily cap on NON-PRO (guest/free) voice sessions, bounding cost when a
# single network cycles anonymous identities into fresh trials. In-memory +
# per-instance (advisory); assert_daily_budget() is the hard backstop. Raise it
# if legitimate shared-IP users (college / office wifi) hit it; env-tunable.
REALTIME_FREE_IP_PER_DAY = int(os.getenv("REALTIME_FREE_IP_PER_DAY", "10"))

# TURN DETECTION — the single biggest driver of "it doesn't wait / it repeats".
#
# semantic_vad (default) uses a classifier on the candidate's WORDS to decide they
# have actually finished a thought, instead of firing on a fixed silence timer.
# A fixed timer is a losing trade: any value either cuts off a candidate who
# pauses mid-calculation or adds dead air after a quick answer. semantic_vad waits
# through the thinking pause and fires when the sentence sounds complete —
# ~2% mid-sentence interruptions vs ~18% for silence timeouts in published tests.
# eagerness "low" = let them take their time (best for someone working out math
# out loud); "high" = chunk as soon as possible. interrupt_response keeps barge-in.
#
# server_vad is kept as an env fallback (set REALTIME_TURN_MODE=server_vad) in case
# a model rejects semantic_vad; REALTIME_VAD_SILENCE_MS then pads the silence
# window well above the ~0.5s default so a mid-calc pause isn't read as "done".
REALTIME_TURN_MODE = os.getenv("REALTIME_TURN_MODE", "semantic_vad")
# Speed note (2026-09-28): "low" gives the candidate the most time -- and so the
# longest silence after a hesitant answer. REALTIME_SEMANTIC_EAGERNESS=medium
# (OpenAI's own default) or =high shortens that wait; it also changes where
# turns are cut, so it is left as an owner decision rather than a default.
REALTIME_SEMANTIC_EAGERNESS = os.getenv("REALTIME_SEMANTIC_EAGERNESS", "low")
REALTIME_VAD_SILENCE_MS = int(os.getenv("REALTIME_VAD_SILENCE_MS", "1400"))

# Input transcription model for realtime voice turns. whisper-1 transcribes the
# committed turn in one batch, which sits directly on the candidate-stops-speaking
# -> interviewer-speaks path. Env-selectable so gpt-4o-mini-transcribe /
# gpt-live-transcribe can be A/B-measured (tools/voice_latency_report.py) without a
# deploy (unchanged default: measured alternatives must win on accuracy too).
REALTIME_TRANSCRIBE_MODEL = os.getenv("REALTIME_TRANSCRIBE_MODEL", "whisper-1")

# Where the voice model's spoken line is generated. "none" = an out-of-band
# response that sees ONLY the approved line (not the session audio): the model
# cannot react to the candidate on its own, and per-line input cost does not grow
# with session length. "auto" = the previous in-band behaviour. Sent to the
# browser so it can be flipped without a frontend deploy.
REALTIME_RESPONSE_CONVERSATION = os.getenv("REALTIME_RESPONSE_CONVERSATION", "none").strip().lower()
if REALTIME_RESPONSE_CONVERSATION not in ("none", "auto"):
    REALTIME_RESPONSE_CONVERSATION = "none"

# STT talk mode over OpenAI realtime TRANSCRIPTION sessions (gpt-live-transcribe).
# The browser only uses it when built with NEXT_PUBLIC_STT_TRANSPORT=live.
STT_LIVE_MODEL = os.getenv("STT_LIVE_MODEL", "gpt-live-transcribe")
STT_LIVE_DELAY = os.getenv("STT_LIVE_DELAY", "low")
STT_LIVE_LANGUAGES = [x.strip() for x in os.getenv("STT_LIVE_LANGUAGES", "en").split(",") if x.strip()]
STT_LIVE_KEYWORDS = [x.strip() for x in os.getenv(
    "STT_LIVE_KEYWORDS",
    "crore,lakh,EBITDA,CAGR,MECE,guesstimate,profitability,revenue,fixed cost,variable cost,market size,"
    "penetration,break-even,unit economics,basis points,percentage points",
).split(",") if x.strip()][:40]


def build_turn_detection() -> dict:
    """Turn-detection config for the realtime session, chosen by REALTIME_TURN_MODE.

    Default is semantic_vad (words-based end-of-turn); server_vad (fixed silence)
    is the env-selectable fallback. Both keep barge-in — the candidate can always
    cut the interviewer off mid-sentence.

    create_response is OFF in both: the realtime model never answers a candidate
    turn on its own. The browser sends the final transcript to V11
    (/attempts/{id}/voice-decision) and only then issues response.create with the
    line V11 approved — or nothing at all, for V11 SILENCE.
    """
    if REALTIME_TURN_MODE == "semantic_vad":
        return {
            "type": "semantic_vad",
            "eagerness": REALTIME_SEMANTIC_EAGERNESS,
            "create_response": False,
            "interrupt_response": True,
        }
    return {
        "type": "server_vad",
        "threshold": 0.5,
        "prefix_padding_ms": 300,
        "silence_duration_ms": REALTIME_VAD_SILENCE_MS,
        "create_response": False,
        "interrupt_response": True,
    }


class RealtimeSessionRequest(BaseModel):
    """`case_id` lets us build the interviewer instructions server-side."""
    case_id: str
    attempt_id: Optional[str] = None


@router.post("/session")
async def create_realtime_session(
    body: RealtimeSessionRequest,
    authorization: Optional[str] = Header(default=None),
    x_forwarded_for: Optional[str] = Header(default=None, alias="X-Forwarded-For"),
):
    """
    Mint a short-lived client secret so the browser can open a WebRTC session
    with OpenAI directly.

    Why the browser connects directly: the pipeline's latency was dominated by
    two network hops (browser -> Render -> OpenAI) on BOTH the transcribe and
    the speak call. Realtime removes them. What we keep is everything that
    matters — the real API key never leaves this process, the tier gate is
    enforced here, and the session's instructions are set here rather than
    accepted from the client.

    The realtime model is only the interviewer's VOICE. MECE Interviewer V11
    decides every turn (/attempts/{id}/voice-decision), exactly as on the typed
    path, so the session carries the voice-renderer instructions and no
    interviewer prompt, rules or case content of its own.
    """
    if not REALTIME_ENABLED:
        raise HTTPException(status_code=503, detail="Voice interview is temporarily unavailable.")
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Realtime is not configured on the server.")

    supabase = get_supabase_client()
    uid, user_obj = get_verified_user(supabase, authorization)          # 401
    # Guests (anonymous auth) may try voice too — the one-time credit trial plus
    # the daily kill-switch bound the cost, and they convert to a real account at
    # the score (GuestSaveWall), which carries their usage over.

    # A session is expensive relative to a chat turn, so this is deliberately
    # tighter than /speak's 40/min.
    check_rate_limit(f"realtime:{uid}", max_calls=6, window_seconds=60)
    assert_daily_budget()                                              # 503 global backstop

    # ONE snapshot serves the tier gate. Do not reintroduce a second read here;
    # see the note in services/ai_usage.assert_tts_quota about round-trips.
    quota = get_ai_input_quota(supabase, uid)
    tier = quota["tier"]
    # Free-trial voice is capped tighter than Pro (advisory — the hard lifetime
    # bound is the credit balance below, burnt down by the per-turn deduction).
    session_cap = MAX_SESSION_SECONDS if tier == "pro" else int(os.getenv("REALTIME_FREE_SESSION_SECONDS", "420"))

    # Per-IP daily cap for NON-PRO (guest/free) voice, so one network can't cycle
    # anonymous identities into unlimited free trials. In-memory + per-instance
    # (advisory); assert_daily_budget() is the hard backstop. Pro is never capped.
    if tier != "pro":
        client_ip = (x_forwarded_for or "").split(",")[0].strip() or "unknown"
        try:
            check_rate_limit(f"rt_ip_day:{client_ip}", max_calls=REALTIME_FREE_IP_PER_DAY, window_seconds=86400)
        except HTTPException:
            raise HTTPException(
                status_code=429,
                detail="This network has reached today's free voice limit. Upgrade to Pro for unlimited voice, or try again tomorrow.",
            )

    # Real-time voice is CREDIT-metered for EVERYONE (it costs ~10x the Groq
    # pipeline). Pro gets a monthly included allowance; a non-Pro user is seeded a
    # ONE-TIME free trial (~2 short interviews) the first time get_balance runs,
    # inside has_credit() just below. So the gate is simply "do you have credit?" —
    # which lets a free user try the best-quality voice once, then hit the upsell.
    if not has_credit(supabase, uid, tier):
        if tier == "pro":
            raise HTTPException(
                status_code=402,
                detail="You're out of real-time interview minutes. Buy a minute pack, or switch to the standard voice mode — it's unlimited on Pro.",
            )
        raise HTTPException(
            status_code=402,
            detail="You've used your free voice interviews. Upgrade to Pro to keep talking through cases out loud.",
        )

    case = (
        supabase.table("cases")
        .select("id, title, type, content")
        .eq("id", body.case_id)
        .limit(1)
        .execute()
    )
    if not case.data:
        raise HTTPException(status_code=404, detail="Case not found")

    # The realtime model is a voice only: V11 decides every interviewer turn via
    # /attempts/{id}/voice-decision, so no interviewer prompt or case content
    # goes into this session (see prompts/voice_renderer.py).
    instructions = VOICE_RENDERER_INSTRUCTIONS

    payload = {
        "session": {
            "type": "realtime",
            "model": REALTIME_MODEL,
            "instructions": instructions,
            "audio": {
                "input": {
                    # Words-based end-of-turn (semantic_vad) by default so a
                    # thinking pause mid-calculation is NOT mistaken for "done" —
                    # that was the #1 cause of the interviewer interrupting and
                    # repeating itself. See build_turn_detection() and the
                    # REALTIME_TURN_MODE note above; barge-in stays on.
                    "turn_detection": build_turn_detection(),
                    "transcription": {"model": REALTIME_TRANSCRIBE_MODEL},
                },
                "output": {"voice": REALTIME_VOICE},
            },
        }
    }

    try:
        t0 = time.time()
        async with httpx.AsyncClient(timeout=20.0) as client:
            res = await client.post(
                CLIENT_SECRETS_URL,
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        latency_ms = int((time.time() - t0) * 1000)

        if res.status_code >= 400:
            # Surface OpenAI's reason rather than a bare 500 — a model-name
            # change is the likeliest failure and it should be obvious.
            print(f"[realtime] client_secrets failed {res.status_code}: {res.text[:500]}")
            raise HTTPException(
                status_code=502,
                detail=f"Could not start the voice session ({res.status_code}).",
            )

        data = res.json()

        # Session creation itself carries no audio, so it books no cost. The
        # REAL metering happens per turn from the client's `response.done`
        # usage, via /attempts/{id}/realtime-turn. Without that, realtime spend
        # is invisible to spend_today_usd() and therefore to the daily-budget
        # kill switch. This row exists so sessions are at least countable.
        log_ai_usage(
            user_id=uid, endpoint="/realtime/session", model=REALTIME_MODEL,
            audio_minutes=0, latency_ms=latency_ms, success=True,
            meta={"case_id": body.case_id, "attempt_id": body.attempt_id},
        )

        return {
            "client_secret": data.get("value") or data.get("client_secret"),
            "expires_at": data.get("expires_at"),
            "model": REALTIME_MODEL,
            "voice": REALTIME_VOICE,
            "max_session_seconds": session_cap,
            "credits": get_balance(supabase, uid, tier),
            # Additive: how the browser should create spoken responses (see
            # REALTIME_RESPONSE_CONVERSATION) and which transcription model is live.
            "response_conversation": REALTIME_RESPONSE_CONVERSATION,
            "transcribe_model": REALTIME_TRANSCRIBE_MODEL,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[realtime] session error: {e}")
        raise HTTPException(status_code=500, detail=f"Could not start the voice session: {e}")


@router.get("/credits")
def realtime_credits_balance(authorization: Optional[str] = Header(default=None)):
    """Current real-time minute balance for the caller (included + purchased).
    The talk-mode UI reads this to show 'X min left' and to know when to show the
    buy-minutes paywall instead of connecting."""
    supabase = get_supabase_client()
    uid, user_obj = get_verified_user(supabase, authorization)
    if is_guest_user(user_obj):
        return {"total_remaining": 0, "included_remaining": 0, "purchased_remaining": 0, "tier": "guest"}
    tier = get_ai_input_quota(supabase, uid)["tier"]
    return get_balance(supabase, uid, tier)


# =============================================================================
# STT talk mode: realtime TRANSCRIPTION session (gpt-live-transcribe)
# =============================================================================
class TranscriptionSessionRequest(BaseModel):
    case_id: Optional[str] = None
    attempt_id: Optional[str] = None


def build_transcription_session() -> dict:
    """Session config for a transcription-only realtime session.

    gpt-live-transcribe does not support server_vad / semantic_vad, so
    turn_detection is null and the browser commits each turn itself (client VAD),
    per developers.openai.com/api/docs/guides/realtime-transcription (2026-09-29).
    """
    transcription = {"model": STT_LIVE_MODEL}
    if "live-transcribe" in STT_LIVE_MODEL:
        transcription["delay"] = STT_LIVE_DELAY
        if STT_LIVE_LANGUAGES:
            transcription["languages"] = STT_LIVE_LANGUAGES
        if STT_LIVE_KEYWORDS:
            transcription["keywords"] = STT_LIVE_KEYWORDS
    elif STT_LIVE_LANGUAGES:
        transcription["language"] = STT_LIVE_LANGUAGES[0]
    return {
        "type": "transcription",
        "audio": {"input": {"transcription": transcription, "turn_detection": None}},
    }


@router.post("/transcription-session")
async def create_transcription_session(
    body: TranscriptionSessionRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Mint a short-lived client secret for a realtime transcription session.

    Same gates as /transcribe (auth, no guests, rate limit, global budget, per-user
    daily voice minutes). The real API key never reaches the browser.
    """
    from services.ai_usage import check_voice_quota
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Live transcription is not configured on the server.")
    supabase = get_supabase_client()
    uid, user_obj = get_verified_user(supabase, authorization)
    if is_guest_user(user_obj):
        raise HTTPException(status_code=403, detail="Create an account to use voice input.")
    check_rate_limit(f"stt-live:{uid}", max_calls=6, window_seconds=60)
    assert_daily_budget()
    check_voice_quota(get_ai_input_quota(supabase, uid))

    payload = {"expires_after": {"anchor": "created_at", "seconds": 600}, "session": build_transcription_session()}
    try:
        t0 = time.time()
        async with httpx.AsyncClient(timeout=20.0) as client:
            res = await client.post(CLIENT_SECRETS_URL, headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}, json=payload)
        latency_ms = int((time.time() - t0) * 1000)
        if res.status_code >= 400:
            print(f"[stt-live] client_secrets failed {res.status_code}: {res.text[:500]}")
            raise HTTPException(status_code=502, detail=f"Could not start live transcription ({res.status_code}).")
        data = res.json()
        log_ai_usage(user_id=uid, endpoint="/realtime/transcription-session", model=STT_LIVE_MODEL,
                     audio_minutes=0, latency_ms=latency_ms, success=True,
                     meta={"case_id": body.case_id, "attempt_id": body.attempt_id})
        return {"client_secret": data.get("value") or data.get("client_secret"), "expires_at": data.get("expires_at"),
                "model": STT_LIVE_MODEL, "delay": STT_LIVE_DELAY}
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"[stt-live] session error: {e}")
        raise HTTPException(status_code=500, detail="Could not start live transcription.")


class TranscriptionUsageRequest(BaseModel):
    seconds: float


@router.post("/transcription-usage")
def transcription_usage(body: TranscriptionUsageRequest, authorization: Optional[str] = Header(default=None)):
    """Meter streamed live-transcription audio under the SAME '/transcribe' endpoint as
    Whisper, so the existing per-user daily voice-minute quota applies unchanged.
    The client reports incremental seconds; one report is clamped to 10 minutes."""
    supabase = get_supabase_client()
    uid, user_obj = get_verified_user(supabase, authorization)
    if is_guest_user(user_obj):
        return {"ok": True}
    check_rate_limit(f"stt-live-usage:{uid}", max_calls=60, window_seconds=60)
    secs = max(0.0, min(float(body.seconds or 0), 600.0))
    if secs > 0:
        log_ai_usage(user_id=uid, endpoint="/transcribe", model=STT_LIVE_MODEL, audio_minutes=secs / 60.0,
                     success=True, meta={"provider": "openai", "src": "live-transcribe"})
    return {"ok": True}
