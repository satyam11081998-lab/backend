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
from prompts.interview_prompts import build_interviewer_messages

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


class RealtimeSessionRequest(BaseModel):
    """`case_id` lets us build the interviewer instructions server-side."""
    case_id: str
    attempt_id: Optional[str] = None


@router.post("/session")
async def create_realtime_session(
    body: RealtimeSessionRequest,
    authorization: Optional[str] = Header(default=None),
):
    """
    Mint a short-lived client secret so the browser can open a WebRTC session
    with OpenAI directly.

    Why the browser connects directly: the pipeline's latency was dominated by
    two network hops (browser -> Render -> OpenAI) on BOTH the transcribe and
    the speak call. Realtime removes them. What we keep is everything that
    matters — the real API key never leaves this process, the tier gate is
    enforced here, and the interviewer's instructions are built here rather
    than accepted from the client.

    The instructions are the product: the prompt, the behavioural guardrails and
    the case content. A browser-supplied `instructions` field would let anyone
    rewrite the interviewer, or read the case's hidden framing. So it is built
    server-side from the same prompt builder the typed path uses.
    """
    if not REALTIME_ENABLED:
        raise HTTPException(status_code=503, detail="Voice interview is temporarily unavailable.")
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Realtime is not configured on the server.")

    supabase = get_supabase_client()
    uid, user_obj = get_verified_user(supabase, authorization)          # 401
    if is_guest_user(user_obj):
        raise HTTPException(status_code=403, detail="Create an account to use voice interview mode.")

    # A session is expensive relative to a chat turn, so this is deliberately
    # tighter than /speak's 40/min.
    check_rate_limit(f"realtime:{uid}", max_calls=6, window_seconds=60)
    assert_daily_budget()                                              # 503 global backstop

    # ONE snapshot serves the tier gate. Do not reintroduce a second read here;
    # see the note in services/ai_usage.assert_tts_quota about round-trips.
    quota = get_ai_input_quota(supabase, uid)
    if quota["tier"] != "pro":
        raise HTTPException(
            status_code=403,
            detail="Voice interview is a Pro feature. Upgrade to talk through a case out loud.",
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
    case_row = case.data[0]

    # Reuse the SAME prompt builder as the typed path so the interviewer sounds
    # identical across transports. build_interviewer_messages returns a chat
    # messages array; realtime wants a single instructions string, so take the
    # system turn — that is where every behavioural rule lives.
    messages = build_interviewer_messages(
        case_content=case_row["content"],
        case_type=case_row["type"],
        transcript=[],
        new_user_message="",
        clarifications_exhausted=False,
    )
    # JOIN every system turn, not just the first. build_interviewer_messages
    # emits TWO: the behavioural rules AND the case context. Taking only the
    # first would hand the realtime model an interviewer with perfect manners
    # and no idea what case it is running.
    system_turns = [m["content"] for m in messages if m.get("role") == "system" and m.get("content")]
    instructions = "\n\n".join(system_turns).strip()
    if len(system_turns) < 2:
        # Loud, because a silently case-less interviewer is very hard to spot
        # from the outside — it just sounds vague.
        print(f"[realtime] WARNING: expected >=2 system turns, got {len(system_turns)}")
    if not instructions:
        raise HTTPException(status_code=500, detail="Could not build interviewer instructions.")

    payload = {
        "session": {
            "type": "realtime",
            "model": REALTIME_MODEL,
            "instructions": instructions,
            "audio": {
                "input": {
                    # Server VAD is what buys real barge-in: OpenAI detects the
                    # candidate starting to speak and interrupts the interviewer
                    # without a round trip through us.
                    "turn_detection": {"type": "server_vad"},
                    "transcription": {"model": "whisper-1"},
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
            "max_session_seconds": MAX_SESSION_SECONDS,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[realtime] session error: {e}")
        raise HTTPException(status_code=500, detail=f"Could not start the voice session: {e}")
