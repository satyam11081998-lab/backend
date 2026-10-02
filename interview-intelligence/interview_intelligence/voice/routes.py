"""Voice I/O (spec §29) — behind `voice.enabled` (default off).

This release ships turn-based voice: the browser records an answer, II transcribes it with
its OWN STT route, the transcript goes through the normal /turns path (so scoring sees the
same document as a typed answer), and interviewer lines can be spoken with II's own TTS.
Streaming STT/TTS with barge-in is designed (docs/BUILD_STATUS.md) but not built."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from ..access import flags
from ..ai.provider import ProviderError
from ..ai.routing import resolve
from ..ai.runner import record_media_run
from ..auth.assertion import Principal
from ..errors import Forbidden, Unavailable, Unprocessable
from ..api.deps import principal, unit, use

router = APIRouter(prefix="/v1/voice")
MAX_AUDIO = 10 * 1024 * 1024
AUDIO_TYPES = {"audio/webm", "audio/ogg", "audio/mpeg", "audio/mp4", "audio/wav", "audio/x-wav", "audio/m4a",
               "audio/aac", "video/webm"}


def _require_voice(db) -> None:
    if not flags.flag(db, "voice.enabled"):
        raise Forbidden("Voice mode isn't available yet. You can type your answers.", code="voice_disabled")


@router.post("/transcribe")
def transcribe(audio: UploadFile = File(...), p: Principal = Depends(principal)):
    data = audio.file.read(MAX_AUDIO + 1)
    if len(data) > MAX_AUDIO:
        raise Unprocessable("That recording is too long. Keep answers under a few minutes.", code="audio_too_large")
    mime = (audio.content_type or "").split(";")[0].strip().lower()
    if mime not in AUDIO_TYPES:
        raise Unprocessable("Unsupported audio format.", code="bad_audio")
    with unit() as db:
        use(db, p, klass="voice")
        _require_voice(db)
    for provider, model in resolve("stt", "stt"):
        t0 = time.perf_counter()
        try:
            res = provider.transcribe(data, filename=audio.filename or "answer.webm", mime=mime, model=model,
                                      timeout_s=60.0)
        except ProviderError as e:
            record_media_run(stage="stt", provider=provider.name, model=model, user_id=p.user_id,
                             latency_ms=int((time.perf_counter() - t0) * 1000), status="error",
                             audio_bytes=len(data), error=str(e))
            continue
        record_media_run(stage="stt", provider=provider.name, model=model, user_id=p.user_id,
                         latency_ms=int((time.perf_counter() - t0) * 1000), status="ok", audio_bytes=len(data))
        return {"text": (res.text or "").strip()}
    raise Unavailable("We couldn't transcribe that. Please type your answer instead.", code="stt_failed")


class SpeakBody(BaseModel):
    text: str


@router.post("/speak")
def speak(body: SpeakBody, p: Principal = Depends(principal)):
    text = (body.text or "").strip()
    if not text or len(text) > 900:
        raise Unprocessable("Nothing to speak.", code="bad_text")
    with unit() as db:
        use(db, p, klass="voice")
        _require_voice(db)
    for provider, model in resolve("tts", "tts"):
        t0 = time.perf_counter()
        try:
            audio = provider.speak(text, model=model, voice="alloy", timeout_s=30.0)
        except ProviderError as e:
            record_media_run(stage="tts", provider=provider.name, model=model, user_id=p.user_id,
                             latency_ms=int((time.perf_counter() - t0) * 1000), status="error", chars=len(text),
                             error=str(e))
            continue
        record_media_run(stage="tts", provider=provider.name, model=model, user_id=p.user_id,
                         latency_ms=int((time.perf_counter() - t0) * 1000), status="ok", chars=len(text))
        return Response(content=audio, media_type="audio/mpeg")
    raise Unavailable("Voice playback is unavailable right now.", code="tts_failed")
