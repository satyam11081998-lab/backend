"""
Backend for the browser E2E voice test (company/qa/e2e-voice/run.cjs + run-stt.cjs).

Runs the REAL FastAPI routes (attempts + realtime) with the unified brain ON, an
in-memory database and a scripted interviewer model, so the browser client can be
driven end to end without Supabase or provider keys. /realtime/session mints its
"client secret" from the mock realtime server instead of api.openai.com.

    python -m tools.e2e_voice_backend --port 8765 --mock http://127.0.0.1:8766
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["INTERVIEWER_BRAIN"] = "on"
os.environ["ADAPTIVE_INTERVIEWER"] = "true"
os.environ.setdefault("LOG_TURN_TIMING", "0")

from tests.interviewer_fakes import FakeDB, FakeLLM, install_route_fakes  # noqa: E402

from fastapi import FastAPI, File, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
import uvicorn  # noqa: E402

import routes.attempts as att  # noqa: E402
import routes.realtime as rt  # noqa: E402
from services.interviewer import dedupe, engine, telemetry  # noqa: E402
from services.interviewer.assessor import Assessment  # noqa: E402

S = {"db": FakeDB([], case_type="guesstimate"), "llm": FakeLLM(), "delay_ms": 0, "timing": [], "turns": []}


class SlowLLM:
    """FakeLLM with an optional delay, to reproduce 'decision arrives after the candidate
    started speaking again'."""

    def complete(self, messages, *, max_tokens):
        if S["delay_ms"]:
            time.sleep(S["delay_ms"] / 1000)
        return S["llm"].complete(messages, max_tokens=max_tokens)

    def stream(self, messages, *, max_tokens, meta):
        if S["delay_ms"]:
            time.sleep(S["delay_ms"] / 1000)
        yield from S["llm"].stream(messages, max_tokens=max_tokens, meta=meta)


def build(mock_base: str) -> FastAPI:
    install_route_fakes(att, lambda: S["db"])
    engine.LLM_FACTORY = lambda plan: SlowLLM()
    engine.default_assessor = lambda uid: (lambda **kw: Assessment(material=False))
    att.log_realtime_usage = lambda **kw: None
    att.deduct_realtime_credit = lambda *a, **k: None

    # realtime session route: same code, fake gates, secrets from the mock far end
    rt.OPENAI_API_KEY = "sk-test"
    rt.CLIENT_SECRETS_URL = mock_base + "/v1/realtime/client_secrets"
    rt.get_supabase_client = lambda: S["db"]
    rt.get_verified_user = lambda sb, auth: ("u1", {"id": "u1"})
    rt.check_rate_limit = lambda *a, **k: None
    rt.assert_daily_budget = lambda *a, **k: None
    rt.get_ai_input_quota = lambda sb, uid: {"tier": "pro", "voice": {"unlimited": True, "remaining_min": 99}}
    rt.has_credit = lambda *a, **k: True
    rt.get_balance = lambda *a, **k: {"total_remaining": 60}
    rt.log_ai_usage = lambda **kw: None

    orig_timing = telemetry.emit_timing
    orig_turn = telemetry.emit_turn

    def cap_timing(report):
        ev = orig_timing(report)
        S["timing"].append(ev)
        return ev

    def cap_turn(event):
        ev = orig_turn(event)
        S["turns"].append(ev)
        return ev

    telemetry.emit_timing = cap_timing
    telemetry.emit_turn = cap_turn

    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(att.router)
    app.include_router(rt.router, prefix="/realtime")

    # ---- STT talk mode (pipeline) fakes: scripted transcription + a short WAV for /speak ----
    import io, math, struct, wave
    from fastapi.responses import Response as _Resp

    def _wav(ms: int = 350) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            n = int(16000 * ms / 1000)
            w.writeframes(b"".join(struct.pack("<h", int(3000 * math.sin(2 * math.pi * 440 * i / 16000))) for i in range(n)))
        return buf.getvalue()

    S.setdefault("stt_script", [])
    S.setdefault("transcribe_calls", 0)
    S.setdefault("speak_texts", [])

    @app.post("/transcribe")
    async def fake_transcribe(file: UploadFile = File(...)):
        data = await file.read()
        S["transcribe_calls"] += 1
        text = S["stt_script"].pop(0) if S["stt_script"] else ""
        S.setdefault("transcribe_sizes", []).append(len(data))
        return {"text": text}

    @app.post("/speak")
    def fake_speak(body: dict):
        S["speak_texts"].append(body.get("text", ""))
        return _Resp(content=_wav(), media_type="audio/wav")

    @app.post("/__e2e/stt-script")
    def stt_script(body: dict):
        S["stt_script"] = list(body.get("texts", []))
        S["transcribe_calls"] = 0
        S["speak_texts"] = []
        return {"ok": True}

    @app.get("/__e2e/stt")
    def stt_state():
        return {"transcribe_calls": S["transcribe_calls"], "speak_texts": S["speak_texts"],
                "remaining": S["stt_script"], "sizes": S.get("transcribe_sizes", [])}

    @app.get("/__e2e/db")
    def db_dump():
        att._await_after_turn("a1")
        return {"messages": S["db"].rows(), "session_state": S["db"].session_state(),
                "llm_calls": [c["move"] for c in S["llm"].calls]}

    @app.get("/__e2e/telemetry")
    def tel():
        return {"timing": S["timing"], "turns": S["turns"]}

    @app.post("/__e2e/llm")
    def set_llm(body: dict):
        S["llm"] = FakeLLM(body.get("mode", "good"))
        S["delay_ms"] = int(body.get("delay_ms", 0))
        return {"ok": True}

    @app.post("/__e2e/reset")
    def reset():
        S["db"] = FakeDB([], case_type="guesstimate")
        S["llm"] = FakeLLM()
        S["delay_ms"] = 0
        S["timing"].clear()
        S["turns"].clear()
        dedupe.LEDGER.__init__()
        dedupe.ROWS.__init__()
        return {"ok": True}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--mock", default="http://127.0.0.1:8766")
    a = ap.parse_args()
    uvicorn.run(build(a.mock), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
