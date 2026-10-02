"""Spec §29, §91 — voice: on by default, access-gated, live calls minted server-side, every
speech call logged and costed."""

import pytest
from sqlalchemy import select

from interview_intelligence.db.models import InterviewSession, ModelRun
from interview_intelligence.db.session import db_session
from tests.conftest import ADMIN_EMAIL, auth, enable_pro, make_settings, mint
from tests.helpers import Candidate


def _cfg(client, values: dict):
    r = client.patch("/v1/admin/config", json={"values": values}, headers=auth(mint(email=ADMIN_EMAIL, tier="free")))
    assert r.status_code == 200, r.text


def test_voice_is_on_by_default_and_can_be_switched_off(client):
    enable_pro(client)
    c = Candidate(client)
    me = client.get("/v1/me", headers=c.h).json()
    assert me["flags"]["voice"] is True and me["flags"]["voice_engine"] == "realtime"
    r = client.post("/v1/voice/speak", json={"text": "Hello"}, headers=c.h)
    assert r.status_code == 200 and r.headers["content-type"] == "audio/mpeg"
    _cfg(client, {"voice.enabled": False})  # applies at once, despite the per-user gate cache
    r = client.post("/v1/voice/speak", json={"text": "Hello"}, headers=c.h)
    assert r.status_code == 403 and r.json()["error"]["code"] == "voice_disabled"
    assert "type" in r.json()["error"]["message"].lower()


def test_voice_requires_access(client):
    free = auth(mint(email="free@example.invalid", tier="free"))
    assert client.post("/v1/voice/speak", json={"text": "Hello"}, headers=free).status_code == 403


def test_speech_round_trip_is_logged(client):
    enable_pro(client)
    c = Candidate(client)
    r = client.post("/v1/voice/speak", json={"text": "Tell me about a decision you owned."}, headers=c.h)
    assert r.status_code == 200 and r.content[:4] == b"RIFF"  # simulator returns playable audio
    r = client.post("/v1/voice/transcribe", headers=c.h,
                    files={"audio": ("a.webm", b"\x1aE\xdf\xa3" + b"0" * 2000, "audio/webm")})
    assert r.status_code == 200 and r.json()["text"]
    with db_session() as db:
        stages = sorted(x.stage for x in db.execute(select(ModelRun)).scalars())
    assert stages == ["stt", "tts"]


def test_bad_audio_rejected(client):
    enable_pro(client)
    c = Candidate(client)
    r = client.post("/v1/voice/transcribe", headers=c.h, files={"audio": ("a.exe", b"MZ" * 10, "application/x-msdownload")})
    assert r.status_code == 422
    r = client.post("/v1/voice/speak", json={"text": "x" * 2000}, headers=c.h)
    assert r.status_code == 422


# ---------------------------------------------------------------- live calls ---------------
class _Resp:
    def __init__(self, status, data):
        self.status_code, self._data = status, data
        self.text = str(data)

    def json(self):
        return self._data


class FakeOpenAI:
    """Stands in for httpx.Client when minting a realtime client secret."""
    calls: list = []
    replies: list = []

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, headers=None, json=None):
        FakeOpenAI.calls.append({"url": url, "headers": headers, "json": json})
        return FakeOpenAI.replies.pop(0) if FakeOpenAI.replies else _Resp(
            200, {"value": "ek_test_secret", "expires_at": 1999999999})


@pytest.fixture
def openai_key(monkeypatch):
    """Call it AFTER the interview is prepared (with the key set, analysis would use OpenAI)."""
    from interview_intelligence import config as cfg
    from interview_intelligence.voice import routes as vr
    FakeOpenAI.calls, FakeOpenAI.replies = [], []
    monkeypatch.setattr(vr, "_http_client", lambda: FakeOpenAI())
    yield lambda: cfg.set_settings_for_tests(make_settings(openai_api_key="sk-test"))
    cfg.set_settings_for_tests(make_settings())


def test_live_call_secret_is_minted_server_side(client, openai_key):
    enable_pro(client)
    c = Candidate(client)
    sid = c.ready_session(duration_minutes=15)
    openai_key()
    r = client.post("/v1/voice/live", json={"session_id": sid, "voice": "cedar"}, headers=c.h)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["client_secret"] == "ek_test_secret" and body["voice"] == "cedar" and body["model"]
    sent = FakeOpenAI.calls[0]
    assert sent["url"].endswith("/realtime/client_secrets") and sent["headers"]["Authorization"] == "Bearer sk-test"
    sess = sent["json"]["session"]
    td = sess["audio"]["input"]["turn_detection"]
    # II decides every line: the speech model never replies by itself; barge-in stays on.
    assert td["type"] == "semantic_vad" and td["create_response"] is False and td["interrupt_response"] is True
    assert sess["audio"]["input"]["transcription"]["language"] == "en"
    # The voice gets no interview content: no CV, JD, role, rubric or plan.
    assert "Brand" not in sess["instructions"] and "Priya" not in sess["instructions"]
    assert "exactly" in sess["instructions"]
    # unknown voice -> safe default
    r = client.post("/v1/voice/live", json={"session_id": sid, "voice": "<script>"}, headers=c.h)
    assert r.json()["voice"] == "marin"


def test_live_call_falls_back_when_transcriber_is_rejected(client, openai_key):
    enable_pro(client)
    c = Candidate(client)
    sid = c.ready_session(duration_minutes=15)
    openai_key()
    FakeOpenAI.replies = [_Resp(400, {"error": "unknown model"}), _Resp(200, {"client_secret": {"value": "ek_2"}})]
    r = client.post("/v1/voice/live", json={"session_id": sid}, headers=c.h)
    assert r.status_code == 200 and r.json()["client_secret"] == "ek_2"
    assert FakeOpenAI.calls[1]["json"]["session"]["audio"]["input"]["transcription"]["model"] == "whisper-1"


def test_live_call_failure_tells_the_client_to_fall_back(client, openai_key):
    enable_pro(client)
    c = Candidate(client)
    sid = c.ready_session(duration_minutes=15)
    openai_key()
    FakeOpenAI.replies = [_Resp(500, {"error": "boom"})]
    r = client.post("/v1/voice/live", json={"session_id": sid}, headers=c.h)
    assert r.status_code == 503 and r.json()["error"]["code"] == "live_failed"


def test_live_call_guards(client, openai_key):
    enable_pro(client)
    c = Candidate(client)
    sid = c.ready_session(duration_minutes=15)
    openai_key()
    other = Candidate(client, email="other@example.invalid")
    assert client.post("/v1/voice/live", json={"session_id": sid}, headers=other.h).status_code == 404
    assert client.post("/v1/voice/live", json={"session_id": "nope"}, headers=c.h).status_code == 404
    free = auth(mint(email="free2@example.invalid", tier="free"))
    assert client.post("/v1/voice/live", json={"session_id": sid}, headers=free).status_code == 403
    _cfg(client, {"voice.engine": "standard"})
    r = client.post("/v1/voice/live", json={"session_id": sid}, headers=c.h)
    assert r.status_code == 403 and r.json()["error"]["code"] == "live_off"
    _cfg(client, {"voice.engine": "realtime"})
    with db_session() as db:
        db.get(InterviewSession, __import__("uuid").UUID(sid)).status = "completed"
    r = client.post("/v1/voice/live", json={"session_id": sid}, headers=c.h)
    assert r.status_code == 409
    bad = client.patch("/v1/admin/config", json={"values": {"voice.engine": "telepathy"}},
                       headers=auth(mint(email=ADMIN_EMAIL, tier="free")))
    assert bad.status_code == 422


def test_live_call_needs_an_openai_key(client):
    enable_pro(client)
    c = Candidate(client)
    sid = c.ready_session(duration_minutes=15)
    r = client.post("/v1/voice/live", json={"session_id": sid}, headers=c.h)
    assert r.status_code == 503 and r.json()["error"]["code"] == "live_unconfigured"


def test_live_usage_is_metered_in_the_daily_budget_not_the_interview_cap(client):
    enable_pro(client)
    c = Candidate(client)
    sid = c.ready_session(duration_minutes=15)
    with db_session() as db:
        before = float(db.get(InterviewSession, __import__("uuid").UUID(sid)).cost_usd or 0)
    usage = {"total_tokens": 1900, "input_tokens": 700, "output_tokens": 1200,
             "input_token_details": {"text_tokens": 100, "audio_tokens": 600,
                                     "cached_tokens_details": {"text_tokens": 50, "audio_tokens": 0}},
             "output_token_details": {"text_tokens": 0, "audio_tokens": 1200}}
    r = client.post("/v1/voice/live/usage", json={"session_id": sid, "usage": usage}, headers=c.h)
    assert r.status_code == 204, r.text
    with db_session() as db:
        run = db.execute(select(ModelRun).where(ModelRun.stage == "live_voice")).scalar_one()
        after = float(db.get(InterviewSession, __import__("uuid").UUID(sid)).cost_usd or 0)
    # 600 audio in @32 + 50 text @4 + 50 cached @0.4 + 1200 audio out @64, per 1M
    expected = (600 * 32 + 50 * 4 + 50 * 0.4 + 1200 * 64) / 1e6
    assert abs(float(run.cost_usd) - round(expected, 6)) < 1e-6
    assert run.validation["interview_id"] == sid
    assert after == before  # never eats the interview's own AI cost cap
    from interview_intelligence.ai.runner import reset_budget_cache, spend_today_usd
    reset_budget_cache()
    assert spend_today_usd() >= expected - 1e-9
    other = Candidate(client, email="other@example.invalid")
    assert client.post("/v1/voice/live/usage", json={"session_id": sid, "usage": usage},
                       headers=other.h).status_code == 404


def test_realtime_cost_is_bounded_and_tolerant():
    from interview_intelligence.ai.pricing import realtime_cost_usd
    assert realtime_cost_usd({})[0] == 0
    assert realtime_cost_usd({"input_token_details": "garbage"})[0] == 0
    huge, _ = realtime_cost_usd({"output_token_details": {"audio_tokens": 10 ** 12}})
    assert huge <= 2_000_000 * 64 / 1e6  # clamped
    legacy, _ = realtime_cost_usd({"input_tokens": 1000, "output_tokens": 1000})
    assert legacy > 0
