"""Spec §29, §91 — turn-based voice: flag-gated, Pro-gated, logged and costed like every model call."""

from sqlalchemy import select

from interview_intelligence.db.models import ModelRun
from interview_intelligence.db.session import db_session
from tests.conftest import ADMIN_EMAIL, auth, enable_pro, mint
from tests.helpers import Candidate


def _voice(client, on: bool):
    r = client.patch("/v1/admin/config", json={"values": {"voice.enabled": on}},
                     headers=auth(mint(email=ADMIN_EMAIL, tier="free")))
    assert r.status_code == 200


def test_voice_is_off_by_default_and_offers_text(client):
    enable_pro(client)
    c = Candidate(client)
    r = client.post("/v1/voice/speak", json={"text": "Hello"}, headers=c.h)
    assert r.status_code == 403 and r.json()["error"]["code"] == "voice_disabled"
    assert "type" in r.json()["error"]["message"].lower()


def test_voice_requires_access(client):
    _voice(client, True)
    free = auth(mint(email="free@example.invalid", tier="free"))
    assert client.post("/v1/voice/speak", json={"text": "Hello"}, headers=free).status_code == 403


def test_speech_round_trip_is_logged(client):
    enable_pro(client)
    _voice(client, True)
    c = Candidate(client)
    r = client.post("/v1/voice/speak", json={"text": "Tell me about a decision you owned."}, headers=c.h)
    assert r.status_code == 200 and r.headers["content-type"] == "audio/mpeg"
    r = client.post("/v1/voice/transcribe", headers=c.h,
                    files={"audio": ("a.webm", b"\x1aE\xdf\xa3" + b"0" * 2000, "audio/webm")})
    assert r.status_code == 200 and r.json()["text"]
    with db_session() as db:
        stages = sorted(x.stage for x in db.execute(select(ModelRun)).scalars())
    assert stages == ["stt", "tts"]


def test_bad_audio_rejected(client):
    enable_pro(client)
    _voice(client, True)
    c = Candidate(client)
    r = client.post("/v1/voice/transcribe", headers=c.h, files={"audio": ("a.exe", b"MZ" * 10, "application/x-msdownload")})
    assert r.status_code == 422
    r = client.post("/v1/voice/speak", json={"text": "x" * 2000}, headers=c.h)
    assert r.status_code == 422
