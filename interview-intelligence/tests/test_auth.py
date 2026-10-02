"""Stage K — entitlement assertion attacks (spec §92)."""

import base64
import json
import time
import uuid

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tests.conftest import OTHER_PRIV, PUB_PEM, auth, enable_pro, mint


def _me(client, token=None, raw_header=None):
    headers = {"Authorization": raw_header} if raw_header is not None else (auth(token) if token else {})
    return client.get("/v1/me", headers=headers)


def test_missing_and_malformed(client):
    assert _me(client).status_code == 401
    assert _me(client, raw_header="Token abc").status_code == 401
    assert _me(client, raw_header="Bearer not.a.jwt.at.all").status_code == 401
    assert _me(client, raw_header="Bearer ").status_code == 401


def test_valid_token(client):
    r = _me(client, mint(tier="free"))
    assert r.status_code == 200
    assert r.json()["access"]["allowed"] is False  # free user, no grant


def test_expired_rejected(client):
    t = mint(iat=int(time.time()) - 1000, lifetime=300)
    r = _me(client, t)
    assert r.status_code == 401 and r.json()["error"]["code"] == "token_expired"


def test_wrong_audience_and_issuer(client):
    assert _me(client, mint(aud="mece-backend")).status_code == 401
    assert _me(client, mint(iss="someone-else")).status_code == 401


def test_signed_by_wrong_key(client):
    assert _me(client, mint(key=OTHER_PRIV)).status_code == 401


def test_alg_none_rejected(client):
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).rstrip(b"=")
    now = int(time.time())
    payload = base64.urlsafe_b64encode(json.dumps({
        "iss": "mece-app", "aud": "mece-interview-intelligence", "sub": str(uuid.uuid4()), "tier": "pro",
        "ent": ["interview_intelligence"], "iat": now, "exp": now + 300, "jti": "x", "ver": 1}).encode()).rstrip(b"=")
    token = (header + b"." + payload + b".").decode()
    assert _me(client, token).status_code == 401


def test_hs256_key_confusion_rejected(client):
    """Classic attack: sign HS256 with the PUBLIC key as the HMAC secret."""
    now = int(time.time())
    claims = {"iss": "mece-app", "aud": "mece-interview-intelligence", "sub": str(uuid.uuid4()), "tier": "pro",
              "ent": ["interview_intelligence"], "iat": now, "exp": now + 300, "jti": "x", "ver": 1}
    import hmac, hashlib
    seg = lambda b: base64.urlsafe_b64encode(b).rstrip(b"=")  # noqa: E731
    h = seg(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    p = seg(json.dumps(claims).encode())
    sig = seg(hmac.new(PUB_PEM.encode(), h + b"." + p, hashlib.sha256).digest())
    assert _me(client, (h + b"." + p + b"." + sig).decode()).status_code == 401


def test_tampered_claims_rejected(client):
    t = mint(tier="free", ent=[])
    h, p, s = t.split(".")
    pad = "=" * (-len(p) % 4)
    claims = json.loads(base64.urlsafe_b64decode(p + pad))
    claims["tier"], claims["ent"] = "pro", ["interview_intelligence"]
    p2 = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    assert _me(client, f"{h}.{p2}.{s}").status_code == 401


def test_lifetime_too_long_rejected(client):
    r = _me(client, mint(lifetime=3600))
    assert r.status_code == 401 and r.json()["error"]["code"] == "lifetime_too_long"


def test_non_uuid_subject_and_bad_version(client):
    assert _me(client, mint(sub="admin")).status_code == 401
    assert _me(client, mint(ver=2)).status_code == 401


def test_issued_in_future_rejected(client):
    assert _me(client, mint(iat=int(time.time()) + 600, lifetime=300)).status_code == 401


def test_frontend_flags_are_ignored(client):
    """A body/query 'isPro' cannot grant anything; only the signed assertion counts."""
    t = mint(tier="free", ent=[], extra={"isPro": True, "is_pro": True})
    r = client.post("/v1/documents/text?isPro=true", json={"kind": "jd", "text": "x" * 200, "isPro": True},
                    headers=auth(t))
    assert r.status_code == 403
    # even a forged 'ent' value inside a FREE token signed by the right key is just data; but tier/ent
    # are what MECE asserted — the point is the browser cannot produce such a token.


def test_replayed_token_after_expiry(client):
    t = mint(iat=int(time.time()) - 290, lifetime=300)
    assert _me(client, t).status_code == 200
    expired = mint(iat=int(time.time()) - 400, lifetime=300)
    assert _me(client, expired).status_code == 401


def test_no_keys_configured_fails_closed(client):
    from interview_intelligence import config as cfg
    from tests.conftest import make_settings
    cfg.set_settings_for_tests(make_settings(assertion_public_keys={}))
    r = _me(client, mint())
    assert r.status_code == 401 and r.json()["error"]["code"] == "assertion_keys_missing"


def test_key_rotation_by_kid(client):
    from interview_intelligence import config as cfg
    from tests.conftest import make_settings
    from cryptography.hazmat.primitives import serialization
    new = Ed25519PrivateKey.generate()
    new_pub = new.public_key().public_bytes(serialization.Encoding.PEM,
                                            serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    cfg.set_settings_for_tests(make_settings(assertion_public_keys={"k1": PUB_PEM, "k2": new_pub}))
    assert _me(client, mint(kid="k1")).status_code == 200
    assert _me(client, mint(key=new, kid="k2")).status_code == 200
