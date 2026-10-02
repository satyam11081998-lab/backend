"""Test harness.

* Database: II_TEST_DATABASE_URL (Postgres recommended — row locks, SKIP LOCKED and the
  2-active-session race are only meaningful there). Falls back to a SQLite file.
* AI: the offline SimulatedProvider (II_ENV=test). Individual tests override stages.
* Auth: a throwaway Ed25519 keypair; `mint()` signs assertions exactly like the MECE route.
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from interview_intelligence import config as cfg

_PRIV = Ed25519PrivateKey.generate()
PRIV_PEM = _PRIV.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                               serialization.NoEncryption()).decode()
PUB_PEM = _PRIV.public_key().public_bytes(serialization.Encoding.PEM,
                                         serialization.PublicFormat.SubjectPublicKeyInfo).decode()
OTHER_PRIV = Ed25519PrivateKey.generate()

ADMIN_EMAIL = "admin-test@example.invalid"


def _db_url() -> str:
    url = os.environ.get("II_TEST_DATABASE_URL", "").strip()
    if url:
        return url
    return "sqlite+pysqlite:///" + str(Path(tempfile.gettempdir()) / f"ii_test_{os.getpid()}.sqlite")


DB_URL = _db_url()
IS_PG = DB_URL.startswith("postgres")


def make_settings(**over):
    base = dict(env="test", database_url=DB_URL, assertion_public_keys={"k1": PUB_PEM}, admin_emails=[ADMIN_EMAIL],
                trust_mece_admin=True, encryption_key="", worker_threads=0, max_active_sessions=2,
                max_sessions_per_day=50)
    base.update(over)
    return cfg.Settings(**base)


cfg.set_settings_for_tests(make_settings())


def mint(sub=None, *, email="pro@example.invalid", tier="pro", ent=None, adm=False, lifetime=300, iat=None,
         aud="mece-interview-intelligence", iss="mece-app", key=None, alg="EdDSA", ver=1, kid="k1", extra=None):
    now = int(time.time()) if iat is None else iat
    claims = {"iss": iss, "aud": aud, "sub": str(sub or uuid.uuid4()), "email": email, "tier": tier,
              "ent": ent if ent is not None else (["interview_intelligence"] if tier == "pro" else []),
              "adm": adm, "iat": now, "nbf": now, "exp": now + lifetime, "jti": str(uuid.uuid4()), "ver": ver}
    if extra:
        claims.update(extra)
    return jwt.encode(claims, key or _PRIV, algorithm=alg, headers={"kid": kid})


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def _reset_state():
    """Fresh schema + caches for every test."""
    from interview_intelligence.access import flags, rate_limit
    from interview_intelligence.ai import routing, runner
    from interview_intelligence.db import session as dbs
    from interview_intelligence.db.models import Base
    from interview_intelligence.security import crypto
    from sqlalchemy import text

    cfg.set_settings_for_tests(make_settings())
    dbs.reset_engine_for_tests()
    crypto.reset_for_tests()
    routing.reset_for_tests()
    rate_limit.reset_for_tests()
    flags.invalidate()
    runner.reset_budget_cache()
    eng = dbs.get_engine()
    if IS_PG:
        with eng.begin() as c:
            c.execute(text("DROP SCHEMA IF EXISTS interview_intel CASCADE"))
            c.execute(text("CREATE SCHEMA interview_intel"))
    else:
        Base.metadata.drop_all(eng)
    Base.metadata.create_all(eng)
    from interview_intelligence.main import _register_job_handlers
    _register_job_handlers()
    yield
    dbs.reset_engine_for_tests()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from interview_intelligence.main import create_app
    with TestClient(create_app()) as c:
        yield c


def run_jobs(max_jobs: int = 500) -> int:
    from interview_intelligence.jobs.queue import run_pending
    return run_pending(max_jobs, include_delayed=True)


def enable_pro(client):
    """Admin flips the launch flag so Pro users get in (default is preview-only)."""
    t = mint(email=ADMIN_EMAIL, tier="free")
    r = client.patch("/v1/admin/config", json={"values": {"ii.enabled_for_pro": True}}, headers=auth(t))
    assert r.status_code == 200, r.text


CV_TEXT = """Priya Sharma
priya@example.com | +91 98765 43210
Date of Birth: 01/01/1998
Summary: Brand marketing professional with 4 years of experience in FMCG.
Experience
Assistant Brand Manager, Acme Foods Jun 2022 - Present
- Led a team of 6 to relaunch the snacks range, growing revenue 18% in two quarters
- Built a retailer promotion model that reduced trade spend by 12%
- Managed a 2 crore media budget across digital and TV
Marketing Associate, Zeta Retail Jan 2020 - May 2022
- Analysed consumer panel data to identify a lapsed-buyer segment worth 8% of volume
- Launched a CRM win-back flow that improved repeat purchase by 9%
Education
MBA, IMI Delhi 2020; B.Com, Delhi University 2018
Skills: Brand management, consumer insight, Nielsen, Excel, SQL, campaign analytics
"""

JD_TEXT = """Brand Manager - Snacks
Company: Nimbus Consumer Products
We are a leading FMCG company with brands across snacks and beverages.
Responsibilities
- Own the brand P&L and annual marketing plan for the snacks portfolio
- Lead consumer insight work and translate it into positioning and campaigns
- Partner with sales on trade marketing and channel strategy
- Measure campaign effectiveness and brand health
Requirements
- 3+ years of brand management experience in FMCG (required)
- Strong understanding of consumer behaviour, positioning and pricing
- Experience with market research and campaign analytics
- Ability to manage agencies and cross-functional stakeholders
- MBA preferred
"""
