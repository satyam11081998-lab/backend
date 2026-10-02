"""Access policy: Pro, launch flag, test grants, admin, lapsed users (spec §3, §4, §89)."""

from tests.conftest import ADMIN_EMAIL, JD_TEXT, auth, enable_pro, mint, run_jobs
from tests.helpers import Candidate


def _paste(client, token):
    return client.post("/v1/documents/text", json={"kind": "jd", "text": JD_TEXT}, headers=auth(token))


def test_pro_blocked_until_launch_flag(client):
    t = mint(tier="pro")
    r = _paste(client, t)
    assert r.status_code == 403 and "preview" in r.json()["error"]["message"].lower()
    enable_pro(client)
    assert _paste(client, t).status_code == 201


def test_free_and_lite_blocked(client):
    enable_pro(client)
    assert _paste(client, mint(tier="free")).status_code == 403
    assert _paste(client, mint(tier="lite", ent=[])).status_code == 403


def test_test_grant_lifecycle(client):
    admin_h = auth(mint(email=ADMIN_EMAIL, tier="free"))
    user = mint(email="Tester.One@Example.invalid", tier="free")
    assert _paste(client, user).status_code == 403
    r = client.post("/v1/admin/access-grants", json={"email": "tester.one@example.invalid", "note": "QA"},
                    headers=admin_h)
    assert r.status_code == 201 and r.json()["active"] is True
    gid = r.json()["id"]
    assert _paste(client, user).status_code == 201, "case-insensitive email match"
    r = client.patch(f"/v1/admin/access-grants/{gid}", json={"status": "disabled"}, headers=admin_h)
    assert r.status_code == 200 and r.json()["active"] is False
    assert _paste(client, user).status_code == 403
    client.patch(f"/v1/admin/access-grants/{gid}", json={"status": "enabled"}, headers=admin_h)
    assert _paste(client, user).status_code == 201
    assert client.delete(f"/v1/admin/access-grants/{gid}", headers=admin_h).status_code == 204
    assert _paste(client, user).status_code == 403
    lst = client.get("/v1/admin/access-grants", headers=admin_h).json()["grants"]
    assert lst == []


def test_bad_email_rejected(client):
    admin_h = auth(mint(email=ADMIN_EMAIL, tier="free"))
    for bad in ["", "nope", "a@b", "two words@x.com"]:
        assert client.post("/v1/admin/access-grants", json={"email": bad}, headers=admin_h).status_code == 422


def test_admin_endpoints_require_admin(client):
    t = auth(mint(tier="pro"))
    for method, path in [("get", "/v1/admin/access-grants"), ("get", "/v1/admin/config"), ("get", "/v1/admin/overview"),
                         ("get", "/v1/admin/sessions"), ("get", "/v1/admin/model-runs"), ("get", "/v1/admin/library"),
                         ("post", "/v1/admin/drive/retry-failed"), ("get", "/v1/admin/audit")]:
        r = getattr(client, method)(path, headers=t)
        assert r.status_code == 403, path


def test_mece_admin_claim_trusted_by_default(client):
    t = auth(mint(email="someone@example.invalid", tier="free", adm=True))
    assert client.get("/v1/admin/config", headers=t).status_code == 200


def test_mece_admin_claim_can_be_distrusted(client):
    from interview_intelligence import config as cfg
    from tests.conftest import make_settings
    cfg.set_settings_for_tests(make_settings(trust_mece_admin=False))
    t = auth(mint(email="someone@example.invalid", tier="free", adm=True))
    assert client.get("/v1/admin/config", headers=t).status_code == 403


def test_master_kill_switch(client):
    enable_pro(client)
    admin_h = auth(mint(email=ADMIN_EMAIL, tier="free"))
    client.patch("/v1/admin/config", json={"values": {"ii.enabled": False}}, headers=admin_h)
    assert _paste(client, mint(tier="pro")).status_code == 403
    assert _paste(client, mint(email=ADMIN_EMAIL, tier="free")).status_code == 201, "admins keep access"


def test_lapsed_user_can_still_read_own_history(client):
    enable_pro(client)
    c = Candidate(client)
    c.ready_session()
    c.tier = "free"  # subscription lapsed: MECE now asserts free
    assert client.get("/v1/sessions", headers=c.h).status_code == 200
    r = c.paste_jd.__func__  # noqa: B018 - just to show the method exists
    r2 = client.post("/v1/documents/text", json={"kind": "jd", "text": JD_TEXT + " v2"}, headers=c.h)
    assert r2.status_code == 403, "cannot start new work without entitlement"


def test_bootstrap_test_emails(client):
    from interview_intelligence import config as cfg
    from interview_intelligence.access.policy import bootstrap_test_grants
    from interview_intelligence.db.session import db_session
    from tests.conftest import make_settings
    cfg.set_settings_for_tests(make_settings(bootstrap_test_emails=["first@example.invalid", "bad-entry"]))
    with db_session() as db:
        assert bootstrap_test_grants(db) == 1
    admin_h = auth(mint(email=ADMIN_EMAIL, tier="free"))
    gid = client.get("/v1/admin/access-grants", headers=admin_h).json()["grants"][0]["id"]
    client.patch(f"/v1/admin/access-grants/{gid}", json={"status": "disabled"}, headers=admin_h)
    with db_session() as db:
        assert bootstrap_test_grants(db) == 0  # never re-enables an admin decision
    g = client.get("/v1/admin/access-grants", headers=admin_h).json()["grants"][0]
    assert g["status"] == "disabled"


def test_users_without_access_are_not_registered_by_reading(client):
    """PII minimisation: a free user who only opens the page leaves no row in II."""
    import uuid as _uuid
    from interview_intelligence.db.models import User
    from interview_intelligence.db.session import db_session
    uid = _uuid.uuid4()
    r = client.get("/v1/me", headers=auth(mint(uid, email="free@example.invalid", tier="free")))
    assert r.status_code == 200 and r.json()["access"]["allowed"] is False
    with db_session() as db:
        assert db.get(User, uid) is None
