"""Multi-tenant isolation (spec §63, §92) + static independence checks (spec §1, §113)."""

import re
from pathlib import Path

from tests.conftest import enable_pro, run_jobs
from tests.helpers import STRONG_ANSWER, Candidate

ROOT = Path(__file__).resolve().parent.parent / "interview_intelligence"


def test_cross_user_access_is_404(client):
    enable_pro(client)
    a = Candidate(client, email="a@example.invalid")
    b = Candidate(client, email="b@example.invalid")
    sid = a.ready_session()
    client.post(f"/v1/sessions/{sid}/start", headers=a.h)
    a.turn(sid, STRONG_ANSWER)
    client.post(f"/v1/sessions/{sid}/end", headers=a.h)
    run_jobs()
    docs = client.get("/v1/documents", headers=a.h).json()["documents"]
    doc_id = docs[0]["id"]
    for path in [f"/v1/documents/{doc_id}", f"/v1/sessions/{sid}", f"/v1/sessions/{sid}/room",
                 f"/v1/sessions/{sid}/transcript", f"/v1/sessions/{sid}/report"]:
        r = client.get(path, headers=b.h)
        assert r.status_code == 404, (path, r.status_code)
    assert client.delete(f"/v1/documents/{doc_id}", headers=b.h).status_code == 404
    for path in ["start", "pause", "resume", "end", "abandon"]:
        assert client.post(f"/v1/sessions/{sid}/{path}", headers=b.h).status_code in (404,), path
    r = client.post(f"/v1/sessions/{sid}/turns", json={"client_turn_id": "x", "content": "hi"}, headers=b.h)
    assert r.status_code == 404
    # B cannot build a session on A's documents
    b_jd = b.paste_jd()
    r = b.create(doc_id, b_jd["id"])
    assert r.status_code == 404
    assert client.get("/v1/sessions", headers=b.h).json()["sessions"] == []


def test_invalid_ids_are_404_not_500(client):
    enable_pro(client)
    c = Candidate(client)
    assert client.get("/v1/sessions/not-a-uuid", headers=c.h).status_code == 404
    assert client.get("/v1/documents/../../etc", headers=c.h).status_code == 404


def test_every_table_is_in_the_isolated_schema():
    from interview_intelligence.db.models import SCHEMA, Base
    assert SCHEMA == "interview_intel"
    for t in Base.metadata.sorted_tables:
        assert t.schema == SCHEMA, t.name
        for fk in t.foreign_keys:
            assert fk.column.table.schema == SCHEMA, f"{t.name} references {fk.column.table}"


def test_no_dependency_on_existing_mece_backend():
    """The II package must not import the old backend's modules or read MECE tables."""
    forbidden_imports = re.compile(r"^\s*(from|import)\s+(services|routes|prompts|supabase)\b", re.MULTILINE)
    forbidden_tables = re.compile(r"\b(public\.users|case_attempts|attempt_messages|submissions|ai_usage_log)\b")
    for py in ROOT.rglob("*.py"):
        src = py.read_text(encoding="utf-8")
        assert not forbidden_imports.search(src), f"{py} imports the MECE backend"
        assert not forbidden_tables.search(src), f"{py} references a MECE table"


def test_migration_matches_models():
    from scripts.generate_migration import render
    committed = (ROOT.parent / "migrations" / "0001_interview_intel.sql").read_text(encoding="utf-8")
    assert render() == committed, "models changed: re-run `python -m scripts.generate_migration`"


def test_no_email_addresses_hardcoded_in_source():
    rx = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(com|in|org|net|io)\b")
    for py in ROOT.rglob("*.py"):
        assert not rx.search(py.read_text(encoding="utf-8")), py
