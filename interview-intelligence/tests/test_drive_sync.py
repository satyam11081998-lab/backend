"""Stage B/K — Google Drive storage against a fake Drive API (httpx.MockTransport).

Checks: folder tree created once and reused; opaque folder names (no email/name); crash-safe
dedupe via appProperties; 5xx retried; 4xx not retried and marked failed; Drive ids never
leave the API; delete propagates; reports exported to Interviews/; nothing happens when
Drive is not configured."""

import itertools
import json
import re
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from sqlalchemy import select

from interview_intelligence import config as cfg
from interview_intelligence.db.models import Document, DriveFile, DriveFolder
from interview_intelligence.db.session import db_session
from interview_intelligence.drive_integration import client as drive
from interview_intelligence.drive_integration.sync import folder_key
from tests.conftest import ADMIN_EMAIL, JD_TEXT, auth, enable_pro, make_settings, mint, run_jobs
from tests.helpers import STRONG_ANSWER, Candidate, docx_bytes


class FakeDrive:
    def __init__(self):
        self.folders = {}  # id -> {name, parent}
        self.files = {}  # id -> {name, parent, appProperties, data}
        self.sessions = {}
        self.ids = itertools.count(1)
        self.fail = []  # [(method, url_substring, status)] consumed in order
        self.calls = []

    def _fail(self, request):
        for i, (m, sub, status) in enumerate(self.fail):
            if request.method == m and sub in str(request.url):
                self.fail.pop(i)
                return httpx.Response(status, text="injected")
        return None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.calls.append((request.method, url.split("?")[0]))
        injected = self._fail(request)
        if injected is not None:
            return injected
        if url.startswith(drive.TOKEN_URL):
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
        assert request.headers.get("authorization") == "Bearer tok"
        q = parse_qs(urlparse(url).query)
        if request.method == "GET" and url.startswith(drive.API):
            query = q.get("q", [""])[0]
            m = re.search(r"key='([^']*)' and value='([^']*)'", query)
            if m:
                hits = [{"id": i, "name": f["name"]} for i, f in self.files.items()
                        if f["appProperties"].get(m.group(1)) == m.group(2)]
            else:
                name = re.search(r"name = '([^']*)'", query).group(1)
                parent = re.search(r"'([^']*)' in parents", query).group(1)
                hits = [{"id": i, "name": f["name"]} for i, f in self.folders.items()
                        if f["name"] == name and f["parent"] == parent]
            return httpx.Response(200, json={"files": hits})
        if request.method == "POST" and url.startswith(drive.UPLOAD):
            sid = f"s{next(self.ids)}"
            self.sessions[sid] = json.loads(request.content)
            return httpx.Response(200, headers={"location": f"https://upload.fake/{sid}"})
        if request.method == "PUT" and url.startswith("https://upload.fake/"):
            meta = self.sessions[url.rsplit("/", 1)[1]]
            fid = f"f{next(self.ids)}"
            self.files[fid] = {"name": meta["name"], "parent": meta["parents"][0],
                               "appProperties": meta.get("appProperties", {}), "data": request.content}
            return httpx.Response(200, json={"id": fid})
        if request.method == "POST" and url.startswith(drive.API):
            body = json.loads(request.content)
            fid = f"d{next(self.ids)}"
            self.folders[fid] = {"name": body["name"], "parent": body["parents"][0]}
            return httpx.Response(200, json={"id": fid})
        if request.method == "DELETE":
            self.files.pop(url.split("?")[0].rsplit("/", 1)[1], None)
            return httpx.Response(204)
        return httpx.Response(404)

    def path_of(self, fid):
        parts, node = [], self.files.get(fid) or self.folders.get(fid)
        while node:
            parts.append(node["name"])
            node = self.folders.get(node["parent"])
        return "/".join(reversed(parts))


@pytest.fixture
def fake():
    cfg.set_settings_for_tests(make_settings(gdrive_root_folder_id="ROOT", gdrive_refresh_token="r",
                                             gdrive_client_id="c", gdrive_client_secret="s",
                                             drive_folder_salt="unit-salt"))
    f = FakeDrive()
    drive.set_http_client(httpx.Client(transport=httpx.MockTransport(f)))
    yield f
    drive.set_http_client(httpx.Client(timeout=60.0))


def _status(doc_id):
    with db_session() as db:
        import uuid
        return db.get(Document, uuid.UUID(doc_id)).storage_status


def test_documents_land_in_an_opaque_per_user_tree_created_once(client, fake):
    enable_pro(client)
    c = Candidate(client, email="someone.real@example.invalid")
    cv = c.upload_cv()
    jd = client.post("/v1/documents", data={"kind": "jd"}, headers=c.h,
                     files={"file": ("jd.docx", docx_bytes(JD_TEXT), "application/octet-stream")}).json()
    cv2 = c.upload_cv(text="Second CV\nExperience\nLed a team of 3 analysts, cutting reporting time 40%\n" * 3,
                      name="cv2.docx")
    run_jobs()
    assert _status(cv["id"]) == _status(jd["id"]) == _status(cv2["id"]) == "synced"
    paths = sorted(fake.path_of(i) for i in fake.files)
    key = folder_key(c.uid)
    assert all(p.startswith(f"Users/{key}/") for p in paths), paths
    assert sum(1 for p in paths if p.startswith(f"Users/{key}/CV/")) == 2
    assert sum(1 for p in paths if p.startswith(f"Users/{key}/JD/")) == 1
    assert len(fake.folders) == 4, "Users, <user>, CV, JD — no duplicate folders"
    blob = json.dumps({"folders": fake.folders, "names": [f["name"] for f in fake.files.values()]})
    assert "someone.real" not in blob and "example.invalid" not in blob, "no emails or names in Drive"
    body = client.get(f"/v1/documents/{cv['id']}", headers=c.h).text
    for fid in list(fake.files) + list(fake.folders):
        assert f'"{fid}"' not in body, "Drive ids never leave the API"


def test_crash_after_upload_does_not_create_a_duplicate(client, fake):
    enable_pro(client)
    c = Candidate(client)
    cv = c.upload_cv()
    # A previous attempt uploaded the file but crashed before our DB commit.
    fake.files["orphan"] = {"name": "cv.docx", "parent": "?", "appProperties": {"ii_target_id": cv["id"]}, "data": b""}
    run_jobs()
    assert _status(cv["id"]) == "synced"
    assert not fake.sessions, "found by appProperties -> no second upload"
    with db_session() as db:
        assert db.execute(select(DriveFile.drive_file_id).where(DriveFile.target_id == cv["id"])).scalar() == "orphan"


def test_transient_errors_are_retried(client, fake):
    enable_pro(client)
    c = Candidate(client)
    fake.fail = [("PUT", "upload.fake", 503), ("POST", drive.TOKEN_URL, 500)]
    cv = c.upload_cv()
    run_jobs()
    assert _status(cv["id"]) == "synced" and len(fake.files) == 1


def test_permission_errors_are_not_retried_and_marked_failed(client, fake):
    enable_pro(client)
    c = Candidate(client)
    fake.fail = [("POST", drive.UPLOAD, 403)]
    cv = c.upload_cv()
    run_jobs()
    assert _status(cv["id"]) == "failed" and not fake.files
    with db_session() as db:
        row = db.execute(select(DriveFile).where(DriveFile.target_id == cv["id"])).scalar_one()
        assert row.storage_status == "failed" and "403" in row.last_error
    # the document itself is still usable: the encrypted original stayed in the II database
    assert client.get(f"/v1/documents/{cv['id']}", headers=c.h).json()["analysis_status"] == "ready"
    # admin retry re-queues it
    r = client.post("/v1/admin/drive/retry-failed", headers=auth(mint(email=ADMIN_EMAIL, tier="free")))
    assert r.status_code == 200
    run_jobs()
    assert _status(cv["id"]) == "synced"


def test_delete_propagates_to_drive(client, fake):
    enable_pro(client)
    c = Candidate(client)
    cv = c.upload_cv()
    run_jobs()
    assert len(fake.files) == 1
    assert client.delete(f"/v1/documents/{cv['id']}", headers=c.h).status_code in (200, 204)
    run_jobs()
    assert not fake.files


def test_report_is_exported_to_interviews_folder(client, fake):
    enable_pro(client)
    c = Candidate(client)
    sid = c.ready_session()
    client.post(f"/v1/sessions/{sid}/start", headers=c.h)
    c.turn(sid, STRONG_ANSWER)
    client.post(f"/v1/sessions/{sid}/end", headers=c.h)
    run_jobs()
    reports = [f for f in fake.files.values() if f["name"].startswith("report_")]
    assert len(reports) == 1
    fid = next(i for i, f in fake.files.items() if f["name"].startswith("report_"))
    assert fake.path_of(fid).startswith(f"Users/{folder_key(c.uid)}/Interviews/")
    assert b"pro@example.invalid" not in reports[0]["data"]


def test_nothing_touches_drive_when_not_configured(client):
    enable_pro(client)
    calls = []
    drive.set_http_client(httpx.Client(transport=httpx.MockTransport(lambda r: calls.append(r) or httpx.Response(500))))
    try:
        c = Candidate(client)
        cv = c.upload_cv()
        run_jobs()
        assert not calls and _status(cv["id"]) == "disabled"
        with db_session() as db:
            assert db.execute(select(DriveFolder)).first() is None
    finally:
        drive.set_http_client(httpx.Client(timeout=60.0))
