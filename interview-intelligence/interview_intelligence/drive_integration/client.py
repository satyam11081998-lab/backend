"""Minimal Google Drive v3 client for Interview Intelligence (docs/K_GOOGLE_DRIVE.md).

Own implementation with II-specific env names; it does not import or share the Deck
Vault's Drive code. Auth: OAuth refresh token of a dedicated storage account, or a service
account (base64 JSON) writing into a Shared Drive. The httpx client is module-level so tests
can swap in an httpx.MockTransport.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from typing import Dict, Optional

import httpx
import jwt

from ..config import get_settings

TOKEN_URL = "https://oauth2.googleapis.com/token"
API = "https://www.googleapis.com/drive/v3/files"
UPLOAD = "https://www.googleapis.com/upload/drive/v3/files"
SCOPE = "https://www.googleapis.com/auth/drive"
FOLDER_MIME = "application/vnd.google-apps.folder"

_http = httpx.Client(timeout=60.0)
_token: Dict[str, float | str] = {"value": "", "exp": 0.0}
_lock = threading.Lock()


class DriveError(RuntimeError):
    def __init__(self, msg: str, *, retryable: bool = True):
        super().__init__(msg)
        self.retryable = retryable


def set_http_client(client: httpx.Client) -> None:
    global _http
    _http = client
    _token.update(value="", exp=0.0)


def _check(resp: httpx.Response, what: str) -> httpx.Response:
    if resp.status_code < 400:
        return resp
    retryable = resp.status_code in (408, 429) or resp.status_code >= 500
    raise DriveError(f"Drive {what} failed: HTTP {resp.status_code} {resp.text[:200]}", retryable=retryable)


def access_token() -> str:
    with _lock:
        if _token["value"] and time.time() < float(_token["exp"]) - 60:
            return str(_token["value"])
        s = get_settings()
        if s.gdrive_refresh_token and s.gdrive_client_id and s.gdrive_client_secret:
            resp = _http.post(TOKEN_URL, data={"client_id": s.gdrive_client_id, "client_secret": s.gdrive_client_secret,
                                               "refresh_token": s.gdrive_refresh_token, "grant_type": "refresh_token"})
        elif s.gdrive_sa_json_b64:
            info = json.loads(base64.b64decode(s.gdrive_sa_json_b64).decode("utf-8"))
            now = int(time.time())
            assertion = jwt.encode({"iss": info["client_email"], "scope": SCOPE, "aud": TOKEN_URL, "iat": now,
                                    "exp": now + 3600}, info["private_key"], algorithm="RS256")
            resp = _http.post(TOKEN_URL, data={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                                               "assertion": assertion})
        else:
            raise DriveError("Drive is not configured", retryable=False)
        _check(resp, "token")
        data = resp.json()
        _token.update(value=data["access_token"], exp=time.time() + float(data.get("expires_in", 3600)))
        return str(_token["value"])


def _h() -> Dict[str, str]:
    return {"Authorization": f"Bearer {access_token()}"}


def _q_escape(v: str) -> str:
    return v.replace("\\", "\\\\").replace("'", "\\'")


def find_folder(name: str, parent: str) -> Optional[str]:
    q = f"'{_q_escape(parent)}' in parents and name = '{_q_escape(name)}' and mimeType = '{FOLDER_MIME}' and trashed = false"
    r = _check(_http.get(API, headers=_h(), params={"q": q, "fields": "files(id,name)", "supportsAllDrives": "true",
                                                    "includeItemsFromAllDrives": "true"}), "folder lookup")
    files = r.json().get("files", [])
    return files[0]["id"] if files else None


def create_folder(name: str, parent: str) -> str:
    r = _check(_http.post(API, headers=_h(), params={"supportsAllDrives": "true"},
                          json={"name": name, "mimeType": FOLDER_MIME, "parents": [parent]}), "folder create")
    return r.json()["id"]


def find_by_app_property(key: str, value: str, parent: Optional[str] = None) -> Optional[dict]:
    q = f"appProperties has {{ key='{_q_escape(key)}' and value='{_q_escape(value)}' }} and trashed = false"
    if parent:
        q += f" and '{_q_escape(parent)}' in parents"
    r = _check(_http.get(API, headers=_h(), params={"q": q, "fields": "files(id,name,parents)",
                                                    "supportsAllDrives": "true", "includeItemsFromAllDrives": "true"}),
               "file lookup")
    files = r.json().get("files", [])
    return files[0] if files else None


def upload(name: str, data: bytes, mime: str, parent: str, app_properties: Dict[str, str]) -> str:
    session = _check(_http.post(UPLOAD, headers={**_h(), "X-Upload-Content-Type": mime,
                                                 "Content-Type": "application/json; charset=UTF-8"},
                                params={"uploadType": "resumable", "supportsAllDrives": "true"},
                                json={"name": name, "parents": [parent], "appProperties": app_properties}),
                     "upload session")
    loc = session.headers.get("location")
    if not loc:
        raise DriveError("Drive did not return an upload URL")
    # The session URI authorises the upload; the bearer token is sent too, as Google's own
    # client libraries do, so a proxy or a future API change cannot turn it into a 401.
    put = _check(_http.put(loc, content=data, headers={**_h(), "Content-Type": mime}), "upload")
    fid = (put.json() or {}).get("id")
    if not fid:
        raise DriveError("Drive upload returned no file id")
    return fid


def delete(file_id: str) -> None:
    r = _http.delete(f"{API}/{file_id}", headers=_h(), params={"supportsAllDrives": "true"})
    if r.status_code not in (200, 204, 404):
        _check(r, "delete")
