"""
Google Drive storage for Deck Vault submissions — Python twin of the web app's
lib/google-drive.ts, sharing the SAME env vars and the same `gdrive:<fileId>`
storage-path convention, so files uploaded here are readable by the Next.js
admin/streaming routes with zero extra setup.

Auth (first match wins — identical to the TS client):
  1. GOOGLE_DRIVE_REFRESH_TOKEN + GOOGLE_DRIVE_CLIENT_ID + GOOGLE_DRIVE_CLIENT_SECRET
     (OAuth2 refresh token — personal Google account)
  2. GOOGLE_SA_CREDENTIALS (base64 JSON) or GOOGLE_SA_CLIENT_EMAIL + GOOGLE_SA_PRIVATE_KEY
     (service account JWT)

Folder: GDRIVE_SUBMISSIONS_FOLDER_ID if set, else GDRIVE_FOLDER_ID (the vault).

Uploads use Drive's resumable protocol server-side (session POST, then one PUT
with the bytes) — no multipart size ceiling, works for 20 MB decks.
"""

import base64
import json
import os
import re
import time
from typing import Optional

import httpx
import jwt  # PyJWT — RS256 via the `cryptography` package (both already pinned)

TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPE = "https://www.googleapis.com/auth/drive"

GDRIVE_PREFIX = "gdrive:"

_token_cache: dict = {"token": None, "expires_at": 0.0}


def is_configured() -> bool:
    """True when any complete Drive auth option + a folder id are present."""
    has_folder = bool(_folder_id())
    has_oauth = all(
        os.environ.get(k, "").strip()
        for k in ("GOOGLE_DRIVE_REFRESH_TOKEN", "GOOGLE_DRIVE_CLIENT_ID", "GOOGLE_DRIVE_CLIENT_SECRET")
    )
    has_sa = bool(os.environ.get("GOOGLE_SA_CREDENTIALS", "").strip()) or (
        bool(os.environ.get("GOOGLE_SA_CLIENT_EMAIL", "").strip())
        and bool(os.environ.get("GOOGLE_SA_PRIVATE_KEY", "").strip())
    )
    return has_folder and (has_oauth or has_sa)


def _folder_id() -> str:
    for var in ("GDRIVE_SUBMISSIONS_FOLDER_ID", "GDRIVE_FOLDER_ID"):
        v = os.environ.get(var, "").strip().strip('"')
        if v:
            return v
    return ""


def _sa_creds() -> tuple[str, str]:
    """(client_email, private_key_pem) — with the same defensive PEM rebuild as the TS client."""
    raw_json = os.environ.get("GOOGLE_SA_CREDENTIALS", "").strip()
    if raw_json:
        data = json.loads(base64.b64decode(raw_json).decode("utf-8"))
        return data["client_email"], data["private_key"]

    email = os.environ.get("GOOGLE_SA_CLIENT_EMAIL", "").strip().strip('"').replace("'", "")
    raw_key = os.environ.get("GOOGLE_SA_PRIVATE_KEY", "")
    if not email or not raw_key:
        raise RuntimeError("Google Drive is not configured")
    b64 = re.sub(
        r"-----(BEGIN|END) PRIVATE KEY-----|\\n|\\r|\s+|[\"']",
        "",
        raw_key,
    )
    if len(b64) < 100:
        raise RuntimeError("GOOGLE_SA_PRIVATE_KEY looks truncated or corrupted")
    wrapped = "\n".join(b64[i : i + 64] for i in range(0, len(b64), 64))
    pem = f"-----BEGIN PRIVATE KEY-----\n{wrapped}\n-----END PRIVATE KEY-----\n"
    return email, pem


def _access_token() -> str:
    if _token_cache["token"] and time.time() < _token_cache["expires_at"] - 60:
        return _token_cache["token"]

    refresh = os.environ.get("GOOGLE_DRIVE_REFRESH_TOKEN", "").strip()
    if refresh and os.environ.get("GOOGLE_DRIVE_CLIENT_ID") and os.environ.get("GOOGLE_DRIVE_CLIENT_SECRET"):
        resp = httpx.post(
            TOKEN_URL,
            data={
                "client_id": os.environ["GOOGLE_DRIVE_CLIENT_ID"].strip(),
                "client_secret": os.environ["GOOGLE_DRIVE_CLIENT_SECRET"].strip(),
                "refresh_token": refresh,
                "grant_type": "refresh_token",
            },
            timeout=20.0,
        )
        resp.raise_for_status()
        data = resp.json()
    else:
        email, pem = _sa_creds()
        now = int(time.time())
        assertion = jwt.encode(
            {"iss": email, "scope": SCOPE, "aud": TOKEN_URL, "iat": now, "exp": now + 3600},
            pem,
            algorithm="RS256",
        )
        resp = httpx.post(
            TOKEN_URL,
            data={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": assertion},
            timeout=20.0,
        )
        resp.raise_for_status()
        data = resp.json()

    _token_cache["token"] = data["access_token"]
    _token_cache["expires_at"] = time.time() + float(data.get("expires_in", 3600))
    return _token_cache["token"]


def upload_bytes(filename: str, data: bytes, content_type: str) -> str:
    """Upload into the vault folder via a server-side resumable session.

    Returns the Drive file id. Raises on any failure (caller decides fallback).
    """
    folder = _folder_id()
    if not folder:
        raise RuntimeError("GDRIVE_FOLDER_ID is missing")
    token = _access_token()

    session = httpx.post(
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&supportsAllDrives=true",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Type": content_type,
        },
        json={"name": filename, "parents": [folder]},
        timeout=30.0,
    )
    session.raise_for_status()
    location = session.headers.get("location")
    if not location:
        raise RuntimeError("Drive did not return an upload session URL")

    put = httpx.put(
        location,
        content=data,
        headers={"Content-Type": content_type},
        timeout=120.0,  # a 20 MB deck on a slow leg needs headroom
    )
    put.raise_for_status()
    file_id = put.json().get("id")
    if not file_id:
        raise RuntimeError("Drive upload returned no file id")
    return file_id


def delete_file(file_id: str) -> None:
    """Best-effort delete (cleanup path). 404 is fine; other errors raise."""
    token = _access_token()
    resp = httpx.delete(
        f"https://www.googleapis.com/drive/v3/files/{file_id}?supportsAllDrives=true",
        headers={"Authorization": f"Bearer {token}"},
        timeout=20.0,
    )
    if resp.status_code not in (200, 204, 404):
        resp.raise_for_status()
