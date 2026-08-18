"""
Google Drive Storage & Hierarchical Sync for Deck Vault.

Supports:
- Service Account JWT / OAuth2 Refresh Token
- Hierarchical folder creation (e.g. Case Decks / 2024 / Asian Paints /)
- Resumable uploads for 100MB+ decks
- Idempotent file synchronization and duplicate detection in Drive
- Sanitized deterministic file naming convention
"""

import base64
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import jwt

TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPE = "https://www.googleapis.com/auth/drive"
GDRIVE_PREFIX = "gdrive:"

_token_cache: dict = {"token": None, "expires_at": 0.0}
_folder_id_cache: Dict[Tuple[str, str], str] = {}


def is_configured() -> bool:
    """True when any complete Drive auth option + a folder id are present."""
    has_folder = bool(get_root_folder_id())
    has_oauth = all(
        os.environ.get(k, "").strip()
        for k in ("GOOGLE_DRIVE_REFRESH_TOKEN", "GOOGLE_DRIVE_CLIENT_ID", "GOOGLE_DRIVE_CLIENT_SECRET")
    )
    has_sa = bool(os.environ.get("GOOGLE_SA_CREDENTIALS", "").strip()) or (
        bool(os.environ.get("GOOGLE_SA_CLIENT_EMAIL", "").strip())
        and bool(os.environ.get("GOOGLE_SA_PRIVATE_KEY", "").strip())
    )
    return has_folder and (has_oauth or has_sa)


def get_root_folder_id() -> str:
    for var in ("GDRIVE_SUBMISSIONS_FOLDER_ID", "GDRIVE_FOLDER_ID", "GOOGLE_DRIVE_ROOT_FOLDER_ID"):
        v = os.environ.get(var, "").strip().strip('"')
        if v:
            return v
    return ""


def _sa_creds() -> tuple[str, str]:
    """(client_email, private_key_pem)"""
    raw_json = os.environ.get("GOOGLE_SA_CREDENTIALS", "").strip()
    if raw_json:
        data = json.loads(base64.b64decode(raw_json).decode("utf-8"))
        return data["client_email"], data["private_key"]

    email = os.environ.get("GOOGLE_SA_CLIENT_EMAIL", "").strip().strip('"').replace("'", "")
    raw_key = os.environ.get("GOOGLE_SA_PRIVATE_KEY", "")
    if not email or not raw_key:
        raise RuntimeError("Google Drive service account credentials are not configured")
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


def get_access_token() -> str:
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


def sanitize_filename_component(name: str) -> str:
    """Sanitize string for safe cross-platform and cloud file naming."""
    if not name:
        return "Unknown"
    # Remove illegal characters: / \ : * ? " < > |
    cleaned = re.sub(r'[\/\\:\*\?"<>\|\.,;!\']+', '', name)
    # Replace multiple spaces/hyphens with single underscore
    cleaned = re.sub(r'[\s\-]+', '_', cleaned).strip('_')
    return cleaned[:40] or "Unknown"


def normalize_deck_filename(
    competition: str,
    company: str,
    case_type: str,
    year: Optional[int],
    ext: str = "pdf",
) -> str:
    """
    Construct canonical deterministic filename:
    [Competition]_[Company]_[CaseType]_[Year].[ext]
    Example: Accenture_B-School_Challenge_Accenture_Strategy_2024.pdf
    """
    comp_clean = sanitize_filename_component(competition)
    comp_clean = comp_clean.replace("HUL_LIME", "HUL_LIME").replace("LOREAL", "Loreal")
    company_clean = sanitize_filename_component(company)
    case_clean = sanitize_filename_component(case_type).title()
    year_str = str(year) if year else "2024"

    # Avoid duplicate company if competition already contains major company tokens
    raw_words = [re.sub(r'[^a-z0-9]+', '', w) for w in re.split(r'[\s_\-]+', company.lower())]
    company_words = [w for w in raw_words if w not in ('ltd', 'limited', 'inc', 'corp', 'corporation', 'pvt', 'private', 'the', 'of', 'and', '') and len(w) > 2]
    comp_lower = comp_clean.lower()
    has_overlap = bool(company_words and all(w in comp_lower for w in company_words))

    parts = [comp_clean]
    if not has_overlap and company_clean.lower() not in comp_lower:
        parts.append(company_clean)
    parts.extend([case_clean, year_str])

    base = "_".join(parts)
    clean_ext = ext.lstrip(".").lower()
    return f"{base}.{clean_ext}"


def get_or_create_subfolder(folder_name: str, parent_id: str) -> str:
    """Find or create a subfolder inside parent_id."""
    cache_key = (parent_id, folder_name)
    if cache_key in _folder_id_cache:
        return _folder_id_cache[cache_key]

    token = get_access_token()
    # Search if folder already exists
    query = f"'{parent_id}' in parents and name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    resp = httpx.get(
        "https://www.googleapis.com/drive/v3/files",
        headers={"Authorization": f"Bearer {token}"},
        params={"q": query, "fields": "files(id, name)", "supportsAllDrives": "true", "includeItemsFromAllDrives": "true"},
        timeout=20.0,
    )
    resp.raise_for_status()
    files = resp.json().get("files", [])
    if files:
        folder_id = files[0]["id"]
        _folder_id_cache[cache_key] = folder_id
        return folder_id

    # Create new folder
    create_resp = httpx.post(
        "https://www.googleapis.com/drive/v3/files?supportsAllDrives=true",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        },
        timeout=20.0,
    )
    create_resp.raise_for_status()
    folder_id = create_resp.json()["id"]
    _folder_id_cache[cache_key] = folder_id
    return folder_id


def ensure_folder_hierarchy(path_segments: List[str], root_id: Optional[str] = None) -> Tuple[str, str]:
    """
    Ensure folder tree exists in Google Drive, e.g. ['Case Decks', '2024', 'Asian Paints'].
    Returns (target_folder_id, full_folder_path_string).
    """
    current_id = root_id or get_root_folder_id()
    if not current_id:
        raise RuntimeError("Google Drive root folder ID is not configured.")

    built_path = []
    for segment in path_segments:
        clean_seg = re.sub(r'[\/\\:\*\?"<>\|]+', ' ', segment).strip()
        if not clean_seg:
            continue
        current_id = get_or_create_subfolder(clean_seg, current_id)
        built_path.append(clean_seg)

    return current_id, " / ".join(built_path)


def find_file_in_folder(folder_id: str, filename: str) -> Optional[Dict[str, Any]]:
    """Check if file with filename already exists inside folder_id."""
    token = get_access_token()
    query = f"'{folder_id}' in parents and name = '{filename}' and trashed = false"
    resp = httpx.get(
        "https://www.googleapis.com/drive/v3/files",
        headers={"Authorization": f"Bearer {token}"},
        params={"q": query, "fields": "files(id, name, webViewLink, webContentLink, size)", "supportsAllDrives": "true", "includeItemsFromAllDrives": "true"},
        timeout=20.0,
    )
    if not resp.is_success:
        return None
    files = resp.json().get("files", [])
    return files[0] if files else None


def upload_bytes(
    filename: str,
    data: bytes,
    content_type: str = "application/pdf",
    folder_id: Optional[str] = None,
) -> str:
    """Upload into a specified or root folder via server-side resumable upload."""
    target_folder = folder_id or get_root_folder_id()
    if not target_folder:
        raise RuntimeError("GDRIVE_FOLDER_ID is missing")
    token = get_access_token()

    session = httpx.post(
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&supportsAllDrives=true",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Type": content_type,
        },
        json={"name": filename, "parents": [target_folder]},
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
        timeout=120.0,
    )
    put.raise_for_status()
    file_id = put.json().get("id")
    if not file_id:
        raise RuntimeError("Drive upload returned no file id")
    return file_id


def upload_or_sync_deck(
    normalized_filename: str,
    data: bytes,
    year: Optional[int],
    company_or_comp: str,
    content_type: str = "application/pdf",
) -> Dict[str, str]:
    """
    Idempotent hierarchical Drive sync:
    1. Resolves / creates hierarchy: Case Decks / <Year> / <Company> /
    2. Checks if file already exists in target folder.
    3. Uploads if not present.
    Returns: {"gdrive_file_id", "gdrive_folder_id", "gdrive_url", "gdrive_path"}
    """
    year_folder = str(year) if year else "2024"
    company_folder = sanitize_filename_component(company_or_comp)

    folder_id, folder_path = ensure_folder_hierarchy(["Case Decks", year_folder, company_folder])

    existing = find_file_in_folder(folder_id, normalized_filename)
    if existing:
        file_id = existing["id"]
        web_url = existing.get("webViewLink") or f"https://drive.google.com/file/d/{file_id}/view"
        return {
            "gdrive_file_id": file_id,
            "gdrive_folder_id": folder_id,
            "gdrive_url": web_url,
            "gdrive_path": f"{folder_path} / {normalized_filename}",
            "status": "reused",
        }

    file_id = upload_bytes(normalized_filename, data, content_type=content_type, folder_id=folder_id)
    web_url = f"https://drive.google.com/file/d/{file_id}/view"
    return {
        "gdrive_file_id": file_id,
        "gdrive_folder_id": folder_id,
        "gdrive_url": web_url,
        "gdrive_path": f"{folder_path} / {normalized_filename}",
        "status": "uploaded",
    }


def download_file_bytes(file_id: str) -> bytes:
    """Download full binary bytes from Google Drive."""
    token = get_access_token()
    resp = httpx.get(
        f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&supportsAllDrives=true",
        headers={"Authorization": f"Bearer {token}"},
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.content


def delete_file(file_id: str) -> None:
    """Delete a file from Google Drive."""
    token = get_access_token()
    resp = httpx.delete(
        f"https://www.googleapis.com/drive/v3/files/{file_id}?supportsAllDrives=true",
        headers={"Authorization": f"Bearer {token}"},
        timeout=20.0,
    )
    if resp.status_code not in (200, 204, 404):
        resp.raise_for_status()
