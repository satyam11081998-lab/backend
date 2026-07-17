"""
Deck Vault Rewards — upload a winning case-competition deck + certificate,
get a Pro discount coupon after manual admin verification.

POST /deck-vault/submit  (multipart)  — deck + certificate + competition details + T&C.
GET  /deck-vault/status               — caller's latest submission + live coupon, for the UI.

Policy (defaults; admin can override the % at approval time in /admin/deck-vault):
  corporate competition podium (winner / runner-up / 2nd runner-up) -> 60% off Pro
  b-school  competition podium                                      -> 40% off Pro

Security model:
  - Caller is identified ONLY from the verified Supabase JWT (never a body field).
  - Files land in the PRIVATE `deck-vault-submissions` bucket (no storage policies,
    service-role only) under {user_id}/{submission_id}/ with fixed names.
  - Extension + declared content-type + magic-byte sniff + size caps.
  - Rate-limited; one pending submission per user (also enforced by a partial
    unique index); a user whose submission was approved can't farm more coupons.
  - Coupons themselves are minted only by the admin approval action, never here.
"""

import os
import time
from typing import Optional

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile

from services.auth import get_verified_user_id
from services.rate_limit import check_rate_limit
from services.supabase_client import get_supabase_client
from services.telegram_notify import notify_deck_submission

router = APIRouter(prefix="/deck-vault", tags=["deck-vault"])

BUCKET = "deck-vault-submissions"
TNC_VERSION = "2026-07-17"

MAX_DECK_BYTES = 20 * 1024 * 1024   # PDFs/PPTX of real competition decks run big
MAX_CERT_BYTES = 10 * 1024 * 1024

# Default discount matrix — mirrored in the frontend copy and admin UI.
DEFAULT_PCT = {"corporate": 60, "bschool": 40}

VALID_TYPES = {"corporate", "bschool"}
VALID_POSITIONS = {"winner", "runner_up", "second_runner_up"}

# extension -> (content-type, magic-byte prefixes)
DECK_FORMATS = {
    ".pdf":  ("application/pdf", (b"%PDF-",)),
    ".pptx": ("application/vnd.openxmlformats-officedocument.presentationml.presentation", (b"PK\x03\x04",)),
    ".ppt":  ("application/vnd.ms-powerpoint", (b"\xd0\xcf\x11\xe0",)),
}
CERT_FORMATS = {
    ".pdf":  ("application/pdf", (b"%PDF-",)),
    ".png":  ("image/png", (b"\x89PNG",)),
    ".jpg":  ("image/jpeg", (b"\xff\xd8\xff",)),
    ".jpeg": ("image/jpeg", (b"\xff\xd8\xff",)),
    ".webp": ("image/webp", (b"RIFF",)),
}


def _ext(filename: Optional[str]) -> str:
    name = (filename or "").lower().strip()
    dot = name.rfind(".")
    return name[dot:] if dot != -1 else ""


async def _read_validated(file: UploadFile, formats: dict, max_bytes: int, label: str) -> tuple[bytes, str, str]:
    """Read an upload and enforce extension + size + magic bytes. Returns (bytes, ext, content_type)."""
    ext = _ext(file.filename)
    if ext not in formats:
        allowed = ", ".join(sorted(e.lstrip(".") for e in formats))
        raise HTTPException(status_code=400, detail=f"{label} must be one of: {allowed}.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"{label} file is empty.")
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"{label} is too large (max {max_bytes // (1024 * 1024)} MB).")
    content_type, magics = formats[ext]
    if not any(data[: len(m)] == m for m in magics):
        raise HTTPException(status_code=400, detail=f"{label} doesn't look like a valid {ext.lstrip('.')} file.")
    return data, ext, content_type


def _clean(text: Optional[str], max_len: int) -> str:
    return " ".join((text or "").split())[:max_len].strip()


@router.post("/submit")
async def submit_deck(
    deck: UploadFile = File(...),
    certificate: UploadFile = File(...),
    competition_name: str = Form(...),
    organizer: str = Form(""),
    competition_type: str = Form(...),
    position: str = Form(...),
    year: int = Form(...),
    tnc_accepted: str = Form(...),
    authorization: Optional[str] = Header(default=None),
):
    supabase = get_supabase_client()
    uid = get_verified_user_id(supabase, authorization)
    # Tight cap: nobody legitimately submits more than a few times an hour.
    check_rate_limit(f"deckvault:{uid}", max_calls=4, window_seconds=3600)

    # ── Field validation ────────────────────────────────────────────────────
    competition_name = _clean(competition_name, 120)
    organizer = _clean(organizer, 120)
    if len(competition_name) < 3:
        raise HTTPException(status_code=400, detail="Please give the competition's full name.")
    if competition_type not in VALID_TYPES:
        raise HTTPException(status_code=400, detail="Competition type must be corporate or bschool.")
    if position not in VALID_POSITIONS:
        raise HTTPException(status_code=400, detail="Position must be winner, runner_up or second_runner_up.")
    current_year = time.gmtime().tm_year
    if not (2015 <= int(year) <= current_year):
        raise HTTPException(status_code=400, detail=f"Year must be between 2015 and {current_year}.")
    if (tnc_accepted or "").lower() != "true":
        raise HTTPException(status_code=400, detail="You must accept the submission terms to continue.")

    # ── State guards: one pending at a time; approved users already got theirs ──
    try:
        existing = (
            supabase.table("deck_submissions")
            .select("id, status")
            .eq("user_id", uid)
            .in_("status", ["pending", "approved"])
            .execute()
        )
    except Exception:
        raise HTTPException(status_code=502, detail="Could not check your submission history. Try again.")
    for row in existing.data or []:
        if row["status"] == "pending":
            raise HTTPException(status_code=409, detail="You already have a submission under review — verification typically takes 5–6 hours.")
        if row["status"] == "approved":
            raise HTTPException(status_code=409, detail="Your deck was already approved — this reward is one per member.")

    # ── File validation (extension + size + magic bytes) ────────────────────
    deck_bytes, deck_ext, deck_ct = await _read_validated(deck, DECK_FORMATS, MAX_DECK_BYTES, "Deck")
    cert_bytes, cert_ext, cert_ct = await _read_validated(certificate, CERT_FORMATS, MAX_CERT_BYTES, "Certificate")

    # ── Store files in the private vault bucket ──────────────────────────────
    import uuid as _uuid

    submission_id = str(_uuid.uuid4())
    deck_path = f"{uid}/{submission_id}/deck{deck_ext}"
    cert_path = f"{uid}/{submission_id}/certificate{cert_ext}"
    storage = supabase.storage.from_(BUCKET)
    try:
        storage.upload(deck_path, deck_bytes, file_options={"content-type": deck_ct})
        storage.upload(cert_path, cert_bytes, file_options={"content-type": cert_ct})
    except Exception:
        # Best-effort cleanup so a retry never hits a half-written path.
        try:
            storage.remove([deck_path, cert_path])
        except Exception:
            pass
        raise HTTPException(status_code=502, detail="Could not store your files — please try again in a minute.")

    # ── Record the submission ────────────────────────────────────────────────
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        supabase.table("deck_submissions").insert({
            "id": submission_id,
            "user_id": uid,
            "competition_name": competition_name,
            "organizer": organizer,
            "competition_type": competition_type,
            "position": position,
            "year": int(year),
            "deck_path": deck_path,
            "certificate_path": cert_path,
            "tnc_accepted_at": now_iso,
            "tnc_version": TNC_VERSION,
            "status": "pending",
        }).execute()
    except Exception:
        try:
            storage.remove([deck_path, cert_path])
        except Exception:
            pass
        # Most likely cause: the one-pending unique index raced a double-click.
        raise HTTPException(status_code=409, detail="Could not record the submission — you may already have one under review.")

    # ── Telegram ping to the admin (fire-and-forget, never blocks) ───────────
    user_name, user_email = "", ""
    try:
        u = supabase.table("users").select("name, email").eq("id", uid).single().execute()
        user_name = (u.data or {}).get("name") or ""
        user_email = (u.data or {}).get("email") or ""
    except Exception:
        pass
    notify_deck_submission(
        submission_id=submission_id,
        user_name=user_name,
        user_email=user_email,
        competition_name=competition_name,
        organizer=organizer,
        competition_type=competition_type,
        position=position,
        year=int(year),
        default_pct=DEFAULT_PCT[competition_type],
    )

    return {
        "submission_id": submission_id,
        "status": "pending",
        "message": "Deck received! Verification typically takes 5–6 hours — your coupon code will appear here once approved.",
    }


@router.get("/status")
async def deck_vault_status(authorization: Optional[str] = Header(default=None)):
    """Latest submission + live coupon for the signed-in user (drives the UI states)."""
    supabase = get_supabase_client()
    uid = get_verified_user_id(supabase, authorization)
    check_rate_limit(f"deckvault-status:{uid}", max_calls=30, window_seconds=60)

    submission = None
    try:
        subs = (
            supabase.table("deck_submissions")
            .select("id, competition_name, competition_type, position, year, status, admin_note, discount_pct, created_at, reviewed_at")
            .eq("user_id", uid)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        submission = (subs.data or [None])[0]
    except Exception:
        raise HTTPException(status_code=502, detail="Could not load your submission status.")

    coupon = None
    try:
        cps = (
            supabase.table("discount_coupons")
            .select("code, discount_pct, tier_scope, status, expires_at")
            .eq("user_id", uid)
            .eq("source", "deck_vault")
            .in_("status", ["active", "redeemed"])
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        coupon = (cps.data or [None])[0]
    except Exception:
        coupon = None  # coupon panel is optional; the submission state still renders

    return {"submission": submission, "coupon": coupon}
