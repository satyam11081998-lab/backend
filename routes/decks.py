"""
Deck processing and AI routes (Admin-only).

Provides endpoints for:
- Rendering PDF pages to WebP and saving to the private storage bucket.
- Generating verified AI executive summaries without hallucinated numbers.
- Full end-to-end deck processing.
"""

import os
from typing import Optional

from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from pydantic import BaseModel

from services import gdrive
from services.ai_usage import assert_daily_budget
from services.auth import get_verified_user, is_guest_user
from services.deck_ai import DeckAIError, generate_deck_summary
from services.deck_render import render_deck_pages
from services.rate_limit import check_rate_limit
from services.supabase_client import get_supabase_client

router = APIRouter(prefix="/decks", tags=["decks"])


def _require_admin(authorization: Optional[str]) -> str:
    """Validate Bearer token and enforce is_admin or trusted service/cron key."""
    token = (authorization or "").replace("Bearer ", "").strip()
    service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    cron_secret = os.environ.get("CRON_SECRET", "").strip()
    if (service_key and token == service_key) or (cron_secret and token == cron_secret):
        return "system_service"

    supabase = get_supabase_client()
    uid, user_obj = get_verified_user(supabase, authorization)
    if is_guest_user(user_obj):
        raise HTTPException(status_code=403, detail="Admins only")

    try:
        res = supabase.table("users").select("is_admin").eq("id", uid).single().execute()
        is_admin = bool((res.data or {}).get("is_admin"))
    except Exception:
        raise HTTPException(status_code=403, detail="Admins only")

    if not is_admin:
        raise HTTPException(status_code=403, detail="Admins only")
    return uid


def _effective_free_pages(supabase, deck_id: str, fallback_page_count: Optional[int]) -> int:
    """Ask POSTGRES for the free-page count. Never recompute it here.

    This was a hand-written "exact replica of the SQL function", which is the
    same thing three copies of the clarification quota were before they drifted
    and produced the 2026-08-01 P0 (CONTRACTS.md C9). A replica is only exact
    until someone edits one of them.

    It decides which pages get a watermark, so a drift here would silently
    watermark the wrong slides — or leave a paid slide unmarked. Reading the
    computed column means the rule has exactly one definition, in
    `public.effective_free_pages`, which the image route and the public page
    already read.
    """
    try:
        res = (
            supabase.table("deck_skeletons")
            .select("effective_free_pages")
            .eq("id", deck_id)
            .single()
            .execute()
        )
        val = (res.data or {}).get("effective_free_pages")
        if val is not None:
            return int(val)
    except Exception as e:  # noqa: BLE001
        print(f"[decks] could not read effective_free_pages for {deck_id}: {e}")

    # Fail CLOSED. If the rule cannot be read, watermark only the first page
    # rather than guessing generously — an over-marked preview is a cosmetic
    # problem, an under-marked one gives away unwatermarked paid slides.
    return 1


def _fetch_deck_bytes(storage_path: str, supabase) -> bytes:
    """Download binary bytes from Google Drive or Supabase storage."""
    if not storage_path:
        raise ValueError("Deck has no storage path")

    if storage_path.startswith("gdrive:"):
        file_id = storage_path[len("gdrive:"):]
        return gdrive.download_file_bytes(file_id)

    # Supabase storage bucket fallback
    bucket = "skeletons"
    clean_path = storage_path
    if "/" in storage_path and storage_path.startswith("deck-vault-submissions/"):
        bucket = "deck-vault-submissions"
        clean_path = storage_path.split("/", 1)[1]

    try:
        return supabase.storage.from_(bucket).download(clean_path)
    except Exception as e:
        # Try alternate bucket
        try:
            return supabase.storage.from_("skeletons").download(clean_path)
        except Exception:
            raise RuntimeError(f"Could not download deck file from storage: {e}")


@router.post("/{deck_id}/process")
async def process_deck(
    deck_id: str,
    pdf: Optional[UploadFile] = File(None),
    authorization: Optional[str] = Header(default=None),
):
    """End-to-end processing: renders WebP pages and generates verified AI summary."""
    uid = _require_admin(authorization)
    check_rate_limit(f"deck_process:{uid}", max_calls=60, window_seconds=3600)
    assert_daily_budget()

    supabase = get_supabase_client()
    deck_res = supabase.table("deck_skeletons").select("*").eq("id", deck_id).maybeSingle().execute()
    deck = deck_res.data
    if not deck:
        raise HTTPException(status_code=404, detail="Deck not found")

    try:
        if pdf is not None:
            pdf_bytes = await pdf.read()
        else:
            pdf_bytes = _fetch_deck_bytes(deck["storage_path"], supabase)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not load PDF bytes: {e}")

    # 1. Render pages
    free_pages = _effective_free_pages(supabase, deck_id, deck.get("page_count"))
    try:
        render_result = render_deck_pages(
            deck_id=deck["id"],
            pdf_bytes=pdf_bytes,
            slug=deck.get("slug") or "",
            effective_free_pages=free_pages,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Page rendering failed: {e}")

    # 2. Generate summary
    try:
        summary = generate_deck_summary(
            deck_id=deck["id"],
            pdf_bytes=pdf_bytes,
            deck_title=deck.get("title", ""),
            competition=deck.get("competition", ""),
            organizer=deck.get("organizer", ""),
            year=deck.get("year"),
            user_id=uid,
        )
    except DeckAIError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        summary = deck.get("summary") or ""

    return {
        "success": True,
        "page_count": render_result["page_count"],
        "pages_rendered_at": render_result["rendered_at"],
        "summary": summary,
    }


@router.post("/{deck_id}/render")
async def render_only(
    deck_id: str,
    pdf: Optional[UploadFile] = File(None),
    authorization: Optional[str] = Header(default=None),
):
    """Rasterise and store WebP page images."""
    uid = _require_admin(authorization)
    check_rate_limit(f"deck_render:{uid}", max_calls=60, window_seconds=3600)

    supabase = get_supabase_client()
    deck_res = supabase.table("deck_skeletons").select("*").eq("id", deck_id).maybeSingle().execute()
    deck = deck_res.data
    if not deck:
        raise HTTPException(status_code=404, detail="Deck not found")

    try:
        if pdf is not None:
            pdf_bytes = await pdf.read()
        else:
            pdf_bytes = _fetch_deck_bytes(deck["storage_path"], supabase)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not load PDF bytes: {e}")

    free_pages = _effective_free_pages(supabase, deck_id, deck.get("page_count"))

    try:
        render_result = render_deck_pages(
            deck_id=deck["id"],
            pdf_bytes=pdf_bytes,
            slug=deck.get("slug") or "",
            effective_free_pages=free_pages,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Page rendering failed: {e}")

    return {
        "success": True,
        "page_count": render_result["page_count"],
        "pages_rendered_at": render_result["rendered_at"],
    }


@router.post("/{deck_id}/summarize")
async def summarize_only(
    deck_id: str,
    authorization: Optional[str] = Header(default=None),
):
    """Generate or regenerate AI summary with zero hallucination check."""
    uid = _require_admin(authorization)
    check_rate_limit(f"deck_summary:{uid}", max_calls=60, window_seconds=3600)
    assert_daily_budget()

    supabase = get_supabase_client()
    deck_res = supabase.table("deck_skeletons").select("*").eq("id", deck_id).maybeSingle().execute()
    deck = deck_res.data
    if not deck:
        raise HTTPException(status_code=404, detail="Deck not found")

    pdf_bytes = _fetch_deck_bytes(deck["storage_path"], supabase)

    try:
        summary = generate_deck_summary(
            deck_id=deck["id"],
            pdf_bytes=pdf_bytes,
            deck_title=deck.get("title", ""),
            competition=deck.get("competition", ""),
            organizer=deck.get("organizer", ""),
            year=deck.get("year"),
            user_id=uid,
        )
    except DeckAIError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "success": True,
        "summary": summary,
    }
