"""
Admin Deck Ingestion & Review Queue Routes.
"""

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel

from routes.decks import _require_admin
from services.deck_ingestion_pipeline import discover_decks, process_single_deck, run_batch_ingestion
from services.supabase_client import get_supabase_client

router = APIRouter(prefix="/decks/ingest", tags=["deck-ingestion"])


class ScanRequest(BaseModel):
    directory_path: str
    dry_run: bool = True
    force: bool = False
    batch_size: Optional[int] = None


class ApproveReviewRequest(BaseModel):
    title: Optional[str] = None
    competition: Optional[str] = None
    company: Optional[str] = None
    case_type: Optional[str] = None
    round_type: Optional[str] = None
    result: Optional[str] = None
    year: Optional[int] = None
    tags: Optional[List[str]] = None


@router.post("/scan")
def trigger_ingestion_scan(
    payload: ScanRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Trigger a batch ingestion scan over a local or mounted directory (Admin only)."""
    uid = _require_admin(authorization)

    if not os.path.exists(payload.directory_path):
        raise HTTPException(status_code=400, detail=f"Directory path does not exist: {payload.directory_path}")

    stats = run_batch_ingestion(
        root_directory=payload.directory_path,
        dry_run=payload.dry_run,
        force=payload.force,
        batch_size=payload.batch_size,
    )
    return {
        "success": True,
        "dry_run": payload.dry_run,
        "stats": stats,
    }


@router.get("/review-queue")
def get_review_queue(
    limit: int = Query(default=50, ge=1, le=100),
    authorization: Optional[str] = Header(default=None),
):
    """List all decks in the database flagged for human review (Admin only)."""
    _require_admin(authorization)
    supabase = get_supabase_client()

    res = (
        supabase.table("deck_skeletons")
        .select("id, title, original_filename, normalized_filename, competition, company, case_type, result, year, metadata_confidence, classification_confidence, processing_status, created_at")
        .eq("processing_status", "needs_review")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return {
        "count": len(res.data or []),
        "items": res.data or [],
    }


@router.post("/approve/{deck_id}")
def approve_deck_review(
    deck_id: str,
    payload: ApproveReviewRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Approve or update a flagged deck's metadata and mark it completed (Admin only)."""
    _require_admin(authorization)
    supabase = get_supabase_client()

    update_fields: Dict[str, Any] = {"processing_status": "completed"}
    if payload.title is not None:
        update_fields["title"] = payload.title
    if payload.competition is not None:
        update_fields["competition"] = payload.competition
    if payload.company is not None:
        update_fields["company"] = payload.company
    if payload.case_type is not None:
        update_fields["case_type"] = payload.case_type
    if payload.round_type is not None:
        update_fields["round_type"] = payload.round_type
    if payload.result is not None:
        update_fields["result"] = payload.result
    if payload.year is not None:
        update_fields["year"] = payload.year
    if payload.tags is not None:
        update_fields["tags"] = payload.tags

    res = supabase.table("deck_skeletons").update(update_fields).eq("id", deck_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Deck not found")

    return {
        "success": True,
        "deck": res.data[0],
    }
