"""
Deck Ingestion Pipeline Orchestrator.

Implements the end-to-end ingestion lifecycle:
DISCOVER -> HASH -> DUP CHECK -> EXTRACT -> PREPROCESS -> CLASSIFY ->
GENERATE METADATA -> RENAME -> UPLOAD DRIVE -> RENDER SLIDES -> SAVE DB -> COMPLETE
"""

import glob
import os
import time
from typing import Any, Callable, Dict, List, Optional

from services import gdrive
from services.deck_ai_gemini import generate_deck_synthesis
from services.deck_classifier import classify_deck_rules
from services.deck_extractor import compute_file_hash, extract_deck
from services.deck_render import render_deck_pages
from services.gdrive import normalize_deck_filename, upload_or_sync_deck
from services.supabase_client import get_supabase_client

SUPPORTED_EXTENSIONS = (".pdf", ".pptx", ".ppt", ".xlsx")

BASELINE_COLUMNS = {
    "title", "source_kind", "competition", "result", "case_type",
    "round_type", "file_type", "slide_count", "page_count",
    "description", "summary", "tags", "storage_path",
    "is_active", "is_indexable", "year", "organizer", "free_pages",
}

_migration_notice_printed = False


def _ensure_seo_description(ai_synthesis: Dict[str, Any], classification: Dict[str, Any]) -> str:
    """Guarantee a 120-160 char meta description. Gemini sometimes returns one that
    is too short, which reads poorly in search results and trips the audit's SEO
    check. Fall back to a richer template built from the deck's own metadata."""
    desc = (ai_synthesis.get("seo_description") or "").strip()
    if len(desc) < 120:
        comp = ai_synthesis.get("competition") or classification.get("competition") or "Case Competition"
        res = ai_synthesis.get("result") or classification.get("result") or "Finalist"
        yr = classification.get("year") or ""
        case_type = ai_synthesis.get("case_type") or classification.get("case_type") or "strategy"
        company = ai_synthesis.get("company") or classification.get("company") or ""
        tail = f" for {company}" if company and company.lower() not in ("company", "corporate") else ""
        desc = (
            f"Verified {str(res).lower()} presentation for {comp} ({yr}). "
            f"Explore the full {case_type} case framework{tail} on MECE Deck Vault."
        )
    return desc[:160]


def _safe_db_write(supabase, table_name: str, payload: Dict[str, Any], deck_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Perform insert or update with automatic fallback to baseline schema if migration 0050 has not been run."""
    global _migration_notice_printed
    try:
        if deck_id:
            res = supabase.table(table_name).update(payload).eq("id", deck_id).execute()
            return res.data[0] if (res and res.data and len(res.data) > 0) else None
        else:
            res = supabase.table(table_name).insert(payload).execute()
            return res.data[0] if (res and res.data and len(res.data) > 0) else None
    except Exception as e:
        err_msg = str(e)
        if "PGRST204" in err_msg or "schema cache" in err_msg.lower() or "column" in err_msg.lower():
            if not _migration_notice_printed:
                print("\n[NOTE] Migration 0050 is not yet applied to Supabase SQL. Falling back to baseline columns.")
                print("       To enable rich AI/SEO columns in PostgreSQL, run: consilio/supabase/migrations/0050_deck_ingestion_metadata.sql\n")
                _migration_notice_printed = True

            fallback_payload = {k: v for k, v in payload.items() if k in BASELINE_COLUMNS}
            if deck_id:
                res = supabase.table(table_name).update(fallback_payload).eq("id", deck_id).execute()
                return res.data[0] if (res and res.data and len(res.data) > 0) else None
            else:
                res = supabase.table(table_name).insert(fallback_payload).execute()
                return res.data[0] if (res and res.data and len(res.data) > 0) else None
        else:
            raise


def discover_decks(root_directory: str) -> List[str]:
    """Recursively discover all candidate deck files in the root directory."""
    if not os.path.exists(root_directory):
        raise FileNotFoundError(f"Directory not found: {root_directory}")

    discovered: List[str] = []
    for root, _, files in os.walk(root_directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                discovered.append(os.path.join(root, file))

    # Sort for deterministic processing order
    discovered.sort()
    return discovered


def check_existing_deck_by_hash(file_hash: str, supabase=None) -> Optional[Dict[str, Any]]:
    """Check if a deck with this SHA-256 hash is already in the database."""
    if not supabase:
        supabase = get_supabase_client()
    try:
        res = (
            supabase.table("deck_skeletons")
            .select("id, slug, title, storage_path, created_at")
            .eq("file_hash", file_hash)
            .execute()
        )
        if res and res.data and len(res.data) > 0:
            return res.data[0]
        return None
    except Exception:
        # If file_hash column does not exist yet
        return None


def process_single_deck(
    file_path: str,
    dry_run: bool = False,
    force: bool = False,
    user_id: Optional[str] = None,
    supabase=None,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    no_llm: bool = False,
    rename_files: bool = True,
) -> Dict[str, Any]:
    """
    Process a single deck through all pipeline stages with fault isolation.

    no_llm:       skip the Gemini call and use deterministic local synthesis (the
                  result is generic and will be held for review, never indexed).
    rename_files: when False, skip the Google Drive normalise/upload step and keep a
                  local storage path (useful for local dry testing without Drive).
    """
    filename = os.path.basename(file_path)
    if progress_callback:
        progress_callback("extracting", {"file": filename})

    if not supabase and not dry_run:
        supabase = get_supabase_client()

    try:
        # 1. Local Content & Structural Extraction
        extracted = extract_deck(file_path)
        file_hash = extracted["file_hash"]

        # 2. Duplicate Detection
        if not force and not dry_run and supabase:
            existing = check_existing_deck_by_hash(file_hash, supabase)
            if existing:
                return {
                    "status": "skipped",
                    "reason": "already_processed",
                    "deck_id": existing["id"],
                    "title": existing["title"],
                    "slug": existing.get("slug"),
                    "file_hash": file_hash,
                    "file_path": file_path,
                }

        if progress_callback:
            progress_callback("classifying", {"file": filename, "hash": file_hash[:8]})

        # 3. 4-Stage Classification & Rule Inference
        classification = classify_deck_rules(extracted)

        # 4. Metadata, Gist & SEO Synthesis
        if progress_callback:
            progress_callback("synthesizing", {"file": filename})

        ai_synthesis = generate_deck_synthesis(
            extracted_deck=extracted,
            rule_classification=classification,
            user_id=user_id,
            force_refresh=force,
            no_llm=no_llm,
        )

        # 5. Deterministic File Naming & Target Drive Hierarchy
        normalized_filename = normalize_deck_filename(
            competition=ai_synthesis.get("competition") or classification["competition"],
            company=ai_synthesis.get("company") or classification["company"],
            case_type=ai_synthesis.get("case_type") or classification["case_type"],
            year=classification.get("year"),
            ext=extracted["file_type"],
        )

        year_str = str(classification.get("year") or 2024)
        company_folder = ai_synthesis.get("company") or classification["company"]
        target_drive_path = f"Case Decks / {year_str} / {company_folder} / {normalized_filename}"

        # Fail loud: a deck is only "completed" (and therefore publishable/indexable)
        # when BOTH the rule classifier is confident AND the summary came from a real,
        # number-verified Gemini read of the deck. If Gemini was unavailable (missing
        # key/package, quota, parse error) the synthesis is generic template text — that
        # must be held for human review, never silently published to a public page.
        synthesis_source = ai_synthesis.get("_synthesis_source", "unknown")
        ai_ok = synthesis_source == "gemini"
        needs_review = bool(classification["needs_review"]) or not ai_ok
        status = "needs_review" if needs_review else "completed"
        if not ai_ok:
            review_reason = (
                "AI summary fell back to generic template text "
                f"(synthesis_source={synthesis_source}); Gemini did not produce a "
                "verified summary. Check GEMINI_API_KEY and that google-generativeai "
                "is installed."
            )
            print(f"[pipeline] HELD FOR REVIEW: {filename} -> {review_reason}")

        result_item = {
            "status": status,
            "original_filename": filename,
            "normalized_filename": normalized_filename,
            "file_hash": file_hash,
            "file_size": extracted["file_size"],
            "slide_count": extracted["slide_count"],
            "title": ai_synthesis.get("title") or classification["title"],
            "competition": ai_synthesis.get("competition") or classification["competition"],
            "company": ai_synthesis.get("company") or classification["company"],
            "organizer": ai_synthesis.get("organizer") or classification["organizer"],
            "industry": ai_synthesis.get("industry") or classification["industry"],
            "case_type": ai_synthesis.get("case_type") or classification["case_type"],
            "function": classification.get("function"),
            "round_type": ai_synthesis.get("round_type") or classification["round_type"],
            "result": ai_synthesis.get("result") or classification["result"],
            "year": classification.get("year"),
            "difficulty": classification.get("difficulty", "medium"),
            "geography": classification.get("geography", "India"),
            "source_kind": classification.get("source_kind", "corporate"),
            "tags": ai_synthesis.get("tags") or classification["tags"],
            "description": ai_synthesis.get("description", ""),
            "summary": ai_synthesis.get("summary") or ai_synthesis.get("executive_summary", ""),
            "executive_summary": ai_synthesis.get("executive_summary", ""),
            "gist": ai_synthesis.get("gist", {}),
            "seo_title": ai_synthesis.get("seo_title", ""),
            "seo_description": _ensure_seo_description(ai_synthesis, classification),
            "ai_summary": ai_synthesis.get("ai_summary", ""),
            "confidence": classification["confidence"],
            "field_confidences": classification["field_confidences"],
            "target_drive_path": target_drive_path,
            "file_path": file_path,
        }

        # 6. Dry Run Return
        if dry_run:
            return result_item

        # 7. Live Execution: Upload to Google Drive
        if progress_callback:
            progress_callback("uploading_drive", {"file": normalized_filename})

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        drive_info = {"gdrive_file_id": None, "gdrive_folder_id": None, "gdrive_url": None, "gdrive_path": target_drive_path}
        storage_path = ""
        if rename_files and gdrive.is_configured():
            try:
                drive_sync_res = upload_or_sync_deck(
                    normalized_filename=normalized_filename,
                    data=file_bytes,
                    year=classification.get("year"),
                    company_or_comp=company_folder,
                    content_type="application/pdf" if extracted["file_type"] == "pdf" else "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                )
                drive_info = drive_sync_res
                storage_path = f"gdrive:{drive_sync_res['gdrive_file_id']}"
            except Exception as drive_err:
                print(f"[pipeline] Drive upload warning for {filename}: {drive_err}")
                storage_path = f"local:{filename}"
        else:
            storage_path = f"local:{filename}"

        # 8. Save Record to Database
        if progress_callback:
            progress_callback("saving_db", {"file": normalized_filename})

        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        db_payload = {
            "title": result_item["title"],
            "source_kind": result_item["source_kind"],
            "competition": result_item["competition"],
            "company": result_item["company"],
            "organizer": result_item["organizer"],
            "industry": result_item["industry"],
            "function": result_item["function"],
            "case_type": result_item["case_type"],
            "round_type": result_item["round_type"],
            "result": result_item["result"],
            "year": result_item["year"],
            "difficulty": result_item["difficulty"],
            "geography": result_item["geography"],
            "file_type": extracted["file_type"],
            "slide_count": extracted["slide_count"],
            "page_count": extracted["slide_count"] if extracted["file_type"] == "pdf" else None,
            "description": result_item["description"],
            "summary": result_item["summary"],
            "executive_summary": result_item["executive_summary"],
            "gist": result_item["gist"],
            "seo_title": result_item["seo_title"],
            "seo_description": result_item["seo_description"],
            "ai_summary": result_item["ai_summary"],
            "tags": result_item["tags"],
            "file_hash": file_hash,
            "original_filename": filename,
            "normalized_filename": normalized_filename,
            "storage_path": storage_path,
            "gdrive_file_id": drive_info.get("gdrive_file_id"),
            "gdrive_folder_id": drive_info.get("gdrive_folder_id"),
            "gdrive_url": drive_info.get("gdrive_url"),
            "gdrive_path": drive_info.get("gdrive_path"),
            "processing_status": status,
            "classification_confidence": result_item["field_confidences"],
            "metadata_confidence": result_item["confidence"],
            "is_active": True,
            # Only fully-verified decks are exposed to search engines. Anything held
            # for review stays out of the index (and off the public page) until a
            # human clears it, so generic/low-confidence text can never rank.
            "is_indexable": status == "completed",
            "error_message": None if ai_ok else (
                f"synthesis_source={synthesis_source}; held for review"
            ),
            "processed_at": now_iso,
        }

        # Check if row with this hash already exists (upsert)
        existing = check_existing_deck_by_hash(file_hash, supabase)
        saved_row = _safe_db_write(supabase, "deck_skeletons", db_payload, deck_id=existing["id"] if existing else None)
        deck_id = (saved_row and saved_row.get("id")) or (existing and existing.get("id"))

        if saved_row and saved_row.get("slug"):
            result_item["slug"] = saved_row["slug"]
        elif not deck_id:
            try:
                re_check = supabase.table("deck_skeletons").select("id, slug").eq("title", db_payload["title"]).execute()
                if re_check and re_check.data and len(re_check.data) > 0:
                    deck_id = re_check.data[0]["id"]
                    result_item["slug"] = re_check.data[0].get("slug")
            except Exception:
                pass

        result_item["deck_id"] = deck_id

        # 9. Slide Rasterization (if PDF and deck_id is present)
        if extracted["file_type"] == "pdf" and deck_id:
            try:
                if progress_callback:
                    progress_callback("rendering_slides", {"file": normalized_filename})
                # This count only controls which preview pages get a watermark.
                # The ACCESS boundary is enforced server-side by the SQL function
                # public.effective_free_pages() in the image route — never here, and
                # never trusted from the client. Kept as 25% (clamped 1-5) so the
                # watermark set matches the default the DB will serve.
                render_deck_pages(
                    deck_id=deck_id,
                    pdf_bytes=file_bytes,
                    slug=result_item.get("slug", ""),
                    effective_free_pages=min(5, max(1, round(extracted["slide_count"] * 0.25))),
                )
            except Exception as render_err:
                print(f"[pipeline] Slide rendering warning for {deck_id}: {render_err}")

        return result_item

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "original_filename": filename,
            "file_path": file_path,
        }


def run_batch_ingestion(
    root_directory: str,
    dry_run: bool = False,
    force: bool = False,
    retry_failed: bool = False,
    batch_size: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Run full batch ingestion over a directory.
    """
    deck_files = discover_decks(root_directory)
    total_found = len(deck_files)

    if batch_size and batch_size > 0:
        deck_files = deck_files[:batch_size]

    supabase = get_supabase_client() if not dry_run else None

    stats = {
        "total_discovered": total_found,
        "total_queued": len(deck_files),
        "completed": 0,
        "needs_review": 0,
        "skipped": 0,
        "failed": 0,
        "results": [],
    }

    for idx, file_path in enumerate(deck_files):
        res = process_single_deck(
            file_path=file_path,
            dry_run=dry_run,
            force=force,
            supabase=supabase,
        )

        st = res.get("status")
        if st == "completed":
            stats["completed"] += 1
        elif st == "needs_review":
            stats["needs_review"] += 1
        elif st == "skipped":
            stats["skipped"] += 1
        else:
            stats["failed"] += 1

        stats["results"].append(res)

        if progress_callback:
            progress_callback(idx + 1, len(deck_files), res)

    return stats
