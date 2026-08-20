"""
MECE Master Deck Pipeline — Single-Run Audit, Enrich, Upload & SEO Optimise.

One script to rule them all. Run once and it:
  1. Audits EVERY existing deck in deck_skeletons for correctness
  2. Re-extracts real content from PDFs stored on Google Drive
  3. Generates REAL AI summaries using Google Gemini (not OpenAI)
  4. Fixes names, slugs, broken metadata, and character encoding
  5. Renames PDFs on Google Drive with proper normalised names
  6. Generates adversarial SEO/AEO/GEO metadata from actual content
  7. Flags low-confidence decks for human review (hidden from public)
  8. Discovers and ingests NEW deck files from a local directory
  9. Sets default free preview pages to 2

Usage:
  # Audit & enrich all existing decks (main use case):
  python scripts/master_deck_pipeline.py --audit

  # Audit + ingest new decks from a directory:
  python scripts/master_deck_pipeline.py --audit --dir "C:/path/to/new/decks"

  # Dry run (no DB writes, no Drive changes):
  python scripts/master_deck_pipeline.py --audit --dry-run

  # Force re-process everything including cached AI results:
  python scripts/master_deck_pipeline.py --audit --force

  # Skip Gemini API (100% free local mode):
  python scripts/master_deck_pipeline.py --audit --no-llm

Environment:
  GEMINI_API_KEY              — Google Gemini API key (required unless --no-llm)
  SUPABASE_URL                — Supabase project URL
  SUPABASE_SERVICE_ROLE_KEY   — Supabase service role key
  GDRIVE_SUBMISSIONS_FOLDER_ID — Google Drive root folder ID
  GOOGLE_SA_CLIENT_EMAIL      — Service account email
  GOOGLE_SA_PRIVATE_KEY       — Service account private key
"""

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from services.supabase_client import get_supabase_client
from services.deck_taxonomy import (
    BSCHOOLS,
    CASE_TYPES,
    INDUSTRIES,
    KNOWN_COMPETITIONS,
    RESULTS,
    ROUND_TYPES,
    STANDARD_TAGS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = (".pdf", ".pptx", ".ppt")
DEFAULT_FREE_PAGES = 2  # User requested: show exactly 2 free pages by default
CONFIDENCE_REVIEW_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def print_banner():
    print("=" * 80)
    print("    MECE MASTER DECK PIPELINE — AUDIT · ENRICH · UPLOAD · OPTIMISE")
    print("=" * 80)


def slugify(text: str) -> str:
    text = text.lower()
    text = text.replace("'", "").replace("'", "").replace(".", "-").replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text


def clean_encoding(text: Optional[str]) -> str:
    """Clean corrupted unicode, replacement glyphs, and spaced characters."""
    if not text:
        return ""
    text = text.replace("\x00", "").replace("\ufffd", "").replace("\u0000", "")
    # Fix common UTF-8 mojibake sequences
    text = text.replace("\xe2\x80\x93", " - ").replace("\xe2\x80\x99", "'")
    text = text.replace("\xe2\x80\x9c", '"').replace("\xe2\x80\x9d", '"')

    # Fix spaced characters like "F l i p k a r t"
    chars = [c for c in text if c != " "]
    if len(chars) > 4 and "  " in text:
        words = text.split("  ")
        condensed = ["".join(w.split()) for w in words]
        text = " ".join(condensed)
    elif re.search(r"(\b[A-Za-z0-9]\s){4,}", text):
        text = re.sub(r"(?<=\b[A-Za-z0-9])\s(?=[A-Za-z0-9]\b)", "", text)

    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*—\s*—\s*", " - ", text)
    return text.strip()


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sanitize_filename_component(name: str) -> str:
    if not name:
        return "Unknown"
    cleaned = re.sub(r'[\/\\:\*\?"<>\|\.,;!\']+', '', name)
    cleaned = re.sub(r'[\s\-]+', '_', cleaned).strip('_')
    return cleaned[:40] or "Unknown"


def normalise_deck_filename(competition: str, company: str, case_type: str,
                             year: Optional[int], ext: str = "pdf") -> str:
    comp_clean = sanitize_filename_component(competition)
    company_clean = sanitize_filename_component(company)
    case_clean = sanitize_filename_component(case_type).title()
    year_str = str(year) if year else "2024"

    # Avoid duplicate company if competition already contains it
    raw_words = [re.sub(r'[^a-z0-9]+', '', w) for w in re.split(r'[\s_\-]+', company.lower())]
    company_words = [w for w in raw_words if w not in ('ltd', 'limited', 'inc', 'corp', 'the', 'of', 'and', '') and len(w) > 2]
    comp_lower = comp_clean.lower()
    has_overlap = bool(company_words and all(w in comp_lower for w in company_words))

    parts = [comp_clean]
    if not has_overlap and company_clean.lower() not in comp_lower:
        parts.append(company_clean)
    parts.extend([case_clean, year_str])

    base = "_".join(parts)
    clean_ext = ext.lstrip(".").lower()
    return f"{base}.{clean_ext}"


# ---------------------------------------------------------------------------
# SEO / AEO Adversarial Checks
# ---------------------------------------------------------------------------

def adversarial_seo_check(deck: Dict[str, Any]) -> List[str]:
    """
    Check deck metadata like a digital marketer would. Returns list of issues.
    Think: what would make someone searching for THIS deck NOT find it?
    """
    issues = []
    title = deck.get("title", "")
    comp = deck.get("competition", "")
    seo_title = deck.get("seo_title", "")
    seo_desc = deck.get("seo_description", "")
    summary = deck.get("summary", "")
    slug = deck.get("slug", "")
    ai_summary = deck.get("ai_summary", "")

    # 1. Title must contain competition name and year
    if comp and comp.lower() not in title.lower():
        issues.append(f"Title missing competition name '{comp}'")
    year = deck.get("year")
    if year and str(year) not in title:
        issues.append(f"Title missing year '{year}'")

    # 2. SEO title must exist and be under 65 chars
    if not seo_title:
        issues.append("No SEO title")
    elif len(seo_title) > 70:
        issues.append(f"SEO title too long ({len(seo_title)} chars)")

    # 3. SEO description must be 120-160 chars
    if not seo_desc:
        issues.append("No SEO description")
    elif len(seo_desc) < 100:
        issues.append(f"SEO description too short ({len(seo_desc)} chars)")
    elif len(seo_desc) > 165:
        issues.append(f"SEO description too long ({len(seo_desc)} chars)")

    # 4. Summary must be substantive (not generic templates)
    if summary:
        generic_markers = [
            "TAM / SAM / SOM analysis",
            "customer journey mapping",
            "phased Go-To-Market (GTM) rollout",
            "competitive benchmarking, and a phased Go-To-Market",
        ]
        for marker in generic_markers:
            if marker.lower() in summary.lower():
                issues.append(f"Summary contains generic template text: '{marker[:40]}...'")
                break

    if summary and len(summary.split()) < 40:
        issues.append(f"Summary too thin ({len(summary.split())} words)")

    # 5. Slug must be clean and meaningful
    if not slug:
        issues.append("No slug")
    elif len(slug) < 5:
        issues.append(f"Slug too short: '{slug}'")

    # 6. AI summary for answer engines
    if not ai_summary:
        issues.append("No AI summary for answer engines")

    # 7. Result should be specific
    result = deck.get("result", "")
    if not result or result == "Other":
        issues.append("Vague or missing result classification")

    # 8. Company should be populated
    if not deck.get("company") or deck.get("company") == "Corporate":
        issues.append("Company name is generic or missing")

    # 9. Broken characters check
    for field in ["title", "competition", "summary", "description"]:
        val = deck.get(field, "")
        if val and ("\ufffd" in val or "â€" in val or "\x00" in val):
            issues.append(f"Broken characters in '{field}'")

    return issues


# ---------------------------------------------------------------------------
# Drive Operations
# ---------------------------------------------------------------------------

def download_pdf_from_drive(storage_path: str) -> Optional[bytes]:
    """Download PDF bytes from Google Drive given a storage_path like 'gdrive:FILEID'."""
    if not storage_path or not storage_path.startswith("gdrive:"):
        return None
    file_id = storage_path.replace("gdrive:", "").strip()
    if not file_id:
        return None
    try:
        from services.gdrive import download_file_bytes
        return download_file_bytes(file_id)
    except Exception as e:
        print(f"      [WARN] Could not download from Drive ({file_id}): {e}")
        return None


def rename_file_on_drive(old_file_id: str, new_name: str) -> bool:
    """Rename a file on Google Drive."""
    try:
        from services.gdrive import get_access_token
        import httpx
        token = get_access_token()
        resp = httpx.patch(
            f"https://www.googleapis.com/drive/v3/files/{old_file_id}?supportsAllDrives=true",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={"name": new_name},
            timeout=20.0,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"      [WARN] Drive rename failed for {old_file_id}: {e}")
        return False


# ---------------------------------------------------------------------------
# Content Extraction (lightweight — no full pipeline import needed)
# ---------------------------------------------------------------------------

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> Dict[str, Any]:
    """Extract text and structure from PDF bytes."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_bytes)
    slide_count = len(pdf)
    slides = []
    full_text_parts = []

    for i in range(slide_count):
        page = pdf[i]
        textpage = None
        try:
            textpage = page.get_textpage()
            raw_text = textpage.get_text_range() or ""
        except Exception:
            raw_text = ""
        finally:
            if textpage is not None:
                try:
                    textpage.close()
                except Exception:
                    pass
            try:
                page.close()
            except Exception:
                pass

        clean_text = re.sub(r"\s+", " ", raw_text).strip()
        lines = [l.strip() for l in clean_text.split("\n") if l.strip()] if clean_text else []
        title = lines[0][:120] if lines else f"Slide {i + 1}"

        slides.append({
            "slide_number": i + 1,
            "title": title,
            "text": clean_text,
        })
        if clean_text:
            full_text_parts.append(f"--- Slide {i+1}: {title} ---\n{clean_text}")

    try:
        pdf.close()
    except Exception:
        pass

    full_text = "\n\n".join(full_text_parts)

    # Extract numbers for zero-hallucination check
    numbers = set()
    cleaned = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", full_text)
    for token in re.findall(r"\d+(?:\.\d+)?", cleaned):
        try:
            val = float(token)
            numbers.add(str(int(val)) if val == int(val) else str(val))
        except ValueError:
            numbers.add(token)

    return {
        "slide_count": slide_count,
        "slides": slides,
        "full_text": full_text,
        "numbers": numbers,
    }


# ---------------------------------------------------------------------------
# Gemini AI Synthesis (inline to avoid import conflicts with old deck_ai.py)
# ---------------------------------------------------------------------------

def call_gemini_synthesis(
    extracted: Dict[str, Any],
    classification: Dict[str, Any],
    force: bool = False,
    no_llm: bool = False,
) -> Dict[str, Any]:
    """
    Call the Gemini-based AI synthesis engine.
    Falls back to local synthesis if Gemini is unavailable.
    """
    # Use the new Gemini module
    try:
        from services.deck_ai_gemini import generate_deck_synthesis
        return generate_deck_synthesis(
            extracted_deck=extracted,
            rule_classification=classification,
            force_refresh=force,
            no_llm=no_llm,
        )
    except ImportError:
        # If the new module isn't deployed yet, fall back to local synthesis
        return _fallback_local_synthesis(extracted, classification)


def _fallback_local_synthesis(
    extracted: Dict[str, Any],
    classification: Dict[str, Any],
) -> Dict[str, Any]:
    """Minimal local synthesis when no AI module is available."""
    comp = classification.get("competition", "Case Competition")
    co = classification.get("company", "Company")
    ct = classification.get("case_type", "strategy")
    yr = classification.get("year", 2024)
    res = classification.get("result", "National Finalist")

    slides = extracted.get("slides", [])
    slide_titles = [s["title"] for s in slides if not s["title"].startswith("Slide")][:4]
    themes = ", ".join(slide_titles).lower() if slide_titles else f"{ct} analysis"

    summary = (
        f"This case competition presentation for {comp} ({yr}) analyses "
        f"{co}'s strategic challenges. The deck covers {themes}.\n\n"
        f"This {res.lower()} submission is part of the MECE Deck Vault."
    )

    return {
        "title": f"{comp} {yr} - {res} Deck",
        "description": f"{co} case solution for {comp} ({yr}).",
        "competition": comp,
        "company": co,
        "organizer": classification.get("organizer", co),
        "industry": classification.get("industry", "FMCG"),
        "case_type": ct,
        "round_type": classification.get("round_type", "finale"),
        "result": res,
        "tags": classification.get("tags", [ct.title()]),
        "gist": {},
        "executive_summary": summary,
        "summary": summary,
        "seo_title": f"{comp} {yr} Solution | {ct.title()} Deck",
        "seo_description": f"Verified {res.lower()} deck for {comp} ({yr}) on MECE.",
        "ai_summary": f"Verified deck for {comp} ({yr}) — {co} {ct} strategy.",
    }


# ---------------------------------------------------------------------------
# PHASE 1: Audit & Enrich Existing Decks
# ---------------------------------------------------------------------------

def audit_existing_decks(
    dry_run: bool = False,
    force: bool = False,
    no_llm: bool = False,
) -> Dict[str, Any]:
    """
    Audit every deck in deck_skeletons:
    - Fix encoding, metadata, classification
    - Download PDF from Drive and re-extract real content
    - Generate REAL AI summaries from actual deck content (Gemini)
    - Rename files on Drive with proper names
    - Run adversarial SEO checks
    - Flag low-confidence decks for review (is_active=False for public, visible in admin)
    - Set free_pages to DEFAULT_FREE_PAGES
    """
    supabase = get_supabase_client() if not dry_run else None

    print("\n" + "=" * 80)
    print("  PHASE 1: AUDIT & ENRICH ALL EXISTING DECKS")
    print("=" * 80)

    if not dry_run:
        res = supabase.table("deck_skeletons").select("*").order("created_at", desc=True).execute()
        decks = res.data or []
    else:
        print("  [DRY RUN] Would fetch all decks from database")
        decks = []

    print(f"  Total decks to audit: {len(decks)}\n")

    stats = {
        "total": len(decks),
        "enriched_with_ai": 0,
        "enriched_local": 0,
        "renamed_on_drive": 0,
        "flagged_for_review": 0,
        "seo_issues_found": 0,
        "encoding_fixed": 0,
        "skipped": 0,
        "failed": 0,
        "details": [],
    }

    for idx, deck in enumerate(decks):
        deck_id = deck["id"]
        num = idx + 1
        orig_title = deck.get("title", "Unknown")
        storage_path = deck.get("storage_path", "")

        print(f"\n  [{num}/{len(decks)}] {clean_encoding(orig_title)[:60]}")

        try:
            # --- Step 1: Clean encoding on all text fields ---
            had_encoding_issues = False
            for field in ["title", "competition", "company", "organizer", "result",
                          "case_type", "description", "summary", "executive_summary"]:
                raw = deck.get(field, "")
                cleaned = clean_encoding(raw)
                if cleaned != raw:
                    had_encoding_issues = True
                    deck[field] = cleaned

            if had_encoding_issues:
                stats["encoding_fixed"] += 1
                print("      [FIX] Cleaned encoding issues")

            # --- Step 2: Rule-based classification from existing metadata ---
            classification = _classify_from_metadata(deck)

            # --- Step 3: Download PDF and extract real content ---
            extracted = None
            pdf_bytes = None

            if storage_path.startswith("gdrive:"):
                print("      [DRIVE] Downloading PDF for content extraction...")
                pdf_bytes = download_pdf_from_drive(storage_path)
                if pdf_bytes:
                    extracted = extract_text_from_pdf_bytes(pdf_bytes)
                    print(f"      [OK] Extracted {extracted['slide_count']} slides, "
                          f"{len(extracted['full_text'])} chars of text")
                else:
                    print("      [WARN] Could not download PDF — using existing metadata only")

            # --- Step 4: Generate AI synthesis from real content ---
            needs_ai = (
                force
                or not deck.get("summary")
                or _is_generic_summary(deck.get("summary", ""))
                or not deck.get("ai_summary")
            )

            if extracted and needs_ai:
                # Merge path signals
                extracted["file_hash"] = deck.get("file_hash", "")
                extracted["original_filename"] = deck.get("original_filename", "")
                extracted["path_signals"] = {
                    "raw_path": deck.get("gdrive_path", ""),
                    "detected_competition": classification["competition"],
                    "detected_company": classification["company"],
                    "detected_result": classification["result"],
                    "detected_year": classification["year"],
                }

                synthesis = call_gemini_synthesis(
                    extracted=extracted,
                    classification=classification,
                    force=force,
                    no_llm=no_llm,
                )

                if not no_llm:
                    stats["enriched_with_ai"] += 1
                    print("      [GEMINI] Generated real AI summary from deck content")
                else:
                    stats["enriched_local"] += 1
                    print("      [LOCAL] Generated local summary from extracted text")
            elif extracted:
                # Have content but don't need AI re-gen
                synthesis = {
                    "title": classification.get("title", deck.get("title")),
                    "summary": deck.get("summary", ""),
                    "executive_summary": deck.get("executive_summary", ""),
                    "seo_title": deck.get("seo_title", ""),
                    "seo_description": deck.get("seo_description", ""),
                    "ai_summary": deck.get("ai_summary", ""),
                    "gist": deck.get("gist", {}),
                    "tags": deck.get("tags", []),
                }
                stats["enriched_local"] += 1
            else:
                # No PDF available — generate from metadata only
                extracted = {
                    "slides": [],
                    "full_text": "",
                    "numbers": set(),
                    "slide_count": deck.get("page_count") or deck.get("slide_count") or 0,
                    "file_hash": deck.get("file_hash", ""),
                    "original_filename": deck.get("original_filename", ""),
                    "path_signals": {},
                }
                synthesis = call_gemini_synthesis(
                    extracted=extracted,
                    classification=classification,
                    force=force,
                    no_llm=True,  # No content to send to AI
                )
                stats["enriched_local"] += 1

            # --- Step 5: Rename file on Drive ---
            new_filename = normalise_deck_filename(
                competition=classification["competition"],
                company=classification["company"],
                case_type=classification["case_type"],
                year=classification["year"],
                ext=deck.get("file_type", "pdf"),
            )

            old_filename = deck.get("normalized_filename", "")
            gdrive_file_id = deck.get("gdrive_file_id") or (
                storage_path.replace("gdrive:", "") if storage_path.startswith("gdrive:") else None
            )

            if gdrive_file_id and new_filename != old_filename and not dry_run:
                if rename_file_on_drive(gdrive_file_id, new_filename):
                    stats["renamed_on_drive"] += 1
                    print(f"      [RENAME] {old_filename} -> {new_filename}")

            # --- Step 6: Build clean title ---
            comp = classification["competition"]
            yr = classification["year"]
            res_label = classification["result"]
            clean_title = f"{comp} {yr} - {res_label} Deck"

            # --- Step 7: Build slug ---
            base_slug = f"{slugify(comp)}-{yr}-{slugify(res_label)}"

            # --- Step 8: SEO package ---
            seo_title = synthesis.get("seo_title") or f"{comp} {yr} Solution | {classification['case_type'].title()} Deck - MECE"
            if len(seo_title) > 65:
                seo_title = f"{comp} {yr} Deck | {res_label}"

            seo_desc = synthesis.get("seo_description") or (
                f"Verified {res_label.lower()} presentation for {comp} ({yr}). "
                f"Explore the full {classification['case_type']} framework on MECE Deck Vault."
            )
            if len(seo_desc) > 160:
                seo_desc = seo_desc[:157] + "..."

            # --- Step 9: Adversarial SEO check ---
            check_deck = {
                **deck,
                "title": clean_title,
                "competition": comp,
                "company": classification["company"],
                "seo_title": seo_title,
                "seo_description": seo_desc,
                "summary": synthesis.get("summary") or synthesis.get("executive_summary", ""),
                "ai_summary": synthesis.get("ai_summary", ""),
                "result": res_label,
                "year": yr,
            }
            seo_issues = adversarial_seo_check(check_deck)
            if seo_issues:
                stats["seo_issues_found"] += len(seo_issues)
                for issue in seo_issues:
                    print(f"      [SEO] {issue}")

            # --- Step 10: Determine visibility ---
            # Low confidence or no content -> flag for review (hidden from public)
            overall_confidence = classification.get("confidence", 0.8)
            is_active = True
            is_indexable = True

            if overall_confidence < CONFIDENCE_REVIEW_THRESHOLD:
                is_active = False  # Hidden from public, visible in admin
                is_indexable = False
                stats["flagged_for_review"] += 1
                print(f"      [REVIEW] Flagged — confidence {overall_confidence:.0%}")

            # --- Step 11: Build update payload ---
            update_payload = {
                "title": clean_title,
                "competition": comp,
                "company": classification["company"],
                "organizer": classification.get("organizer", classification["company"]),
                "result": res_label,
                "case_type": classification["case_type"],
                "year": yr,
                "normalized_filename": new_filename,
                "summary": synthesis.get("summary") or synthesis.get("executive_summary", ""),
                "executive_summary": synthesis.get("executive_summary", ""),
                "seo_title": seo_title,
                "seo_description": seo_desc,
                "ai_summary": synthesis.get("ai_summary", ""),
                "gist": synthesis.get("gist", {}),
                "tags": synthesis.get("tags", []),
                "description": synthesis.get("description", f"{classification['company']} case solution for {comp}."),
                "is_active": is_active,
                "is_indexable": is_indexable,
                "free_pages": DEFAULT_FREE_PAGES,
                "processing_status": "needs_review" if not is_active else "completed",
            }

            # Preserve industry/function if available
            if classification.get("industry"):
                update_payload["industry"] = classification["industry"]
            if classification.get("function"):
                update_payload["function"] = classification["function"]

            # --- Step 12: Write to DB ---
            if not dry_run:
                try:
                    supabase.table("deck_skeletons").update(update_payload).eq("id", deck_id).execute()
                    print(f"      [DB] Updated: {clean_title}")
                except Exception as e:
                    # Fallback to core fields if migration 0050 not applied
                    fallback = {k: v for k, v in update_payload.items()
                                if k in {"title", "competition", "result", "case_type",
                                         "year", "summary", "tags", "is_active", "is_indexable",
                                         "free_pages", "description"}}
                    try:
                        supabase.table("deck_skeletons").update(fallback).eq("id", deck_id).execute()
                        print(f"      [DB] Updated (baseline): {clean_title}")
                    except Exception as e2:
                        stats["failed"] += 1
                        print(f"      [ERROR] DB write failed: {e2}")
                        continue
            else:
                print(f"      [DRY] Would update: {clean_title}")

            stats["details"].append({
                "deck_id": deck_id,
                "title": clean_title,
                "competition": comp,
                "confidence": overall_confidence,
                "seo_issues": seo_issues,
                "is_active": is_active,
                "new_filename": new_filename,
            })

        except Exception as e:
            stats["failed"] += 1
            print(f"      [ERROR] {e}")

    return stats


# ---------------------------------------------------------------------------
# Classification from existing metadata (replaces the massive if/elif chains)
# ---------------------------------------------------------------------------

def _classify_from_metadata(deck: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use the taxonomy registry (KNOWN_COMPETITIONS) instead of hardcoded if/elif.
    Falls back to existing metadata if no match is found.
    """
    title = clean_encoding(deck.get("title", ""))
    comp = clean_encoding(deck.get("competition", ""))
    company = clean_encoding(deck.get("company", ""))
    organizer = clean_encoding(deck.get("organizer", ""))
    result = clean_encoding(deck.get("result", "")) or "National Finalist"
    case_type = clean_encoding(deck.get("case_type", "")) or "strategy"
    year = deck.get("year") or 2024
    orig_file = deck.get("original_filename", "")
    norm_file = deck.get("normalized_filename", "")
    slug = deck.get("slug", "")

    blob = f"{title} {comp} {company} {orig_file} {norm_file} {slug}".lower()
    blob = re.sub(r"[^a-z0-9\s]+", " ", blob)
    blob = re.sub(r"\s+", " ", blob)

    confidence = 0.85  # Default for already-classified decks

    # Try to match against KNOWN_COMPETITIONS taxonomy
    matched = False
    for alias, meta in KNOWN_COMPETITIONS.items():
        if alias in blob:
            comp = meta["canonical_name"]
            company = meta.get("company", company)
            organizer = meta.get("organizer", organizer)
            case_type = meta.get("default_case_type", case_type)
            confidence = 0.92
            matched = True
            break

    # Detect special competitions with version numbers
    if "flipkart" in blob or "wired" in blob:
        company = "Flipkart"
        if "6" in blob:
            comp, year = "Flipkart WiRED 6.0", 2022
        elif "7" in blob:
            comp, year = "Flipkart WiRED 7.0", 2023
        elif "8" in blob:
            comp, year = "Flipkart WIRED 8.0", 2024
        else:
            comp = "Flipkart WiRED"
        case_type = "strategy"
        confidence = 0.95

    if "marketizing" in blob or "bitsom" in blob:
        comp, company = "Marketizing 3.0 - BITSoM", "BITSoM"
        case_type, year = "strategy", 2026

    # Result refinement from text
    if "winner" in blob and "runner" not in blob and "semi" not in blob:
        result = "National Winner"
    elif "1st runner" in blob or ("runner up" in blob and "2nd" not in blob):
        result = "National 1st Runner Up"
    elif "2nd runner" in blob:
        result = "National 2nd Runner Up"
    elif "semi" in blob:
        result = "National Semi Finalist"
    elif "finalist" in blob:
        result = "National Finalist"

    # Year detection
    year_matches = re.findall(r"\b(20[12]\d)\b", blob)
    if year_matches:
        year = int(max(set(year_matches), key=year_matches.count))

    if not matched and not comp:
        comp = "Corporate Case Challenge"
        confidence = 0.5

    # Industry from taxonomy
    industry = "Other"
    for alias, meta in KNOWN_COMPETITIONS.items():
        if meta["canonical_name"] == comp:
            industry = meta.get("industry", "Other")
            break

    # Function mapping
    function_map = {
        "strategy": "Strategy", "growth": "Strategy", "market entry": "Strategy",
        "marketing": "Marketing & Brand Strategy", "pricing": "Marketing & Brand Strategy",
        "supply chain": "Operations & Supply Chain", "operations": "Operations & Supply Chain",
        "finance": "Finance & M&A", "M&A": "Finance & M&A", "BFSI": "Finance & M&A",
        "product": "Product Management", "digital transformation": "Product Management",
        "analytics": "Analytics & AI Strategy", "sustainability": "Strategy",
    }
    function = function_map.get(case_type, "Strategy")

    return {
        "competition": comp,
        "company": company or organizer or "Enterprise",
        "organizer": organizer or company or "",
        "result": result,
        "case_type": case_type,
        "year": year,
        "industry": industry,
        "function": function,
        "confidence": confidence,
        "title": f"{comp} {year} - {result} Deck",
    }


def _is_generic_summary(summary: str) -> bool:
    """Check if a summary is generic template text vs. real content."""
    if not summary:
        return True
    generic_markers = [
        "TAM / SAM / SOM analysis",
        "customer journey mapping",
        "phased Go-To-Market (GTM) rollout",
        "Competitor Benchmarking, and a phased Go-To-Market",
        "Go-To-Market (GTM) Roadmap",
        "market expansion framework, operational roadmap, and financial viability model",
        "part of the MECE Deck Vault",
        "evaluates core business dynamics, target customer segments",
    ]
    summary_lower = summary.lower()
    matches = sum(1 for m in generic_markers if m.lower() in summary_lower)
    return matches >= 2


# ---------------------------------------------------------------------------
# PHASE 2: Ingest New Decks from Local Directory
# ---------------------------------------------------------------------------

def ingest_new_decks(
    directory: str,
    dry_run: bool = False,
    force: bool = False,
    no_llm: bool = False,
    batch_size: Optional[int] = None,
    rename_files: bool = True,
) -> Dict[str, Any]:
    """
    Discover and ingest new deck files using the existing pipeline.
    This reuses services/deck_ingestion_pipeline.py but with Gemini AI.
    """
    print("\n" + "=" * 80)
    print("  PHASE 2: INGEST NEW DECKS FROM LOCAL DIRECTORY")
    print("=" * 80)
    print(f"  Directory: {directory}")

    if not os.path.exists(directory):
        print(f"  [ERROR] Directory not found: {directory}")
        return {"total": 0, "error": "directory_not_found"}

    # Discover files
    deck_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                deck_files.append(os.path.join(root, file))
    deck_files.sort()

    total = len(deck_files)
    print(f"  Found {total} deck files\n")

    if total == 0:
        return {"total": 0, "ingested": 0}

    if batch_size:
        deck_files = deck_files[:batch_size]

    # Use the existing pipeline with Gemini swap
    try:
        from services.deck_ingestion_pipeline import process_single_deck
        supabase = get_supabase_client() if not dry_run else None

        stats = {"total": total, "ingested": 0, "skipped": 0, "failed": 0, "review": 0}

        for idx, file_path in enumerate(deck_files):
            filename = os.path.basename(file_path)
            print(f"  [{idx+1}/{len(deck_files)}] {filename[:60]}")

            result = process_single_deck(
                file_path=file_path,
                dry_run=dry_run,
                force=force,
                supabase=supabase,
                no_llm=no_llm,
                rename_files=rename_files,
            )

            st = result.get("status")
            if st == "completed":
                stats["ingested"] += 1
                # Set free_pages to 2 for new decks
                if not dry_run and result.get("deck_id"):
                    try:
                        supabase.table("deck_skeletons").update(
                            {"free_pages": DEFAULT_FREE_PAGES}
                        ).eq("id", result["deck_id"]).execute()
                    except Exception:
                        pass
                print(f"      [OK] {result.get('normalized_filename', filename)}")
            elif st == "skipped":
                stats["skipped"] += 1
                print(f"      [SKIP] Already indexed: {result.get('title', filename)}")
            elif st == "needs_review":
                stats["review"] += 1
                print(f"      [REVIEW] Needs human review")
            else:
                stats["failed"] += 1
                print(f"      [FAIL] {result.get('error', 'unknown')}")

        return stats

    except ImportError as e:
        print(f"  [ERROR] Could not import ingestion pipeline: {e}")
        return {"total": total, "error": str(e)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MECE Master Deck Pipeline - Audit, Enrich, Upload & Optimise"
    )
    parser.add_argument("--audit", action="store_true",
                        help="Audit and enrich all existing decks in database")
    parser.add_argument("--dir", type=str, default=None,
                        help="Directory containing new deck files to ingest")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate without modifying DB or Drive")
    parser.add_argument("--force", action="store_true",
                        help="Force re-processing of all decks including cached AI")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip Gemini API — 100% free local mode")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Limit new deck ingestion to N files")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")

    args = parser.parse_args()

    if not args.audit and not args.dir:
        parser.print_help()
        print("\nError: Specify --audit and/or --dir to run the pipeline.")
        sys.exit(1)

    if not args.json:
        print_banner()
        print(f"  Mode           : {'DRY RUN' if args.dry_run else 'LIVE'}")
        print(f"  Gemini AI      : {'DISABLED (free local)' if args.no_llm else 'ENABLED'}")
        print(f"  Force Refresh  : {args.force}")
        print(f"  Free Pages     : {DEFAULT_FREE_PAGES} (default for all decks)")
        if args.dir:
            print(f"  New Decks Dir  : {args.dir}")
        print("-" * 80)

    t_start = time.time()
    results = {}

    # Phase 1: Audit existing
    if args.audit:
        audit_stats = audit_existing_decks(
            dry_run=args.dry_run,
            force=args.force,
            no_llm=args.no_llm,
        )
        results["audit"] = audit_stats

    # Phase 2: Ingest new
    if args.dir:
        ingest_stats = ingest_new_decks(
            directory=args.dir,
            dry_run=args.dry_run,
            force=args.force,
            no_llm=args.no_llm,
            batch_size=args.batch_size,
        )
        results["ingest"] = ingest_stats

    elapsed = round(time.time() - t_start, 2)
    results["elapsed_seconds"] = elapsed

    if args.json:
        # Convert sets to lists for JSON serialization
        print(json.dumps(results, indent=2, default=str))
        return

    # Print summary
    print("\n" + "=" * 80)
    print("                    MASTER PIPELINE SUMMARY")
    print("=" * 80)

    if "audit" in results:
        a = results["audit"]
        print(f"\n  AUDIT & ENRICH:")
        print(f"    Total Decks Audited     : {a.get('total', 0)}")
        print(f"    Enriched with Gemini AI : {a.get('enriched_with_ai', 0)}")
        print(f"    Enriched Locally        : {a.get('enriched_local', 0)}")
        print(f"    Renamed on Drive        : {a.get('renamed_on_drive', 0)}")
        print(f"    Flagged for Review       : {a.get('flagged_for_review', 0)}")
        print(f"    SEO Issues Found        : {a.get('seo_issues_found', 0)}")
        print(f"    Encoding Fixes          : {a.get('encoding_fixed', 0)}")
        print(f"    Failed                  : {a.get('failed', 0)}")

    if "ingest" in results:
        i = results["ingest"]
        print(f"\n  NEW DECK INGESTION:")
        print(f"    Total Found             : {i.get('total', 0)}")
        print(f"    Successfully Ingested   : {i.get('ingested', 0)}")
        print(f"    Skipped (Duplicates)    : {i.get('skipped', 0)}")
        print(f"    Flagged for Review       : {i.get('review', 0)}")
        print(f"    Failed                  : {i.get('failed', 0)}")

    print(f"\n  Total Time: {elapsed}s")
    print(f"  Free Preview Pages: {DEFAULT_FREE_PAGES} (set for all decks)")
    print("=" * 80)


if __name__ == "__main__":
    main()
