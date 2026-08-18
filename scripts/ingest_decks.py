"""
Case Deck Automated Ingestion CLI.

Scans deck folders, extracts text locally, classifies against MECE taxonomy,
generates verified AI summaries & SEO metadata, renames and organizes into
Google Drive hierarchy, and stores structured metadata into PostgreSQL.

Usage:
  python scripts/ingest_decks.py --dir "C:/path/to/decks"
  python scripts/ingest_decks.py --dir "C:/path/to/decks" --dry-run
  python scripts/ingest_decks.py --dir "C:/path/to/decks" --batch-size 10
  python scripts/ingest_decks.py --dir "C:/path/to/decks" --force
"""

import argparse
import json
import os
import sys
import time

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from services.deck_ingestion_pipeline import discover_decks, process_single_deck
from services.supabase_client import get_supabase_client


def print_banner():
    print("=" * 80)
    print("      MECE AUTOMATED CASE DECK INGESTION & INDEXING PIPELINE")
    print("=" * 80)


def format_confidence_badge(score: float) -> str:
    if score >= 0.85:
        return f"[HIGH {int(score*100)}%]"
    elif score >= 0.70:
        return f"[MED  {int(score*100)}%]"
    else:
        return f"[LOW  {int(score*100)}%]"


def main():
    parser = argparse.ArgumentParser(description="MECE Automated Deck Ingestion Pipeline")
    parser.add_argument(
        "--dir",
        type=str,
        default=os.getenv("DECK_SOURCE_DIRECTORY", r"C:\Users\satya\Downloads\cases\IMI Unstop Igniters Case Decks\Corporate Case Comps\some of them"),
        help="Root directory containing deck files (PDF, PPTX)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Simulate pipeline without modifying DB or uploading files")
    parser.add_argument("--force", action="store_true", help="Force re-processing of already indexed files")
    parser.add_argument("--batch-size", type=int, default=None, help="Process at most N decks")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed or unindexed decks")
    parser.add_argument("--json", action="store_true", help="Output full results as JSON")

    args = parser.parse_args()

    if not args.json:
        print_banner()
        print(f"Target Directory : {args.dir}")
        print(f"Execution Mode   : {'DRY RUN (Simulation)' if args.dry_run else 'LIVE INGESTION'}")
        print(f"Force Reprocess  : {args.force}")
        if args.batch_size:
            print(f"Batch Limit      : {args.batch_size} decks")
        print("-" * 80)

    try:
        deck_files = discover_decks(args.dir)
    except Exception as e:
        print(f"ERROR: Could not discover decks: {e}", file=sys.stderr)
        sys.exit(1)

    total_discovered = len(deck_files)
    if not args.json:
        print(f"Discovered {total_discovered} candidate deck files.")
        print("-" * 80)

    if total_discovered == 0:
        if not args.json:
            print("No PDF/PPTX files found in target directory.")
        return

    if args.batch_size and args.batch_size > 0:
        deck_files = deck_files[:args.batch_size]

    supabase = get_supabase_client() if not args.dry_run else None

    completed_count = 0
    needs_review_count = 0
    skipped_count = 0
    failed_count = 0
    results = []

    t_start = time.time()

    for idx, file_path in enumerate(deck_files):
        num = idx + 1
        total = len(deck_files)
        orig_name = os.path.basename(file_path)

        if not args.json:
            print(f"\n[{num}/{total}] Processing: {orig_name[:60]}")

        res = process_single_deck(
            file_path=file_path,
            dry_run=args.dry_run,
            force=args.force,
            supabase=supabase,
        )

        st = res.get("status")
        results.append(res)

        if st == "completed":
            completed_count += 1
            if not args.json:
                badge = format_confidence_badge(res.get("confidence", 0.9))
                print(f"      Status       : OK {badge}")
                print(f"      Classification: {res.get('competition')} | {res.get('company')} | {res.get('case_type')} | {res.get('result')}")
                print(f"      Normalized   : {res.get('normalized_filename')}")
                print(f"      Drive Path   : {res.get('target_drive_path')}")
        elif st == "needs_review":
            needs_review_count += 1
            if not args.json:
                badge = format_confidence_badge(res.get("confidence", 0.6))
                print(f"      Status       : NEEDS REVIEW {badge}")
                print(f"      Classification: {res.get('competition')} | {res.get('company')} | {res.get('case_type')}")
                print(f"      Normalized   : {res.get('normalized_filename')}")
        elif st == "skipped":
            skipped_count += 1
            if not args.json:
                print(f"      Status       : SKIPPED (Already in database: {res.get('title')})")
        else:
            failed_count += 1
            if not args.json:
                print(f"      Status       : FAILED ({res.get('error')})")

    elapsed = round(time.time() - t_start, 2)

    if args.json:
        output = {
            "total_discovered": total_discovered,
            "total_processed": len(deck_files),
            "completed": completed_count,
            "needs_review": needs_review_count,
            "skipped": skipped_count,
            "failed": failed_count,
            "elapsed_seconds": elapsed,
            "results": results,
        }
        print(json.dumps(output, indent=2))
        return

    print("\n" + "=" * 80)
    print("                      INGESTION BATCH SUMMARY")
    print("=" * 80)
    print(f"Total Discovered   : {total_discovered}")
    print(f"Processed In Batch : {len(deck_files)}")
    print(f"Successfully Ready : {completed_count}")
    print(f"Flagged For Review : {needs_review_count}")
    print(f"Skipped Duplicates : {skipped_count}")
    print(f"Failed Extractions : {failed_count}")
    print(f"Total Time Taken   : {elapsed}s")
    print("=" * 80)

    if needs_review_count > 0:
        print("\nItems Flagged For Human Review:")
        for r in results:
            if r.get("status") == "needs_review":
                print(f" - {r.get('original_filename')} (Confidence: {int(r.get('confidence',0)*100)}%) -> Proposed: {r.get('title')}")


if __name__ == "__main__":
    main()
