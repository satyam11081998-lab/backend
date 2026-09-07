"""
Sync all local case deck PDFs from disk to Supabase Storage `skeletons/` bucket.
Ensures every deck has a permanent, downloadable PDF in storage so that re-processing
and DRM slide rendering work 100% reliably in both Next.js and FastAPI.
"""

import hashlib
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.supabase_client import get_supabase_client


def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def sync_all_pdfs():
    supabase = get_supabase_client()
    print("=" * 80)
    print("      SYNCING LOCAL PDF FILES TO SUPABASE STORAGE 'skeletons' BUCKET")
    print("=" * 80)

    # 1. Index all local files by hash and basename
    cases_root = r"C:\Users\satya\Downloads\cases"
    local_files_by_hash = {}
    local_files_by_name = {}

    for root, _, files in os.walk(cases_root):
        for f in files:
            full_path = os.path.join(root, f)
            if not os.path.isfile(full_path):
                continue
            name_lower = f.lower()
            local_files_by_name[name_lower] = full_path
            try:
                f_hash = compute_sha256(full_path)
                local_files_by_hash[f_hash] = full_path
            except Exception:
                pass

    print(f"Discovered {len(local_files_by_name)} local files on disk.")

    # 2. Get all decks from deck_skeletons
    res = supabase.table("deck_skeletons").select("id, slug, title, storage_path, original_filename, normalized_filename, file_hash").execute()
    decks = res.data or []
    print(f"Auditing {len(decks)} decks in database...\n")

    synced_count = 0

    for idx, deck in enumerate(decks):
        deck_id = deck["id"]
        storage_path = deck.get("storage_path") or ""
        file_hash = deck.get("file_hash")
        orig_fn = (deck.get("original_filename") or "").lower()
        norm_fn = (deck.get("normalized_filename") or "").lower()

        # Find matching local file
        matched_local_path = None
        if file_hash and file_hash in local_files_by_hash:
            matched_local_path = local_files_by_hash[file_hash]
        elif orig_fn and orig_fn in local_files_by_name:
            matched_local_path = local_files_by_name[orig_fn]
        elif norm_fn and norm_fn in local_files_by_name:
            matched_local_path = local_files_by_name[norm_fn]
        elif storage_path.startswith("local:"):
            loc_name = storage_path[len("local:"):].lower()
            if loc_name in local_files_by_name:
                matched_local_path = local_files_by_name[loc_name]

        if not matched_local_path:
            # Try fuzzy match on title or keywords
            title_slug = deck.get("slug") or ""
            for name, path in local_files_by_name.items():
                clean_n = name.replace("-", "").replace("_", "").replace(" ", "")
                clean_s = title_slug.replace("-", "")
                if clean_n[:12] in clean_s or clean_s[:12] in clean_n:
                    matched_local_path = path
                    break

        if matched_local_path and os.path.isfile(matched_local_path):
            with open(matched_local_path, "rb") as f:
                pdf_bytes = f.read()

            target_path = f"{deck_id}.pdf"
            # Upload to 'skeletons' bucket
            try:
                supabase.storage.from_("skeletons").upload(
                    path=target_path,
                    file=pdf_bytes,
                    file_options={"content-type": "application/pdf", "upsert": "true"},
                )
                # Update storage_path in database
                supabase.table("deck_skeletons").update({
                    "storage_path": target_path,
                    "file_hash": compute_sha256(matched_local_path),
                }).eq("id", deck_id).execute()
                synced_count += 1
                print(f"[{idx+1}/{len(decks)}] [SYNCED] {deck['title']} -> skeletons/{target_path}")
            except Exception as err:
                print(f"[{idx+1}/{len(decks)}] [ERROR uploading {deck_id}]: {err}")
        else:
            print(f"[{idx+1}/{len(decks)}] [SKIPPED - Remote or Drive]: {deck['title']} (path: {storage_path[:30]})")

    print("\n" + "=" * 80)
    print(f"SYNC COMPLETE! {synced_count} PDF files uploaded to Supabase Storage 'skeletons' bucket.")
    print("=" * 80)


if __name__ == "__main__":
    sync_all_pdfs()
