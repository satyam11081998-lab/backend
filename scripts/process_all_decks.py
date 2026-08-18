"""Script to render and summarize all existing decks in the database."""
import os
import sys

from services.supabase_client import get_supabase_client
from routes.decks import _fetch_deck_bytes, _effective_free_pages
from services.deck_render import render_deck_pages
from services.deck_ai import generate_deck_summary

def main():
    supabase = get_supabase_client()
    res = supabase.table("deck_skeletons").select("*").execute()
    decks = res.data or []
    print(f"Found {len(decks)} decks to process.")

    for d in decks:
        title = d.get("title")
        deck_id = d.get("id")
        slug = d.get("slug")
        storage_path = d.get("storage_path")
        print(f"\n==========================================")
        print(f"Processing: {title}")
        print(f"Deck ID: {deck_id} | Slug: {slug}")
        print(f"Storage Path: {storage_path}")
        print(f"==========================================")

        try:
            print(f"1. Downloading PDF bytes...")
            pdf_bytes = _fetch_deck_bytes(storage_path, supabase)
            print(f"   Downloaded {len(pdf_bytes)} bytes.")

            free_pages = _effective_free_pages(supabase, deck_id, d.get("page_count"))
            print(f"2. Rendering pages with watermark (effective free pages: {free_pages})...")
            render_res = render_deck_pages(deck_id, pdf_bytes, slug or "", free_pages)
            print(f"   Successfully rendered {render_res['rendered_pages']} pages to WebP + OG JPEG!")

            if not d.get("summary"):
                print(f"3. Generating verified AI executive summary...")
                try:
                    summary = generate_deck_summary(
                        deck_id, pdf_bytes, title, d.get("competition", ""),
                        d.get("organizer", ""), d.get("year")
                    )
                    print(f"   Summary generated successfully:\n{summary[:150]}...\n")
                except Exception as e_ai:
                    print(f"   AI summary failed: {e_ai}")
            else:
                print("3. Summary already exists in database.")

        except Exception as e:
            print(f"   ERROR processing {title}: {e}")

    print("\nProcessing complete for all decks!")

if __name__ == "__main__":
    main()