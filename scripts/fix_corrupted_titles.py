"""
Fix corrupted spaced dashes in database titles, competitions, case types, and results.
"""

import os
import re
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.supabase_client import get_supabase_client
from services.deck_taxonomy import (
    CASE_TYPES,
    INDUSTRIES,
    KNOWN_COMPETITIONS,
    RESULTS,
)


def clean_corrupted_string(text: str) -> str:
    """Strip spaced em-dashes and artifact spaces between single letters."""
    if not text:
        return ""

    # Replace null bytes and replacement character
    text = text.replace("\x00", "").replace("\ufffd", "").replace("", "")
    
    # If the text has single characters separated by dashes or spaces:
    # e.g. "— L — ' — O — r — é — a — l" or "— D — I — G — I — T — A — L —"
    # Remove dashes/spaces between single characters
    for _ in range(8):
        text = re.sub(r"(?<=\b[A-Za-z0-9éÉ'’])\s*[-—]\s*(?=[A-Za-z0-9éÉ'’]\b)", "", text)
        text = re.sub(r"(?<=\b[A-Za-z0-9éÉ'’])\s+(?=[A-Za-z0-9éÉ'’]\b)", "", text)

    # Clean leading or trailing dashes
    text = re.sub(r"^[—\-\s]+", "", text)
    text = re.sub(r"[—\-\s]+$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def repair_all_database_decks():
    supabase = get_supabase_client()
    print("=" * 80)
    print("      REPAIRING CORRUPTED TITLES AND METADATA IN DECK VAULT")
    print("=" * 80)

    res = supabase.table("deck_skeletons").select("*").execute()
    decks = res.data or []
    print(f"Auditing {len(decks)} decks...\n")

    repaired_count = 0

    for idx, deck in enumerate(decks):
        deck_id = deck["id"]
        raw_title = deck.get("title") or ""
        raw_comp = deck.get("competition") or ""
        raw_co = deck.get("company") or ""
        raw_ct = deck.get("case_type") or ""
        raw_res = deck.get("result") or ""
        raw_org = deck.get("organizer") or ""
        year = deck.get("year") or 2024
        slug = deck.get("slug") or ""
        orig_fn = deck.get("original_filename") or ""
        norm_fn = deck.get("normalized_filename") or ""

        # Clean all fields
        clean_comp = clean_corrupted_string(raw_comp)
        clean_co = clean_corrupted_string(raw_co)
        clean_ct = clean_corrupted_string(raw_ct).lower()
        clean_res = clean_corrupted_string(raw_res)
        clean_org = clean_corrupted_string(raw_org)

        # Standardize Case Type
        valid_case_types = [c.lower() for c in CASE_TYPES]
        if clean_ct not in valid_case_types:
            # Match closest
            matched_ct = "strategy"
            for vct in valid_case_types:
                if vct in clean_ct or clean_ct in vct:
                    matched_ct = vct
                    break
            clean_ct = matched_ct

        # Standardize Result
        if "winner" in clean_res.lower() and "runner" not in clean_res.lower() and "semi" not in clean_res.lower():
            clean_res = "National Winner"
        elif "runner" in clean_res.lower() or "1st runner" in clean_res.lower():
            clean_res = "National 1st Runner Up"
        elif "semi" in clean_res.lower():
            clean_res = "National Semi Finalist"
        elif "finalist" in clean_res.lower():
            clean_res = "National Finalist"
        elif clean_res.lower() in ("other", ""):
            clean_res = "National Finalist"

        # Standardize Competition & Company
        combined = f"{clean_comp} {clean_co} {orig_fn} {norm_fn} {slug}".lower()
        combined = re.sub(r"[^a-z0-9\s]+", " ", combined)
        combined = re.sub(r"\s+", " ", combined)

        if "flipkart" in combined or "wired" in combined:
            if "6" in combined:
                clean_comp = "Flipkart WiRED 6.0"
                year = 2022
            elif "7" in combined:
                clean_comp = "Flipkart WiRED 7.0"
                year = 2023
            else:
                clean_comp = "Flipkart WIRED 8.0"
                year = 2024
            clean_co = "Flipkart"
            clean_ct = "strategy"
        elif "marketizing" in combined or "bitsom" in combined:
            clean_comp = "Marketizing 3.0 — BITSoM"
            clean_co = "BITSoM"
            clean_ct = "strategy"
            year = 2026
        elif "loreal" in combined or "or al" in combined or "oral" in combined:
            clean_comp = "L'Oréal Sustainability Challenge"
            clean_co = "L'Oréal"
            clean_ct = "sustainability"
        elif "accenture" in combined:
            clean_comp = "Accenture B-School Challenge"
            clean_co = "Accenture"
            clean_ct = "digital transformation"
        elif "corporatecasechallenge" in combined or "corporate" in combined:
            clean_comp = "Corporate Case Challenge"
            clean_co = "Corporate"
            clean_ct = clean_ct or "strategy"
        elif "makemytrip" in combined or "trek" in combined:
            clean_comp = "MakeMyTrip Trek"
            clean_co = "MakeMyTrip"
            clean_ct = "product"
        elif "galderma" in combined or "grad" in combined:
            clean_comp = "Galderma G.R.A.D. Challenge"
            clean_co = "Galderma"
            clean_ct = "marketing"
        elif "mondelez" in combined:
            clean_comp = "Mondelez Supply Track"
            clean_co = "Mondelez International"
            clean_ct = "supply chain"
        elif "delphique" in combined or "union bank" in combined:
            clean_comp = "Union Bank Delphique Cerebro"
            clean_co = "Union Bank of India"
            clean_ct = "BFSI"
        elif "apc" in combined or "canvas" in combined or "asian paints" in combined:
            clean_comp = "Asian Paints Canvas"
            clean_co = "Asian Paints"
            clean_ct = "growth"
        elif "lime" in combined or "unilever" in combined or "hul" in combined:
            clean_comp = "HUL L.I.M.E."
            clean_co = "Hindustan Unilever"
            clean_ct = "marketing"
        elif "transcend" in combined or "colgate" in combined:
            clean_comp = "Colgate Transcend"
            clean_co = "Colgate-Palmolive"
            clean_ct = "marketing"
        elif "wavemakers" in combined or "boat" in combined:
            clean_comp = "boAt WaveMakers Challenge"
            clean_co = "boAt"
            clean_ct = "marketing"
        elif "cummins" in combined:
            clean_comp = "Cummins Case Challenge"
            clean_co = "Cummins"
            clean_ct = "finance"
        elif "jsw" in combined:
            clean_comp = "JSW Steel Challenge"
            clean_co = "JSW Steel"
            clean_ct = "operations"
        elif "tata steel" in combined or "steel a thon" in combined or "steelathon" in combined:
            clean_comp = "Tata Steel Steel-a-thon"
            clean_co = "Tata Steel"
            clean_ct = "operations"
        elif "sugar" in combined:
            clean_comp = "SUGAR Cosmetics Valuation Challenge"
            clean_co = "SUGAR Cosmetics"
            clean_ct = "finance"
        elif "marico" in combined:
            clean_comp = "Marico Over The Wall"
            clean_co = "Marico"
            clean_ct = "growth"
        elif "reckitt" in combined:
            clean_comp = "Reckitt Global Challenge"
            clean_co = "Reckitt Benckiser"
            clean_ct = "sustainability"
        elif "kotak" in combined:
            clean_comp = "Kotak Life Growth Manager Challenge"
            clean_co = "Kotak Life Insurance"
            clean_ct = "growth"
        elif "happ" in combined:
            clean_comp = "Happ Coach Case Challenge"
            clean_co = "Happ Coach"
            clean_ct = "growth"
        elif "amazon" in combined:
            clean_comp = "Amazon Advertising Smart Challenge"
            clean_co = "Amazon"
            clean_ct = "product"
        elif "abinbev" in combined or "bud" in combined:
            clean_comp = "AB InBev The BUD Challenge"
            clean_co = "AB InBev"
            clean_ct = "marketing"
        elif "pidilite" in combined:
            clean_comp = "Bond with Pidilite Case Challenge"
            clean_co = "Pidilite Industries"
            clean_ct = "marketing"
        elif "publicis" in combined or "sapient" in combined:
            clean_comp = "Publicis Sapient Product Spotlight"
            clean_co = "Publicis Sapient"
            clean_ct = "product"
        elif "stylbiz" in combined or "myntra" in combined:
            clean_comp = "Myntra StylBiz"
            clean_co = "Myntra"
            clean_ct = "product"
        elif "icici" in combined:
            clean_comp = "ICICI Bank Case Challenge"
            clean_co = "ICICI Bank"
            clean_ct = "BFSI"
        elif "samsung" in combined or "edge" in combined:
            clean_comp = "Samsung E.D.G.E."
            clean_co = "Samsung"
            clean_ct = "product"
        elif "epic" in combined or "tvs" in combined:
            clean_comp = "TVS Credit E.P.I.C. Challenge"
            clean_co = "TVS Credit"
            clean_ct = "BFSI"

        if not clean_comp or clean_comp in ("Corporate Case Challenge", "Case Competition"):
            clean_comp = "Corporate Case Challenge"
            clean_co = "Corporate"

        if not clean_co:
            clean_co = clean_comp

        clean_title = f"{clean_comp} {year} — {clean_res} Deck"
        clean_description = f"{clean_co} case solution for {clean_comp} ({year}) focusing on {clean_ct}."

        clean_payload = {
            "title": clean_title,
            "competition": clean_comp,
            "company": clean_co,
            "organizer": clean_org or clean_co,
            "result": clean_res,
            "case_type": clean_ct,
            "year": year,
            "description": clean_description,
            "seo_title": f"{clean_comp} {year} Solution Deck | {clean_res} — MECE",
            "seo_description": f"Verified {clean_res.lower()} presentation for {clean_comp} ({year}). Explore the full {clean_ct} framework, market analysis, and slide deck on MECE.",
            "is_active": True,
            "is_indexable": True,
        }

        try:
            supabase.table("deck_skeletons").update(clean_payload).eq("id", deck_id).execute()
            repaired_count += 1
            print(f"[{idx+1}/{len(decks)}] [REPAIRED] {clean_title} | Comp: {clean_comp} | CaseType: {clean_ct}")
        except Exception as err:
            # Baseline update
            fallback = {
                "title": clean_title,
                "competition": clean_comp,
                "result": clean_res,
                "case_type": clean_ct,
                "year": year,
                "description": clean_description,
                "is_active": True,
                "is_indexable": True,
            }
            supabase.table("deck_skeletons").update(fallback).eq("id", deck_id).execute()
            repaired_count += 1
            print(f"[{idx+1}/{len(decks)}] [REPAIRED BASELINE] {clean_title}")

    print("\n" + "=" * 80)
    print(f"REPAIR COMPLETE! All {repaired_count} / {len(decks)} decks cleaned.")
    print("=" * 80)


if __name__ == "__main__":
    repair_all_database_decks()
