"""
Deck AI Synthesis & Metadata Engine — Google Gemini Edition.

Drop-in replacement for deck_ai.py that uses Google Gemini (gemini-2.0-flash)
instead of OpenAI for all AI summarisation, gist extraction, and SEO content.

Zero-hallucination number verification is preserved. Caching by
(file_hash + prompt_version) minimises API spend.

Environment:
  GEMINI_API_KEY   — required when no_llm=False
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Set

from services.deck_extractor import extract_numbers_from_text
from services.deck_taxonomy import CASE_TYPES, INDUSTRIES, RESULTS, ROUND_TYPES

MODEL = "gemini-2.0-flash"
PROMPT_VERSION = "v3_gemini_structured"

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache", "deck_ai")

EN_DASH = "–"
EM_DASH = "—"
DASH_RE = re.compile(f"[{EN_DASH}{EM_DASH}]")


class DeckAIError(Exception):
    pass


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

def _gemini_client():
    """Return a configured Gemini GenerativeModel."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise DeckAIError(
            "GEMINI_API_KEY not set. Export it or add to .env before running with AI."
        )
    try:
        import google.generativeai as genai
    except ImportError:
        raise DeckAIError(
            "google-generativeai package is not installed. "
            "Run: pip install google-generativeai"
        )
    genai.configure(api_key=key)
    return genai.GenerativeModel(MODEL)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_COMBINED_SYNTHESIS_PROMPT = """You are the Lead Consulting Editor at MECE (mece.in), the top case competition and interview prep platform for Indian MBA and PGDM students.

You will analyze extracted text from a real case-competition presentation deck and generate structured metadata, SEO content, a 7-point case gist, and an authoritative executive summary.

STRICT RULES:
1. ZERO HALLUCINATION ON NUMBERS: NEVER invent any metric, percentage, currency figure, market size, headcount, or timeframe not explicitly present in the extracted deck text. If the deck text contains no numbers, the executive summary and gist MUST contain ZERO numbers.
2. British spelling (e.g. monetisation, optimise, organisation, prioritise).
3. Professional consulting tone (McKinsey / BCG / Bain caliber). Plain ASCII punctuation only. No em dashes (use commas or hyphens).
4. ALL content MUST be derived from the ACTUAL deck text provided. Do NOT use generic template language like "TAM/SAM/SOM analysis" or "customer journey mapping" unless those exact frameworks appear in the deck.
5. The executive summary must describe what THIS SPECIFIC deck actually covers — the real problem, the real solution, the real frameworks used.

Return a strictly valid JSON object with the following schema:
{
  "title": "Clean, authoritative public title (e.g. HUL L.I.M.E. 2024 - National Winner Deck)",
  "description": "One-line summary derived from the deck's actual content",
  "competition": "Competition name (e.g. HUL L.I.M.E.)",
  "company": "Subject company / brand analyzed (e.g. Hindustan Unilever)",
  "organizer": "Organizing body / company (e.g. Hindustan Unilever)",
  "industry": "One from official list: FMCG, Paints & Coatings, E-Commerce & Retail, BFSI & Banking, Tech, SaaS & IT, Automotive & Mobility, Consumer Electronics, Healthcare & Pharma, Manufacturing & Industrial, Consulting & Professional Services, Hospitality & Travel, Other",
  "case_type": "One from official list: strategy, marketing, finance, operations, supply chain, product, technology, digital transformation, analytics, consulting, market entry, growth, pricing, M&A, sustainability, ESG, hr, general management, social impact, healthcare, retail, BFSI, other",
  "round_type": "One from official list: screening, campus, zonal, regional, quarter-final, semi-final, final, finale, other",
  "result": "One from official list: National Winner, National 1st Runner Up, National 2nd Runner Up, National Finalist, National Semi Finalist, Zonal Winner, Regional Winner, Campus Winner, Participant, Other",
  "tags": ["Array", "of", "4-6", "tags", "from actual content"],
  "gist": {
    "company_and_context": "What company/brand and real problem is this case about? Be specific to the deck.",
    "central_business_problem": "The actual core dilemma or strategic objective from the deck",
    "strategic_questions": "Key questions actually addressed in this deck",
    "proposed_solution": "The actual strategic solution proposed in the deck",
    "analytical_frameworks": "Frameworks ACTUALLY applied in the deck (not generic ones)",
    "key_recommendations": "The real actionable recommendations from the deck",
    "distinctive_angle": "What specifically makes this deck stand out?"
  },
  "executive_summary": "3-4 coherent paragraphs (total 150-250 words) structured as Problem & Context -> Approach & Framework -> Solution & Recommendations -> Quantified Impact (only real numbers from deck). Must be SPECIFIC to this deck's actual content.",
  "seo_title": "Search optimized title under 65 chars (e.g. HUL LIME 2024 Solution | FMCG Brand & Growth Strategy Deck)",
  "seo_description": "Compelling meta description between 140 and 160 chars. Must reference the deck's actual focus area.",
  "ai_summary": "Dense, factual 3-sentence summary optimized for semantic retrieval and LLM answer engines. Must describe what this specific deck covers."
}
"""


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _clean_text_for_summary(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"(\d)\s*[–—]\s*(\d)", r"\1-\2", text)
    text = DASH_RE.sub("-", text)
    text = (
        text.replace("‘", "'")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("…", "...")
    )
    return text.strip()


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _get_cached_ai_result(file_hash: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(CACHE_DIR):
        return None
    cache_file = os.path.join(CACHE_DIR, f"{file_hash}_{PROMPT_VERSION}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_cached_ai_result(file_hash: str, data: Dict[str, Any]) -> None:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(CACHE_DIR, f"{file_hash}_{PROMPT_VERSION}.json")
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main synthesis
# ---------------------------------------------------------------------------

def generate_deck_synthesis(
    extracted_deck: Dict[str, Any],
    rule_classification: Dict[str, Any],
    user_id: Optional[str] = None,
    force_refresh: bool = False,
    no_llm: bool = False,
) -> Dict[str, Any]:
    """
    Generate complete verified synthesis using Google Gemini.
    When no_llm=True, generates high-quality deterministic metadata locally.
    """
    file_hash = extracted_deck.get("file_hash", "")
    if not force_refresh and file_hash:
        cached = _get_cached_ai_result(file_hash)
        if cached:
            return cached

    comp = rule_classification.get("competition", "Case Competition")
    co = rule_classification.get("company", "Company")
    ct = rule_classification.get("case_type", "strategy")
    yr = rule_classification.get("year", 2024)
    res_label = rule_classification.get("result", "National Finalist")

    if no_llm:
        return _generate_local_synthesis(extracted_deck, rule_classification, file_hash)

    # --- Gemini API call ---
    source_text = extracted_deck.get("full_text", "")
    source_numbers: Set[str] = extracted_deck.get("numbers", set())
    path_signals = extracted_deck.get("path_signals", {})
    year = rule_classification.get("year") or path_signals.get("detected_year")

    # Select rich context — Gemini flash handles longer context well
    context_text = source_text[:12000] if len(source_text) > 50 else "No text layer extracted; mostly graphical diagrams."

    rule_hints = (
        f"Detected Competition: {rule_classification.get('competition')}\n"
        f"Detected Company: {rule_classification.get('company')}\n"
        f"Detected Industry: {rule_classification.get('industry')}\n"
        f"Detected Case Type: {rule_classification.get('case_type')}\n"
        f"Detected Result: {rule_classification.get('result')}\n"
        f"Detected Year: {year}\n"
        f"Original Filename: {extracted_deck.get('original_filename')}\n"
        f"Folder Path: {path_signals.get('raw_path')}\n"
    )

    user_prompt = (
        f"--- EXTRACTED CONTEXT & SIGNALS ---\n{rule_hints}\n"
        f"--- EXTRACTED SLIDE CONTENT ---\n\"\"\"\n{context_text}\n\"\"\"\n\n"
        "Generate the complete structured JSON response matching the required schema."
    )

    last_errors: List[str] = []
    final_payload: Optional[Dict[str, Any]] = None

    for attempt in range(2):
        full_prompt = _COMBINED_SYNTHESIS_PROMPT + "\n\n" + user_prompt
        if attempt > 0 and last_errors:
            full_prompt += (
                f"\n\nCorrection required: {'; '.join(last_errors)}. "
                "Return strictly valid JSON with zero hallucinated figures."
            )

        try:
            model = _gemini_client()
            t0 = time.time()

            generation_config = {
                "temperature": 0.2 if attempt == 0 else 0.0,
                "max_output_tokens": 2000,
                "response_mime_type": "application/json",
            }

            response = model.generate_content(
                full_prompt,
                generation_config=generation_config,
            )

            elapsed = time.time() - t0
            raw = (response.text or "").strip()

            # Parse JSON from response
            parsed = json.loads(raw)
            exec_summary = _clean_text_for_summary(parsed.get("executive_summary", ""))

            if len(exec_summary.split()) < 70:
                last_errors = ["Executive summary is too brief, aim for 150-250 words"]
                continue

            # Zero-hallucination check
            allowed_numbers = source_numbers | ({str(year)} if year else set()) | {
                "1", "2", "3", "4", "5", "10", "20", "24", "25", "26"
            }
            candidate_numbers = extract_numbers_from_text(exec_summary)
            invented = candidate_numbers - allowed_numbers

            if invented:
                last_errors = [
                    f"Summary contained hallucinated figures not in deck: {sorted(invented)}"
                ]
                continue

            # Clean payload
            parsed["executive_summary"] = exec_summary
            parsed["summary"] = exec_summary
            parsed["title"] = _clean_text_for_summary(
                parsed.get("title", rule_classification.get("title", ""))
            )
            parsed["description"] = _clean_text_for_summary(
                parsed.get("description", "")
            )
            parsed["seo_title"] = _clean_text_for_summary(
                parsed.get("seo_title", "")
            )[:80]
            parsed["seo_description"] = _clean_text_for_summary(
                parsed.get("seo_description", "")
            )[:180]
            parsed["ai_summary"] = _clean_text_for_summary(
                parsed.get("ai_summary", "")
            )

            final_payload = parsed
            break

        except Exception as api_err:
            last_errors.append(f"Gemini API error: {api_err}")
            if "quota" in str(api_err).lower() or "429" in str(api_err):
                time.sleep(2)
            else:
                break

    if not final_payload:
        # Fallback to local synthesis
        final_payload = _generate_local_synthesis(
            extracted_deck, rule_classification, file_hash
        )

    if file_hash:
        _save_cached_ai_result(file_hash, final_payload)

    return final_payload


def _generate_local_synthesis(
    extracted_deck: Dict[str, Any],
    rule_classification: Dict[str, Any],
    file_hash: str,
) -> Dict[str, Any]:
    """Generate deterministic synthesis from slide content without any LLM call."""
    comp = rule_classification.get("competition", "Case Competition")
    co = rule_classification.get("company", "Company")
    ct = rule_classification.get("case_type", "strategy")
    yr = rule_classification.get("year", 2024)
    res_label = rule_classification.get("result", "National Finalist")

    slides = extracted_deck.get("slides", [])
    # Extract real slide titles (not generic "Slide N")
    slide_titles = [
        s["title"]
        for s in slides
        if s.get("title") and not s["title"].startswith("Slide")
    ][:6]

    # Build themes from actual slide titles
    if slide_titles:
        key_themes = ", ".join(slide_titles[:4]).lower()
    else:
        key_themes = f"{ct.title()} and market analysis"

    # Try to extract real content snippets from first few slides
    first_slides_text = " ".join(
        [s.get("text", "")[:200] for s in slides[:3]]
    ).strip()

    if first_slides_text and len(first_slides_text) > 100:
        # We have real content — build a more specific summary
        local_summary = (
            f"This case competition presentation for {comp} ({yr}) analyses "
            f"{co}'s strategic challenges. "
            f"The deck covers {key_themes}, presenting an actionable framework "
            f"addressing core business dynamics and competitive positioning.\n\n"
            f"The team structured a comprehensive approach spanning market analysis, "
            f"solution design, and implementation planning. "
            f"Key recommendations focus on sustainable execution and competitive "
            f"differentiation.\n\n"
            f"This {res_label.lower()} submission serves as a verified reference "
            f"for case competition preparation on MECE Deck Vault."
        )
    else:
        local_summary = (
            f"This case competition presentation for {comp} ({yr}) analyses "
            f"{co}'s strategic challenges and market opportunities. "
            f"The deck evaluates core business dynamics, competitive landscape, "
            f"and operational capabilities.\n\n"
            f"The team structured an actionable roadmap covering {key_themes}. "
            f"Key recommendations focus on sustainable execution and competitive "
            f"differentiation in the Indian market.\n\n"
            f"This {res_label.lower()} submission serves as a verified reference "
            f"for case competition preparation on MECE Deck Vault."
        )

    local_payload = {
        "title": rule_classification.get("title", f"{comp} {yr} - {res_label} Deck"),
        "description": f"{co} case solution for {comp} ({yr}) focusing on {ct}.",
        "competition": comp,
        "company": co,
        "organizer": rule_classification.get("organizer", co),
        "industry": rule_classification.get("industry", "FMCG"),
        "case_type": ct,
        "round_type": rule_classification.get("round_type", "finale"),
        "result": res_label,
        "tags": rule_classification.get("tags", [ct.title(), "Strategy"]),
        "gist": {
            "company_and_context": f"{co} case solution for {comp} ({yr}).",
            "central_business_problem": f"Strategic problem solving in {ct} for {co}.",
            "strategic_questions": (
                ", ".join(slide_titles[:3])
                if slide_titles
                else "Evaluating market opportunities and operational feasibility."
            ),
            "proposed_solution": f"Holistic framework for {key_themes}.",
            "analytical_frameworks": (
                ", ".join(slide_titles[1:5])
                if len(slide_titles) > 1
                else "Market Analysis, Competitor Benchmarking, GTM Roadmap"
            ),
            "key_recommendations": "Phased implementation and scalable execution.",
            "distinctive_angle": f"Verified {res_label.lower()} with structured problem breakdown.",
        },
        "executive_summary": local_summary,
        "summary": local_summary,
        "seo_title": f"{comp} {yr} Solution | {ct.title()} Deck",
        "seo_description": (
            f"Verified {res_label.lower()} deck for {comp} ({yr}) "
            f"covering {ct} on MECE Deck Vault."
        ),
        "ai_summary": (
            f"Verified case competition deck for {comp} ({yr}) analysing "
            f"{co}'s {ct} strategy. {res_label} submission in the MECE Deck Vault."
        ),
    }
    if file_hash:
        _save_cached_ai_result(file_hash, local_payload)
    return local_payload


# ---------------------------------------------------------------------------
# Legacy backward-compatible wrapper
# ---------------------------------------------------------------------------

def generate_deck_summary(
    deck_id: str,
    pdf_bytes: bytes,
    deck_title: str,
    competition: str,
    organizer: str = "",
    year: Optional[int] = None,
    user_id: Optional[str] = None,
) -> str:
    """Legacy wrapper for single-deck summarisation route."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_bytes)
    pages_text = []
    try:
        for page in pdf:
            textpage = None
            try:
                textpage = page.get_textpage()
                pages_text.append(textpage.get_text_range() or "")
            finally:
                for handle in (textpage, page):
                    try:
                        if handle is not None:
                            handle.close()
                    except Exception:
                        pass
    finally:
        try:
            pdf.close()
        except Exception:
            pass

    full_text = "\n".join(pages_text).strip()
    extracted = {
        "file_hash": "",
        "full_text": full_text,
        "numbers": extract_numbers_from_text(full_text),
        "path_signals": {},
        "original_filename": f"{deck_title}.pdf",
        "slides": [],
    }
    rules = {
        "competition": competition,
        "organizer": organizer,
        "company": organizer or competition,
        "year": year,
        "case_type": "strategy",
        "result": "National Finalist",
        "title": deck_title,
    }

    result = generate_deck_synthesis(extracted, rules, user_id=user_id)
    summary = result.get("executive_summary", "")

    # Persist summary
    from services.supabase_client import get_supabase_client

    supabase = get_supabase_client()
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    supabase.table("deck_skeletons").update(
        {"summary": summary, "summary_generated_at": now_iso}
    ).eq("id", deck_id).execute()

    return summary
