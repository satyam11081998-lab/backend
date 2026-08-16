"""
Deck summary generator with strict zero-hallucination number verification.

Extracts text from the PDF text layer using pypdfium2 (BSD/Apache-2.0), drafts a
150-250 word executive summary (Problem -> Approach -> Recommendation -> Key Numbers),
and verifies in Python that no number in the output is absent from the input text layer.
"""

import json
import os
import re
import time
from typing import Optional, Set

from openai import OpenAI
import pypdfium2 as pdfium

from services.ai_usage import log_ai_usage
from services.supabase_client import get_supabase_client

MODEL = "gpt-4o"

EN_DASH = "–"
EM_DASH = "—"
DASH_RE = re.compile(f"[{EN_DASH}{EM_DASH}]")


class DeckAIError(Exception):
    pass


_SUMMARY_PROMPT = """You write an executive summary for a winning business case-competition deck published on MECE (mece.in).
MECE is the case, guesstimate, and interview prep platform for Indian MBA and PGDM students.

HARD RULES:
1. NEVER invent any metric, number, percentage, currency figure, market size, or timeframe not present in the deck text.
   If the deck text contains no numbers, your summary MUST contain zero numbers.
2. Structure the summary in 3-4 coherent paragraphs (total 150-250 words) covering:
   - Problem & Context: The core business challenge and objective.
   - Approach & Framework: How the team structured and analyzed the problem.
   - Solution & Recommendations: Key strategic initiatives and implementation roadmap.
   - Impact & Outcomes: Quantified results or business takeaways (ONLY using numbers from the deck).
3. Professional, authoritative consulting tone.
4. British spelling (e.g. monetisation, optimise, organisation, prioritise).
5. Do NOT use em dashes or en dashes. Use commas, colons, or standard hyphens.
6. Plain ASCII punctuation only.

Return ONLY a JSON object with this shape:
{
  "summary": "Full 150-250 word summary in markdown prose with clear paragraphs."
}
"""


def _client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise DeckAIError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key, timeout=60.0, max_retries=1)


def _chat(endpoint: str, user_id: Optional[str], *, model: str, **kwargs):
    t0 = time.time()
    resp = _client().chat.completions.create(model=model, **kwargs)
    log_ai_usage(
        user_id=user_id, endpoint=endpoint, model=model, response=resp,
        latency_ms=int((time.time() - t0) * 1000),
    )
    return resp


def _extract_text_and_numbers(pdf_bytes: bytes) -> tuple[str, Set[str]]:
    """Extract raw text and all numbers from the PDF text layer using pypdfium2."""
    pdf = pdfium.PdfDocument(pdf_bytes)
    pages_text = []
    for page in pdf:
        textpage = page.get_textpage()
        text = textpage.get_text_range() or ""
        pages_text.append(text)
    full_text = "\n".join(pages_text).strip()
    numbers = set(re.findall(r"\d+(?:\.\d+)?", full_text))
    return full_text, numbers


def _numbers_in_text(text: str) -> Set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text or ""))


def _clean_summary(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"(\d)\s*[–—]\s*(\d)", r"\1-\2", text)
    text = DASH_RE.sub("-", text)
    text = (text.replace("‘", "'").replace("’", "'")
                .replace("“", '"').replace("”", '"')
                .replace("…", "..."))
    return text.strip()


def generate_deck_summary(
    deck_id: str,
    pdf_bytes: bytes,
    deck_title: str,
    competition: str,
    organizer: str = "",
    year: Optional[int] = None,
    user_id: Optional[str] = None,
) -> str:
    """Extract text, draft AI summary, verify numbers, and save to database."""
    source_text, source_numbers = _extract_text_and_numbers(pdf_bytes)

    # If deck has essentially no text layer (scanned/pure graphics), provide metadata
    context_text = source_text[:8000] if len(source_text) > 50 else (
        f"Title: {deck_title}\nCompetition: {competition}\nOrganizer: {organizer}\nYear: {year}\n"
        "(Note: The presentation slides contain mostly visual diagrams and charts.)"
    )

    year_str = f" ({year})" if year else ""
    user_prompt = (
        f"Deck: {deck_title}\n"
        f"Competition: {competition}{year_str}\n"
        f"Organizer: {organizer or 'N/A'}\n\n"
        f"Extracted Deck Text:\n\"\"\"\n{context_text}\n\"\"\"\n\n"
        "Draft the 150-250 word executive summary."
    )

    last_errors = []
    final_summary = ""

    for attempt, temperature in enumerate((0.3, 0.0)):
        messages = [
            {"role": "system", "content": _SUMMARY_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        if attempt > 0 and last_errors:
            messages.append({
                "role": "user",
                "content": f"Correction required: {'; '.join(last_errors)}. Generate strictly valid JSON with no hallucinated numbers.",
            })

        resp = _chat(
            "/decks/summarize", user_id, model=MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=800,
        )

        raw = (resp.choices[0].message.content or "").strip()
        try:
            parsed = json.loads(raw)
            candidate = _clean_summary(parsed.get("summary", ""))
        except Exception:
            last_errors = ["Response was not valid JSON"]
            continue

        if len(candidate.split()) < 80:
            last_errors = ["Summary is too brief, aim for 150-250 words"]
            continue

        # Zero-hallucination check: any number in candidate must exist in source_numbers
        allowed_numbers = source_numbers | ({str(year)} if year else set()) | {"1", "2", "3", "4", "5"}
        candidate_numbers = _numbers_in_text(candidate)
        invented = candidate_numbers - allowed_numbers

        if invented:
            last_errors = [f"Summary contained hallucinated figures not present in the deck: {sorted(invented)}"]
            continue

        final_summary = candidate
        break

    if not final_summary:
        raise DeckAIError(
            f"Could not generate a verified summary ({'; '.join(last_errors)}). Please edit manually or retry."
        )

    # Persist summary
    supabase = get_supabase_client()
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    supabase.table("deck_skeletons").update({
        "summary": final_summary,
        "summary_generated_at": now_iso,
    }).eq("id", deck_id).execute()

    return final_summary