"""
Growth Agent — programmatic-SEO writer.

Turns a real, current, GD-worthy business headline (from news_headlines) into a
GENUINELY USEFUL public article that teaches an aspirant how to break the topic
down as a case / group discussion, and funnels them into practising on MECE.

Design rules (the anti-spam guardrails that make this 100x instead of a penalty):
  * GROUNDED: the factual anchor is a real headline the platform already fetched.
    The model may reason about method but must NOT invent statistics, deals, or
    numbers that aren't in the source.
  * TEACHES METHOD, doesn't give away the graded product: the value is "here's how
    to structure this", with a CTA to the interactive (paywalled) practice.
  * SELF-CRITIQUED: every draft is scored 0..100 by a second (cheap) model for
    usefulness / grounding / originality. Low scores are flagged; NOTHING is ever
    auto-published — an admin approves.
  * COST-TIERED: generation runs on the `seo_writer` provider feature (gpt-4o by
    default, toggleable to Groq in the admin), critique on `seo_critique`
    (Groq by default). Both are logged to ai_usage_log.

Nothing here raises to the client except through the route, which maps failures
to clean HTTP errors. The daily-budget kill switch is checked by the route.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from services.ai_providers import chat_with_fallback
from services.ai_usage import log_ai_usage

SITE = "https://mece.in"

_WRITER_SYSTEM = """You are MECE's Growth Editor. MECE is an Indian case-interview and group-discussion \
prep platform. You write a single, genuinely useful public article that helps an aspirant turn a REAL \
current business story into case / GD practice, and gently points them to practise on MECE.

HARD RULES
- GROUND every factual claim in the provided headline/source. Do NOT invent statistics, deal sizes, \
dates, or company decisions that are not in the source. When you reason beyond the source, frame it as \
method ("here's how you'd size this", "the structure to use"), not as fact.
- TEACH THE METHOD. The value is showing how to structure the problem (MECE issue trees, answer-first \
synthesis, sizing in Rs/crore), not dumping opinions. Indian-English register, money in Rs/crore.
- Do NOT give away a full graded answer — the point is to make them want to practise the interactive way.
- No hype, no fluff, no "in today's fast-paced world". No clickbait that the body doesn't deliver.
- Be concrete and genuinely helpful to someone with an interview in two weeks.

Return ONLY a JSON object with this exact shape:
{
  "title": "compelling, <= 62 chars, includes the topic; reads like a real article title",
  "meta_description": "<= 155 chars, specific, no clickbait",
  "dek": "one-line subtitle that states the payoff",
  "keywords": ["4-8 short search phrases an aspirant would type"],
  "content": {
    "intro": "1-2 short paragraphs setting up the story and why it's good practice",
    "why_it_matters": "1 short paragraph on the business stakes, grounded in the source",
    "framework": {"heading": "How to structure this in an interview", "steps": ["3-6 concrete steps"]},
    "sections": [{"heading": "...", "paragraphs": ["..."], "bullets": ["optional"]}],
    "takeaways": ["3-5 crisp, do-this lines"],
    "practice_prompt": "one specific guesstimate or mini-case the reader can try right now"
  }
}
Keep it tight and skimmable. Short lines beat paragraphs."""


_CRITIQUE_SYSTEM = """You are a strict SEO + editorial QA reviewer for a case-interview prep site. \
You are protecting the domain from thin, spammy, or ungrounded auto-generated content. Score the DRAFT \
0-100 on: usefulness to a real aspirant, grounding in the source (no invented facts), originality (not \
generic filler), correctness, and topical fit. Be harsh — a page that adds nothing should score < 40. \
Return ONLY JSON: {"score": int 0-100, "publishable": bool, "notes": "one or two sentences on the \
single biggest issue or why it's strong"}. publishable is true only if score >= 70 AND it invents no facts."""


def _slugify(title: str) -> str:
    s = (title or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:70].strip("-") or "insight"


def _unique_slug(supabase, base: str) -> str:
    slug = base
    for attempt in range(6):
        try:
            r = supabase.table("seo_pages").select("id").eq("slug", slug).limit(1).execute()
            if not (r.data or []):
                return slug
        except Exception:
            return slug  # if the check errors, use the base; DB unique index is the backstop
        slug = f"{base[:60]}-{int(time.time() * 1000) % 100000}"
    return slug


def list_candidate_headlines(supabase, limit: int = 12) -> List[Dict[str, Any]]:
    """Fresh, GD-worthy headlines not yet turned into an SEO page."""
    try:
        used = supabase.table("seo_pages").select("source_headline_id").execute()
        used_ids = {r["source_headline_id"] for r in (used.data or []) if r.get("source_headline_id")}
    except Exception:
        used_ids = set()
    try:
        r = (supabase.table("news_headlines")
             .select("id, title, description, category, source_name, source_url, "
                     "gd_worthiness_score, published_at")
             .order("gd_worthiness_score", desc=True)
             .order("published_at", desc=True)
             .limit(limit * 3).execute())
        rows = r.data or []
    except Exception:
        return []
    out = [h for h in rows if h.get("id") not in used_ids]
    return out[:limit]


def _pick_headline(supabase, headline_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if headline_id:
        try:
            r = supabase.table("news_headlines").select(
                "id, title, description, category, source_name, source_url, gd_worthiness_score, published_at"
            ).eq("id", headline_id).maybe_single().execute()
            return dict(r.data) if r.data else None
        except Exception:
            return None
    cands = list_candidate_headlines(supabase, limit=1)
    return cands[0] if cands else None


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty model response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # tolerate a fenced or padded object
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def generate_seo_page(
    supabase,
    user_id: Optional[str],
    *,
    headline_id: Optional[str] = None,
    topic: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate ONE grounded SEO draft, self-critique it, and insert it as a
    draft in seo_pages. Returns the inserted row (dict). Raises ValueError with a
    clean message on unrecoverable problems."""
    headline = _pick_headline(supabase, headline_id)
    if headline is None and not topic:
        raise ValueError("No fresh GD-worthy headline available to write about. "
                         "Refresh the news pipeline, or pass an explicit topic.")

    src_title = (headline or {}).get("title") or (topic or "")
    src_desc = (headline or {}).get("description") or ""
    src_name = (headline or {}).get("source_name") or ""
    src_url = (headline or {}).get("source_url") or ""
    category = (headline or {}).get("category") or ""

    user_prompt = (
        f"SOURCE HEADLINE: {src_title}\n"
        f"SOURCE SUMMARY: {src_desc}\n"
        f"SOURCE: {src_name}\n"
        f"CATEGORY: {category}\n"
        f"AUDIENCE: Indian MBA / placement aspirants preparing for consulting/finance/product case & GD rounds.\n"
        f"Write the article as specified. Ground every fact in the source above; reason about METHOD freely."
    )

    # --- generation (seo_writer feature) ---
    t0 = time.time()
    resp, model, provider = chat_with_fallback(
        "seo_writer",
        messages=[{"role": "system", "content": _WRITER_SYSTEM},
                  {"role": "user", "content": user_prompt}],
        response_format={"type": "json_object"},
        temperature=0.6,
        max_tokens=1800,
    )
    try:
        log_ai_usage(user_id=user_id, endpoint="/seo/generate", model=model,
                     response=resp, latency_ms=int((time.time() - t0) * 1000),
                     meta={"provider": provider, "stage": "write"})
    except Exception:
        pass

    data = _extract_json(resp.choices[0].message.content)
    title = (data.get("title") or src_title or "Untitled").strip()[:120]
    meta = (data.get("meta_description") or src_desc or title).strip()[:300]
    dek = (data.get("dek") or "").strip()[:300]
    content = data.get("content") if isinstance(data.get("content"), dict) else {}
    keywords = [str(k)[:60] for k in (data.get("keywords") or []) if isinstance(k, (str, int))][:8]

    # --- critique (seo_critique feature) ---
    score: Optional[int] = None
    notes = ""
    try:
        t1 = time.time()
        cresp, cmodel, cprov = chat_with_fallback(
            "seo_critique",
            messages=[{"role": "system", "content": _CRITIQUE_SYSTEM},
                      {"role": "user", "content": f"SOURCE: {src_title}\n\nDRAFT:\n{json.dumps({'title': title, 'meta': meta, 'content': content})[:6000]}"}],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=300,
        )
        try:
            log_ai_usage(user_id=user_id, endpoint="/seo/generate", model=cmodel,
                         response=cresp, latency_ms=int((time.time() - t1) * 1000),
                         meta={"provider": cprov, "stage": "critique"})
        except Exception:
            pass
        cjson = _extract_json(cresp.choices[0].message.content)
        raw = cjson.get("score")
        score = int(raw) if isinstance(raw, (int, float)) else None
        notes = str(cjson.get("notes") or "")[:600]
    except Exception:
        score, notes = None, "Critique step failed; review manually."

    source_refs: List[Dict[str, str]] = []
    if src_url:
        source_refs.append({"label": src_name or "Source", "url": src_url})

    row = {
        "slug": _unique_slug(supabase, _slugify(title)),
        "kind": "news_case",
        "title": title,
        "meta_description": meta,
        "dek": dek,
        "content": content,
        "source_refs": source_refs,
        "topic": src_title[:280],
        "keywords": keywords,
        "status": "draft",
        "quality_score": score,
        "quality_notes": notes,
        "model": model,
        "agent_meta": {"provider": provider, "category": category},
        "source_headline_id": (headline or {}).get("id"),
        "created_by": user_id,
    }
    try:
        ins = supabase.table("seo_pages").insert(row).execute()
        return (ins.data or [row])[0]
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Draft generated but could not be saved: {type(e).__name__}: {e}")
