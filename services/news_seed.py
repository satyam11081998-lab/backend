"""
Pick today's news story to build the daily guesstimate from.

Why this is not just "reuse the star headline": `headline_classifier` scores
GD-worthiness, and GD-worthiness is close to the OPPOSITE of what a guesstimate
needs. Its prompt rewards a headline with "TWO defensible sides" and scores
product launches at zero. A guesstimate needs none of that — it needs a
COUNTABLE quantity:

    "Delhi's AQI crosses 450"            → how many air purifiers sell this month?  ✓
    "RBI holds the repo rate at 6.5%"    → nothing to count                         ✗
    "Govt clears ₹12,000 cr metro line"  → the number is already in the headline    ✗

So this module scores the same stored headlines on a different axis, in ONE cheap
model call per market per day, through the metered `news_classify` lane.

Every failure path returns None, and the caller then generates exactly as it does
today. A missing key, an empty news table, a rate limit, a malformed response —
none of them may leave the daily surface without a guesstimate.
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from services.ai_providers import chat_with_fallback
from services.ai_usage import log_ai_usage
from services.model_json import parse_model_json
from services.supabase_client import get_supabase_client

# How far back a story can be and still feel like "today's news".
LOOKBACK_HOURS = 48
# Candidates sent to the model. One call, small prompt.
MAX_CANDIDATES = 12
# Below this, no headline is worth seeding with — better a good invented
# guesstimate than a forced one about the budget session.
MIN_SEED_SCORE = 55


SEED_SYSTEM_PROMPT = """You pick news stories that make good GUESSTIMATE questions for MBA case-interview practice.

A guesstimate asks the candidate to estimate a COUNTABLE quantity from scratch, using public anchors (population, households, income, ownership rates) and a chain of assumptions. It is not a debate and not an opinion.

Score each headline 0-100 on how well it seeds such a question:

90-100 — the story immediately suggests a specific countable thing whose size nobody knows, and which a smart person could estimate in ten minutes.
  "Delhi air quality hits severe for the fifth day" -> how many air purifiers will Delhi households buy this winter?
  "Cab aggregators roll out bike taxis in 12 cities" -> how many bike-taxi rides happen in Bengaluru in a day?
  "Monsoon arrives two weeks late in Maharashtra" -> how many litres of bottled water does Mumbai sell in a late-monsoon week?

50-80 — a countable quantity is there but needs a step of interpretation, or the market is narrow.

0-30 — score LOW, and be strict about this:
  - Macro policy, rates, inflation, indices, budgets, GDP: there is nothing a candidate can count.
  - The headline already states the quantity (a deal size, a revenue figure, a user count). If the number is given, there is nothing to estimate.
  - Pure opinion, debate, politics, diplomacy, awards, appointments, obituaries.
  - Sport, entertainment, crime, accidents.
  - Stock tips, "top 5 picks", generic PR launches.

For the single best headline, also write `quantity`: the one countable thing to estimate, phrased as a question of 12 words or fewer, in the market's own currency and geography.

Return ONLY JSON:
{"scored": [{"index": 0, "score": 83, "quantity": "How many air purifiers will Delhi households buy this winter?"}, {"index": 1, "score": 10, "quantity": null}]}
Every input index must appear exactly once. `quantity` may be null for anything you scored under 50."""


def _recent_stored_headlines(lookback_hours: int = LOOKBACK_HOURS) -> List[Dict[str, Any]]:
    """India candidates: whatever the 6 AM news pipeline already stored."""
    try:
        supabase = get_supabase_client()
        since = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).isoformat()
        res = (
            supabase.table("news_headlines")
            .select("title, description, source_name, source_url, published_at")
            .gte("published_at", since)
            .order("published_at", desc=True)
            .limit(40)
            .execute()
        )
        return [r for r in (res.data or []) if r.get("title")]
    except Exception as exc:  # noqa: BLE001 — a missing table must not break the cron
        print(f"news_seed: could not read stored headlines ({type(exc).__name__}: {exc})")
        return []


def _live_us_headlines() -> List[Dict[str, Any]]:
    """
    US candidates, fetched live and NOT stored.

    `news_headlines` has no market column and feeds the India-only GD briefs, so
    writing US rows into it would quietly pollute that product. One extra GNews
    call a day is the cheaper trade.
    """
    key = os.environ.get("GNEWS_API_KEY", "").strip()
    if not key:
        return []
    try:
        from services.news_fetcher import fetch_from_gnews

        raw = fetch_from_gnews(key, max_results=30, page=1, country="us")
        return [
            {
                "title": h["title"],
                "description": h.get("description"),
                "source_name": h.get("source_name"),
                "source_url": h.get("source_url"),
                "published_at": h.get("published_at"),
            }
            for h in raw
            if h.get("title")
        ]
    except Exception as exc:  # noqa: BLE001
        print(f"news_seed: US live fetch failed ({type(exc).__name__}: {exc})")
        return []


def _dedupe(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        k = " ".join(str(r.get("title") or "").lower().split())
        if k and k not in seen:
            seen.add(k)
            out.append(r)
    return out


def _has_own_number(title: str) -> bool:
    """
    A headline that already states a big quantity leaves nothing to estimate.
    Cheap pre-filter so the model's budget is spent on real candidates.
    """
    t = title.lower()
    units = ("crore", "lakh", "billion", "million", "trillion", "bn", "mn", "%", " per cent", "percent")
    return any(u in t for u in units)


def pick_guesstimate_seed(market: str = "IN") -> Optional[Dict[str, Any]]:
    """
    Today's seed story for `market`, or None.

    Returns {"title", "description", "source_name", "source_url", "published_at",
             "quantity", "score"} — `quantity` is the countable thing the model
    thinks the guesstimate should ask for, which the generator uses as a hint
    rather than as finished copy.

    Never raises.
    """
    try:
        rows = _live_us_headlines() if market == "US" else _recent_stored_headlines()
        rows = _dedupe(rows)
        if not rows:
            return None

        # Prefer headlines that do not already carry their own figure, but fall
        # back to the full set rather than returning nothing.
        preferred = [r for r in rows if not _has_own_number(str(r.get("title") or ""))]
        candidates = (preferred or rows)[:MAX_CANDIDATES]

        if not any(os.environ.get(k, "").strip() for k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY")):
            return None

        payload = [
            {
                "index": i,
                "title": c["title"],
                "description": (c.get("description") or "")[:300],
            }
            for i, c in enumerate(candidates)
        ]
        user_msg = (
            f"Market: {'United States' if market == 'US' else 'India'}.\n"
            f"Score these {len(payload)} headlines for how well each seeds a guesstimate, "
            f"and give the quantity to estimate for the best one.\n\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

        t0 = time.time()
        response, used_model, _prov = chat_with_fallback(
            "news_classify",
            messages=[
                {"role": "system", "content": SEED_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=1200,
        )
        log_ai_usage(
            endpoint="/cron/schedule-daily",
            model=used_model,
            response=response,
            latency_ms=int((time.time() - t0) * 1000),
            meta={"step": "guesstimate_seed", "market": market, "candidates": len(payload)},
        )

        raw = response.choices[0].message.content
        if not raw:
            return None
        scored = parse_model_json(raw).get("scored")
        if not isinstance(scored, list) or not scored:
            return None

        best: Optional[Dict[str, Any]] = None
        for item in scored:
            try:
                idx = int(item.get("index"))
                score = int(item.get("score", 0))
            except (TypeError, ValueError):
                continue
            if not (0 <= idx < len(candidates)):
                continue
            if best is None or score > best["score"]:
                q = item.get("quantity")
                best = {
                    "score": max(0, min(100, score)),
                    "idx": idx,
                    "quantity": q.strip() if isinstance(q, str) and q.strip() else None,
                }

        if not best or best["score"] < MIN_SEED_SCORE:
            return None

        chosen = candidates[best["idx"]]
        return {
            "title": chosen["title"],
            "description": chosen.get("description"),
            "source_name": chosen.get("source_name"),
            "source_url": chosen.get("source_url"),
            "published_at": chosen.get("published_at"),
            "quantity": best["quantity"],
            "score": best["score"],
        }
    except Exception as exc:  # noqa: BLE001 — the daily surface matters more than the seed
        print(f"news_seed: seed selection failed ({type(exc).__name__}: {exc})")
        return None
