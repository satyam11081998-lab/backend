"""
Growth Agent — the daily blog.

One article a day on /insights for the people MECE serves: MBA students preparing for
placements and MBA aspirants preparing for GD / PI / WAT rounds. The bar is a good business
explainer (Mint, ET Prime, a McKinsey "Insight"): a current, non-controversial business
story, real and sourced facts and figures, clear analysis, then what it means for a GD, an
interview or a case — with a link to practise on MECE.

Pipeline (each step degrades instead of raising; nothing half-done is ever published):

  1. TOPIC    today's news_headlines (last 72 h) in business / economy / tech / jobs / policy,
              minus anything political, tragic, criminal or a stock tip, minus stories already
              written and topics too close to the last 45 days of posts, ranked by GD-worthiness,
              MBA relevance, freshness and variety. A cheap "editor" model picks the broadest of
              the top few and frames the angle one level above the headline. Evergreen topics
              (EVERGREEN) back this up, rotated so a later attempt tries different ones.
  2. RESEARCH Gemini with Google Search grounding, asked for 12-16 dated facts with their
              publisher. A fact is kept with a link when ANY of these ties it to a page the
              search actually read: the grounding "supports" (normalised text match), the
              publisher it names (matched to the result's domain/title), or its numbers found on
              that page's text. Facts the search returned but that can't be linked are kept as
              "attributed" (named source, no link) only alongside enough linked ones. The news
              article itself is a linked fact. A second, context-focused search runs when the
              first is thin; research models fall back (DAILY_BLOG_RESEARCH_MODEL -> GEMINI_MODEL
              -> gemini-2.5-flash -> gemini-2.0-flash). A topic needs >= 3 linked facts.
  3. WRITE    1,000-1,400 words in a house style written for people, from those facts only, each
              cited inline ([F2]). DAILY_BLOG_WRITER_MODEL (an OpenAI model name) if set, else the
              `seo_writer` feature (gpt-4o by default).
  4. CHECKS   deterministic: every number traceable to a fact, citations valid, length, parts
              present, banned machine phrases, em dashes, controversy terms, title/meta length.
              Up to two repair rounds with the exact problems.
  5. CRITIC   `seo_critique` scores 0-100 against a publication-quality rubric.
  6. REVIEW   saved as a draft (kind='daily') and sent to the admin's Telegram chat in full.
              Replying "publish" there publishes it (services/growth/telegram_review.py).
              DAILY_BLOG_AUTOPUBLISH=1 publishes without review when every check passes and the
              score >= DAILY_BLOG_MIN_SCORE.

Persistence ("keep going until there is an article"): one run tries topics until one works or
DAILY_BLOG_TIME_BUDGET_S runs out; the cron fires every 30 minutes through the morning and does
nothing once today's post exists; the reviewer can ask for "another" at any time.

Switches (env, all OFF by default):
  DAILY_BLOG_ENABLED      the cron route does nothing until this is on
  DAILY_BLOG_AUTOPUBLISH  off = every post waits for "publish" on Telegram (or Admin -> Growth)
  DAILY_BLOG_MIN_SCORE    critic score needed to auto-publish (default 80)
"""

from __future__ import annotations

import html as _html
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

IST = timezone(timedelta(hours=5, minutes=30))
KIND = "daily"
FORMAT = "daily-2"
SITE = "https://mece.in"

_run_lock = threading.Lock()


def _flag(name: str, default: bool = False) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(os.getenv(name, str(default)))))
    except ValueError:
        return default


def research_models() -> List[str]:
    out: List[str] = []
    for m in (os.getenv("DAILY_BLOG_RESEARCH_MODEL"), os.getenv("GEMINI_MODEL"), "gemini-2.5-flash", "gemini-2.0-flash"):
        m = (m or "").strip()
        if m and m not in out:
            out.append(m)
    return out


def config() -> Dict[str, Any]:
    return {
        "enabled": _flag("DAILY_BLOG_ENABLED"),
        "autopublish": _flag("DAILY_BLOG_AUTOPUBLISH"),
        "min_score": _int_env("DAILY_BLOG_MIN_SCORE", 80, 0, 100),
        "min_words": 850,
        "max_words": 1700,
        "min_linked_facts": 3,
        "time_budget_s": _int_env("DAILY_BLOG_TIME_BUDGET_S", 480, 60, 1500),
        "max_topics": _int_env("DAILY_BLOG_MAX_TOPICS", 8, 1, 20),
        "research_available": research_available(),
        "research_models": research_models(),
        "writer_model": (os.getenv("DAILY_BLOG_WRITER_MODEL") or "").strip() or "seo_writer (gpt-4o by default)",
    }


# =============================================================================
# 1. Topic selection
# =============================================================================
ALLOWED_CATEGORIES = {"business", "macro", "micro", "tech", "jobs", "policy"}

# Political, divisive, tragic or criminal: never the hook for a prep-site article.
_BLOCK = [
    r"elections?", r"poll(s|ing)?(?! ?(of|on) (consumers|customers))", r"bjp", r"congress party", r"aap", r"tmc",
    r"opposition part(y|ies)", r"protests?", r"riots?", r"communal", r"religio\w*", r"temples?", r"mosques?", r"churche?s?",
    r"castes?", r"hindus?", r"muslims?", r"(?<!price )(?<!talent )(?<!bidding )(?<!discount )wars?", r"terror\w*",
    r"militants?", r"missiles?", r"arm(y|ies)", r"border clash\w*", r"ceasefire", r"killed", r"dead", r"deaths?",
    r"murder\w*", r"rape\w*", r"assault\w*", r"arrest(ed|s)?", r"ed raids?", r"raids?", r"money laundering",
    r"scams?", r"fraud\w*", r"bail", r"chargesheet\w*", r"lawsuits?", r"sued", r"verdicts?", r"suicides?",
    r"accidents?", r"crash(ed|es)?", r"stampede", r"flood(s|ing)?", r"earthquakes?", r"cyclones?", r"pakistan",
    r"china border", r"tariff war", r"prime ministers?", r"modi", r"rahul gandhi", r"chief ministers?", r"lok sabha",
    r"rajya sabha", r"parliament\w*", r"controvers\w*", r"boycott\w*", r"hate|hatred", r"scandal\w*",
]
_BLOCK_RE = re.compile(r"\b(" + "|".join(_BLOCK) + r")\b", re.I)

# Market noise: one stock's move, a tip, an IPO grey market — too narrow to teach anything.
_NOISE = [r"shares? (jump|fall|surge|slump|rise|drop|rall)\w*", r"stock(s)? to buy", r"buy or sell", r"target price",
          r"multibagger", r"\bgmp\b", r"sensex today", r"nifty today", r"market live", r"stocks? in focus",
          r"q[1-4] results? live", r"dividend", r"record date", r"bonus issue", r"stock split", r"upper circuit",
          r"lower circuit", r"\bf&o\b", r"intraday"]
_NOISE_RE = re.compile(r"(" + "|".join(_NOISE) + r")", re.I)

# What an MBA student or aspirant studies and gets asked about. A headline touching more of these
# is a better lesson; the domain also drives day-to-day variety.
MBA_DOMAINS: Dict[str, List[str]] = {
    "strategy": ["strategy", "market share", "competition", "competitor", "acquisition", "acquire", "merger", "expansion",
                 "enter", "entry", "exit", "diversif", "partnership", "stake", "consolidat", "moat", "business model"],
    "marketing": ["brand", "consumer", "customer", "pricing", "price", "launch", "advertis", "campaign", "fmcg", "retail",
                  "d2c", "premium", "rural", "demand", "festive", "loyalty", "distribution"],
    "finance": ["profit", "loss", "revenue", "margin", "valuation", "funding", "raises", "ipo", "debt", "investment",
                "capex", "earnings", "fund", "bank", "credit", "loan", "insurance", "fintech", "upi", "payments"],
    "operations": ["supply chain", "logistics", "manufactur", "factory", "capacity", "plant", "warehouse", "delivery",
                   "quick commerce", "inventory", "production", "ev", "semiconductor", "airline", "railway", "port"],
    "tech_product": ["ai", "artificial intelligence", "app", "platform", "startup", "saas", "software", "digital", "data",
                     "cloud", "telecom", "5g", "subscription", "users", "product"],
    "economy": ["gdp", "inflation", "repo", "rbi", "interest rate", "growth", "exports", "imports", "rupee", "fdi",
                "budget", "gst", "tax", "consumption", "economy", "manufacturing pmi", "monsoon"],
    "careers": ["hiring", "jobs", "layoff", "salary", "talent", "workforce", "campus", "placement", "skills", "employees",
                "work from office", "gig"],
}

_STOP = set("""a an the of to in on for and or but with by from at as is are was were be been it its this that these those
into over under after before about than then so such not no yes new says said will would can could may might also more most
how why what when who whom which india indian crore lakh rs per cent percent year years month months week day days today""".split())

# Broad, always-relevant topics (business explainers and classic GD/WAT themes, none partisan).
# Research still pulls current figures, so an evergreen post is dated and sourced like any other.
EVERGREEN: List[Dict[str, str]] = [
    {"topic": "How quick commerce makes money in India", "domain": "operations"},
    {"topic": "Why airlines in India find it so hard to stay profitable", "domain": "operations"},
    {"topic": "The economics of IPL franchises", "domain": "finance"},
    {"topic": "How UPI changed payments in India, and how it makes money", "domain": "finance"},
    {"topic": "Why FMCG companies are betting on rural India", "domain": "marketing"},
    {"topic": "The unit economics of food delivery apps", "domain": "strategy"},
    {"topic": "How India's EV two-wheeler market is shaping up", "domain": "operations"},
    {"topic": "Why D2C brands move to offline retail", "domain": "marketing"},
    {"topic": "How the RBI's repo rate reaches your EMI", "domain": "economy"},
    {"topic": "The business of OTT streaming in India", "domain": "tech_product"},
    {"topic": "How India's semiconductor push works", "domain": "operations"},
    {"topic": "Why premium products are growing faster than mass ones in India", "domain": "marketing"},
    {"topic": "How GCCs (global capability centres) are changing Indian jobs", "domain": "careers"},
    {"topic": "The economics of Indian Railways", "domain": "economy"},
    {"topic": "How quick-service restaurants grow in India", "domain": "strategy"},
    {"topic": "How credit cards and BNPL compete for India's young spenders", "domain": "finance"},
    {"topic": "Why India's telecom companies keep raising tariffs", "domain": "strategy"},
    {"topic": "How the festive season shapes Indian retail", "domain": "marketing"},
    {"topic": "The business of India's private hospitals", "domain": "operations"},
    {"topic": "How edtech in India reset after the pandemic boom", "domain": "tech_product"},
    {"topic": "How India's coffee chains compete", "domain": "strategy"},
    {"topic": "Why companies are moving manufacturing to India", "domain": "economy"},
    {"topic": "How the gig economy works for delivery partners and platforms", "domain": "careers"},
    {"topic": "The economics of India's renewable energy push", "domain": "economy"},
    {"topic": "Will AI take entry-level jobs in Indian IT services?", "domain": "careers"},
    {"topic": "Why India's women workforce participation matters for growth", "domain": "economy"},
    {"topic": "How India's startup funding cycle works, boom to winter", "domain": "finance"},
    {"topic": "What a four-day work week would mean for Indian companies", "domain": "careers"},
    {"topic": "How India's data centre boom is being built", "domain": "tech_product"},
    {"topic": "Why Indian households are moving savings into mutual funds", "domain": "finance"},
    {"topic": "How the sports business in India is growing beyond cricket", "domain": "marketing"},
    {"topic": "What makes a unicorn: India's startup economy in numbers", "domain": "strategy"},
]


def _norm_words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) > 2 and w not in _STOP]


def _tokens(text: str) -> set:
    return set(_norm_words(text))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def is_blocked(text: str) -> bool:
    return bool(_BLOCK_RE.search(text or ""))


# Short words must match whole ("ev" is not "every", "app" is not "apparel"); longer ones match as
# prefixes ("acquire" covers "acquires", "manufactur" covers "manufacturing").
_DOMAIN_RES = {dom: [re.compile(r"(?<![a-z0-9])" + re.escape(w) + (r"(?![a-z0-9])" if len(w) <= 3 else ""))
                     for w in words] for dom, words in MBA_DOMAINS.items()}


def domain_hits(text: str) -> Dict[str, int]:
    t = (text or "").lower()
    out = {}
    for dom, pats in _DOMAIN_RES.items():
        n = sum(1 for p in pats if p.search(t))
        if n:
            out[dom] = n
    return out


def _parse_ts(v: Any) -> Optional[datetime]:
    if not v:
        return None
    try:
        s = str(v).replace("Z", "+00:00")
        d = datetime.fromisoformat(s)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def score_headline(h: Dict[str, Any], recent: List[Dict[str, Any]], now: datetime) -> Tuple[float, List[str], str]:
    """(score, reasons, domain). score < 0 = not eligible."""
    title = h.get("title") or ""
    text = f"{title}. {h.get('description') or ''}"
    cat = (h.get("category") or "").lower()
    if cat not in ALLOWED_CATEGORIES:
        return -1, [f"category {cat or 'none'} is not a business topic"], ""
    if is_blocked(text):
        return -1, ["political, tragic, criminal or divisive"], ""
    reasons: List[str] = []
    gd = float(h.get("gd_worthiness_score") or 0)
    score = gd
    reasons.append(f"GD-worthiness {gd:g}")
    hits = domain_hits(text)
    domain = max(hits, key=hits.get) if hits else ""
    rel = min(sum(hits.values()), 6)
    score += rel * 0.8
    if hits:
        reasons.append("MBA relevance: " + ", ".join(sorted(hits)))
    else:
        score -= 3
        reasons.append("no MBA domain")
    if _NOISE_RE.search(text):
        score -= 5
        reasons.append("market noise (one stock / a tip)")
    if h.get("is_star"):
        score += 1
        reasons.append("top story")
    pub = _parse_ts(h.get("published_at"))
    if pub:
        age_h = max(0.0, (now - pub).total_seconds() / 3600)
        score += max(0.0, 2.0 - age_h / 24)  # fresher is better, up to +2
    tt = _tokens(title)
    for p in recent:
        if _jaccard(tt, _tokens(f"{p.get('title') or ''} {p.get('topic') or ''}")) >= 0.3:
            return -1, ["too close to a recent post: " + (p.get("title") or "")[:60]], domain
    last_domains = [((p.get("agent_meta") or {}).get("domain") or "") for p in recent[:3]]
    if domain and last_domains.count(domain) >= 2:
        score -= 2.5
        reasons.append(f"{domain} two of the last three days")
    return round(score, 2), reasons, domain


def recent_posts(supabase, now: datetime, days: int = 45) -> List[Dict[str, Any]]:
    try:
        r = (supabase.table("seo_pages").select("id, title, topic, keywords, kind, status, agent_meta, created_at, "
                                                "source_headline_id")
             .gte("created_at", (now - timedelta(days=days)).isoformat())
             .order("created_at", desc=True).limit(200).execute())
        return list(r.data or [])
    except Exception:
        return []


def candidate_topics(supabase, now: Optional[datetime] = None, limit: int = 6) -> List[Dict[str, Any]]:
    """The best few fresh stories for today's post, with why they rank where they do."""
    now = now or datetime.now(timezone.utc)
    recent = recent_posts(supabase, now)
    used = {p.get("source_headline_id") for p in recent if p.get("source_headline_id")}
    try:
        r = (supabase.table("news_headlines")
             .select("id, title, description, category, source_name, source_url, gd_worthiness_score, published_at, "
                     "keywords, is_star")
             .gte("published_at", (now - timedelta(hours=72)).isoformat())
             .order("gd_worthiness_score", desc=True).limit(80).execute())
        rows = list(r.data or [])
    except Exception:
        rows = []
    out = []
    for h in rows:
        if h.get("id") in used:
            continue
        s, reasons, dom = score_headline(h, recent, now)
        if s < 0:
            continue
        out.append({"headline": h, "score": s, "reasons": reasons, "domain": dom})
    out.sort(key=lambda c: c["score"], reverse=True)
    return out[:limit]


def _slot(now: datetime) -> int:
    """Which scheduled attempt of the day this is (half-hour slots from 07:00 IST), so a later run tries other topics."""
    t = now.astimezone(IST)
    return max(0, (t.hour * 60 + t.minute - 420) // 30)


def evergreen_topics(supabase, now: datetime, n: int = 3) -> List[Dict[str, Any]]:
    recent = recent_posts(supabase, now, days=120)
    start = (now.astimezone(IST).timetuple().tm_yday * 3 + _slot(now) * n) % len(EVERGREEN)
    out = []
    for i in range(len(EVERGREEN)):
        e = EVERGREEN[(start + i) % len(EVERGREEN)]
        tt = _tokens(e["topic"])
        if all(_jaccard(tt, _tokens(f"{p.get('title') or ''} {p.get('topic') or ''}")) < 0.3 for p in recent):
            out.append({"headline": None, "score": 0, "reasons": ["evergreen"], "domain": e["domain"], "angle": e["topic"]})
        if len(out) >= n:
            break
    return out


def evergreen_topic(supabase, now: datetime) -> Dict[str, Any]:
    got = evergreen_topics(supabase, now, 1)
    if got:
        return got[0]
    e = EVERGREEN[0]
    return {"headline": None, "score": 0, "reasons": ["evergreen"], "domain": e["domain"], "angle": e["topic"]}


_EDITOR_SYSTEM = """You are the editor of MECE Insights, read by Indian MBA students preparing for placements and by \
MBA aspirants preparing for GD, PI and WAT rounds. From the candidate news stories, pick the ONE that makes the best \
article today: a business or economy question many readers would search for and could be asked about in a GD or an \
interview this season; broad enough to interest many, not a one-day stock move, never political or divisive.
Frame the ANGLE one level broader than the headline: the business question behind the story, as a reader would \
search it (headline "App X raises platform fee by Rs 2" -> angle "Why food delivery apps keep raising platform fees"). \
6-12 words, no clickbait.
Return ONLY JSON: {"order": [indices, best first], "angle": "<angle for the first>", "why": "one sentence"}"""


def editor_pick(cands: List[Dict[str, Any]], chat: Callable) -> List[Dict[str, Any]]:
    """Order the candidates (best first) and frame a broader angle for the first. Falls back to score order."""
    ordered = [dict(c, angle=(c["headline"] or {}).get("title") or "") for c in cands]
    if not cands:
        return ordered
    lines = [f"[{i}] {(c['headline'] or {}).get('title')} — {((c['headline'] or {}).get('description') or '')[:220]} "
             f"(category {(c['headline'] or {}).get('category')})" for i, c in enumerate(cands)]
    try:
        resp, _, _ = chat("seo_critique", messages=[{"role": "system", "content": _EDITOR_SYSTEM},
                                                     {"role": "user", "content": "\n".join(lines)}],
                          response_format={"type": "json_object"}, temperature=0.2, max_tokens=250)
        data = _extract_json(resp.choices[0].message.content)
        order = [int(i) for i in (data.get("order") or [data.get("pick", 0)]) if str(i).lstrip("-").isdigit()]
        order = [i for i in dict.fromkeys(order) if 0 <= i < len(cands)]
        order += [i for i in range(len(cands)) if i not in order]
        angle = str(data.get("angle") or "").strip()
        out = [ordered[i] for i in order]
        if angle and not is_blocked(angle) and len(angle) <= 110:
            out[0]["angle"] = angle
            out[0]["reasons"] = list(out[0]["reasons"]) + ["editor: " + str(data.get("why") or "")[:160]]
        return out
    except Exception:
        return ordered


# =============================================================================
# 2. Research (Gemini + Google Search grounding)
# =============================================================================
def research_available() -> bool:
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        return False
    try:
        import google.genai  # noqa: F401
        return True
    except Exception:
        return False


_RESEARCH_PROMPT = """You are researching an article for an Indian business publication read by MBA students. \
Topic: "{angle}".{hook}

Search the live web and collect 12 to 16 facts a reader could check:
- figures: market size, growth, revenue, profit, users, prices, market shares, costs, volumes;
- the key players and what each did, with dates;
- official data and statements (RBI, ministries, regulators, company filings, industry bodies), attributed;
- what changed recently and why.
Prefer primary sources and established business press (Economic Times, Mint, Business Standard, Reuters, \
Bloomberg, company filings, RBI, government data, industry reports). Prefer the most recent figures; give the \
period or date of each. No opinions, no forecasts stated as facts, nothing political.

Write each fact on its own line, in exactly this form (no bullets, no numbering, no bold):
FACT: <one sentence with the figure or decision, what it measures, and the period or date> | SOURCE: <publisher or organisation>
Nothing else."""

_CONTEXT_PROMPT = """Research the background to this business topic for an article read by Indian MBA students: \
"{angle}".{hook}

Search the live web for 8 to 12 more checkable facts, different from obvious headline numbers:
- how the business or market works (revenue model, costs, margins, scale);
- the main companies and their shares or sizes;
- the history: when it started, the turning points, with years;
- what regulators, companies or industry bodies have said or decided, attributed.
Prefer primary sources and established business press, with dates.

Write each fact on its own line, in exactly this form (no bullets, no numbering, no bold):
FACT: <one sentence> | SOURCE: <publisher or organisation>
Nothing else."""


def _gemini_grounded(prompt: str, model: str) -> Dict[str, Any]:
    """{text, chunks: [{uri, title, domain}], supports: [{text, chunks}], usage, model} — raises on failure."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    resp = client.models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())], temperature=0.2),
    )
    return extract_grounding(resp, model)


def extract_grounding(resp: Any, model: str = "") -> Dict[str, Any]:
    """The text, the pages the search read, and which sentences each page supports, from a
    google-genai GenerateContentResponse."""
    text = getattr(resp, "text", "") or ""
    chunks, supports = [], []
    cand = (getattr(resp, "candidates", None) or [None])[0]
    gm = getattr(cand, "grounding_metadata", None)
    for ch in (getattr(gm, "grounding_chunks", None) or []):
        web = getattr(ch, "web", None)
        chunks.append({"uri": getattr(web, "uri", "") or "", "title": getattr(web, "title", "") or "",
                       "domain": getattr(web, "domain", "") or ""} if web else {"uri": "", "title": "", "domain": ""})
    for s in (getattr(gm, "grounding_supports", None) or []):
        seg = getattr(s, "segment", None)
        supports.append({"text": getattr(seg, "text", "") or "", "chunks": list(getattr(s, "grounding_chunk_indices", None) or [])})
    um = getattr(resp, "usage_metadata", None)
    usage = {"prompt": getattr(um, "prompt_token_count", None), "completion": getattr(um, "candidates_token_count", None)}
    return {"text": text, "chunks": chunks, "supports": supports, "usage": usage, "model": model}


def grounded_with_fallback(prompt: str, grounded: Callable) -> Dict[str, Any]:
    """Try each research model in turn; returns the first answer that came back with search results."""
    errors = []
    last: Dict[str, Any] = {}
    for model in research_models():
        try:
            g = grounded(prompt, model)
        except TypeError:  # an injected single-argument fake (tests)
            g = grounded(prompt)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{model}: {type(e).__name__}: {str(e)[:120]}")
            continue
        last = g
        if g.get("chunks"):
            g["errors"] = errors
            return g
        errors.append(f"{model}: answered without searching")
    last = dict(last or {"text": "", "chunks": [], "supports": []})
    last["errors"] = errors
    return last


def _resolve_url(uri: str) -> str:
    """Grounding links are Google redirect URLs that expire: store where they point instead."""
    if not uri or "grounding-api-redirect" not in uri:
        return uri
    try:
        import requests
        r = requests.head(uri, allow_redirects=False, timeout=6)
        loc = r.headers.get("location") or r.headers.get("Location")
        if loc and loc.startswith("http"):
            return loc
        r = requests.get(uri, allow_redirects=True, timeout=8, stream=True)
        if r.url and "grounding-api-redirect" not in r.url:
            return r.url
    except Exception:
        pass
    return ""


def _fetch_text(url: str) -> str:
    """Plain text of a page (for checking a fact's numbers against it). '' on any failure."""
    if not url.startswith("http"):
        return ""
    try:
        import requests
        r = requests.get(url, timeout=7, headers={"User-Agent": "Mozilla/5.0 (compatible; MECE-Insights/1.0)"})
        if r.status_code >= 400:
            return ""
        body = r.text[:900_000]
        body = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", body)
        body = re.sub(r"(?s)<[^>]+>", " ", body)
        return _html.unescape(re.sub(r"\s+", " ", body))
    except Exception:
        return ""


def _domain(url: str) -> str:
    m = re.match(r"https?://([^/]+)", url or "")
    return (m.group(1) if m else "").lower().removeprefix("www.")


def _squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _norm_text(s: str) -> str:
    s = re.sub(r"[*_`#>]+", " ", (s or "").lower())
    s = re.sub(r"[^a-z0-9.%₹ ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# Words too common in publisher names to identify one ("Times", "India", "Business" …).
_GENERIC_SOURCE_WORDS = {"india", "indian", "times", "business", "news", "report", "reports", "annual", "data", "ministry",
                         "government", "govt", "department", "limited", "group", "global", "world", "economic",
                         "financial", "daily", "press", "research", "survey", "national", "bank", "official", "company",
                         "statement", "website", "media", "agency", "association", "council", "board", "reserve",
                         "statistics", "office", "union", "federation", "institute", "india's"}

_FACT_RE = re.compile(r"^\W*(?:\d{1,2}[.)]\s*)?\W*fact\W{0,3}\s*[:\-–]\s*(.+)$", re.I)
_SOURCE_SPLIT = re.compile(r"\s*[|•]\s*\**\s*source\s*\**\s*[:\-–]\s*", re.I)


def parse_fact_lines(text: str) -> List[Dict[str, str]]:
    """FACT lines, however the model dressed them ('* **FACT:** …', '1. FACT - …'), with the named source.
    If it ignored the format entirely, numbered or bulleted lines that carry a figure are taken instead."""
    out: List[Dict[str, str]] = []
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for ln in lines:
        m = _FACT_RE.match(ln.replace("**", "").replace("__", ""))
        if m:
            body = m.group(1).strip()
            parts = _SOURCE_SPLIT.split(body, maxsplit=1)
            fact = parts[0].strip().strip("*").strip()
            src = parts[1].strip().strip("*").strip() if len(parts) > 1 else ""
            src = re.split(r"\s*\|\s*", src)[0]
            if len(fact) >= 25:
                out.append({"text": fact, "source": src[:120]})
    if len(out) >= 3:
        return out
    have = {_norm_text(o["text"]) for o in out}
    for ln in lines:
        clean = re.sub(r"^\W*(?:\d{1,2}[.)])?\s*", "", ln.replace("**", "").replace("__", "")).strip()
        m = _FACT_RE.match(clean)
        clean = m.group(1).strip() if m else clean
        if len(clean) >= 40 and re.search(r"\d", clean) and not clean.lower().startswith(("here", "sure", "below")):
            parts = _SOURCE_SPLIT.split(clean, maxsplit=1)
            item = {"text": parts[0].strip().strip("*").strip(), "source": parts[1].strip()[:120] if len(parts) > 1 else ""}
            if _norm_text(item["text"]) not in have:
                have.add(_norm_text(item["text"]))
                out.append(item)
    return out


def _key_numbers(text: str) -> set:
    return {n for n in _numbers(text) if not _exempt(n)}


def _page_has_numbers(page: str, nums: set) -> bool:
    if not page or not nums:
        return False
    flat = page.replace(",", "")
    return all(re.search(r"(?<![\d.])" + re.escape(n) + r"(?![\d])", flat) for n in nums)


def link_facts(g: Dict[str, Any], resolve: Callable, fetch: Callable) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(linked, attributed). Linked facts carry the web pages that back them; attributed ones only a named source."""
    chunks = g.get("chunks") or []
    supports = [(s, _norm_text(s.get("text") or "")) for s in (g.get("supports") or [])]
    resolved: Dict[int, Dict[str, str]] = {}

    def source(i: int) -> Optional[Dict[str, str]]:
        if i in resolved:
            return resolved[i] or None
        ch = chunks[i] if 0 <= i < len(chunks) else {}
        url = resolve(ch.get("uri") or "")
        label = (ch.get("title") or ch.get("domain") or _domain(url) or "").strip()
        resolved[i] = {"url": url, "label": label, "domain": (ch.get("domain") or _domain(url) or label).lower()} \
            if (url or label) else {}
        return resolved[i] or None

    facts = [f for f in parse_fact_lines(g.get("text") or "") if not is_blocked(f["text"])]
    linked, attributed, pending = [], [], []
    for f in facts:
        fn = _norm_text(f["text"])
        ftok = _tokens(fn)
        fnum = _key_numbers(f["text"])
        idx: List[int] = []
        for s, sn in supports:
            if not sn:
                continue
            if (len(sn) >= 20 and (sn in fn or fn in sn)) or _jaccard(_tokens(sn), ftok) >= 0.5 or \
                    (fnum and fnum <= _key_numbers(sn) and _jaccard(_tokens(sn), ftok) >= 0.2):
                idx.extend(s.get("chunks") or [])
        how = "support" if idx else ""
        if not idx and f["source"]:
            want = _squash(re.sub(r"(?i)^the\s+", "", f["source"]))
            first = next((w for w in re.findall(r"[a-z0-9]+", f["source"].lower())
                          if len(w) >= 4 and w not in _GENERIC_SOURCE_WORDS), "")
            for i, ch in enumerate(chunks):
                have = _squash(ch.get("domain") or "") + "|" + _squash(ch.get("title") or "")
                if (len(want) >= 3 and want in have) or (first and first in have):
                    idx.append(i)
                    how = "publisher"
                    break
        srcs, seen = [], set()
        for i in idx:
            src = source(int(i))
            if src and (src["url"] or src["label"]) not in seen:
                seen.add(src["url"] or src["label"])
                srcs.append(src)
        item = {"text": f["text"][:400], "named_source": f["source"], "sources": srcs[:3], "how": how}
        if srcs:
            linked.append(item)
        else:
            pending.append(item)
    # last resort: is the fact's figure printed on one of the pages the search read?
    if pending and chunks:
        need = [p for p in pending if _key_numbers(p["text"])]
        if need:
            idxs = list(range(min(len(chunks), 10)))
            with ThreadPoolExecutor(max_workers=6) as ex:
                pages = dict(zip(idxs, ex.map(lambda i: fetch((source(i) or {}).get("url") or ""), idxs)))
            for p in need:
                for i, page in pages.items():
                    if _page_has_numbers(page, _key_numbers(p["text"])):
                        p["sources"] = [source(i)]
                        p["how"] = "page"
                        break
        for p in pending:
            (linked if p["sources"] else attributed).append(p)
    else:
        attributed.extend(pending)
    attributed = [a for a in attributed if a["named_source"]]
    return linked, attributed


def research(angle: str, hook: Optional[Dict[str, Any]] = None, *, grounded: Callable = _gemini_grounded,
             resolve: Callable = _resolve_url, fetch: Callable = _fetch_text) -> Dict[str, Any]:
    """Sourced facts for the angle: {facts: [{id, text, sources, linked}], linked, attributed, errors, usage}."""
    hook = hook or {}
    hook_line = f' News hook: "{hook.get("title")}" ({hook.get("source_name") or "news"}).' if hook.get("title") else ""
    linked: List[Dict[str, Any]] = []
    attributed: List[Dict[str, Any]] = []
    errors: List[str] = []
    usage = {"prompt": 0, "completion": 0}
    if hook.get("title") and hook.get("source_url"):
        desc = (hook.get("description") or "").strip()
        linked.append({"text": f"{hook['title'].strip().rstrip('.')}." + (f" {desc}" if desc else ""),
                       "named_source": hook.get("source_name") or "",
                       "sources": [{"url": hook["source_url"], "label": hook.get("source_name") or _domain(hook["source_url"]),
                                    "domain": _domain(hook["source_url"])}], "how": "news"})
    for prompt in (_RESEARCH_PROMPT, _CONTEXT_PROMPT):
        g = grounded_with_fallback(prompt.format(angle=angle, hook=hook_line), grounded)
        errors += g.get("errors") or []
        u = g.get("usage") or {}
        usage["prompt"] += u.get("prompt") or 0
        usage["completion"] += u.get("completion") or 0
        lk, at = link_facts(g, resolve, _fetch_text if fetch is None else fetch)
        seen = {_norm_text(f["text"]) for f in linked + attributed}
        linked += [f for f in lk if _norm_text(f["text"]) not in seen]
        seen = {_norm_text(f["text"]) for f in linked + attributed}
        attributed += [f for f in at if _norm_text(f["text"]) not in seen]
        if len(linked) >= 8:
            break
    # attributed-only facts never outnumber linked ones
    attributed = attributed[: max(0, len(linked))]
    facts = []
    for f in linked[:16] + attributed[:6]:
        facts.append({"id": f"F{len(facts) + 1}", "text": f["text"], "sources": f["sources"],
                      "linked": bool(f["sources"]), "named_source": f.get("named_source") or "", "how": f.get("how", "")})
    return {"facts": facts, "linked": sum(1 for f in facts if f["linked"]), "errors": errors[:6], "usage": usage}


# =============================================================================
# 3. Writing
# =============================================================================
_STRONG_TELLS = ["fast-paced", "delve", "tapestry", "game-changer", "game changer", "in conclusion", "it's worth noting",
                 "it is worth noting", "ever-evolving", "testament to", "navigate the", "navigating the", "realm",
                 "unleash", "embark", "revolutioni", "in today's world", "in the world of", "a myriad", "plethora",
                 "paradigm", "synergy", "buckle up", "let's dive", "dive into", "deep dive", "unlock the", "unlocking"]
_SOFT_TELLS = ["crucial", "pivotal", "landscape", "leverage", "robust", "seamless", "holistic", "moreover", "furthermore",
               "underscore", "boasts", "cutting-edge", "when it comes to", "not just", "key player", "significant"]

_WRITER_SYSTEM = """You are a senior business writer at MECE Insights. Write one article that stands with the best \
Indian business explainers (Mint, ET Prime, The Ken) and consulting insight pieces (McKinsey, BCG): clear, specific, \
evidence-led, worth sharing. Readers: MBA students preparing for placements, MBA aspirants preparing for GD, PI and WAT \
rounds, and young professionals.

FACTS
- Use ONLY the numbered facts provided for anything factual. End every sentence that uses a fact with its id in \
square brackets: "Orders grew 40% last year [F3]." Several: [F2][F5].
- Attribute in the sentence where it helps the reader: "according to RBI data", "the company said in its annual report".
- Every number in the article must come from a fact, written the same way. Only the PI questions, the WAT prompt and \
the case question may use numbers for an estimate.
- If the facts don't support a claim, leave it out. Never invent a decision, a quote, a forecast or a figure.
- Facts marked (attributed) have a named source but no link: use them only with the attribution in the sentence.

QUALITY
- Lead with the news and why it matters now. Then explain the business underneath: how money is made, who wins and \
loses, what drives it, what the numbers say. Give both sides where there is a real debate. End with what to watch.
- Analysis, not summary: connect facts ("that is twice the growth of…", "which means each order…") and explain \
mechanisms. Concrete beats general. One idea per paragraph.
- Plain, precise English as written in India. Money in ₹ with crore and lakh. Paragraphs of 2-4 sentences, varied \
sentence length, active verbs.
- Never use: delve, landscape, navigate, crucial, pivotal, robust, seamless, holistic, leverage, realm, tapestry, \
game-changer, unlock, moreover, furthermore, "in conclusion", "it's worth noting", "in today's fast-paced world", \
"when it comes to", "not just X but Y", "dive into". No exclamation marks. No em dashes (use commas, colons or full \
stops). No hype, no politics, no preaching.

SHAPE: 1,000-1,400 words in total. Return ONLY JSON:
{
  "title": "<= 70 characters, specific and informative, what a reader would search for; no clickbait",
  "meta_description": "<= 155 characters, specific, with the key figure",
  "dek": "one sentence that says what the reader will understand",
  "keywords": ["5-8 search phrases"],
  "content": {
    "key_points": ["exactly 3 short sentences: the most important things to know, cited"],
    "lede": "2-3 sentences: what happened and why it matters now, with the key figure, cited",
    "sections": [
      {"heading": "a specific heading (at least two of the headings phrased as questions)",
       "paragraphs": ["2-4 paragraphs"], "bullets": ["optional, 3-5 items"]}
    ],
    "numbers": [{"figure": "as written in the fact, e.g. ₹1,200 crore", "what": "what it measures", "fact": "F2"}],
    "framework": {"name": "a consulting lens that fits, e.g. profit tree, Porter's five forces, 3Cs, value chain",
                  "heading": "How to break it down", "steps": ["3-5 steps applying the lens to THIS story"]},
    "what_to_watch": ["3 specific things to watch next, phrased as questions or indicators, no predictions"],
    "aspirants": {"gd_topic": "a GD topic this story fits",
                  "for": ["2-3 strong points on one side"], "against": ["2-3 strong points on the other"],
                  "pi_questions": ["3 questions an interviewer could ask about it"],
                  "wat_prompt": "a WAT essay prompt on it",
                  "case_question": "how it could come up as a case in a placement interview (one or two sentences)"},
    "faq": [{"q": "a question people search", "a": "2-3 sentences, cited"}]
  }
}
4-5 sections, 4-6 numbers, 3 FAQs."""


def _chat_json(chat: Callable, feature: str, system: str, user: str, *, max_tokens: int, temperature: float,
               model_override: str = ""):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if model_override:
        try:
            from services.ai_providers import openai_client
            cli = openai_client()
            if cli is not None:
                resp = cli.chat.completions.create(model=model_override, messages=messages, temperature=temperature,
                                                   max_tokens=max_tokens, response_format={"type": "json_object"})
                return _extract_json(resp.choices[0].message.content), resp, model_override, "openai"
        except Exception as e:  # noqa: BLE001 - fall back to the configured feature
            print(f"[daily_blog] writer model {model_override} failed: {type(e).__name__}: {e}")
    resp, model, provider = chat(feature, messages=messages, response_format={"type": "json_object"},
                                 temperature=temperature, max_tokens=max_tokens)
    return _extract_json(resp.choices[0].message.content), resp, model, provider


def _facts_block(facts: List[Dict[str, Any]]) -> str:
    out = []
    for f in facts:
        src = f.get("named_source") or ", ".join(s.get("label", "") for s in f.get("sources") or [])
        out.append(f"[{f['id']}] {f['text']}" + (f" (source: {src})" if src else "")
                   + ("" if f.get("linked", True) else " (attributed)"))
    return "\n".join(out)


def write_article(chat: Callable, angle: str, hook: Dict[str, Any], facts: List[Dict[str, Any]],
                  problems: Optional[List[str]] = None, previous: Optional[Dict[str, Any]] = None):
    user = (f"TOPIC: {angle}\n"
            + (f"NEWS HOOK: {hook.get('title')} ({hook.get('source_name') or 'news'}, "
               f"{str(hook.get('published_at') or '')[:10]})\n" if hook else "")
            + f"TODAY: {datetime.now(IST).strftime('%d %B %Y')}\n\nFACTS (the only facts you may use):\n{_facts_block(facts)}")
    if problems and previous:
        user += ("\n\nYOUR PREVIOUS DRAFT FAILED THESE CHECKS. Fix every one, keep everything else that was good, "
                 "and return the full article again:\n- " + "\n- ".join(problems)
                 + "\n\nPREVIOUS DRAFT:\n" + json.dumps(previous, ensure_ascii=False)[:12000])
    return _chat_json(chat, "seo_writer", _WRITER_SYSTEM, user, max_tokens=4500, temperature=0.55,
                      model_override=(os.getenv("DAILY_BLOG_WRITER_MODEL") or "").strip())


# =============================================================================
# 4. Checks
# =============================================================================
_NUM_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{2,3})+|\d+)(?:\.(\d+))?")
_CITE_RE = re.compile(r"\[(F\d+)\]")


def _numbers(text: str) -> set:
    out = set()
    for m in _NUM_RE.finditer(_CITE_RE.sub(" ", text or "")):
        whole = m.group(1).replace(",", "")
        frac = (m.group(2) or "").rstrip("0")
        out.add(whole + ("." + frac if frac else ""))
    return out


def _exempt(n: str) -> bool:
    try:
        v = float(n)
    except ValueError:
        return True
    return (v.is_integer() and v <= 10) or (v.is_integer() and 1990 <= v <= 2040)


def _sections_texts(c: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for s in c.get("sections") or []:
        out += [s.get("heading") or ""] + list(s.get("paragraphs") or []) + list(s.get("bullets") or [])
    return out


def _checked_texts(article: Dict[str, Any]) -> List[str]:
    """Every reader-facing string that states facts (PI questions, the WAT prompt, the case question and the
    practice prompt may ask the reader to estimate, so their numbers are not checked)."""
    c = article.get("content") or {}
    out = [article.get("title") or "", article.get("meta_description") or "", article.get("dek") or "",
           c.get("summary") or "", c.get("lede") or ""] + list(c.get("key_points") or []) + _sections_texts(c)
    for n in c.get("numbers") or []:
        out += [str(n.get("figure") or ""), str(n.get("what") or "")]
    out += list((c.get("framework") or {}).get("steps") or [])
    out += list(c.get("what_to_watch") or [])
    asp = c.get("aspirants") or {}
    out += [asp.get("gd_topic") or ""] + list(asp.get("for") or []) + list(asp.get("against") or [])
    ia = c.get("interview_angle") or {}
    out += [ia.get("case") or "", ia.get("gd") or ""]
    for f in c.get("faq") or []:
        out += [f.get("q") or "", f.get("a") or ""]
    out += list(c.get("takeaways") or [])
    return [str(x) for x in out if x]


def _all_texts(article: Dict[str, Any]) -> List[str]:
    c = article.get("content") or {}
    asp = c.get("aspirants") or {}
    ia = c.get("interview_angle") or {}
    return (_checked_texts(article) + [c.get("practice_prompt") or "", asp.get("wat_prompt") or "",
                                       asp.get("case_question") or ""]
            + list(asp.get("pi_questions") or []) + list(ia.get("questions") or []))


def word_count(article: Dict[str, Any]) -> int:
    texts = _all_texts({"content": article.get("content") or {}})
    return sum(len(re.findall(r"[A-Za-z0-9₹%]+", _CITE_RE.sub("", t))) for t in texts)


def check_article(article: Dict[str, Any], facts: List[Dict[str, Any]], cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    """Deterministic checks for the daily format. [] = passes."""
    cfg = cfg or config()
    p: List[str] = []
    c = article.get("content") or {}
    fact_ids = {f["id"] for f in facts}
    allowed = set()
    for f in facts:
        allowed |= _numbers(f["text"])
    title = (article.get("title") or "").strip()
    meta = (article.get("meta_description") or "").strip()
    if not title or len(title) > 75:
        p.append(f"title must be 1-70 characters (is {len(title)})")
    if not meta or len(meta) > 160:
        p.append(f"meta_description must be 1-155 characters (is {len(meta)})")
    words = word_count(article)
    if words < cfg["min_words"] or words > cfg["max_words"]:
        p.append(f"length must be {cfg['min_words']}-{cfg['max_words']} words (is {words}); aim for 1,000-1,400")
    if len(c.get("key_points") or []) < 3:
        p.append("key_points needs 3 items")
    if len((c.get("lede") or "").split()) < 20:
        p.append("lede needs 2-3 full sentences")
    if len(c.get("sections") or []) < 4:
        p.append(f"needs 4-5 sections (has {len(c.get('sections') or [])})")
    if len(c.get("numbers") or []) < 4:
        p.append("needs at least 4 entries in numbers")
    if len((c.get("framework") or {}).get("steps") or []) < 3:
        p.append("framework needs 3-5 steps")
    if len(c.get("what_to_watch") or []) < 2:
        p.append("what_to_watch needs 3 items")
    asp = c.get("aspirants") or {}
    if not (asp.get("gd_topic") and asp.get("for") and asp.get("against") and asp.get("pi_questions")
            and asp.get("wat_prompt")):
        p.append("aspirants needs gd_topic, for, against, pi_questions and wat_prompt")
    if len(c.get("faq") or []) < 3:
        p.append("needs 3 FAQs")
    checked = " \n".join(_checked_texts(article))
    unsupported = sorted(n for n in _numbers(checked) if n not in allowed and not _exempt(n))
    if unsupported:
        p.append("these numbers are not in the facts (remove them, or use a fact's figure exactly as written): "
                 + ", ".join(unsupported[:12]))
    cited = set(_CITE_RE.findall(checked))
    bad = sorted(cited - fact_ids)
    if bad:
        p.append("these fact ids do not exist: " + ", ".join(bad))
    if len(cited & fact_ids) < min(5, len(fact_ids)):
        p.append(f"cite at least {min(5, len(fact_ids))} different facts inline, like [F1]")
    for n in c.get("numbers") or []:
        if n.get("fact") not in fact_ids:
            p.append(f"numbers entry '{str(n.get('figure'))[:30]}' must name its fact id")
            break
    everything = " \n".join(_all_texts(article)).lower()
    strong = [t for t in _STRONG_TELLS if t in everything]
    soft = [t for t in _SOFT_TELLS if t in everything]
    if strong:
        p.append("remove these phrases: " + ", ".join(strong))
    if len(soft) >= 3:
        p.append("too many stock phrases, rewrite plainly: " + ", ".join(soft))
    dashes = everything.count("—")
    if dashes > 3:
        p.append(f"too many em dashes ({dashes}); use commas, colons or full stops")
    if "!" in everything:
        p.append("no exclamation marks")
    if is_blocked(everything):
        p.append("touches a political, tragic or divisive subject: " + (_BLOCK_RE.search(everything).group(0)))
    return p


def _serious(problems: List[str]) -> List[str]:
    """Problems a reviewer must see before publishing (facts and safety), as opposed to style."""
    keys = ("numbers are not in the facts", "fact ids do not exist", "political", "cite at least")
    return [x for x in problems if any(k in x for k in keys)]


# =============================================================================
# 5. Critic, related practice, citations
# =============================================================================
_CRITIC_SYSTEM = """You are the editor-in-chief of MECE Insights. Would this article hold its own next to a good Mint \
or ET Prime explainer or a consulting insight piece? Readers: Indian MBA students and MBA aspirants (GD/PI/WAT).
Score 0-100, harshly, on: insight (does it explain the business underneath, not just repeat news?), accuracy and \
grounding (claims match the listed facts, nothing invented), clarity and voice (reads like a skilled human writer), \
usefulness to the reader (could they speak on this in a GD or interview?), and structure. Generic or padded: under 50.
Return ONLY JSON: {"score": <int>, "publishable": <bool>, "notes": "the two most important improvements, or why it \
is strong, in one or two sentences"}. publishable is true only if score >= 80 and nothing is invented."""


def critique(chat: Callable, article: Dict[str, Any], facts: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        data, resp, model, provider = _chat_json(
            chat, "seo_critique", _CRITIC_SYSTEM,
            f"FACTS:\n{_facts_block(facts)}\n\nDRAFT:\n{json.dumps(article, ensure_ascii=False)[:12000]}",
            max_tokens=300, temperature=0.0)
        raw = data.get("score")
        return {"score": int(raw) if isinstance(raw, (int, float)) else None,
                "publishable": bool(data.get("publishable")), "notes": str(data.get("notes") or "")[:600],
                "model": model, "provider": provider}
    except Exception as e:  # noqa: BLE001
        return {"score": None, "publishable": False, "notes": f"Critique failed ({type(e).__name__}); review manually."}


def related_practice(supabase, article: Dict[str, Any], angle: str) -> Dict[str, Any]:
    """The closest case and guesstimate in MECE's India bank, by shared words (none if nothing is close)."""
    try:
        r = (supabase.table("cases").select("id, title, type, skill_cluster, market, is_active")
             .eq("is_active", True).limit(1000).execute())
        rows = [x for x in (r.data or []) if (x.get("market") or "IN") == "IN"]
    except Exception:
        return {}
    want = _tokens(" ".join([angle, article.get("title") or "", " ".join(article.get("keywords") or [])]))
    out: Dict[str, Any] = {}
    for kind in ("case", "guesstimate"):
        best, best_s = None, 0.0
        for x in rows:
            if (x.get("type") or "case") != kind:
                continue
            s = len(want & _tokens(f"{x.get('title') or ''} {x.get('skill_cluster') or ''}"))
            if s > best_s:
                best, best_s = x, s
        if best is not None and best_s >= 1:
            out[kind] = {"id": best["id"], "title": best.get("title") or "", "type": kind}
    return out


def apply_citations(article: Dict[str, Any], facts: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """[F3] -> [2] (the fact's source numbers), and the numbered source list the page shows. An attributed
    fact (named source, no link) gets a numbered source without a URL."""
    by_id = {f["id"]: f for f in facts}
    sources: List[Dict[str, Any]] = []
    index: Dict[str, int] = {}

    def nums_for(fid: str) -> List[int]:
        f = by_id.get(fid)
        if not f:
            return []
        srcs = f.get("sources") or ([{"label": f.get("named_source"), "url": ""}] if f.get("named_source") else [])
        ns = []
        for s in srcs:
            key = s.get("url") or (s.get("label") or "").lower()
            if not key:
                continue
            if key not in index:
                index[key] = len(sources) + 1
                sources.append({"n": index[key], "label": s.get("label") or _domain(s.get("url") or ""),
                                "url": s.get("url") or ""})
            ns.append(index[key])
        return ns

    def fix(text: Any) -> Any:
        if not isinstance(text, str):
            return text

        def rep(m):
            ns = nums_for(m.group(1))
            return "[" + ",".join(str(n) for n in ns) + "]" if ns else ""
        out = _CITE_RE.sub(rep, text)
        out = re.sub(r"\](\s*)\[", ",", out)  # [1][3] -> [1,3]
        return re.sub(r"\[(\d+(?:,\d+)*)\]", lambda m: "[" + ",".join(dict.fromkeys(m.group(1).split(","))) + "]", out)

    def walk(v: Any) -> Any:
        if isinstance(v, str):
            return fix(v)
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, dict):
            return {k: (walk(x) if k != "fact" else x) for k, x in v.items()}
        return v

    c = walk(article.get("content") or {})
    for n in c.get("numbers") or []:
        n["sources"] = nums_for(n.get("fact") or "")
    art = dict(article)
    art["content"] = c
    for k in ("title", "meta_description", "dek"):
        art[k] = _CITE_RE.sub("", art.get(k) or "").strip()
    return art, sources


# =============================================================================
# 6. The daily run
# =============================================================================
def _extract_json(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    if not t:
        raise ValueError("empty model response")
    t = re.sub(r"^```(?:json)?", "", t).rstrip("`").strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def _slugify(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (title or "").lower()).strip("-")
    return s[:70].strip("-") or "insight"


def _unique_slug(supabase, base: str) -> str:
    slug = base
    for _ in range(6):
        try:
            r = supabase.table("seo_pages").select("id").eq("slug", slug).limit(1).execute()
            if not (r.data or []):
                return slug
        except Exception:
            return slug
        slug = f"{base[:60]}-{int(time.time() * 1000) % 100000}"
    return slug


def todays_post(supabase, now: datetime) -> Optional[Dict[str, Any]]:
    start = now.astimezone(IST).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    try:
        r = (supabase.table("seo_pages").select("id, slug, title, status, quality_score, created_at, published_at, "
                                                "agent_meta")
             .eq("kind", KIND).gte("created_at", start.isoformat()).order("created_at", desc=True).limit(1).execute())
        rows = r.data or []
        return rows[0] if rows else None
    except Exception:
        return None


def _default_deps() -> Dict[str, Callable]:
    from services.ai_providers import chat_with_fallback
    from services.ai_usage import log_ai_usage
    from services.growth import telegram_review
    return {"chat": chat_with_fallback, "grounded": _gemini_grounded, "resolve": _resolve_url, "fetch": _fetch_text,
            "log": log_ai_usage, "review": telegram_review}


def run_daily(supabase, *, user_id: Optional[str] = None, force: bool = False, publish: Optional[bool] = None,
              dry_run: bool = False, now: Optional[datetime] = None, deps: Optional[Dict[str, Callable]] = None,
              notify_failure: bool = False, exclude_titles: Optional[List[str]] = None,
              rotate: bool = False) -> Dict[str, Any]:
    """Write today's post. Returns {status: published|draft|exists|skipped|preview, reason, page?, trace}.
    `publish`: None = follow DAILY_BLOG_AUTOPUBLISH; False = always a draft for review. Never raises."""
    if not _run_lock.acquire(blocking=False):
        return {"status": "skipped", "reason": "a daily post is already being written"}
    try:
        return _run(supabase, user_id=user_id, force=force, publish=publish, dry_run=dry_run,
                    now=now or datetime.now(timezone.utc), deps=deps or _default_deps(),
                    notify_failure=notify_failure, exclude_titles=exclude_titles or [], rotate=rotate)
    except Exception as e:  # noqa: BLE001
        return {"status": "skipped", "reason": f"failed: {type(e).__name__}: {e}"[:300]}
    finally:
        _run_lock.release()


def _log(log: Callable, user_id, stage: str, model: str, resp, t0: float, **meta) -> None:
    try:
        log(user_id=user_id, endpoint="/growth/daily-blog", model=model, response=resp,
            latency_ms=int((time.time() - t0) * 1000), meta={"stage": stage, **meta})
    except Exception:
        pass


def _run(supabase, *, user_id, force, publish, dry_run, now, deps, notify_failure, exclude_titles, rotate) -> Dict[str, Any]:
    cfg = config()
    chat, log = deps["chat"], deps.get("log") or (lambda **k: None)
    review = deps.get("review")
    deadline = time.time() + cfg["time_budget_s"]
    trace: Dict[str, Any] = {"ist_date": now.astimezone(IST).date().isoformat(), "research": []}
    if not force and not dry_run:
        existing = todays_post(supabase, now)
        if existing:
            return {"status": "exists", "reason": "today's post is already written", "page": existing}

    cands = candidate_topics(supabase, now)
    skip = {_norm_text(t) for t in exclude_titles}
    cands = [c for c in cands if _norm_text((c["headline"] or {}).get("title") or "") not in skip]
    trace["candidates"] = [{"title": (c["headline"] or {}).get("title"), "score": c["score"], "reasons": c["reasons"]}
                           for c in cands]
    order = editor_pick(cands, chat) if cands else []
    if rotate and order:  # a later scheduled run of the day starts further down the list
        shift = min(_slot(now), len(order) - 1)
        order = order[shift:] + order[:shift]
    order += evergreen_topics(supabase, now, 3)
    order = order[: cfg["max_topics"]]

    for cand in order:
        if time.time() > deadline:
            trace["stopped"] = "time budget used"
            break
        hook = cand.get("headline") or {}
        angle = cand.get("angle") or hook.get("title") or ""
        t0 = time.time()
        res = research(angle, hook, grounded=deps["grounded"], resolve=deps["resolve"], fetch=deps.get("fetch"))
        u = res.get("usage") or {}
        _log(log, user_id, "research", (cfg["research_models"] or ["gemini"])[0],
             SimpleNamespace(id=None, usage=SimpleNamespace(prompt_tokens=u.get("prompt"), completion_tokens=u.get("completion"),
                                                            total_tokens=(u.get("prompt") or 0) + (u.get("completion") or 0))),
             t0, facts=len(res.get("facts") or []), linked=res.get("linked"))
        trace["research"].append({"angle": angle, "facts": len(res.get("facts") or []), "linked": res.get("linked", 0),
                                  "errors": res.get("errors")})
        if res.get("linked", 0) < cfg["min_linked_facts"]:
            continue
        page = _write_and_save(supabase, cand, angle, res["facts"], cfg=cfg, deps=deps, user_id=user_id, now=now,
                               publish=publish, dry_run=dry_run, trace=trace)
        if page is not None:
            return page
    reason = f"tried {len(trace['research'])} topic(s); none had {cfg['min_linked_facts']} facts tied to a source"
    if notify_failure and review is not None:
        try:
            review.send_text(f"MECE daily blog: no article yet today. {reason}. Will try again at the next run; "
                             "reply 'another' to try now.")
        except Exception:
            pass
    return {"status": "skipped", "reason": reason, "trace": trace}


def _write_and_save(supabase, cand, angle, facts, *, cfg, deps, user_id, now, publish, dry_run, trace):
    chat, log, review = deps["chat"], deps.get("log") or (lambda **k: None), deps.get("review")
    hook = cand.get("headline") or {}
    t0 = time.time()
    try:
        article, resp, model, provider = write_article(chat, angle, hook, facts)
    except Exception as e:  # noqa: BLE001
        trace.setdefault("write_errors", []).append(f"{type(e).__name__}: {str(e)[:160]}")
        return None
    _log(log, user_id, "write", model, resp, t0, provider=provider)
    problems = check_article(article, facts, cfg)
    trace["first_draft_problems"] = problems
    for rnd in range(2):
        if not problems:
            break
        t0 = time.time()
        try:
            fixed, resp, model, provider = write_article(chat, angle, hook, facts, problems, article)
        except Exception as e:  # noqa: BLE001
            trace["repair_error"] = f"{type(e).__name__}"
            break
        _log(log, user_id, f"repair{rnd + 1}", model, resp, t0, provider=provider)
        fixed_problems = check_article(fixed, facts, cfg)
        if len(_serious(fixed_problems)) < len(_serious(problems)) or \
                (len(_serious(fixed_problems)) == len(_serious(problems)) and len(fixed_problems) <= len(problems)):
            article, problems = fixed, fixed_problems
    review_note = critique(chat, article, facts)
    related = related_practice(supabase, article, angle)
    cited, sources = apply_citations(article, facts)
    content = dict(cited.get("content") or {})
    content.update({"format": FORMAT, "sources": sources, "related": related, "words": word_count(article)})

    want_publish = cfg["autopublish"] if publish is None else bool(publish)
    passes = (not problems and review_note.get("score") is not None and review_note["score"] >= cfg["min_score"]
              and review_note.get("publishable"))
    status = "published" if (want_publish and passes) else "draft"
    title = (cited.get("title") or angle).strip()[:120]
    row = {
        "slug": _unique_slug(supabase, _slugify(title)),
        "kind": KIND,
        "title": title,
        "meta_description": (cited.get("meta_description") or "").strip()[:300],
        "dek": (cited.get("dek") or "").strip()[:300],
        "content": content,
        "source_refs": [{"label": s["label"], "url": s["url"]} for s in sources if s.get("url")],
        "topic": angle[:280],
        "keywords": [str(k)[:60] for k in (cited.get("keywords") or [])][:8],
        "status": status,
        "quality_score": review_note.get("score"),
        "quality_notes": ("; ".join(problems) + (" | " if problems else "") + (review_note.get("notes") or ""))[:900],
        "model": model,
        "agent_meta": {"ist_date": trace["ist_date"], "domain": cand.get("domain"), "reasons": cand.get("reasons"),
                       "topic_source": "news" if hook else "evergreen", "facts": len(facts),
                       "linked_facts": sum(1 for f in facts if f.get("linked")),
                       "problems": problems, "serious_problems": _serious(problems),
                       "critic": {k: review_note.get(k) for k in ("score", "publishable", "model")},
                       "writer_provider": provider, "research_models": cfg["research_models"],
                       "review": "auto" if status == "published" else "pending"},
        "source_headline_id": hook.get("id"),
        "created_by": user_id,
        "published_at": now.isoformat() if status == "published" else None,
    }
    if dry_run:
        return {"status": "preview", "reason": "dry run: not saved", "page": row, "trace": trace}
    try:
        ins = supabase.table("seo_pages").insert(row).execute()
        page = (ins.data or [row])[0]
    except Exception as e:  # noqa: BLE001
        trace.setdefault("write_errors", []).append(f"not saved: {type(e).__name__}")
        return None
    sent = None
    if review is not None:
        try:
            sent = review.send_for_review(supabase, page)
        except Exception as e:  # noqa: BLE001
            trace["telegram_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    why = "published" if status == "published" else (
        "sent to Telegram for review: reply 'publish' there" if sent else
        "waiting for review in Admin -> Growth (Telegram is not set up)")
    return {"status": status, "reason": why, "page": page, "trace": trace}


def publish_page(supabase, page_id: str, *, via: str = "admin", now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    """Publish one seo_pages row (also used by the Telegram reply). Returns the updated row or None."""
    now = now or datetime.now(timezone.utc)
    try:
        cur = supabase.table("seo_pages").select("id, slug, title, status, agent_meta").eq("id", page_id).limit(1).execute()
        rows = cur.data or []
        if not rows:
            return None
        meta = dict(rows[0].get("agent_meta") or {})
        meta.update({"review": "published", "reviewed_via": via, "reviewed_at": now.isoformat()})
        up = (supabase.table("seo_pages").update({"status": "published", "published_at": now.isoformat(),
                                                  "agent_meta": meta}).eq("id", page_id).execute())
        return (up.data or [dict(rows[0], status="published")])[0]
    except Exception:
        return None


def set_review_state(supabase, page_id: str, *, status: str, review: str, via: str = "admin",
                     extra: Optional[Dict[str, Any]] = None) -> bool:
    try:
        cur = supabase.table("seo_pages").select("agent_meta").eq("id", page_id).limit(1).execute()
        rows = cur.data or []
        if not rows:
            return False
        meta = dict(rows[0].get("agent_meta") or {})
        meta.update({"review": review, "reviewed_via": via, **(extra or {})})
        patch: Dict[str, Any] = {"agent_meta": meta}
        if status:
            patch["status"] = status
            if status != "published":
                patch["published_at"] = None
        supabase.table("seo_pages").update(patch).eq("id", page_id).execute()
        return True
    except Exception:
        return False


def status(supabase, now: Optional[datetime] = None) -> Dict[str, Any]:
    """For /admin/growth: switches, today's post, the last fortnight, and the topics it would pick now."""
    now = now or datetime.now(timezone.utc)
    try:
        r = (supabase.table("seo_pages").select("id, slug, title, status, quality_score, quality_notes, created_at, "
                                                "published_at, agent_meta")
             .eq("kind", KIND).order("created_at", desc=True).limit(14).execute())
        recent = list(r.data or [])
    except Exception:
        recent = []
    try:
        from services.growth import telegram_review
        tg = telegram_review.state()
    except Exception as e:  # noqa: BLE001
        tg = {"configured": False, "error": type(e).__name__}
    return {"config": config(), "telegram": tg, "today": todays_post(supabase, now), "recent": recent,
            "candidates": [{"title": (c["headline"] or {}).get("title"), "score": c["score"], "reasons": c["reasons"],
                            "domain": c["domain"]} for c in candidate_topics(supabase, now)]}
