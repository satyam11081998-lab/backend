"""
Growth Agent — the daily blog autopilot.

One article a day on /insights, written for the people MECE serves (MBA aspirants and
students preparing for placements): a current, non-controversial business story, told with
real, sourced facts and figures, then turned into interview practice — how it can come up as
a case or a GD, the questions to expect, and a link to a related case on MECE.

Pipeline (each step degrades instead of raising; nothing half-done is ever published):

  1. TOPIC   — today's news_headlines (last 72 h) in business / economy / tech / jobs / policy,
               minus anything political, tragic, criminal or a stock tip, minus stories already
               written and topics too close to the last 45 days of posts. Ranked by the news
               classifier's GD-worthiness, MBA relevance, freshness and variety (the same
               domain is not picked three days running). A cheap "editor" model then picks the
               broadest of the top five and frames the angle one level up from the headline
               ("why delivery apps keep raising fees", not "X raises fee by Rs 2"). No fresh
               story qualifies -> an evergreen topic from EVERGREEN, rotated.
  2. RESEARCH — Gemini with Google Search grounding returns dated facts with numbers. Only
               facts the grounding metadata ties to a web source survive; < 3 -> next topic.
  3. WRITE   — `seo_writer` (gpt-4o by default) writes from those facts only, citing them
               inline ([F2]) in a house style that bans the usual machine-written tells.
  4. GATES   — deterministic: length band, every number traceable to a fact, citations
               valid, banned phrases, em dashes, controversy terms, title/meta lengths,
               required parts. One repair round with the exact problems; still failing -> draft.
  5. CRITIC  — `seo_critique` scores 0-100 (usefulness, grounding, human voice, fit).
  6. SAVE    — seo_pages kind='daily'. Published only if DAILY_BLOG_AUTOPUBLISH is on, every
               gate passed and the score >= DAILY_BLOG_MIN_SCORE; otherwise a draft for
               /admin/growth. One post per IST day (idempotent), Telegram ping to the admin.

Switches (env, all OFF by default — ships dormant):
  DAILY_BLOG_ENABLED      the cron route does nothing until this is on
  DAILY_BLOG_AUTOPUBLISH  off = every post waits for an admin's Publish click
  DAILY_BLOG_MIN_SCORE    critic score needed to auto-publish (default 75)
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

IST = timezone(timedelta(hours=5, minutes=30))
KIND = "daily"
SITE = "https://mece.in"
RESEARCH_MODEL = os.getenv("DAILY_BLOG_RESEARCH_MODEL") or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

_run_lock = threading.Lock()


def _flag(name: str, default: bool = False) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "on"}


def config() -> Dict[str, Any]:
    try:
        min_score = int(os.getenv("DAILY_BLOG_MIN_SCORE", "75"))
    except ValueError:
        min_score = 75
    return {
        "enabled": _flag("DAILY_BLOG_ENABLED"),
        "autopublish": _flag("DAILY_BLOG_AUTOPUBLISH"),
        "min_score": max(0, min(100, min_score)),
        "min_words": 650,
        "max_words": 1300,
        "research_available": research_available(),
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

# What an MBA aspirant studies and gets asked about. A headline touching more of these
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

# A broad, always-relevant topic when the news has nothing suitable. Research still pulls
# current figures, so an evergreen post is dated and sourced like any other.
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
    # variety: not the same story or the same domain again and again
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


def candidate_topics(supabase, now: Optional[datetime] = None, limit: int = 5) -> List[Dict[str, Any]]:
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


def evergreen_topic(supabase, now: datetime) -> Dict[str, Any]:
    recent = recent_posts(supabase, now, days=120)
    start = now.astimezone(IST).timetuple().tm_yday % len(EVERGREEN)
    for i in range(len(EVERGREEN)):
        e = EVERGREEN[(start + i) % len(EVERGREEN)]
        tt = _tokens(e["topic"])
        if all(_jaccard(tt, _tokens(f"{p.get('title') or ''} {p.get('topic') or ''}")) < 0.3 for p in recent):
            return {"headline": None, "score": 0, "reasons": ["evergreen (no fresh story qualified)"],
                    "domain": e["domain"], "angle": e["topic"]}
    e = EVERGREEN[start]
    return {"headline": None, "score": 0, "reasons": ["evergreen"], "domain": e["domain"], "angle": e["topic"]}


_EDITOR_SYSTEM = """You are the editor of MECE Insights, read by Indian MBA students and aspirants preparing for \
placements (consulting, product, marketing, finance, general management). From the candidate news stories, pick the \
ONE that makes the most useful, broadly interesting article for them today: a business question many readers would \
search for, not a niche or a one-day stock move, and never political or divisive.
Then frame the ANGLE one level broader than the headline: the business question behind the story, phrased as a \
reader would search it (e.g. headline "App X raises platform fee by Rs 2" -> angle "Why food delivery apps keep \
raising platform fees"). 6-12 words, no clickbait.
Return ONLY JSON: {"pick": <index>, "angle": "...", "why": "one sentence"}"""


def editor_pick(cands: List[Dict[str, Any]], chat: Callable) -> Dict[str, Any]:
    """Let a cheap model choose among the top few and frame a broader angle. Falls back to #1."""
    best = dict(cands[0])
    best["angle"] = (best["headline"] or {}).get("title") or ""
    lines = []
    for i, c in enumerate(cands):
        h = c["headline"] or {}
        lines.append(f"[{i}] {h.get('title')} — {(h.get('description') or '')[:220]} (category {h.get('category')})")
    try:
        resp, _, _ = chat("seo_critique", messages=[{"role": "system", "content": _EDITOR_SYSTEM},
                                                     {"role": "user", "content": "\n".join(lines)}],
                          response_format={"type": "json_object"}, temperature=0.2, max_tokens=200)
        data = _extract_json(resp.choices[0].message.content)
        idx = int(data.get("pick", 0))
        angle = str(data.get("angle") or "").strip()
        if 0 <= idx < len(cands) and angle and not is_blocked(angle) and len(angle) <= 110:
            pick = dict(cands[idx])
            pick["angle"] = angle
            pick["reasons"] = list(pick["reasons"]) + ["editor: " + str(data.get("why") or "")[:160]]
            return pick
    except Exception:
        pass
    return best


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


_RESEARCH_PROMPT = """Research this business topic from the live web for an article in an Indian business-education \
publication: "{angle}".{hook}

Find 8 to 12 facts that a reader could check: numbers (market size, growth, revenue, users, prices, share, \
costs), dates and named decisions. Prefer primary sources (company filings and statements, RBI, government \
statistics, industry bodies) and established business press. Prefer the most recent figures and say the period \
or date each one is for. Do not include opinions, forecasts presented as facts, or anything political.

Write each fact on its own line, exactly in this form:
FACT: <one sentence with the figure, what it measures, and the period or date>
Nothing else."""


def _gemini_grounded(prompt: str) -> Dict[str, Any]:
    """{text, chunks: [{uri, title}], supports: [{text, chunks: [int]}], usage} — raises on failure."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    resp = client.models.generate_content(
        model=RESEARCH_MODEL, contents=prompt,
        config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())], temperature=0.2),
    )
    text = getattr(resp, "text", "") or ""
    chunks, supports = [], []
    cand = (getattr(resp, "candidates", None) or [None])[0]
    gm = getattr(cand, "grounding_metadata", None)
    for ch in (getattr(gm, "grounding_chunks", None) or []):
        web = getattr(ch, "web", None)
        chunks.append({"uri": getattr(web, "uri", "") or "", "title": getattr(web, "title", "") or ""} if web else
                      {"uri": "", "title": ""})
    for s in (getattr(gm, "grounding_supports", None) or []):
        seg = getattr(s, "segment", None)
        supports.append({"text": getattr(seg, "text", "") or "", "chunks": list(getattr(s, "grounding_chunk_indices", None) or [])})
    um = getattr(resp, "usage_metadata", None)
    usage = {"prompt": getattr(um, "prompt_token_count", None), "completion": getattr(um, "candidates_token_count", None)}
    return {"text": text, "chunks": chunks, "supports": supports, "usage": usage}


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


def _domain(url: str) -> str:
    m = re.match(r"https?://([^/]+)", url or "")
    return (m.group(1) if m else "").removeprefix("www.")


def research(angle: str, hook: str = "", *, grounded: Callable = _gemini_grounded,
             resolve: Callable = _resolve_url) -> Dict[str, Any]:
    """Sourced facts for the angle. Only a fact the grounding ties to a web page is kept."""
    prompt = _RESEARCH_PROMPT.format(angle=angle, hook=f" News hook: {hook}" if hook else "")
    try:
        g = grounded(prompt)
    except Exception as e:  # noqa: BLE001
        return {"facts": [], "error": f"{type(e).__name__}: {e}"[:300], "usage": {}}
    lines = [ln.strip() for ln in (g.get("text") or "").splitlines() if ln.strip().upper().startswith("FACT:")]
    resolved: Dict[int, Dict[str, str]] = {}

    def source(i: int) -> Optional[Dict[str, str]]:
        if i in resolved:
            return resolved[i] or None
        ch = (g.get("chunks") or [])[i] if i < len(g.get("chunks") or []) else {}
        url = resolve(ch.get("uri") or "")
        title = (ch.get("title") or "").strip()
        resolved[i] = {"url": url, "label": title or _domain(url)} if (url or title) else {}
        return resolved[i] or None

    facts = []
    for ln in lines:
        body = ln[5:].strip()
        if not body or is_blocked(body):
            continue
        idx: List[int] = []
        for s in g.get("supports") or []:
            st = (s.get("text") or "").strip()
            if st and (st in body or body in st or _jaccard(_tokens(st), _tokens(body)) >= 0.6):
                idx.extend(s.get("chunks") or [])
        srcs, seen = [], set()
        for i in idx:
            src = source(int(i))
            if src and (src["url"] or src["label"]) not in seen:
                seen.add(src["url"] or src["label"])
                srcs.append(src)
        if srcs:
            facts.append({"id": f"F{len(facts) + 1}", "text": body[:400], "sources": srcs[:3]})
    return {"facts": facts[:14], "usage": g.get("usage") or {}, "raw_facts": len(lines)}


# =============================================================================
# 3. Writing
# =============================================================================
_STRONG_TELLS = ["fast-paced", "delve", "tapestry", "game-changer", "game changer", "in conclusion", "it's worth noting",
                 "it is worth noting", "ever-evolving", "testament to", "navigate the", "navigating the", "realm",
                 "unleash", "embark", "revolutioni", "in today's world", "in the world of", "a myriad", "plethora",
                 "paradigm", "synergy", "buckle up", "let's dive", "dive into", "deep dive", "unlock the", "unlocking"]
_SOFT_TELLS = ["crucial", "pivotal", "landscape", "leverage", "robust", "seamless", "holistic", "moreover", "furthermore",
               "underscore", "boasts", "cutting-edge", "when it comes to", "not just", "key player", "significant"]

_WRITER_SYSTEM = """You write for MECE Insights: clear, factual business articles for Indian MBA students and \
aspirants preparing for placement interviews and group discussions. A reader should finish knowing what happened, \
why it matters to a business, and how it could come up in their interview, then want to practise it on MECE.

FACTS
- Use ONLY the numbered facts you are given for anything factual. Every sentence that uses a fact ends with its \
id in square brackets, e.g. "Orders grew 40% last year [F3]." Several: [F2][F5].
- Every number in the article must come from a fact, written the same way. The only exceptions: the practice \
prompt and the interview questions, which may ask the reader to estimate.
- If the facts don't support a claim, leave the claim out. Never invent a company decision, a quote or a figure.

VOICE (this matters: it must read like a sharp person wrote it, not a machine)
- Plain, specific English as written in India. Money in ₹ with crore and lakh. Short paragraphs of 2-4 sentences. \
Vary sentence length. Concrete nouns, active verbs.
- Do not use these words or phrases: delve, landscape, navigate, crucial, pivotal, robust, seamless, holistic, \
leverage, realm, tapestry, game-changer, unlock, moreover, furthermore, "in conclusion", "it's worth noting", \
"in today's fast-paced world", "when it comes to", "not just X but Y", "dive into". No exclamation marks. \
No em dashes: use a comma, a colon or a full stop.
- No hype and no opinion on politics. Explain, don't preach.

SHAPE (800-1,100 words in total)
Return ONLY JSON:
{
  "title": "<= 65 characters, the question or plain claim a reader would search for",
  "meta_description": "<= 155 characters, specific",
  "dek": "one line that says what the reader will get",
  "keywords": ["5-8 search phrases"],
  "content": {
    "summary": "40-60 words that answer the title directly, with the key figure",
    "sections": [
      {"heading": "a question, e.g. What happened?", "paragraphs": ["..."], "bullets": ["optional"]},
      {"heading": "a question about why it matters for the business", "paragraphs": ["..."]},
      {"heading": "a question about the economics or the strategy behind it", "paragraphs": ["..."]}
    ],
    "numbers": [{"figure": "the number as in the fact, e.g. ₹1,200 crore", "what": "what it measures", "fact": "F2"}],
    "framework": {"heading": "How to structure it", "steps": ["3-5 steps a candidate would use to break this down"]},
    "interview_angle": {"case": "how this becomes a case question (2-3 sentences)",
                        "gd": "a GD topic it fits and the two sides in one line each",
                        "questions": ["3 questions an interviewer could ask about it"]},
    "practice_prompt": "one estimation or mini-case the reader can try now",
    "faq": [{"q": "...", "a": "2-3 sentences, cited"}],
    "takeaways": ["3 short, specific lines"]
  }
}
3-4 sections, 3-6 numbers, 3 FAQs."""


def _chat_json(chat: Callable, feature: str, system: str, user: str, *, max_tokens: int, temperature: float):
    resp, model, provider = chat(feature, messages=[{"role": "system", "content": system},
                                                    {"role": "user", "content": user}],
                                 response_format={"type": "json_object"}, temperature=temperature,
                                 max_tokens=max_tokens)
    return _extract_json(resp.choices[0].message.content), resp, model, provider


def _facts_block(facts: List[Dict[str, Any]]) -> str:
    return "\n".join(f"[{f['id']}] {f['text']}" for f in facts)


def write_article(chat: Callable, angle: str, hook: Dict[str, Any], facts: List[Dict[str, Any]],
                  problems: Optional[List[str]] = None, previous: Optional[Dict[str, Any]] = None):
    user = (f"TOPIC: {angle}\n"
            + (f"NEWS HOOK: {hook.get('title')} ({hook.get('source_name') or 'news'}, "
               f"{str(hook.get('published_at') or '')[:10]})\n" if hook else "")
            + f"TODAY: {datetime.now(IST).strftime('%d %B %Y')}\n\nFACTS (the only facts you may use):\n{_facts_block(facts)}")
    if problems and previous:
        user += ("\n\nYOUR PREVIOUS DRAFT FAILED THESE CHECKS. Fix every one and return the full article again:\n- "
                 + "\n- ".join(problems) + "\n\nPREVIOUS DRAFT:\n" + json.dumps(previous)[:9000])
    return _chat_json(chat, "seo_writer", _WRITER_SYSTEM, user, max_tokens=3200, temperature=0.55)


# =============================================================================
# 4. Gates
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


def _checked_texts(article: Dict[str, Any]) -> List[str]:
    """Every reader-facing string that states facts (the practice prompt and interview questions may ask
    the reader to estimate, so they are not checked for numbers)."""
    c = article.get("content") or {}
    out = [article.get("title") or "", article.get("meta_description") or "", article.get("dek") or "",
           c.get("summary") or ""]
    for s in c.get("sections") or []:
        out += [s.get("heading") or ""] + list(s.get("paragraphs") or []) + list(s.get("bullets") or [])
    for n in c.get("numbers") or []:
        out += [str(n.get("figure") or ""), str(n.get("what") or "")]
    out += list((c.get("framework") or {}).get("steps") or [])
    ia = c.get("interview_angle") or {}
    out += [ia.get("case") or "", ia.get("gd") or ""]
    for f in c.get("faq") or []:
        out += [f.get("q") or "", f.get("a") or ""]
    out += list(c.get("takeaways") or [])
    return [str(x) for x in out if x]


def _all_texts(article: Dict[str, Any]) -> List[str]:
    c = article.get("content") or {}
    ia = c.get("interview_angle") or {}
    return _checked_texts(article) + [c.get("practice_prompt") or ""] + list(ia.get("questions") or [])


def word_count(article: Dict[str, Any]) -> int:
    c = article.get("content") or {}
    texts = _all_texts({"content": c})
    return sum(len(re.findall(r"[A-Za-z0-9₹%]+", _CITE_RE.sub("", t))) for t in texts)


def check_article(article: Dict[str, Any], facts: List[Dict[str, Any]], cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    """Deterministic checks. [] = passes."""
    cfg = cfg or config()
    p: List[str] = []
    c = article.get("content") or {}
    fact_ids = {f["id"] for f in facts}
    allowed = set()
    for f in facts:
        allowed |= _numbers(f["text"])
    title = (article.get("title") or "").strip()
    meta = (article.get("meta_description") or "").strip()
    if not title or len(title) > 70:
        p.append(f"title must be 1-70 characters (is {len(title)})")
    if not meta or len(meta) > 160:
        p.append(f"meta_description must be 1-160 characters (is {len(meta)})")
    words = word_count(article)
    if words < cfg["min_words"] or words > cfg["max_words"]:
        p.append(f"length must be {cfg['min_words']}-{cfg['max_words']} words (is {words}); aim for 800-1,100")
    summary_words = len((c.get("summary") or "").split())
    if not 25 <= summary_words <= 80:
        p.append(f"summary must be 40-60 words (is {summary_words})")
    if len(c.get("sections") or []) < 3:
        p.append("needs at least 3 sections")
    if len(c.get("numbers") or []) < 3:
        p.append("needs at least 3 entries in numbers")
    ia = c.get("interview_angle") or {}
    if not (ia.get("case") and ia.get("questions")):
        p.append("interview_angle needs a case and questions")
    if not c.get("practice_prompt"):
        p.append("needs a practice_prompt")
    if len(c.get("faq") or []) < 2:
        p.append("needs at least 2 FAQs")
    checked = " \n".join(_checked_texts(article))
    unsupported = sorted(n for n in _numbers(checked) if n not in allowed and not _exempt(n))
    if unsupported:
        p.append("these numbers are not in the facts (remove them or use a fact's figure exactly): "
                 + ", ".join(unsupported[:12]))
    cited = set(_CITE_RE.findall(checked))
    bad = sorted(cited - fact_ids)
    if bad:
        p.append("these fact ids do not exist: " + ", ".join(bad))
    if len(cited & fact_ids) < 3:
        p.append("cite at least 3 different facts inline, like [F1]")
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


# =============================================================================
# 5. Critic, related practice, citations
# =============================================================================
_CRITIC_SYSTEM = """You review articles for MECE Insights (readers: Indian MBA students and placement aspirants). \
Score the DRAFT 0-100, harshly, on: usefulness to that reader (do they learn something they could say in an \
interview or GD?), grounding (claims match the listed facts, nothing invented), voice (reads like a sharp human \
writer, not generic machine text), fit (business, broad enough to interest many, not political or divisive), and \
structure. A generic or padded article scores under 50.
Return ONLY JSON: {"score": <int>, "publishable": <bool>, "notes": "one or two sentences: the biggest weakness or \
why it is strong"}. publishable is true only if score >= 75 and nothing is invented."""


def critique(chat: Callable, article: Dict[str, Any], facts: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        data, resp, model, provider = _chat_json(
            chat, "seo_critique", _CRITIC_SYSTEM,
            f"FACTS:\n{_facts_block(facts)}\n\nDRAFT:\n{json.dumps(article)[:9000]}", max_tokens=300, temperature=0.0)
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
    """[F3] -> [2] (the fact's source numbers), and the numbered source list the page shows."""
    by_id = {f["id"]: f for f in facts}
    sources: List[Dict[str, Any]] = []
    index: Dict[str, int] = {}

    def nums_for(fid: str) -> List[int]:
        f = by_id.get(fid)
        if not f:
            return []
        ns = []
        for s in f["sources"]:
            key = s.get("url") or s.get("label")
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
        return re.sub(r"\](\s*)\[", ",", out)  # [1][3] -> [1,3]

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
        ns = nums_for(n.get("fact") or "")
        n["sources"] = ns
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
        r = (supabase.table("seo_pages").select("id, slug, title, status, quality_score, created_at, published_at")
             .eq("kind", KIND).gte("created_at", start.isoformat()).order("created_at", desc=True).limit(1).execute())
        rows = r.data or []
        return rows[0] if rows else None
    except Exception:
        return None


def _default_deps() -> Dict[str, Callable]:
    from services.ai_providers import chat_with_fallback
    from services.ai_usage import log_ai_usage
    return {"chat": chat_with_fallback, "grounded": _gemini_grounded, "resolve": _resolve_url, "log": log_ai_usage}


def _notify(text: str) -> None:
    try:
        from services.telegram_notify import send_admin_alert
        send_admin_alert(text)
    except Exception:
        pass


def run_daily(supabase, *, user_id: Optional[str] = None, force: bool = False, publish: Optional[bool] = None,
              dry_run: bool = False, now: Optional[datetime] = None, deps: Optional[Dict[str, Callable]] = None
              ) -> Dict[str, Any]:
    """Write today's post. Returns {status: published|draft|exists|skipped|preview, reason, page?, trace}.
    `publish`: None = follow DAILY_BLOG_AUTOPUBLISH; False = always a draft. Never raises."""
    if not _run_lock.acquire(blocking=False):
        return {"status": "skipped", "reason": "a daily post is already being written"}
    try:
        return _run(supabase, user_id=user_id, force=force, publish=publish, dry_run=dry_run,
                    now=now or datetime.now(timezone.utc), deps=deps or _default_deps())
    except Exception as e:  # noqa: BLE001
        return {"status": "skipped", "reason": f"failed: {type(e).__name__}: {e}"[:300]}
    finally:
        _run_lock.release()


def _run(supabase, *, user_id, force, publish, dry_run, now, deps) -> Dict[str, Any]:
    cfg = config()
    chat, log = deps["chat"], deps.get("log") or (lambda **k: None)
    trace: Dict[str, Any] = {"ist_date": now.astimezone(IST).date().isoformat()}
    if not force and not dry_run:
        existing = todays_post(supabase, now)
        if existing:
            return {"status": "exists", "reason": "today's post is already written", "page": existing}

    cands = candidate_topics(supabase, now)
    trace["candidates"] = [{"title": (c["headline"] or {}).get("title"), "score": c["score"], "reasons": c["reasons"]}
                           for c in cands]
    order: List[Dict[str, Any]] = []
    if cands:
        first = editor_pick(cands, chat)
        order = [first] + [dict(c, angle=(c["headline"] or {}).get("title") or "") for c in cands
                           if c["headline"] is not (first.get("headline"))][:2]
    order.append(evergreen_topic(supabase, now))

    pick, facts = None, []
    for cand in order:
        hook = cand.get("headline") or {}
        t0 = time.time()
        res = research(cand["angle"], hook.get("title") or "", grounded=deps["grounded"], resolve=deps["resolve"])
        try:
            u = res.get("usage") or {}
            shim = SimpleNamespace(id=None, usage=SimpleNamespace(
                prompt_tokens=u.get("prompt"), completion_tokens=u.get("completion"),
                total_tokens=(u.get("prompt") or 0) + (u.get("completion") or 0)))
            log(user_id=user_id, endpoint="/growth/daily-blog", model=RESEARCH_MODEL, response=shim,
                latency_ms=int((time.time() - t0) * 1000),
                meta={"stage": "research", "facts": len(res.get("facts") or [])})
        except Exception:
            pass
        trace.setdefault("research", []).append({"angle": cand["angle"], "facts": len(res.get("facts") or []),
                                                 "error": res.get("error")})
        if len(res.get("facts") or []) >= 3:
            pick, facts = cand, res["facts"]
            break
    if pick is None:
        _notify("MECE daily blog: no post today — no topic had at least 3 sourced facts.")
        return {"status": "skipped", "reason": "no topic had at least 3 sourced facts", "trace": trace}

    hook = pick.get("headline") or {}
    angle = pick["angle"]
    t0 = time.time()
    article, resp, model, provider = write_article(chat, angle, hook, facts)
    log(user_id=user_id, endpoint="/growth/daily-blog", model=model, response=resp,
        latency_ms=int((time.time() - t0) * 1000), meta={"stage": "write", "provider": provider})
    problems = check_article(article, facts, cfg)
    trace["first_draft_problems"] = problems
    if problems:
        t0 = time.time()
        try:
            fixed, resp, model, provider = write_article(chat, angle, hook, facts, problems, article)
            log(user_id=user_id, endpoint="/growth/daily-blog", model=model, response=resp,
                latency_ms=int((time.time() - t0) * 1000), meta={"stage": "repair", "provider": provider})
            fixed_problems = check_article(fixed, facts, cfg)
            if len(fixed_problems) <= len(problems):
                article, problems = fixed, fixed_problems
        except Exception as e:  # noqa: BLE001
            trace["repair_error"] = f"{type(e).__name__}"
    review = critique(chat, article, facts)
    related = related_practice(supabase, article, angle)
    cited, sources = apply_citations(article, facts)
    content = dict(cited.get("content") or {})
    content.update({"format": "daily-1", "sources": sources, "related": related, "words": word_count(article)})

    want_publish = cfg["autopublish"] if publish is None else bool(publish)
    passes = (not problems and review.get("score") is not None and review["score"] >= cfg["min_score"]
              and review.get("publishable"))
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
        "quality_score": review.get("score"),
        "quality_notes": ("; ".join(problems) + (" | " if problems else "") + (review.get("notes") or ""))[:900],
        "model": model,
        "agent_meta": {"ist_date": trace["ist_date"], "domain": pick.get("domain"), "reasons": pick.get("reasons"),
                       "topic_source": "news" if hook else "evergreen", "facts": len(facts),
                       "problems": problems, "critic": {k: review.get(k) for k in ("score", "publishable", "model")},
                       "writer_provider": provider, "research_model": RESEARCH_MODEL,
                       "autopublish": want_publish},
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
        return {"status": "skipped", "reason": f"written but not saved: {type(e).__name__}", "trace": trace}
    why = "" if status == "published" else (
        "waiting for you (auto-publish is off)" if not want_publish else
        f"held as a draft: score {review.get('score')}, {len(problems)} check(s) failed")
    _notify(f"MECE daily blog — {status}: {title}\n{SITE}/insights/{row['slug']}\n{why}".strip())
    return {"status": status, "reason": why or "published", "page": page, "trace": trace}


def status(supabase, now: Optional[datetime] = None) -> Dict[str, Any]:
    """For /admin/growth: switches, today's post, the last fortnight, and tomorrow's likely topics."""
    now = now or datetime.now(timezone.utc)
    try:
        r = (supabase.table("seo_pages").select("id, slug, title, status, quality_score, quality_notes, created_at, "
                                                "published_at, agent_meta")
             .eq("kind", KIND).order("created_at", desc=True).limit(14).execute())
        recent = list(r.data or [])
    except Exception:
        recent = []
    return {"config": config(), "today": todays_post(supabase, now), "recent": recent,
            "candidates": [{"title": (c["headline"] or {}).get("title"), "score": c["score"], "reasons": c["reasons"],
                            "domain": c["domain"]} for c in candidate_topics(supabase, now)]}
