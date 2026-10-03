"""
Daily blog autopilot (services/growth/daily_blog.py). Standard library only:

    python -m tests.test_daily_blog

No network, no keys: research, the writer and the critic are fakes injected through
`deps`, and a FakeSupabase stands in for PostgREST. What is pinned here is the part that
must never regress silently: which topics are allowed, that only sourced facts survive,
that an invented number or a machine-sounding phrase keeps a post from publishing, the
switches, and one post per IST day.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests import _sdk_stubs  # noqa: E402

_sdk_stubs.install()
try:  # pragma: no cover - fastapi only for HTTPException in imported modules
    import fastapi  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    import types as _t
    _f = _t.ModuleType("fastapi")

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str = ""):
            super().__init__(detail)
            self.status_code, self.detail = status_code, detail

    _f.HTTPException = HTTPException
    sys.modules["fastapi"] = _f

from services.growth import daily_blog as db  # noqa: E402

NOW = datetime(2026, 10, 4, 1, 30, tzinfo=timezone.utc)  # 07:00 IST


# ---------------------------------------------------------------------------- fakes
class _Q:
    def __init__(self, sb, name):
        self.sb, self.name, self.filters, self._limit, self._insert = sb, name, [], None, None

    def select(self, *_a, **_k):
        return self

    def eq(self, col, val):
        self.filters.append(lambda r: r.get(col) == val)
        return self

    def gte(self, col, val):
        self.filters.append(lambda r: str(r.get(col) or "") >= str(val))
        return self

    def order(self, *_a, **_k):
        return self

    def limit(self, n):
        self._limit = n
        return self

    def insert(self, row):
        self._insert = row
        return self

    def execute(self):
        rows = self.sb.tables.setdefault(self.name, [])
        if self._insert is not None:
            row = dict(self._insert, id=f"id{len(rows) + 1}", created_at=NOW.isoformat())
            rows.append(row)
            return SimpleNamespace(data=[row])
        out = [r for r in rows if all(f(r) for f in self.filters)]
        return SimpleNamespace(data=out[: self._limit] if self._limit else out)


class FakeSupabase:
    def __init__(self, tables):
        self.tables = tables

    def table(self, name):
        return _Q(self, name)


def headline(i, title, category="business", gd=7, hours_ago=6, desc="", star=False):
    return {"id": f"h{i}", "title": title, "description": desc, "category": category, "source_name": "Business Daily",
            "source_url": f"https://news.example/{i}", "gd_worthiness_score": gd, "is_star": star,
            "published_at": (NOW - timedelta(hours=hours_ago)).isoformat(), "keywords": []}


FACTS_TEXT = """FACT: Quick-commerce orders in India grew 74% in FY25, according to the industry body.
FACT: The three largest quick-commerce apps had about 5,000 dark stores in March 2026.
FACT: The average order value on quick-commerce apps was ₹520 in 2025.
FACT: Prime ministers met to discuss trade.
FACT: An unsourced claim with no grounding at all."""


def fake_grounded(_prompt):
    lines = FACTS_TEXT.splitlines()
    return {"text": FACTS_TEXT,
            "chunks": [{"uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc", "title": "industry.example"},
                       {"uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/def", "title": "biz.example"}],
            "supports": [{"text": lines[0][6:], "chunks": [0]}, {"text": lines[1][6:], "chunks": [1]},
                         {"text": lines[2][6:], "chunks": [0, 1]}, {"text": lines[3][6:], "chunks": [1]}],
            "usage": {"prompt": 100, "completion": 300}}


def fake_resolve(uri):
    return {"abc": "https://industry.example/report", "def": "https://biz.example/story"}[uri.rsplit("/", 1)[-1]]


def paragraph(n_words, cite="[F1]"):
    words = ("Quick commerce changed how Indian cities buy groceries and the reason is simple convenience "
             "that people will pay for when the basket is small and the need is urgent").split()
    out = " ".join((words * (n_words // len(words) + 1))[:n_words])
    return out + f". {cite}"


def good_article(**over):
    a = {
        "title": "How quick commerce makes money in India",
        "meta_description": "Orders grew 74% in FY25. Here is how dark stores and order values decide who profits.",
        "dek": "The economics behind ten-minute delivery, and how it comes up in interviews.",
        "keywords": ["quick commerce india", "dark stores", "unit economics"],
        "content": {
            "summary": "Quick commerce makes money when a dark store fills enough orders of a high enough value. "
                       "Orders grew 74% in FY25 [F1], and with about 5,000 dark stores [F2] the race is now about "
                       "order value, which averaged ₹520 [F3].",
            "sections": [{"heading": "What happened?", "paragraphs": [paragraph(140, "[F1]")]},
                         {"heading": "Why does it matter?", "paragraphs": [paragraph(140, "[F2]")]},
                         {"heading": "How do the economics work?", "paragraphs": [paragraph(160, "[F3]")]}],
            "numbers": [{"figure": "74%", "what": "order growth in FY25", "fact": "F1"},
                        {"figure": "5,000", "what": "dark stores", "fact": "F2"},
                        {"figure": "₹520", "what": "average order value", "fact": "F3"}],
            "framework": {"heading": "How to structure it", "steps": ["Revenue per order", "Cost per order", "Orders per store per day"]},
            "interview_angle": {"case": "A client asks whether to open dark stores in a new city. " + paragraph(30, ""),
                                "gd": "Is quick commerce good for kiranas? Yes: reach. No: margins.",
                                "questions": ["How many orders does a dark store need a day to break even?",
                                              "What would you change first, order value or delivery cost?",
                                              "Estimate the quick-commerce market in a city of 50 lakh."]},
            "practice_prompt": "Estimate how many orders a dark store in Pune serves on a weekday.",
            "faq": [{"q": "Is quick commerce profitable?", "a": paragraph(40, "[F3]")},
                    {"q": "How many dark stores are there?", "a": "About 5,000 across the three largest apps [F2]."},
                    {"q": "What drives profit?", "a": paragraph(30, "[F1]")}],
            "takeaways": ["Order value decides profit.", "Density beats reach.", "Know the store-level maths."],
        },
    }
    a.update(over)
    return a


class Chat:
    """Fake chat_with_fallback: editor -> writer (scripted drafts) -> critic."""

    def __init__(self, drafts, score=86, editor=None):
        self.drafts, self.score, self.editor, self.calls = list(drafts), score, editor, []

    def __call__(self, feature, messages, **_kw):
        system = messages[0]["content"]
        self.calls.append(feature)
        if "editor of MECE Insights" in system:
            body = self.editor or {"pick": 0, "angle": "How quick commerce makes money in India", "why": "broad"}
        elif feature == "seo_writer":
            body = self.drafts.pop(0) if len(self.drafts) > 1 else self.drafts[0]
        else:
            body = {"score": self.score, "publishable": self.score >= 75, "notes": "clear and grounded"}
        msg = SimpleNamespace(content=json.dumps(body))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None, id=None), "fake-model", "fake"


def deps(chat):
    return {"chat": chat, "grounded": fake_grounded, "resolve": fake_resolve, "log": lambda **k: None}


def base_tables():
    return {
        "news_headlines": [
            headline(1, "Quick-commerce apps add dark stores as order values climb", desc="Delivery apps expand.", gd=8),
            headline(2, "State elections: parties promise free power", category="policy", gd=9),
            headline(3, "XYZ shares jump 12% after results", gd=8),
            headline(4, "Cricket league final draws record viewers", category="other", gd=9),
            headline(5, "Five killed in factory fire in Gujarat", gd=9),
        ],
        "seo_pages": [],
        "cases": [{"id": "c1", "title": "Dark store expansion for a grocery app", "type": "case", "is_active": True, "market": None},
                  {"id": "c2", "title": "US airline pricing", "type": "case", "is_active": True, "market": "US"},
                  {"id": "g1", "title": "Estimate daily orders for a quick commerce dark store", "type": "guesstimate",
                   "is_active": True, "market": "IN"}],
    }


passed = 0


def check(name, cond):
    global passed
    if not cond:
        print("  ✗", name)
        raise SystemExit(1)
    passed += 1
    print("  ✓", name)


def env(**kv):
    for k in ("DAILY_BLOG_ENABLED", "DAILY_BLOG_AUTOPUBLISH", "DAILY_BLOG_MIN_SCORE"):
        os.environ.pop(k, None)
    for k, v in kv.items():
        os.environ[k] = v


# ---------------------------------------------------------------------------- topics
print("topic selection")
sb = FakeSupabase(base_tables())
cands = db.candidate_topics(sb, NOW)
titles = [c["headline"]["title"] for c in cands]
check("a business story is a candidate", "Quick-commerce apps add dark stores as order values climb" in titles)
check("politics is never a candidate", not any("elections" in t for t in titles))
check("tragedy is never a candidate", not any("killed" in t for t in titles))
check("an 'other' category story is never a candidate", not any("Cricket" in t for t in titles))
check("a one-stock move ranks below a real story", titles.index("XYZ shares jump 12% after results") > 0
      if "XYZ shares jump 12% after results" in titles else True)
check("'price war' is business, 'trade war' is not", not db.is_blocked("Telecom price war returns")
      and db.is_blocked("Trade war fears hit exporters"))
check("company names are not caught by the blocklist", not db.is_blocked("Hindustan Unilever raises prices")
      and not db.is_blocked("RBI Governor holds the repo rate"))
used = FakeSupabase(dict(base_tables(), seo_pages=[{"title": "Why quick commerce apps add dark stores", "topic": "",
                                                    "created_at": (NOW - timedelta(days=3)).isoformat(), "agent_meta": {}}]))
check("a topic too close to a recent post is skipped",
      not any("Quick-commerce" in c["headline"]["title"] for c in db.candidate_topics(used, NOW)))
ev = db.evergreen_topic(FakeSupabase({"seo_pages": []}), NOW)
check("an evergreen topic is always available", ev["angle"] and ev["headline"] is None)

# ---------------------------------------------------------------------------- research
print("research")
r = db.research("How quick commerce makes money", grounded=fake_grounded, resolve=fake_resolve)
check("only grounded, non-political facts survive", len(r["facts"]) == 3)
check("redirect links are resolved to the real page", r["facts"][0]["sources"][0]["url"] == "https://industry.example/report")
check("a broken research call gives no facts, not an error",
      db.research("x", grounded=lambda p: (_ for _ in ()).throw(RuntimeError("quota")), resolve=fake_resolve)["facts"] == [])

# ---------------------------------------------------------------------------- gates
print("gates")
facts = r["facts"]
cfg = {"min_words": 650, "max_words": 1300}
check("a good article passes every check", db.check_article(good_article(), facts, cfg) == [], )
bad = good_article()
bad["content"]["sections"][0]["paragraphs"].append("Revenue hit ₹9,999 crore last year [F1].")
check("an invented number is caught", any("9999" in p for p in db.check_article(bad, facts, cfg)))
bad = good_article()
bad["content"]["summary"] += " Let's delve into this ever-evolving landscape."
probs = db.check_article(bad, facts, cfg)
check("machine-written phrases are caught", any("delve" in p for p in probs))
bad = good_article()
bad["content"]["sections"][1]["paragraphs"] = ["Short."]
check("a thin article is caught", any("length" in p for p in db.check_article(bad, facts, cfg)))
bad = good_article()
bad["content"]["faq"][0]["a"] += " [F9]"
check("a citation to a fact that doesn't exist is caught", any("F9" in p for p in db.check_article(bad, facts, cfg)))
ok = good_article()
ok["content"]["practice_prompt"] = "Estimate orders for a city of 40 lakh people with 12 dark stores."
check("the practice prompt may use made-up numbers to estimate with", db.check_article(ok, facts, cfg) == [])
art, sources = db.apply_citations(good_article(), facts)
check("[F#] becomes numbered source links", "[1]" in art["content"]["summary"] and "[F1]" not in art["content"]["summary"]
      and sources[0]["url"].startswith("https://"))

# ---------------------------------------------------------------------------- the run
print("the daily run")
env()
check("off by default (ships dormant)", db.config()["enabled"] is False and db.config()["autopublish"] is False)
sb = FakeSupabase(base_tables())
res = db.run_daily(sb, now=NOW, deps=deps(Chat([good_article()])))
page = res["page"]
check("a passing post is a DRAFT while auto-publish is off", res["status"] == "draft" and page["status"] == "draft"
      and page["published_at"] is None)
check("it is a daily post with sources and a related case and guesstimate",
      page["kind"] == "daily" and page["content"]["sources"] and page["content"]["related"]["case"]["id"] == "c1"
      and page["content"]["related"]["guesstimate"]["id"] == "g1")
check("the US bank is never linked", all(v["id"] != "c2" for v in page["content"]["related"].values()))
check("one post per IST day", db.run_daily(sb, now=NOW + timedelta(hours=3), deps=deps(Chat([good_article()])))["status"] == "exists")

env(DAILY_BLOG_ENABLED="1", DAILY_BLOG_AUTOPUBLISH="1")
sb = FakeSupabase(base_tables())
res = db.run_daily(sb, now=NOW, deps=deps(Chat([good_article()])))
check("auto-publish on + every check passed + a good score -> published",
      res["status"] == "published" and res["page"]["published_at"])
sb = FakeSupabase(base_tables())
res = db.run_daily(sb, now=NOW, deps=deps(Chat([good_article()], score=60)))
check("a low critic score keeps it a draft even with auto-publish on", res["status"] == "draft")
fabricated = good_article()
fabricated["content"]["summary"] += " Profits were ₹4,321 crore [F1]."
sb = FakeSupabase(base_tables())
chat = Chat([fabricated, fabricated])
res = db.run_daily(sb, now=NOW, deps=deps(chat))
check("an invented number that survives the repair round stays a draft", res["status"] == "draft"
      and "4321" in res["page"]["quality_notes"])
check("the writer got one repair round", chat.calls.count("seo_writer") == 2)
sb = FakeSupabase(base_tables())
chat = Chat([fabricated, good_article()])
res = db.run_daily(sb, now=NOW, deps=deps(chat))
check("a repaired article can publish", res["status"] == "published")
sb = FakeSupabase(base_tables())
res = db.run_daily(sb, now=NOW, publish=False, deps=deps(Chat([good_article()])))
check("an admin run without publish is always a draft", res["status"] == "draft")
sb = FakeSupabase(base_tables())
res = db.run_daily(sb, now=NOW, dry_run=True, deps=deps(Chat([good_article()])))
check("a dry run saves nothing", res["status"] == "preview" and sb.tables["seo_pages"] == [])
no_facts = dict(deps(Chat([good_article()])), grounded=lambda p: {"text": "", "chunks": [], "supports": []})
res = db.run_daily(FakeSupabase(base_tables()), now=NOW, deps=no_facts)
check("no sourced facts for any topic -> no post at all", res["status"] == "skipped")
env()

print(f"\n{passed} checks passed")
