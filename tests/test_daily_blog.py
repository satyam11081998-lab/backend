"""
Daily blog (services/growth/daily_blog.py + telegram_review.py). Standard library only:

    python -m tests.test_daily_blog

No network, no keys: research, the writer, the critic and Telegram are fakes, and a FakeSupabase
stands in for PostgREST. Pinned here: which topics are allowed; that research keeps facts however
the model formats them and links them to the pages the search read (supports, publisher, figures on
the page); that an invented number is caught and repaired; that the run keeps going until a topic
works; one post per IST day; and the Telegram review loop (publish / another / reject, admin chat
only, secret token).
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
from services.growth import telegram_review as tg  # noqa: E402

NOW = datetime(2026, 10, 6, 1, 30, tzinfo=timezone.utc)  # 07:00 IST


# ---------------------------------------------------------------------------- fakes
class _Q:
    def __init__(self, sb, name):
        self.sb, self.name, self.filters, self._limit, self._insert, self._update = sb, name, [], None, None, None

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

    def update(self, patch):
        self._update = patch
        return self

    def execute(self):
        rows = self.sb.tables.setdefault(self.name, [])
        if self._insert is not None:
            row = dict(self._insert, id=f"id{len(rows) + 1}", created_at=self.sb.now.isoformat())
            rows.append(row)
            return SimpleNamespace(data=[row])
        out = [r for r in rows if all(f(r) for f in self.filters)]
        if self._update is not None:
            for r in out:
                r.update(json.loads(json.dumps(self._update)))
            return SimpleNamespace(data=out)
        out = sorted(out, key=lambda r: str(r.get("created_at") or ""), reverse=True)
        return SimpleNamespace(data=[dict(r) for r in (out[: self._limit] if self._limit else out)])


class FakeSupabase:
    def __init__(self, tables, now=NOW):
        self.tables, self.now = tables, now

    def table(self, name):
        return _Q(self, name)


def headline(i, title, category="business", gd=7, hours_ago=6, desc="", star=False):
    return {"id": f"h{i}", "title": title, "description": desc, "category": category, "source_name": "Business Daily",
            "source_url": f"https://news.example/{i}", "gd_worthiness_score": gd, "is_star": star,
            "published_at": (NOW - timedelta(hours=hours_ago)).isoformat(), "keywords": []}


# What Gemini really sends back: markdown bullets and bold, a preface line, a fact whose support
# segment is only part of the sentence, one matched only by its publisher, one only by its figure on
# the page, one political, one unsourced.
GEMINI_TEXT = """Here are the facts I found:
* **FACT:** Quick-commerce orders in India grew 74% in FY25, according to the industry body. | SOURCE: Redseer
1. FACT: The three largest quick-commerce apps had about 5,000 dark stores in March 2026. | SOURCE: Economic Times
- FACT - The average order value on quick-commerce apps was ₹520 in 2025. | SOURCE: Mint
FACT: Dark stores typically break even at about 1,300 orders a day, a 2025 brokerage note estimated. | SOURCE: a brokerage
FACT: Prime ministers met to discuss trade. | SOURCE: Reuters
FACT: Quick commerce now accounts for two-thirds of e-grocery orders in metro cities. | SOURCE: an analyst"""


def fake_grounded(_prompt, model="gemini-test"):
    return {"text": GEMINI_TEXT, "model": model,
            "chunks": [{"uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc", "title": "redseer.com", "domain": "redseer.com"},
                       {"uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/def", "title": "economictimes.indiatimes.com", "domain": "economictimes.indiatimes.com"},
                       {"uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/ghi", "title": "livemint.com", "domain": "livemint.com"},
                       {"uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/jkl", "title": "broker.example", "domain": "broker.example"}],
            "supports": [{"text": "Quick-commerce orders in India grew 74% in FY25", "chunks": [0]},
                         {"text": "Prime ministers met to discuss trade.", "chunks": [1]}],
            "usage": {"prompt": 100, "completion": 300}}


def fake_resolve(uri):
    return {"abc": "https://redseer.com/report", "def": "https://economictimes.indiatimes.com/q-commerce",
            "ghi": "https://livemint.com/aov", "jkl": "https://broker.example/note"}[uri.rsplit("/", 1)[-1]]


def fake_fetch(url):
    return "Our note: a dark store breaks even at 1,300 orders per day." if "broker" in url else ""


def para(n_words, cite):
    words = ("Quick commerce changed how Indian cities buy groceries because people will pay for speed when the "
             "basket is small and the need is urgent and the store is close").split()
    return " ".join((words * (n_words // len(words) + 1))[:n_words]) + f". {cite}"


def good_article(**over):
    a = {
        "title": "How quick commerce makes money in India",
        "meta_description": "Orders grew 74% in FY25. How dark stores and order values decide who profits.",
        "dek": "The economics of ten-minute delivery, and how it comes up in GDs and interviews.",
        "keywords": ["quick commerce india", "dark stores", "unit economics"],
        "content": {
            "key_points": ["Orders grew 74% in FY25 [F2].", "About 5,000 dark stores run today [F3].",
                           "Average orders reached ₹520 [F4]."],
            "lede": "India's quick-commerce apps grew orders 74% in FY25 [F2], and the race has moved from opening "
                    "stores to filling them, with about 5,000 dark stores now running [F3].",
            "sections": [{"heading": "What happened?", "paragraphs": [para(150, "[F2]"),
                                                                      "The apps keep adding dark stores [F1]."]},
                         {"heading": "Why does it matter?", "paragraphs": [para(150, "[F3]")]},
                         {"heading": "How do the economics work?", "paragraphs": [para(170, "[F4]")]},
                         {"heading": "Where is the debate?", "paragraphs": [para(150, "[F5]")]}],
            "numbers": [{"figure": "74%", "what": "order growth in FY25", "fact": "F2"},
                        {"figure": "5,000", "what": "dark stores", "fact": "F3"},
                        {"figure": "₹520", "what": "average order value", "fact": "F4"},
                        {"figure": "1,300", "what": "orders a day to break even", "fact": "F5"}],
            "framework": {"name": "Profit tree", "heading": "How to break it down",
                          "steps": ["Revenue per order", "Cost per order", "Orders per store per day"]},
            "what_to_watch": ["Do order values keep rising?", "Do stores reach break-even?", "Does density improve?"],
            "aspirants": {"gd_topic": "Is quick commerce good for kirana stores?",
                          "for": ["Reach", "Jobs"], "against": ["Margins", "Kirana pressure"],
                          "pi_questions": ["How many orders does a dark store need a day to break even?",
                                           "Would you raise basket size or cut delivery cost first?",
                                           "Estimate the market in a city of 50 lakh."],
                          "wat_prompt": "Ten-minute delivery: convenience or a race to the bottom?",
                          "case_question": "A grocery app asks whether to open 40 more stores in a city."},
            "topic": "Operations",
            "pull_quote": "The race has moved from opening stores to filling them [F3].",
            "art": {"hero": {"prompt": "A dark store at night in Bengaluru, shelves lit, a rider waiting outside",
                             "alt": "A lit dark store at night", "caption": "Where ten-minute orders are packed"},
                    "inline": [{"after_section": 1, "prompt": "Crates of vegetables in a warehouse aisle",
                                "alt": "Warehouse aisle", "caption": "Stock close to the customer"}]},
            "faq": [{"q": "Is quick commerce profitable?", "a": para(40, "[F5]")},
                    {"q": "How many dark stores are there?", "a": "About 5,000 across the three largest apps [F3]."},
                    {"q": "What drives profit?", "a": para(40, "[F4]")}],
        },
    }
    a.update(over)
    return a


class Chat:
    """Fake chat_with_fallback: editor -> writer (scripted drafts) -> critic."""

    def __init__(self, drafts, score=86, fail_writes=0):
        self.drafts, self.score, self.calls, self.fail_writes = list(drafts), score, [], fail_writes

    def __call__(self, feature, messages, **_kw):
        system = messages[0]["content"]
        self.calls.append(feature)
        if "editor of MECE Insights" in system:
            body = {"order": [0], "angle": "How quick commerce makes money in India", "why": "broad"}
        elif feature == "seo_writer":
            if self.fail_writes:
                self.fail_writes -= 1
                raise RuntimeError("model overloaded")
            body = self.drafts.pop(0) if len(self.drafts) > 1 else self.drafts[0]
        else:
            body = {"score": self.score, "publishable": self.score >= 80, "notes": "clear and grounded"}
        msg = SimpleNamespace(content=json.dumps(body))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None, id=None), "fake-model", "fake"


class Review:
    def __init__(self):
        self.sent, self.texts = [], []

    def send_for_review(self, _sb, page):
        self.sent.append(page)
        return True

    def send_text(self, text, **_k):
        self.texts.append(text)
        return 1


made_images = []


def fake_images(_sb, slug, art):
    made_images.append((slug, art))
    return {"hero": {"url": f"https://cdn.example/{slug}/hero.webp", "og_url": f"https://cdn.example/{slug}/og.jpg",
                     "width": 2000, "height": 1125, "alt": art["hero"]["alt"], "caption": art["hero"]["caption"],
                     "credit": "Image generated with Gemini for MECE Insights"},
            "images": [{"url": f"https://cdn.example/{slug}/inline-1.webp", "after_section": 1, "alt": "a", "caption": "c",
                        "credit": "Image generated with Gemini for MECE Insights"}], "errors": []}


def deps(chat, review=None, grounded=fake_grounded):
    return {"chat": chat, "grounded": grounded, "resolve": fake_resolve, "fetch": fake_fetch,
            "log": lambda **k: None, "review": review, "images": fake_images}


def base_tables():
    return {
        "news_headlines": [
            headline(1, "Quick-commerce apps add dark stores as order values climb",
                     desc="Delivery apps expand their dark-store networks.", gd=8),
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
    for k in ("DAILY_BLOG_ENABLED", "DAILY_BLOG_AUTOPUBLISH", "DAILY_BLOG_MIN_SCORE", "DAILY_BLOG_WRITER_MODEL",
              "TELEGRAM_BOT_TOKEN", "TELEGRAM_ADMIN_CHAT_ID", "TELEGRAM_CHAT_ID", "RENDER_EXTERNAL_URL",
              "DAILY_BLOG_RESEARCH_MODEL", "GEMINI_MODEL"):
        os.environ.pop(k, None)
    for k, v in kv.items():
        os.environ[k] = v


env()
# ---------------------------------------------------------------------------- topics
print("topic selection")
sb = FakeSupabase(base_tables())
cands = db.candidate_topics(sb, NOW)
titles = [c["headline"]["title"] for c in cands]
check("a business story is a candidate", "Quick-commerce apps add dark stores as order values climb" in titles)
check("politics, tragedy and 'other' are never candidates",
      not any(("elections" in t) or ("killed" in t) or ("Cricket" in t) for t in titles))
check("'price war' is business, 'trade war' is not", not db.is_blocked("Telecom price war returns")
      and db.is_blocked("Trade war fears hit exporters"))
check("company names are not caught by the blocklist", not db.is_blocked("Hindustan Unilever raises prices")
      and not db.is_blocked("RBI Governor holds the repo rate"))
check("three evergreen back-ups are always available", len(db.evergreen_topics(FakeSupabase({"seo_pages": []}), NOW, 3)) == 3)
later = NOW + timedelta(hours=2)
check("a later run of the day tries different evergreen topics",
      db.evergreen_topics(FakeSupabase({"seo_pages": []}), NOW, 3)[0]["angle"]
      != db.evergreen_topics(FakeSupabase({"seo_pages": []}), later, 3)[0]["angle"])

# ---------------------------------------------------------------------------- research
print("research")
parsed = db.parse_fact_lines(GEMINI_TEXT)
check("facts are read however the model formats them (bullets, bold, numbering, dashes)", len(parsed) == 6
      and parsed[0]["text"].startswith("Quick-commerce orders") and parsed[0]["source"] == "Redseer")
r = db.research("How quick commerce makes money", headline(1, "Quick-commerce apps add dark stores",
                desc="Delivery apps expand."), grounded=fake_grounded, resolve=fake_resolve, fetch=fake_fetch)
by_how = {}
for f in r["facts"]:
    by_how.setdefault(f["how"], []).append(f)
check("the news article itself is a linked fact", by_how.get("news") and by_how["news"][0]["sources"][0]["url"].startswith("https://news.example"))
check("a fact backed by part of a sentence is linked (search support)", any("74%" in f["text"] for f in by_how.get("support", [])))
check("a fact is linked by the publisher it names (Economic Times, Mint)",
      sum(1 for f in by_how.get("publisher", [])) >= 2)
check("a fact is linked when its figure is printed on a page the search read",
      any("1,300" in f["text"] for f in by_how.get("page", [])))
check("political facts are dropped", not any("Prime ministers" in f["text"] for f in r["facts"]))
check("a fact with only a vague named source is kept as attributed, not linked",
      any((not f["linked"]) and "two-thirds" in f["text"] for f in r["facts"]))
check("enough linked facts to write", r["linked"] >= 5)
check("redirect links are resolved to the real page",
      any(s["url"] == "https://economictimes.indiatimes.com/q-commerce" for f in r["facts"] for s in f["sources"]))
calls = []


def flaky(prompt, model):
    calls.append(model)
    if model != "gemini-3.6-flash":
        raise RuntimeError("404 NOT_FOUND. This model is no longer available to new users")
    return fake_grounded(prompt, model)


env(GEMINI_MODEL="gemini-9-pro")
db._dead_models.clear()
r2 = db.research("x", grounded=flaky, resolve=fake_resolve, fetch=fake_fetch)
check("research skips a retired model and uses one that works",
      r2["linked"] >= 3 and calls[:2] == ["gemini-9-pro", "gemini-3.6-flash"] and r2["engines"] == ["gemini:gemini-3.6-flash"])
calls.clear()
db.research("y", grounded=flaky, resolve=fake_resolve, fetch=fake_fetch)
check("a retired model is remembered: it costs one failed call, not one per topic", "gemini-9-pro" not in calls)
env()
db._dead_models.clear()

# every Gemini model gone (as on 2026-10-06): ask the API which exist, then fall back to OpenAI web search
db._discovered.update(at=10 ** 12, models=["gemini-4.0-flash"])
seen_models = []


def all_gone(prompt, model):
    seen_models.append(model)
    raise RuntimeError("404 NOT_FOUND: model no longer available")


OPENAI_TEXT = ("FACT: UPI processed 20.6 billion transactions in August 2026, NPCI data show. | SOURCE: NPCI "
               "([npci.org.in](https://www.npci.org.in/stats?utm_source=openai))\n"
               "FACT: About 69 million Indians held mutual fund investments in 2026. | SOURCE: AMFI "
               "([amfiindia.com](https://www.amfiindia.com/data?utm_source=openai))\n"
               "FACT: Around 50 million people bought shares directly in 2026. | SOURCE: Mint "
               "([livemint.com](https://www.livemint.com/x?utm_source=openai))")


def openai_search(prompt):
    chunks = [{"uri": "https://www.npci.org.in/stats", "title": "npci.org.in", "domain": "npci.org.in"},
              {"uri": "https://www.amfiindia.com/data", "title": "amfiindia.com", "domain": "amfiindia.com"},
              {"uri": "https://www.livemint.com/x", "title": "livemint.com", "domain": "livemint.com"}]
    supports = [{"text": db._strip_links(ln), "chunks": [i]} for i, ln in enumerate(OPENAI_TEXT.split("\n"))]
    return {"text": OPENAI_TEXT, "chunks": chunks, "supports": supports, "usage": {}, "engine": "openai:gpt-4.1"}


r3 = db.research("Why are millions of UPI users not investing?", grounded=all_gone, resolve=lambda u: u,
                 fetch=fake_fetch, web_search=openai_search)
check("when no Gemini model works, OpenAI web search does the research",
      r3["linked"] >= 3 and r3["engines"] == ["openai:gpt-4.1"] and "gemini-4.0-flash" in seen_models)
check("inline citation links are stripped from the facts",
      all("](" not in f["text"] and "utm_source" not in f["text"] for f in r3["facts"]))
db._discovered.update(at=0.0, models=[])
db._dead_models.clear()
db._discovered.update(at=10 ** 12, models=[])
check("a broken research call gives no facts, not an error",
      db.research("x", grounded=lambda p, m: (_ for _ in ()).throw(RuntimeError("quota")), resolve=fake_resolve,
                  fetch=fake_fetch)["facts"] == [])
db._discovered.update(at=0.0, models=[])
db._dead_models.clear()

# ---------------------------------------------------------------------------- checks
print("checks")
facts = r["facts"]
ids = {f["id"]: f["text"] for f in facts}
# renumber the article's citations to whatever ids research produced for these figures
fid = {k: next(i for i, t in ids.items() if k in t) for k in ("74%", "5,000", "₹520", "1,300")}


def renumber(a):
    s = json.dumps(a, ensure_ascii=False)
    for old, key in (("F2", "74%"), ("F3", "5,000"), ("F4", "₹520"), ("F5", "1,300")):
        s = s.replace(f'"{old}"', f'"__{fid[key]}"').replace(f"[{old}]", f"[__{fid[key]}]")
    return json.loads(s.replace("__F", "F"))


cfg = dict(db.config())
GOOD = renumber(good_article())
check("a good article passes every check", db.check_article(GOOD, facts, cfg) == [])
bad = renumber(good_article())
bad["content"]["sections"][0]["paragraphs"].append(f"Revenue hit ₹9,999 crore last year [{fid['74%']}].")
check("an invented number is caught", any("9999" in p for p in db.check_article(bad, facts, cfg)))
bad = renumber(good_article())
bad["content"]["lede"] += " Let's delve into this ever-evolving landscape."
check("machine-written phrases are caught", any("delve" in p for p in db.check_article(bad, facts, cfg)))
bad = renumber(good_article())
bad["content"]["sections"] = bad["content"]["sections"][:2]
check("a thin article is caught", any("sections" in p or "length" in p for p in db.check_article(bad, facts, cfg)))
bad = renumber(good_article())
del bad["content"]["aspirants"]
check("the MBA aspirant section is required", any("aspirants" in p for p in db.check_article(bad, facts, cfg)))
art, sources = db.apply_citations(GOOD, facts)
check("[F#] becomes numbered source links", "[F" not in json.dumps(art) and sources and sources[0]["n"] == 1)

# ---------------------------------------------------------------------------- the run
print("the daily run")
check("off by default", db.config()["enabled"] is False and db.config()["autopublish"] is False)
sb = FakeSupabase(base_tables())
rev = Review()
res = db.run_daily(sb, now=NOW, deps=deps(Chat([GOOD]), rev))
page = res["page"]
check("a good post is saved as a draft and sent to Telegram for review",
      res["status"] == "draft" and page["status"] == "draft" and rev.sent and rev.sent[0]["id"] == page["id"])
check("it is the new format, with sources and a related case and guesstimate",
      page["content"]["format"] == "daily-2" and page["content"]["sources"]
      and page["content"]["related"]["case"]["id"] == "c1" and page["content"]["related"]["guesstimate"]["id"] == "g1")
check("one post per IST day", db.run_daily(sb, now=NOW + timedelta(hours=3), deps=deps(Chat([GOOD]), rev))["status"] == "exists")
check("every post gets Gemini pictures from the writer's art direction (hero and inline)",
      page["content"]["hero"]["url"].endswith("/hero.webp") and page["content"]["images"][0]["after_section"] == 1
      and made_images and made_images[-1][0] == page["slug"] and "art" not in page["content"]
      and page["agent_meta"]["art"]["hero"]["prompt"].startswith("A dark store"))
check("topic label and pull quote are kept, without citation markers",
      page["content"]["topic_label"] == "Operations" and "[" not in page["content"]["pull_quote"])
n_before = len(made_images)
db.run_daily(FakeSupabase(base_tables()), now=NOW, dry_run=True, deps=deps(Chat([GOOD]), Review()))
check("a dry run commissions no pictures", len(made_images) == n_before)

print("images")
from services.growth import images as im  # noqa: E402
import io as _io  # noqa: E402
from PIL import Image as _Image  # noqa: E402


def png(w=2400, h=1350):
    b = _io.BytesIO()
    _Image.new("RGB", (w, h), (120, 90, 60)).save(b, format="PNG")
    return b.getvalue()


uploads, prompts = [], []


def fake_gen(prompt, aspect):
    prompts.append((prompt, aspect))
    return {"bytes": png(), "mime": "image/png", "model": "gemini-x-image"}


def fake_upload(_sb, path, data, ctype):
    uploads.append((path, len(data), ctype))
    return f"https://store.example/{path}"


art = {"hero": {"prompt": "Scene A", "alt": "A", "caption": "Cap A"},
       "inline": [{"after_section": 3, "prompt": "Scene C", "alt": "C", "caption": "Cap C"},
                  {"after_section": 1, "prompt": "Scene B", "alt": "B", "caption": "Cap B"}]}
out = im.make_images(None, "my-post", art, generate=fake_gen, upload=fake_upload)
check("hero (16:9) and inline (3:2) pictures are generated in one house style",
      out["hero"]["width"] == 2000 and out["hero"]["height"] == 1125 and len(out["images"]) == 2
      and all("No text" in p or "no text" in p for p, _ in prompts) and {a for _, a in prompts} == {"16:9", "3:2"})
check("pictures are stored as WebP for the page and a JPEG link preview for the hero",
      any(p.startswith("my-post/hero-") and t in ("image/webp", "image/jpeg") for p, _, t in uploads)
      and any(p.startswith("my-post/og-") and t == "image/jpeg" for p, _, t in uploads)
      and out["hero"]["og_url"].startswith("https://store.example/my-post/og-"))
check("inline pictures keep their place in the article", [i["after_section"] for i in out["images"]] == [1, 3])
broken = im.make_images(None, "p", art, generate=lambda p, a: (_ for _ in ()).throw(RuntimeError("quota")),
                        upload=fake_upload)
check("a failed picture never breaks the article", broken.get("hero") is None and broken["errors"])


class _ArtChat:
    def __call__(self, feature, **_k):
        msg = SimpleNamespace(content=json.dumps(art))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)]), None, None


old_post = {"id": "p-old", "slug": "old-post", "title": "How UPI makes money", "dek": "d", "status": "published",
            "content": {"format": "daily-1", "summary": "s", "sections": [{"heading": "H", "paragraphs": ["p"]}]},
            "agent_meta": {"domain": "finance"}}
sb = FakeSupabase({"seo_pages": [json.loads(json.dumps(old_post))]})
res = db.add_images(sb, "p-old", chat=_ArtChat(),
                    make=lambda s, slug, a: im.make_images(s, slug, a, generate=fake_gen, upload=fake_upload))
row = sb.tables["seo_pages"][0]
check("pictures can be added to an older post (art planned from the article)",
      res["ok"] and row["content"]["hero"]["url"].startswith("https://store.example/old-post/hero-")
      and len(row["content"]["images"]) == 2 and row["agent_meta"]["art"]["hero"]["prompt"] == "Scene A")
check("an older post gets its topic label for the magazine layout", row["content"]["topic_label"] == "Finance")
check("adding pictures to a missing post says so", db.add_images(sb, "nope", chat=_ArtChat())["ok"] is False)

env(DAILY_BLOG_AUTOPUBLISH="1")
sb = FakeSupabase(base_tables())
res = db.run_daily(sb, now=NOW, deps=deps(Chat([GOOD]), Review()))
check("auto-publish on + every check passed + a good score -> published", res["status"] == "published")
sb = FakeSupabase(base_tables())
res = db.run_daily(sb, now=NOW, deps=deps(Chat([GOOD], score=70), Review()))
check("a low score keeps it for review even with auto-publish on", res["status"] == "draft")
env()

fabricated = renumber(good_article())
fabricated["content"]["lede"] += f" Profits were ₹4,321 crore [{fid['74%']}]."
chat = Chat([fabricated, fabricated, fabricated])
sb = FakeSupabase(base_tables())
res = db.run_daily(sb, now=NOW, deps=deps(chat, Review()))
check("two repair rounds, then the reviewer sees the problem", chat.calls.count("seo_writer") == 3
      and "4321" in res["page"]["quality_notes"] and res["page"]["agent_meta"]["serious_problems"])
chat = Chat([fabricated, GOOD])
sb = FakeSupabase(base_tables())
res = db.run_daily(sb, now=NOW, deps=deps(chat, Review()))
check("a repaired article comes out clean", res["page"]["agent_meta"]["problems"] == [])

thin = {"text": "", "chunks": [], "supports": []}
seen = []


def thin_then_good(prompt, model):
    seen.append(prompt)
    return thin if len(seen) <= 4 else fake_grounded(prompt, model)  # the first topic finds nothing


sb = FakeSupabase(base_tables())
res = db.run_daily(sb, now=NOW, deps=deps(Chat([GOOD]), Review(), grounded=thin_then_good))
check("it keeps going to the next topic until one has sourced facts",
      res["status"] == "draft" and len(res["trace"]["research"]) >= 2)
sb = FakeSupabase(base_tables())
res = db.run_daily(sb, now=NOW, deps=deps(Chat([GOOD], fail_writes=1), Review()))
check("a writer failure moves on to the next topic instead of giving up", res["status"] == "draft")
rev = Review()
res = db.run_daily(FakeSupabase(base_tables()), now=NOW, notify_failure=True,
                   deps=deps(Chat([GOOD]), rev, grounded=lambda p, m: thin))
check("nothing anywhere -> no post, and the late run says so on Telegram", res["status"] == "skipped" and rev.texts)
sb = FakeSupabase(base_tables())
res = db.run_daily(sb, now=NOW, dry_run=True, deps=deps(Chat([GOOD]), Review()))
check("a dry run saves nothing", res["status"] == "preview" and sb.tables["seo_pages"] == [])

# ---------------------------------------------------------------------------- Telegram review
print("telegram review")
env(TELEGRAM_BOT_TOKEN="123:abc", TELEGRAM_ADMIN_CHAT_ID="777", RENDER_EXTERNAL_URL="https://mece-api.onrender.com")
api = []


def fake_call(method, payload=None, timeout=10.0):
    api.append((method, payload or {}))
    if method == "getWebhookInfo":
        return {"ok": True, "result": {"url": ""}}
    if method == "sendMessage":
        return {"ok": True, "result": {"message_id": 100 + len(api)}}
    return {"ok": True, "result": True}


tg._call = fake_call
tg._webhook_ok["url"] = ""
sb = FakeSupabase(base_tables())
res = db.run_daily(sb, now=NOW, deps=dict(deps(Chat([GOOD]), tg)))
page = sb.tables["seo_pages"][0]
sent = [p for m, p in api if m == "sendMessage"]
check("the webhook registers itself with a secret token",
      any(m == "setWebhook" and p["url"] == "https://mece-api.onrender.com/seo/telegram/webhook"
          and p["secret_token"] == tg.webhook_secret() for m, p in api))
check("the whole draft goes to the admin chat, each message under Telegram's limit",
      len(sent) >= 3 and all(p["chat_id"] == "777" and len(p["text"]) <= 4000 for p in sent)
      and "Reply <b>publish</b>" in sent[-1]["text"])
ids = page["agent_meta"]["telegram"]["message_ids"]
check("the message ids are remembered on the draft", len(ids) == len(sent))

started = []


def run_another(exclude):
    started.append(exclude)


r = tg.handle_update(sb, {"update_id": 1, "message": {"message_id": 5, "chat": {"id": 999}, "text": "publish"}},
                     run_another=run_another)
check("a stranger's chat is ignored", r.get("ignored") and page["status"] == "draft")
r = tg.handle_update(sb, {"update_id": 2, "message": {"message_id": 6, "chat": {"id": 777}, "text": "ok"}},
                     run_another=run_another)
check("a bare 'ok' does not publish (it gets the help text)", r.get("action") == "help" and page["status"] == "draft")
r = tg.handle_update(sb, {"update_id": 3, "message": {"message_id": 7, "chat": {"id": 777}, "text": "/publish@MeceBot",
                                                      "reply_to_message": {"message_id": ids[0]}}},
                     run_another=run_another)
check("replying 'publish' publishes that draft", r.get("action") == "published" and page["status"] == "published"
      and page["published_at"] and page["agent_meta"]["review"] == "published"
      and page["agent_meta"]["reviewed_via"] == "telegram")
check("and the live link comes back", "mece.in/insights/" + page["slug"] in [p for m, p in api if m == "sendMessage"][-1]["text"])
r = tg.handle_update(sb, {"update_id": 3, "message": {"message_id": 7, "chat": {"id": 777}, "text": "publish"}},
                     run_another=run_another)
check("Telegram re-sending the same update does nothing twice", r.get("duplicate"))

sb = FakeSupabase(base_tables())
api.clear()
db.run_daily(sb, now=NOW, deps=dict(deps(Chat([GOOD]), tg)))
draft = sb.tables["seo_pages"][0]
r = tg.handle_update(sb, {"update_id": 10, "message": {"message_id": 8, "chat": {"id": 777}, "text": "Another one 🙏"}},
                     run_another=run_another)
import time as _time  # noqa: E402
_time.sleep(0.05)
check("'another' drops the draft and starts a new one", r.get("action") == "another" and draft["status"] == "rejected"
      and started and draft["agent_meta"]["review"] == "replaced")
db.run_daily(sb, now=NOW, force=True, deps=dict(deps(Chat([GOOD]), tg)))
second = sb.tables["seo_pages"][-1]
r = tg.handle_update(sb, {"update_id": 11, "message": {"message_id": 9, "chat": {"id": 777}, "text": "reject"}},
                     run_another=run_another)
check("'reject' drops the newest waiting draft", r.get("action") == "rejected" and second["status"] == "rejected")
check("commands are read loosely but safely",
      tg.parse_command("Publish!") == "publish" and tg.parse_command("approve it") == "publish"
      and tg.parse_command("yes") == "" and tg.parse_command("/another") == "another")
env()

print(f"\n{passed} checks passed")
