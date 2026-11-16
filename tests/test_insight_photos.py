"""
The cover photo of a MECE Insights article must not ship empty (services/growth/images.py: find_photo).

    python tests/test_insight_photos.py

Standard library only; no network, no keys. images.py is loaded by path so this does not need the database
settings that importing the `services` package asks for.

What is pinned (all four were live failures on 2026-11-16, when 8 of 32 published articles had no cover):
  1. precise photos found, judge picks none, titles do not overlap the subject  -> a sector photo, not nothing
  2. the judge call crashes                                                     -> still a photo
  3. every precise and sector search comes back empty                           -> a general business photo (hero only)
  4. the shared time budget is already spent                                    -> the hero still gets its search
and that an optional inline slot is still allowed to stay empty, and a good precise match still wins.
"""
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("insight_images", os.path.join(HERE, "..", "services", "growth", "images.py"))
im = importlib.util.module_from_spec(spec)
spec.loader.exec_module(im)

failed = []


def check(name, ok):
    print(("  ok   " if ok else "  FAIL ") + name)
    if not ok:
        failed.append(name)


def cand(title, url, w=2400, h=1500):
    return {"source": "Wikimedia Commons", "title": title, "image_url": url, "thumb_url": url + "?t", "page_url": url,
            "width": w, "height": h, "license": "CC BY-SA 4.0", "license_url": "", "creator": "x"}


HERO = {"subject": "An Indian professional reviewing vertical video on a smartphone in an office",
        "search": ["Google India Hyderabad campus", "Rigi app creators"], "alt": "a", "prompt": "p"}
PRECISE = [cand("Cyber Towers at dusk", "u1"), cand("Hitec City road", "u2")]
SECTOR = [cand("Server racks in a data centre", "s1"), cand("Technology park Bangalore", "s2")]
GENERAL = [cand("Mumbai skyline", "g1")]


def searcher(table, log=None):
    def search(queries, used=None, errors=None, deadline=None, **kw):
        if log is not None:
            log.append((tuple(queries), deadline))
        if deadline and time.time() > deadline:
            if errors is not None:
                errors.append("photo search hit its time budget")
            return []
        for key, rows in table:
            if key(queries):
                return [c for c in rows if c["image_url"] not in (used or set())]
        return []
    return search


is_precise = lambda qs: "Google India Hyderabad campus" in qs  # noqa: E731
is_default = lambda qs: qs and qs[0] in im._DEFAULT_QUERIES  # noqa: E731
is_sector = lambda qs: not is_precise(qs) and not is_default(qs)  # noqa: E731
none = lambda c, want, title: None  # noqa: E731
fetch = lambda url, *a: b"img:" + url.encode()  # noqa: E731


def crash(c, want, title):
    raise RuntimeError("no Gemini vision model answered")


print("cover photo fallbacks")
errs = []
got = im.find_photo(dict(HERO), "Why Google paid ten crore for a fintech app", set(), errors=errs, slug="a", must=True,
                    search=searcher([(is_precise, PRECISE), (is_sector, SECTOR)]), pick=none, fetch=fetch)
check("judge picks none of the precise photos -> a sector photo is used, not nothing",
      bool(got) and got["meta"]["image_url"] in ("s1", "s2") and any("sector photo" in e for e in errs))

errs = []
got = im.find_photo(dict(HERO), "Why Google paid ten crore for a fintech app", set(), errors=errs, slug="a", must=True,
                    search=searcher([(is_precise, PRECISE), (is_sector, SECTOR)]), pick=crash, fetch=fetch)
check("the judge crashing still yields a photo, and the crash is recorded",
      bool(got) and any("judge unavailable (RuntimeError)" in e for e in errs))

errs = []
got = im.find_photo(dict(HERO), "a fintech story", set(), errors=errs, slug="a", must=True,
                    search=searcher([(is_default, GENERAL)]), pick=none, fetch=fetch)
check("nothing precise and nothing for the sector -> the hero takes a general business photo",
      bool(got) and got["meta"]["image_url"] == "g1" and any("general business photo" in e for e in errs))

errs = []
got = im.find_photo(dict(HERO, after_section=2), "a fintech story", set(), errors=errs, slug="a", must=False,
                    search=searcher([(is_default, GENERAL)]), pick=none, fetch=fetch)
check("an inline slot may stay empty: it never takes the general photo",
      got is None and any("no open-licensed photo found" in e for e in errs))

log, errs = [], []
spent = time.time() - 1
got = im.find_photo(dict(HERO), "a fintech story", set(), errors=errs, slug="a", must=True, deadline=spent,
                    search=searcher([(is_precise, PRECISE), (is_sector, SECTOR)], log), pick=none, fetch=fetch)
check("with the shared budget already spent, the hero is still given time to search",
      bool(got) and any(d and d > time.time() for _q, d in log[1:]))
got = im.find_photo(dict(HERO, after_section=1), "a fintech story", set(), errors=[], slug="a", must=False, deadline=spent,
                    search=searcher([(is_precise, PRECISE), (is_sector, SECTOR)]), pick=none, fetch=fetch)
check("an inline slot gets no such extension", got is None)

good = [cand("Google Hyderabad campus building", "p9"), cand("Some other tower", "p8")]
got = im.find_photo(dict(HERO), "t", set(), errors=[], slug="a", must=True,
                    search=searcher([(is_precise, good), (is_sector, SECTOR)]), pick=crash, fetch=fetch)
check("with the judge down, a title matching two words of a precise query beats a sector photo",
      bool(got) and got["meta"]["image_url"] == "p9")
got = im.find_photo(dict(HERO), "t", set(), errors=[], slug="a", must=True,
                    search=searcher([(is_precise, PRECISE), (is_sector, SECTOR)]),
                    pick=lambda c, want, title: c[1], fetch=fetch)
check("when the judge does choose, its choice is used", bool(got) and got["meta"]["image_url"] == "u2")

used = set()
a = im.find_photo(dict(HERO), "a fintech story", used, errors=[], slug="a", must=True,
                  search=searcher([(is_sector, SECTOR)]), pick=none, fetch=fetch)
b = im.find_photo(dict(HERO), "a fintech story", used, errors=[], slug="a", must=True,
                  search=searcher([(is_sector, SECTOR)]), pick=none, fetch=fetch)
check("two slots never take the same photo", a and b and a["meta"]["image_url"] != b["meta"]["image_url"])

portrait = [cand("Google Hyderabad campus building", "tall", w=1000, h=1600)]
got = im.find_photo(dict(HERO), "a fintech story", set(), errors=[], slug="a", must=True,
                    search=searcher([(is_precise, portrait), (is_sector, SECTOR)]), pick=crash, fetch=fetch)
check("a portrait-shaped photo is never taken on the judge-free path", bool(got) and got["meta"]["image_url"] != "tall")

for title, expect in (("Why Google paid ten crore rupees for a sliver of Rigi", "technology park Bangalore"),
                      ("Air India's widebody ambitions and cost strategy", "Air India aircraft"),
                      ("India's bold move banning black mass exports", None)):
    item = {"subject": "A creator filming vertical video for a digital app"} if "Google" in title else {"subject": ""}
    w = im.widen_queries(item, title)
    check(f"sector scenes for: {title[:40]}", w and w != im._DEFAULT_QUERIES and (expect is None or expect in w))

print()
if failed:
    print(f"{len(failed)} FAILED")
    sys.exit(1)
print("all passed")
