#!/usr/bin/env python3
"""
gd_feed_probe.py — settle the GD Dossier source roster empirically.

WHY THIS EXISTS
---------------
The source list for the news pipeline must not be a hardcoded Python constant
guessed by whoever wrote it (today: `news_fetcher.INDIAN_DOMAINS`). Which feeds
are reachable depends on the IP doing the fetching — a publisher that answers
from a laptop may 403 a datacentre, and the reverse happens too.

So: run this ON RENDER (render shell, or a one-off job), not on your laptop.
What matters is what the production egress can reach.

WHAT IT DOES
------------
For every candidate feed it reports:
  HTTP status · item count · % of items carrying a summary · median item age
  · whether the feed looks like an aggregator (many outlets) or a single outlet
and then prints a ready-to-paste `news_sources` seed INSERT containing only
the feeds that passed.

Standard library only — no pip install, so it runs in any shell you can get.

USAGE
-----
  python scripts/gd_feed_probe.py                 # probe the built-in roster
  python scripts/gd_feed_probe.py --extra urls.txt  # one URL per line, added
  python scripts/gd_feed_probe.py --json out.json   # machine-readable too
  python scripts/gd_feed_probe.py --timeout 25

PASS CRITERIA (tune at the top of the file)
  >= MIN_ITEMS items, >= MIN_SUMMARY_PCT of them with a summary,
  and a median age under MAX_MEDIAN_AGE_DAYS.
"""

from __future__ import annotations

import argparse
import json
import re
import ssl
import statistics
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime

# ---------------------------------------------------------------- pass criteria
MIN_ITEMS = 8
MIN_SUMMARY_PCT = 50.0
MAX_MEDIAN_AGE_DAYS = 10.0

UA = ("Mozilla/5.0 (compatible; ConsilioNewsBot/1.0; "
      "+https://consilio.app) feed-probe")

# ---------------------------------------------------------------- the roster
# kind:   primary  = a regulator/ministry publishing its own documents
#         press    = a newsroom
#         global   = multilateral / international desk
# weight: used by story_score.py for source breadth. Primary docs are worth more
#         because a figure found in one is citable without hedging.
CANDIDATES = [
    # --- primary sources: the reason the fact table can say "verified" ---
    ("RBI press releases",        "primary", 1.5, "https://www.rbi.org.in/pressreleases_rss.xml"),
    ("RBI notifications",         "primary", 1.5, "https://www.rbi.org.in/notifications_rss.xml"),
    ("RBI speeches",              "primary", 1.3, "https://www.rbi.org.in/Speeches_rss.xml"),
    ("SEBI",                      "primary", 1.4, "https://www.sebi.gov.in/sebirss.xml"),
    ("PIB releases",              "primary", 1.4, "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3"),
    ("PIB economic affairs",      "primary", 1.4, "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3"),
    ("TRAI press releases",       "primary", 1.3, "https://www.trai.gov.in/rss.xml"),
    ("PRS Legislative",           "primary", 1.3, "https://prsindia.org/rss.xml"),

    # --- national business press ---
    ("Business Standard · economy",  "press", 1.0, "https://www.business-standard.com/rss/economy-policy-102.rss"),
    ("Business Standard · companies","press", 1.0, "https://www.business-standard.com/rss/companies-101.rss"),
    ("Business Standard · markets",  "press", 0.9, "https://www.business-standard.com/rss/markets-106.rss"),
    ("Business Standard · opinion",  "press", 0.9, "https://www.business-standard.com/rss/opinion-105.rss"),
    ("The Hindu BusinessLine · economy", "press", 1.0, "https://www.thehindubusinessline.com/economy/feeder/default.rss"),
    ("The Hindu BusinessLine · companies","press", 1.0, "https://www.thehindubusinessline.com/companies/feeder/default.rss"),
    ("The Hindu · business",         "press", 1.0, "https://www.thehindu.com/business/feeder/default.rss"),
    ("Mint · economy",               "press", 1.0, "https://www.livemint.com/rss/economy"),
    ("Mint · companies",             "press", 1.0, "https://www.livemint.com/rss/companies"),
    ("Mint · markets",               "press", 0.9, "https://www.livemint.com/rss/markets"),
    ("Moneycontrol · economy",       "press", 0.9, "https://www.moneycontrol.com/rss/economy.xml"),
    ("Moneycontrol · business",      "press", 0.9, "https://www.moneycontrol.com/rss/business.xml"),
    ("Economic Times · top",         "press", 1.0, "https://economictimes.indiatimes.com/rssfeedstopstories.cms"),
    ("Economic Times · industry",    "press", 1.0, "https://economictimes.indiatimes.com/industry/rssfeeds/13352306.cms"),
    ("Financial Express · economy",  "press", 0.9, "https://www.financialexpress.com/about/economy/feed/"),
    ("Financial Express · industry", "press", 0.9, "https://www.financialexpress.com/about/industry/feed/"),
    ("NDTV Profit",                  "press", 0.9, "https://feeds.feedburner.com/ndtvprofit-latest"),
    ("Hindustan Times · business",   "press", 0.8, "https://www.hindustantimes.com/feeds/rss/business/rssfeed.xml"),

    # --- international desks: the "India in the world" framing panels like ---
    ("Reuters · business",           "global", 1.1, "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best"),
    ("World Bank blogs",             "global", 1.1, "https://blogs.worldbank.org/en/rss.xml"),
    ("IMF blog",                     "global", 1.1, "https://www.imf.org/en/Blogs/rss"),
    ("BIS speeches",                 "global", 1.0, "https://www.bis.org/doclist/cbspeeches.rss"),
]

NS = {
    "content": "http://purl.org/rss/1.0/modules/content/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "atom": "http://www.w3.org/2005/Atom",
}


def _fetch(url: str, timeout: float):
    """Return (status, body_bytes, error_str)."""
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
        "Accept-Language": "en-IN,en;q=0.9",
    })
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return r.getcode(), r.read(), None
    except urllib.error.HTTPError as e:
        return e.code, b"", f"HTTP {e.code}"
    except Exception as e:                                  # noqa: BLE001
        return 0, b"", f"{type(e).__name__}: {e}"


def _text(el):
    return (el.text or "").strip() if el is not None else ""


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", " ", s or "").strip()


def _parse_date(raw: str):
    if not raw:
        return None
    raw = raw.strip()
    try:
        d = parsedate_to_datetime(raw)
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc)
    except Exception:                                       # noqa: BLE001
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S", "%d %b, %Y", "%Y-%m-%d"):
        try:
            d = datetime.strptime(raw.replace("Z", "+0000"), fmt)
            return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
        except Exception:                                   # noqa: BLE001
            continue
    return None


def _parse_feed(body: bytes) -> dict:
    """Handle RSS 2.0 and Atom without a dependency."""
    try:
        root = ET.fromstring(body)
    except ET.ParseError as e:
        return {"ok": False, "error": f"XML parse error: {e}"}

    items = root.findall(".//item")
    kind = "rss"
    if not items:
        items = root.findall(".//atom:entry", NS) or root.findall(
            ".//{http://www.w3.org/2005/Atom}entry")
        kind = "atom" if items else kind

    now = datetime.now(timezone.utc)
    parsed, ages, outlets, with_summary = [], [], set(), 0

    for it in items:
        title = _text(it.find("title"))
        link = _text(it.find("link"))
        if not link:                                        # atom style
            le = it.find("{http://www.w3.org/2005/Atom}link")
            if le is not None:
                link = le.get("href") or ""
        summary = (_strip_html(_text(it.find("description")))
                   or _strip_html(_text(it.find("{http://www.w3.org/2005/Atom}summary")))
                   or _strip_html(_text(it.find("content:encoded", NS))))
        raw_date = (_text(it.find("pubDate"))
                    or _text(it.find("dc:date", NS))
                    or _text(it.find("{http://www.w3.org/2005/Atom}published"))
                    or _text(it.find("{http://www.w3.org/2005/Atom}updated")))
        d = _parse_date(raw_date)
        if d:
            ages.append((now - d).total_seconds() / 86400.0)
        if len(summary) >= 40:
            with_summary += 1
        src = it.find("source")
        if src is not None and (src.text or "").strip():
            outlets.add(src.text.strip())
        elif link:
            m = re.match(r"https?://(?:www\.)?([^/]+)", link)
            if m:
                outlets.add(m.group(1))
        parsed.append({"title": title, "link": link,
                       "summary_len": len(summary), "date": raw_date})

    n = len(parsed)
    return {
        "ok": n > 0,
        "format": kind,
        "items": n,
        "with_summary": with_summary,
        "summary_pct": round(100.0 * with_summary / n, 1) if n else 0.0,
        "median_age_days": round(statistics.median(ages), 2) if ages else None,
        "newest_age_days": round(min(ages), 2) if ages else None,
        "dated_items": len(ages),
        "distinct_outlets": len(outlets),
        "sample": parsed[:3],
    }


def _verdict(r: dict) -> tuple[bool, str]:
    if not r.get("ok"):
        return False, r.get("error", "no items")
    if r["items"] < MIN_ITEMS:
        return False, f"only {r['items']} items (need {MIN_ITEMS})"
    if r["summary_pct"] < MIN_SUMMARY_PCT:
        return False, f"summaries on {r['summary_pct']}% (need {MIN_SUMMARY_PCT}%)"
    age = r.get("median_age_days")
    if age is None:
        return False, "no parseable dates"
    if age > MAX_MEDIAN_AGE_DAYS:
        return False, f"median age {age}d (max {MAX_MEDIAN_AGE_DAYS}d)"
    return True, "pass"


def _sql_escape(s: str) -> str:
    return s.replace("'", "''")


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe candidate news feeds for the GD dossier pipeline.")
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--extra", help="file with additional feed URLs, one per line")
    ap.add_argument("--json", dest="json_out", help="also write full results as JSON here")
    args = ap.parse_args()

    roster = list(CANDIDATES)
    if args.extra:
        with open(args.extra, encoding="utf-8") as fh:
            for line in fh:
                u = line.strip()
                if u and not u.startswith("#"):
                    roster.append((u, "press", 0.8, u))

    print(f"Probing {len(roster)} feeds  ·  timeout {args.timeout}s  ·  "
          f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
    header = f"{'':2} {'SOURCE':34} {'HTTP':>5} {'ITEMS':>6} {'SUMM%':>6} {'MED AGE':>8}  VERDICT"
    print(header)
    print("-" * len(header))

    results, passed = [], []
    for name, kind, weight, url in roster:
        status, body, err = _fetch(url, args.timeout)
        if err or not body:
            row = {"name": name, "kind": kind, "weight": weight, "url": url,
                   "http": status, "ok": False, "error": err or "empty body"}
            print(f"{'✗':2} {name[:34]:34} {status or '---':>5} {'-':>6} {'-':>6} {'-':>8}  {err}")
            results.append(row)
            continue

        parsed = _parse_feed(body)
        ok, why = _verdict(parsed)
        row = {"name": name, "kind": kind, "weight": weight, "url": url,
               "http": status, **parsed, "verdict": why, "passed": ok}
        results.append(row)
        if ok:
            passed.append(row)
        print(f"{'✓' if ok else '✗':2} {name[:34]:34} {status:>5} "
              f"{parsed.get('items', 0):>6} {parsed.get('summary_pct', 0):>6} "
              f"{str(parsed.get('median_age_days', '-')):>8}  {why}")

    total_items = sum(r.get("items", 0) for r in passed)
    primaries = sum(1 for r in passed if r["kind"] == "primary")

    print("\n" + "=" * 72)
    print(f"PASSED: {len(passed)} of {len(roster)} feeds  ·  "
          f"{total_items} items in one sweep  ·  {primaries} primary sources")
    print("GATE 0 needs >= 12 passing feeds and >= 3 primary sources.")
    if len(passed) >= 12 and primaries >= 3:
        print("=> GATE 0 GREEN. Seed INSERT below.")
    else:
        print("=> GATE 0 RED. Stop and report before starting Phase 1 — the design "
              "assumes breadth, and a thin roster changes the plan.")
    print("=" * 72 + "\n")

    if passed:
        print("-- seed for migration 0064_gd_dossier.sql (idempotent)")
        print("insert into public.news_sources (name, kind, url, weight, enabled) values")
        rows = [
            f"  ('{_sql_escape(r['name'])}', '{r['kind']}', "
            f"'{_sql_escape(r['url'])}', {r['weight']}, true)"
            for r in passed
        ]
        print(",\n".join(rows))
        print("on conflict (url) do update set\n"
              "  name = excluded.name, kind = excluded.kind,\n"
              "  weight = excluded.weight, enabled = true;")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"probed_at": datetime.now(timezone.utc).isoformat(),
                       "criteria": {"min_items": MIN_ITEMS,
                                    "min_summary_pct": MIN_SUMMARY_PCT,
                                    "max_median_age_days": MAX_MEDIAN_AGE_DAYS},
                       "results": results}, fh, indent=2, ensure_ascii=False)
        print(f"\nFull results written to {args.json_out}")

    return 0 if len(passed) >= 12 and primaries >= 3 else 1


if __name__ == "__main__":
    sys.exit(main())
