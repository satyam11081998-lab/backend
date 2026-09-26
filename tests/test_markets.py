"""
International markets (migration 0070) — backend rules. Standard library only:

    python -m tests.test_markets

Covers the pieces that decide what an account may practise and which day a
daily attempt belongs to. A FakeSupabase stands in for PostgREST so the query
shapes (table, filters, fallbacks) are exercised without a network.
"""

import os
import sys
import types
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# fastapi is only needed for HTTPException; stub it when the venv is absent.
try:  # pragma: no cover
    import fastapi  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    fastapi_stub = types.ModuleType("fastapi")

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str = ""):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    fastapi_stub.HTTPException = HTTPException
    sys.modules["fastapi"] = fastapi_stub

from fastapi import HTTPException  # noqa: E402
from services import markets as mk  # noqa: E402


class _Res:
    def __init__(self, data):
        self.data = data


class _Q:
    def __init__(self, db, table):
        self.db, self.table, self.filters, self.single, self.lim = db, table, [], False, None
        self.order_desc = False

    def select(self, cols):
        self.cols = cols
        if self.db.missing_column and "market" in cols:
            raise Exception("column users.market does not exist (42703)")
        return self

    def eq(self, k, v):
        self.filters.append(("eq", k, v)); return self

    def lte(self, k, v):
        self.filters.append(("lte", k, v)); return self

    def gte(self, k, v):
        self.filters.append(("gte", k, v)); return self

    def in_(self, k, vals):
        self.filters.append(("in", k, list(vals))); return self

    def order(self, k, desc=False):
        self.order_desc = desc; return self

    def limit(self, n):
        self.lim = n; return self

    def maybe_single(self):
        self.single = True; return self

    def execute(self):
        if self.db.raise_on and self.table == self.db.raise_on:
            raise Exception("relation does not exist")
        rows = [dict(r) for r in self.db.tables.get(self.table, [])]
        for op, k, v in self.filters:
            if op == "eq":
                rows = [r for r in rows if r.get(k) == v]
            elif op == "lte":
                rows = [r for r in rows if str(r.get(k)) <= str(v)]
            elif op == "gte":
                rows = [r for r in rows if str(r.get(k)) >= str(v)]
            elif op == "in":
                rows = [r for r in rows if r.get(k) in v]
        if self.order_desc:
            rows.sort(key=lambda r: str(r.get("scheduled_date")), reverse=True)
        if self.lim is not None:
            rows = rows[: self.lim]
        if self.single:
            return _Res(rows[0] if rows else None)
        return _Res(rows)


class FakeSupabase:
    def __init__(self, tables, missing_column=False, raise_on=None):
        self.tables, self.missing_column, self.raise_on = tables, missing_column, raise_on

    def table(self, name):
        return _Q(self, name)


passed = 0


def check(name, cond):
    global passed
    if not cond:
        print("  ✗", name)
        raise SystemExit(1)
    passed += 1
    print("  ✓", name)


def raises_403(fn):
    try:
        fn()
    except HTTPException as e:
        return e.status_code == 403
    return False


print("normalisation")
check("None → IN", mk.normalize_market(None) == "IN")
check("junk → IN", mk.normalize_market("XX") == "IN")
check("eu lower → EU", mk.normalize_market("eu") == "EU")
check("EU practises the US bank", mk.content_market_of("EU") == "US")
check("case without market is India", mk.case_market({"id": "x"}) == "IN")
check("US case", mk.case_market({"market": "US"}) == "US")

print("user_market")
db = FakeSupabase({"users": [{"id": "u1", "market": "US"}, {"id": "u2", "market": None}]})
check("stamped US", mk.user_market(db, "u1") == "US")
check("unstamped → IN", mk.user_market(db, "u2") == "IN")
check("pre-0070 (no column) → IN", mk.user_market(FakeSupabase({"users": []}, missing_column=True), "u1") == "IN")
try:
    mk.user_market(FakeSupabase({}, raise_on="users"), "u1")
    check("outage re-raises", False)
except Exception:
    check("outage re-raises", True)

print("assert_market_access")
db = FakeSupabase({"users": [{"id": "us", "market": "US"}, {"id": "eu", "market": "EU"}, {"id": "in", "market": "IN"}]})
check("US user, US case → allowed", mk.assert_market_access(db, "us", {"market": "US"}) == "US")
check("EU user, US case → allowed", mk.assert_market_access(db, "eu", {"market": "US"}) == "US")
check("IN user, India case → allowed", mk.assert_market_access(db, "in", {"market": "IN"}) == "IN")
check("IN user, legacy case (no market) → allowed", mk.assert_market_access(db, "in", {}) == "IN")
check("US user, India case → 403", raises_403(lambda: mk.assert_market_access(db, "us", {"market": "IN"})))
check("US user, legacy case → 403", raises_403(lambda: mk.assert_market_access(db, "us", {})))
check("IN user, US case → 403", raises_403(lambda: mk.assert_market_access(db, "in", {"market": "US"})))
check("owner of a private India case → allowed", mk.assert_market_access(db, "us", {"market": "IN", "owner_id": "us"}) == "US")
adb = FakeSupabase({"users": [{"id": "adm", "market": "IN", "is_admin": True}, {"id": "usadm", "market": "US", "is_admin": True}]})
check("India admin, US case → allowed under US rules", mk.assert_market_access(adb, "adm", {"market": "US"}) == "US")
check("US admin, India case → allowed under India rules", mk.assert_market_access(adb, "usadm", {"market": "IN"}) == "IN")
check("admin, own-market case → unchanged", mk.assert_market_access(adb, "adm", {"market": "IN"}) == "IN")
check("non-admin still 403 (is_admin missing)", raises_403(lambda: mk.assert_market_access(db, "in", {"market": "US"})))
check("admin read error → 403 (fail closed)", raises_403(lambda: mk.assert_market_access(FakeSupabase({"users": [{"id": "in", "market": "IN"}]}), "in", {"market": "US"})))

print("intl_daily_ids")
db = FakeSupabase({"market_daily_schedule": [
    {"market": "US", "scheduled_date": "2026-09-23", "case_id": "c23", "guesstimate_id": "g23"},
    {"market": "US", "scheduled_date": "2026-09-24", "case_id": "c24", "guesstimate_id": "g24"},
]})
check("exact day", mk.intl_daily_ids(db, "US", "2026-09-24") == {"c24", "g24"})
check("exact day missing → empty", mk.intl_daily_ids(db, "US", "2026-09-25") == set())
check("fallback to most recent", mk.intl_daily_ids(db, "US", "2026-09-25", exact=False) == {"c24", "g24"})
check("pre-0070 table missing → empty, no raise", mk.intl_daily_ids(FakeSupabase({}, raise_on="market_daily_schedule"), "US", "2026-09-25") == set())

print("calendar")
check("US day start is New York midnight (EDT)", mk.market_day_start_iso("US", "2026-09-25").endswith("-04:00"))
check("US day start is New York midnight (EST)", mk.market_day_start_iso("US", "2026-12-01").endswith("-05:00"))
check("India day start is IST", mk.market_day_start_iso("IN", "2026-09-25").endswith("+05:30"))
d_us = mk.market_today("US"); d_in = mk.market_today("IN")
check("today strings are ISO dates", len(d_us) == 10 and len(d_in) == 10)

print("interviewer register")
india_case = {"content": "A kirana chain in Pune.", "market": "IN"}
us_case = {"content": "A Midwest grocery chain.", "market": "US"}
check("India content byte-identical", mk.llm_case_content(india_case) == "A kirana chain in Pune.")
check("legacy content byte-identical", mk.llm_case_content({"content": "x"}) == "x")
out = mk.llm_case_content(us_case)
check("US content keeps the prompt", out.startswith("A Midwest grocery chain."))
check("US content carries the dollar register", "US dollars" in out and "never Rs, lakh or crore" in out)

print("access_guard (international path)")
from services import access_guard as ag  # noqa: E402

today = mk.market_today("US")


def world(tier="free", attempts=(), guest=False, perk=None, market="US"):
    return FakeSupabase({
        "users": [{"id": "u", "market": market, "subscription_tier": tier, "subscription_expires_at": None,
                   "is_guest": guest, "linkedin_follow_claimed_at": perk}],
        "market_daily_schedule": [{"market": "US", "scheduled_date": today, "case_id": "daily-c", "guesstimate_id": "daily-g"}],
        "case_attempts": [dict(a) for a in attempts],
        "cases": [{"id": "bank-c1", "type": "profitability"}, {"id": "bank-c2", "type": "pricing"},
                  {"id": "bank-g1", "type": "guesstimate"}, {"id": "daily-c", "type": "growth"}],
        # India schedule must NOT be consulted for a US account:
        "daily_schedule": [{"scheduled_date": today, "case_id": "bank-c1", "guesstimate_code": None}],
    })


def allowed(db, case):
    try:
        ag.assert_can_attempt(db, "u", case)
        return True
    except HTTPException:
        return False


US = lambda cid, typ="profitability": {"id": cid, "type": typ, "market": "US"}  # noqa: E731
check("free: today's US daily case", allowed(world(), US("daily-c", "growth")))
check("free: one bank case", allowed(world(), US("bank-c1")))
check("free: India schedule ignored (bank-c1 is not a US daily)",
      not allowed(world(attempts=[{"user_id": "u", "case_id": "bank-c2", "is_first_attempt": True, "counted_for_daily": False, "created_at": today + "T12:00:00+00:00"}]), US("bank-c1")))
check("free: second bank case refused", not allowed(world(attempts=[{"user_id": "u", "case_id": "bank-c1", "is_first_attempt": True, "counted_for_daily": False, "created_at": "x"}]), US("bank-c2")))
check("free: LinkedIn perk grants one more", allowed(world(perk="2026-01-01", attempts=[{"user_id": "u", "case_id": "bank-c1", "is_first_attempt": True, "counted_for_daily": False, "created_at": "x"}]), US("bank-c2")))
check("free: daily re-attempt refused", not allowed(world(attempts=[{"user_id": "u", "case_id": "daily-c", "is_first_attempt": True, "counted_for_daily": True, "created_at": "x"}]), US("daily-c", "growth")))
check("guest: daily allowed", allowed(world(guest=True), US("daily-c", "growth")))
check("guest: bank refused", not allowed(world(guest=True), US("bank-c1")))
check("pro: anything in the US bank", allowed(world(tier="pro"), US("bank-c2")))
check("pro: India case still refused", not allowed(world(tier="pro"), {"id": "in-1", "type": "profitability", "market": "IN"}))
check("India account: US case refused", not allowed(world(market="IN"), US("bank-c1")))
# India path unchanged: the India daily (daily_schedule, IST) is attemptable.
_in = world(market="IN")
_in.tables["daily_schedule"] = [{"scheduled_date": mk.market_today("IN"), "case_id": "in-daily", "guesstimate_code": None}]
check("India account: India daily allowed (unchanged path)", allowed(_in, {"id": "in-daily", "type": "profitability", "market": "IN"}))
check("India account: legacy case row (no market key) allowed", allowed(_in, {"id": "in-daily", "type": "profitability"}))

print(f"\n{passed} checks passed.")
