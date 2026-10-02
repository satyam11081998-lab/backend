"""
Broadcast targeted practice, per market (2026-10-02). Standard library only:

    python -m tests.test_broadcast_market

Pins three things:
  1. India generation is byte-for-byte the pre-markets prompt (a golden copy of
     the old system + user messages lives below), so existing India broadcasts
     are unchanged.
  2. A US broadcast asks for the US register (dollars, US firms, the US bank's
     nine case types) and stamps every option with market "US".
  3. Saving files the case under the right bank: US options insert
     cases.market='US'; India inserts are the same row as before (no market
     key, so the column default 'IN' applies); an option generated for one
     market cannot be saved under the other.
"""

import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── stand-ins for the provider + metering modules (no SDK, no network) ──────
_calls = []
_reply = {"value": None}


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]


def _chat_with_fallback(feature, **kw):
    _calls.append({"feature": feature, **kw})
    return _Resp(json.dumps(_reply["value"])), "test-model", "test-provider"


_ai_providers = types.ModuleType("services.ai_providers")
_ai_providers.chat_with_fallback = _chat_with_fallback
sys.modules["services.ai_providers"] = _ai_providers
_usage_meta = []
_ai_usage = types.ModuleType("services.ai_usage")
_ai_usage.log_ai_usage = lambda **kw: _usage_meta.append(kw.get("meta"))
sys.modules["services.ai_usage"] = _ai_usage

from services import broadcast_gen as bg  # noqa: E402

passed = 0


def check(name, cond):
    global passed
    if not cond:
        print("  ✗", name)
        raise SystemExit(1)
    passed += 1
    print("  ✓", name)


def raises_value_error(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


# ── golden India prompt (copied from services/broadcast_gen.py before markets) ──
_IN_SYSTEM = (
    "You are an expert McKinsey/BCG/Bain interviewer writing ORIGINAL, India-flavoured "
    "(Rs/crore, Indian sectors/cities/firms) practice material for MBA placement aspirants. "
    "Output strict JSON only."
)
_IN_CASE_SHAPE = (
    '{"focus":"a short, correctly-spelled, presentable 2-4 word label naming the REAL brand or sector this targets (extract the actual company, FIX typos, Title Case; e.g. bluestone jhwellery -> BlueStone Jewellery). NEVER echo the user phrasing verbatim.",'
    '"options":[{'
    '"title":"short candidate-facing title (no real brand name unless generic)",'
    '"type":"profitability|market_sizing|growth",'
    '"difficulty":"easy|medium|hard",'
    '"hook":"ONE line (<=90 chars) that makes an aspirant want to try it; concrete, no hype",'
    '"scenario":"3-5 sentences: situation, a CXO/PE/founder protagonist, the explicit decision, '
    'and the 2-3 concrete Rs numbers the candidate needs",'
    '"quant_ask":"one specific quantity to compute, as a sentence",'
    '"framework_hint":"one line nudging structure without giving the answer",'
    '"solution":"4-8 sentence worked model solution, shown AFTER submit"'
    "}]}"
)


def _in_user(n, what, topic, diff, shape):
    return (
        f'Create {n} DISTINCT {what} options an aspirant could practise, all grounded in this '
        f'topic/target: "{topic}".\nDIFFICULTY: {diff}.\n'
        "Make the options genuinely different from one another (different angle AND mechanic), each "
        "self-contained and freshly invented (never a real published casebook scenario). "
        f"Return ONLY JSON of EXACTLY this shape, with {n} entries in options:\n{shape}"
    )


CASE_REPLY = {
    "focus": "Chick-fil-A",
    "options": [
        {"title": "Drive-thru throughput", "type": "Market Entry", "difficulty": "hard", "hook": "h1",
         "scenario": "A QSR chain ...", "quant_ask": "q", "framework_hint": "f", "solution": "s"},
        {"title": "Breakfast menu", "type": "growth_strategy", "difficulty": "medium", "hook": "h2",
         "scenario": "The CEO ...", "solution": "s"},
        {"title": "Catering", "type": "M&A", "difficulty": "easy", "hook": "h3", "scenario": "A PE partner ..."},
    ],
}
GUESS_REPLY = {
    "focus": "Pickleball",
    "options": [
        {"title": "Paddles sold", "difficulty": "medium", "hook": "h", "prompt": "Estimate ...",
         "approach_hint": "a", "solution": "s"},
        {"title": "Courts", "difficulty": "medium", "hook": "h", "prompt": "Estimate courts ..."},
    ],
}

print("India generation (unchanged)")
_calls.clear(); _usage_meta.clear()
_reply["value"] = {"focus": "Titan", "options": [{"title": "t", "type": "growth", "difficulty": "hard",
                                                   "hook": "h", "scenario": "s"}]}
opts = bg.generate_options("Titan jewellery", "case", "hard", 3)
msgs = _calls[0]["messages"]
check("default market is India", opts[0]["market"] == "IN")
check("India system prompt is byte-identical", msgs[0]["content"] == _IN_SYSTEM)
check("India user prompt is byte-identical",
      msgs[1]["content"] == _in_user(3, "case-interview scenario", "Titan jewellery", "hard", _IN_CASE_SHAPE))
check("India usage meta carries no market key", "market" not in (_usage_meta[0] or {}))
check("India case types keep the India set", opts[0]["type"] == "growth")
_calls.clear()
bg.generate_options("Titan jewellery", "case", "hard", 3, market="IN")
check("explicit IN == default", _calls[0]["messages"][1]["content"] == msgs[1]["content"])
_calls.clear()
bg.generate_options("Titan jewellery", "case", "hard", 3, market=None)
check("None == default (old callers)", _calls[0]["messages"][1]["content"] == msgs[1]["content"])

print("US generation")
_calls.clear(); _usage_meta.clear()
_reply["value"] = CASE_REPLY
opts = bg.generate_options("chick fil a", "case", "hard", 3, market="US")
sys_p, user_p = _calls[0]["messages"][0]["content"], _calls[0]["messages"][1]["content"]
check("US system asks for US dollars", "US dollars" in sys_p and "never Rs, lakh or crore" in sys_p)
check("US system: American English, US recruiting", "American English" in sys_p and "US consulting" in sys_p)
check("US prompt has no India register", all(w not in user_p for w in ("Rs ", "crore", "aspirant", "India")))
check("US prompt offers the nine US case types", "market entry|growth|pricing|m&a" in user_p)
check("every option stamped US", all(o["market"] == "US" for o in opts))
check("US type 'Market Entry' -> 'market entry'", opts[0]["type"] == "market entry")
check("US type 'growth_strategy' -> 'growth'", opts[1]["type"] == "growth")
check("US type 'M&A' -> 'm&a'", opts[2]["type"] == "m&a")
check("focus label carried", all(o.get("focus") == "Chick-fil-A" for o in opts))
check("usage meta records the US market", (_usage_meta[0] or {}).get("market") == "US")
_calls.clear()
_reply["value"] = GUESS_REPLY
gopts = bg.generate_options("pickleball paddles", "guesstimate", "medium", 2, market="US")
gp = _calls[0]["messages"][1]["content"]
check("US guesstimate is asked for as market sizing", "MARKET SIZING question" in gp and "US geography" in gp)
check("US guesstimate options keep kind/type guesstimate", all(o["kind"] == "guesstimate" and o["type"] == "guesstimate" for o in gopts))
_calls.clear()
_reply["value"] = CASE_REPLY
bg.generate_options("pickleball", "case", "medium", 2, market="EU")
check("EU is generated as US", "US dollars" in _calls[0]["messages"][0]["content"])
check("unknown market refused", raises_value_error(lambda: bg.generate_options("x", "case", "medium", 2, market="UK")))

print("Saving")


class _Ins:
    def __init__(self, db, row):
        self.db, self.row = db, row

    def execute(self):
        self.db.rows.append(dict(self.row))
        return types.SimpleNamespace(data=[{"id": f"case-{len(self.db.rows)}"}])


class _Tbl:
    def __init__(self, db):
        self.db = db

    def insert(self, row):
        return _Ins(self.db, row)


class FakeDB:
    def __init__(self):
        self.rows = []

    def table(self, name):
        assert name == "cases"
        return _Tbl(self)


db = FakeDB()
saved = bg.save_option(db, "admin-1", opts[0], "chick fil a")
row = db.rows[-1]
check("US option (no request market) saves as US", row.get("market") == "US" and saved["market"] == "US")
check("US row is unlisted + inactive", row["unlisted"] is True and row["is_active"] is False)
check("US row keeps the US case type", row["type"] == "market entry")
check("US generated_for records the market", row["generated_for"].get("market") == "US")

in_opt = {"kind": "case", "title": "Margins", "type": "profitability", "difficulty": "medium",
          "scenario": "A Pune FMCG firm ...", "quant_ask": "Compute X"}
bg.save_option(db, "admin-1", in_opt, "fmcg")
row = db.rows[-1]
check("legacy India option (no market anywhere) saves without a market key", "market" not in row)
check("legacy India row: generated_for unchanged", "market" not in row["generated_for"])
check("legacy India row: content shape unchanged",
      row["content"] == "A Pune FMCG firm ...\n\n**Quantitative ask:** Compute X")
check("India case type coercion unchanged", row["type"] == "profitability")

bg.save_option(db, "admin-1", dict(in_opt, market="IN"), "fmcg", market="IN")
check("explicit India save also has no market key", "market" not in db.rows[-1])

before = len(db.rows)
check("US option refused under India", raises_value_error(lambda: bg.save_option(db, "a", opts[0], "", market="IN")))
check("India option refused under US", raises_value_error(lambda: bg.save_option(db, "a", dict(in_opt, market="IN"), "", market="US")))
check("refusals insert nothing", len(db.rows) == before)

bg.save_option(db, "admin-1", gopts[0], "pickleball", market="US")
row = db.rows[-1]
check("US guesstimate saves as US guesstimate", row["market"] == "US" and row["type"] == "guesstimate")

print(f"\n{passed} checks passed.")
