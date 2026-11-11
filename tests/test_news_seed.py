"""
News-seeded guesstimates (services/news_seed.py + content_generator) — on the
standard library alone, no keys and no network:

    python -m tests.test_news_seed

What matters here is not that a seed is found. It is that a seed is never
REQUIRED: the daily case and guesstimate are the only thing a free-tier user can
attempt, so every failure path in the seed lookup has to fall back silently to
the behaviour that shipped before it existed. Most of the checks below are
failure paths for that reason.
"""

import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

passed = 0
failed = 0


def check(label, ok):
    global passed, failed
    if ok:
        passed += 1
        print(f"  ok   {label}")
    else:
        failed += 1
        print(f"  FAIL {label}")


# ── stubs ──────────────────────────────────────────────────────────────────
# news_seed imports the provider layer at module scope. Stub only what is
# missing, so a machine with the real SDK still exercises the real import.

_calls = []          # every chat_with_fallback invocation
_next_reply = {}     # what the stub provider returns next


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]
        self.usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=10, total_tokens=20)


def _fake_chat(lane, **kw):
    _calls.append({"lane": lane, **kw})
    if isinstance(_next_reply.get("raise"), Exception):
        raise _next_reply["raise"]
    return _Resp(_next_reply.get("content", "{}")), "stub-model", "stub"


providers = types.ModuleType("services.ai_providers")
providers.chat_with_fallback = _fake_chat
providers.openai_client = lambda *a, **k: None
sys.modules["services.ai_providers"] = providers

usage = types.ModuleType("services.ai_usage")
usage.log_ai_usage = lambda **kw: None
sys.modules["services.ai_usage"] = usage

try:
    from services.model_json import parse_model_json  # noqa: F401
except Exception:  # pragma: no cover — stdlib stand-in
    mj = types.ModuleType("services.model_json")

    def _parse(raw):
        t = str(raw).strip()
        if t.startswith("```"):
            t = t.split("```")[1]
            if t.lstrip().startswith("json"):
                t = t.lstrip()[4:]
        return json.loads(t)

    mj.parse_model_json = _parse
    sys.modules["services.model_json"] = mj

_tables = {}


class _Res:
    def __init__(self, data):
        self.data = data


class _Q:
    def __init__(self, table):
        self.table = table

    def select(self, *a, **k):
        return self

    def gte(self, *a, **k):
        return self

    def order(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def execute(self):
        return _Res(list(_tables.get(self.table, [])))


class _DB:
    def table(self, name):
        return _Q(name)


sb = types.ModuleType("services.supabase_client")
sb.get_supabase_client = lambda: _DB()
sys.modules["services.supabase_client"] = sb

from services import news_seed as ns  # noqa: E402

AQI = "Delhi air quality stays in the severe band for a fifth day"
RBI = "RBI holds the repo rate steady"
DEAL = "Zomato acquires a logistics startup for Rs 1,200 crore"


def head(title, desc=""):
    return {
        "title": title,
        "description": desc,
        "source_name": "Test Wire",
        "source_url": "https://example.com/x",
        "published_at": "2026-11-11T04:00:00+00:00",
    }


def reply(scored):
    _next_reply.clear()
    _next_reply["content"] = json.dumps({"scored": scored})


def reset(rows=None, key="test-key"):
    _calls.clear()
    _next_reply.clear()
    _tables.clear()
    _tables["news_headlines"] = rows or []
    os.environ["GEMINI_API_KEY"] = key


# ── the cheap pre-filter ───────────────────────────────────────────────────
print("\nPre-filter — a headline that states its own number has nothing to estimate")
check("crore figure is filtered", ns._has_own_number(DEAL))
check("percentage is filtered", ns._has_own_number("Inflation eases to 4.2% in October"))
check("million is filtered", ns._has_own_number("App crosses 5 million downloads"))
check("a qualitative headline is kept", not ns._has_own_number(AQI))

# ── the fallback paths: every one must return None, never raise ────────────
print("\nFallbacks — none of these may raise, and all must return None")

reset(rows=[])
check("no stored headlines -> None", ns.pick_guesstimate_seed("IN") is None)
check("no stored headlines -> no model call (no spend)", len(_calls) == 0)

reset(rows=[head(AQI)], key="")
for k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"):
    os.environ.pop(k, None)
check("no provider key -> None", ns.pick_guesstimate_seed("IN") is None)
check("no provider key -> no model call", len(_calls) == 0)

reset(rows=[head(AQI)])
_next_reply["raise"] = RuntimeError("429 rate limited")
check("provider raises -> None, swallowed", ns.pick_guesstimate_seed("IN") is None)

reset(rows=[head(AQI)])
_next_reply["content"] = "not json at all {{{"
check("unparseable reply -> None", ns.pick_guesstimate_seed("IN") is None)

reset(rows=[head(AQI)])
_next_reply["content"] = json.dumps({"scored": []})
check("empty scored array -> None", ns.pick_guesstimate_seed("IN") is None)

reset(rows=[head(AQI)])
reply([{"index": 0, "score": ns.MIN_SEED_SCORE - 1, "quantity": "how many purifiers"}])
check("best score below the floor -> None (no forced seed)", ns.pick_guesstimate_seed("IN") is None)

reset(rows=[head(AQI)])
reply([{"index": 99, "score": 95, "quantity": "x"}])
check("out-of-range index -> None, no crash", ns.pick_guesstimate_seed("IN") is None)

reset(rows=[head(AQI)])
reply([{"index": "nope", "score": "bad"}])
check("non-numeric fields -> None, no crash", ns.pick_guesstimate_seed("IN") is None)

# ── the happy path ─────────────────────────────────────────────────────────
print("\nSelection")

reset(rows=[head(RBI), head(AQI)])
reply([
    {"index": 0, "score": 12, "quantity": None},
    {"index": 1, "score": 88, "quantity": "How many air purifiers will Delhi households buy this winter?"},
])
seed = ns.pick_guesstimate_seed("IN")
check("picks the highest scorer, not the first", bool(seed) and seed["title"] == AQI)
check("carries the quantity through", bool(seed) and "air purifiers" in (seed.get("quantity") or ""))
check("carries the score through", bool(seed) and seed["score"] == 88)
check("one model call only", len(_calls) == 1)
check("runs on the metered cheap lane", _calls[0]["lane"] == "news_classify")
check("asks for JSON mode", _calls[0].get("response_format", {}).get("type") == "json_object")

reset(rows=[head(DEAL), head(AQI)])
reply([{"index": 0, "score": 90, "quantity": "q"}])
ns.pick_guesstimate_seed("IN")
sent = json.loads(_calls[0]["messages"][1]["content"].split("\n\n", 1)[1])
check("the self-numbered headline never reaches the model", all(DEAL not in s["title"] for s in sent))
check("the qualitative headline does", any(AQI in s["title"] for s in sent))

reset(rows=[head(AQI), head(AQI)])
reply([{"index": 0, "score": 90, "quantity": "q"}])
ns.pick_guesstimate_seed("IN")
check("duplicate titles collapse to one candidate", len(json.loads(_calls[0]["messages"][1]["content"].split("\n\n", 1)[1])) == 1)

reset(rows=[head(AQI)])
reply([{"index": 0, "score": 90, "quantity": "q"}])
ns.pick_guesstimate_seed("IN")
check("India prompt says India", "India" in _calls[0]["messages"][1]["content"])

# US must never read the India table — it has no market column, so a US run that
# fell back to stored rows would seed US users with Indian news.
reset(rows=[head(AQI)])
os.environ.pop("GNEWS_API_KEY", None)
check("US with no GNews key -> None (never the India table)", ns.pick_guesstimate_seed("US") is None)
check("US with no GNews key -> no model call", len(_calls) == 0)

# ── the prompt block ──────────────────────────────────────────────────────
print("\nPrompt block")

block = ns and None
from services.content_generator import _seed_instruction, generate_daily_content, GeneratorError  # noqa: E402

block = _seed_instruction({"title": AQI, "description": "AQI above 450 across the NCR.", "quantity": "How many air purifiers?"})
check("includes the headline", AQI in block)
check("includes the context", "AQI above 450" in block)
check("includes the suggested quantity", "How many air purifiers?" in block)
check("demands a self-contained question", "SELF-CONTAINED" in block)
check("forbids inventing reported statistics", "Do not invent statistics" in block)
check("leaves the case unseeded", "CASE is NOT seeded" in block)
check("scopes itself to the guesstimate", "GUESSTIMATE ONLY" in block)
check("a seed with no title yields nothing", _seed_instruction({"title": "   "}) == "")
check("a seed with no quantity still works", AQI in _seed_instruction({"title": AQI}))

# ── generate_daily_content wiring ─────────────────────────────────────────
print("\ngenerate_daily_content")

os.environ["OPENAI_API_KEY"] = "test-key"
BOTH = json.dumps({
    "case": {"title": "c", "type": "growth", "difficulty": "medium", "scenario": "s", "quant_ask": "q",
             "framework_hint": "h", "solution": "sol"},
    "guesstimate": {"title": "g", "difficulty": "medium", "prompt": "p", "approach_hint": "a", "solution": "sol"},
})

_calls.clear()
_next_reply.clear()
_next_reply["content"] = BOTH
out = generate_daily_content([], market="IN", seed_headline={"title": AQI, "quantity": "How many purifiers?"})
up = _calls[0]["messages"][1]["content"]
check("seeded run reaches the model with the headline", AQI in up)
check("seeded run returns the parsed pair", out.get("case", {}).get("title") == "c")

_calls.clear()
_next_reply["content"] = BOTH
generate_daily_content(["Old title"], market="IN")
up_unseeded = _calls[0]["messages"][1]["content"]
check("unseeded prompt carries no news block", "NEWS SEED" not in up_unseeded)
check("unseeded prompt keeps the anti-repeat list", "Old title" in up_unseeded)

_calls.clear()
_next_reply["content"] = BOTH
generate_daily_content([], market="IN", seed_headline=None)
check("seed_headline=None is identical to omitting it", "NEWS SEED" not in _calls[0]["messages"][1]["content"])

_calls.clear()
_next_reply["content"] = BOTH
generate_daily_content([], market="US", seed_headline={"title": "US storm season opens early"})
check("US run still gets the US system prompt", "UNITED STATES" in _calls[0]["messages"][0]["content"])
check("US run gets its seed", "US storm season" in _calls[0]["messages"][1]["content"])

_calls.clear()
_next_reply.clear()
_next_reply["raise"] = RuntimeError("provider down")
try:
    generate_daily_content([], market="IN", seed_headline={"title": AQI})
    check("a dead provider still raises GeneratorError", False)
except GeneratorError:
    check("a dead provider still raises GeneratorError", True)
except Exception as exc:
    check(f"a dead provider still raises GeneratorError (got {type(exc).__name__})", False)

print(f"\n{passed} checks passed, {failed} failed.")
sys.exit(1 if failed else 0)
