"""Step 3: a streamed reply filtered down to nothing is retried once with the reason, before a stock line ships.

Before 2026-11-18 an all-praise streamed reply became a fixed line from _MODE_FALLBACK ("Let's step back and look
at the broader picture."). The non-streamed path already retried; the streamed one did not. No network, no keys.
"""
from __future__ import annotations

import os

for _k, _v in {"OPENAI_API_KEY": "sk-test", "SUPABASE_URL": "https://example.supabase.co",
               "SUPABASE_SERVICE_ROLE_KEY": "test"}.items():
    os.environ.setdefault(_k, _v)
os.environ["ADAPTIVE_INTERVIEWER"] = "true"

from services import interviewer_decision as idc  # noqa: E402
from tests.test_v121_real_sessions import _SeqLLM, _stream, U, A  # noqa: E402

HIST = [U("I'd look at market size first, then our share of it"), A("Go on.")]
MSG = "I don't know."  # routes to TEACH_STEP, a streamed reply
STOCK = set(idc._MODE_FALLBACK.values()) | {"Okay, continue."}


def test_an_all_praise_stream_is_retried_and_the_retry_ships():
    llm = _SeqLLM(["Great job! Excellent work!", "Start with total online furniture spend: how many households buy furniture online each year?"])
    out, ctl = _stream(llm, MSG, HIST)
    assert ctl["render"] == "model" and ctl["function"] == "TEACH_STEP"
    assert llm.calls[0].get("stream")
    assert out.strip() == "Start with total online furniture spend: how many households buy furniture online each year?", out
    assert len(llm.calls) == 2 and "only praise" in llm.calls[1]["messages"][-1]["content"]


def test_the_stock_line_remains_only_when_the_retry_also_fails():
    llm = _SeqLLM(["Great job! Excellent work!", "Brilliant! Outstanding!"])
    out, _ctl = _stream(llm, MSG, HIST)
    assert out.strip() and len(llm.calls) == 2
    assert out.strip() in STOCK or not idc.is_praise_only(out)


def test_a_normal_stream_makes_no_extra_call():
    llm = _SeqLLM(["Start with households: how many buy furniture online each year?"])
    out, _ctl = _stream(llm, MSG, HIST)
    assert len(llm.calls) == 1 and "households" in out
