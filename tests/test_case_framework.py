"""services/case_framework.py — the worked approach per case.

The rules under test are the ones that protect a PAID teaching product: a generated
framework is a draft until a human approves it, a malformed tree is refused before it is
stored rather than when it fails to render, and the automatic note screen cannot approve
anything by itself.
"""
import json

from services import case_framework as F


class _FakeLLM:
    def __init__(self, text):
        self.text = text
        self.chat = self
        self.completions = self

    def create(self, **kw):
        class M:
            pass
        m, c, r = M(), M(), M()
        m.content = self.text
        c.message = m
        r.choices = [c]
        return r


_GOOD = json.dumps({
    "tree": {"label": "Profit", "op": "-", "value": "70 crore", "driver": True, "children": [
        {"label": "Revenue", "op": "*", "driver": True, "children": [
            {"label": "Price", "value": "12"}, {"label": "Volume", "value": "40 lakh"}]},
        {"label": "Total costs", "driver": False},
    ]},
    "facts": ["Profits down 15% over one year", "Costs fell and so did revenue"],
    "observations": ["The decline was limited to the client, not the industry"],
    "recommendations": ["Investigate product mix before cutting cost"],
})


# ---------------------------------------------------------------- generation is a draft
def test_a_generated_framework_is_never_published_by_generating_it():
    """A framework reads with the authority of a model answer, so a wrong one teaches the
    error. Generation drafts; a human publishes."""
    row = F.generate("c1", "Snacks maker", "profit down", "worked solution text",
                     llm=_FakeLLM(_GOOD))
    assert row["status"] == "draft"
    assert row["source"] == "generated"
    assert row["model"]


def test_the_tree_comes_back_in_the_shape_the_renderer_already_draws():
    row = F.generate("c1", "t", "c", "s", llm=_FakeLLM(_GOOD))
    t = row["tree"]
    assert t["label"] == "Profit" and t["op"] == "-" and t["kind"] == "driver"
    rev = t["children"][0]
    assert rev["op"] == "*" and rev["role"] == "derived"
    assert rev["children"][0]["role"] == "input"
    assert t["children"][1]["kind"] == "stated"       # driver false -> ordinary


def test_a_thin_solution_produces_nothing_rather_than_a_guess():
    assert F.generate("c1", "t", "c", "s", llm=_FakeLLM('{"insufficient": true}')) is None
    assert F.generate("c1", "t", "c", "", llm=_FakeLLM(_GOOD)) is None


def test_a_tree_with_no_branches_is_refused():
    one = json.dumps({"tree": {"label": "Profit"}})
    assert F.generate("c1", "t", "c", "s", llm=_FakeLLM(one)) is None


def test_malformed_output_is_refused_before_it_is_stored():
    for bad in ("not json at all", "{}", '{"tree": null}', '{"tree": {"label": ""}}'):
        assert F.generate("c1", "t", "c", "s", llm=_FakeLLM(bad)) is None


def test_a_fenced_response_is_still_read():
    row = F.generate("c1", "t", "c", "s", llm=_FakeLLM("```json\n" + _GOOD + "\n```"))
    assert row and row["tree"]["label"] == "Profit"


def test_depth_and_breadth_are_capped_before_storage():
    deep = {"label": "L0"}
    cur = deep
    for i in range(1, 12):
        cur["children"] = [{"label": f"L{i}"}]
        cur = cur["children"][0]
    row = F.generate("c1", "t", "c", "s", llm=_FakeLLM(json.dumps({"tree": deep})))
    def depth(n, k=1):
        return max([depth(c, k + 1) for c in n.get("children", [])] or [k])
    assert row and depth(row["tree"]) <= 6


def test_lists_are_capped_and_never_padded():
    many = json.dumps({"tree": json.loads(_GOOD)["tree"],
                       "facts": [f"fact {i}" for i in range(40)],
                       "observations": []})
    row = F.generate("c1", "t", "c", "s", llm=_FakeLLM(many))
    assert len(row["facts"]) == 6 and row["observations"] == []


# ---------------------------------------------------------------- note moderation
def test_the_automatic_screen_cannot_approve_a_note_by_itself():
    """It returns a verdict. Nothing here writes status='approved' -- a model clearing
    text to sit beside that same model's framework is grading its own homework."""
    out = F.review_note("the denominator trap bites here", "t", "s",
                        llm=_FakeLLM('{"verdict":"accept","reason":"useful"}'))
    assert out["verdict"] == "accept"
    assert "status" not in out


def test_an_unreadable_review_defaults_to_unsure_not_accept():
    out = F.review_note("something", "t", "s", llm=_FakeLLM("the model rambled"))
    assert out["verdict"] == "unsure"


def test_an_empty_note_is_rejected_without_a_model_call():
    assert F.review_note("   ", "t", "s", llm=None)["verdict"] == "reject"


def test_a_failing_model_never_lets_a_note_through():
    class Boom:
        chat = property(lambda self: (_ for _ in ()).throw(RuntimeError("down")))
    assert F.review_note("x", "t", "s", llm=Boom())["verdict"] == "unsure"


# ---------------------------------------------------------------- guesstimates
# A guesstimate decomposes AND computes, and the computation is the half candidates get
# wrong. These tests exist because the first cut generated the tree and nothing else,
# which would have shipped half a worked answer on every guesstimate in the bank.
_EVM = {
    "tree": {"label": "Population of India", "value": "1400 Mn", "driver": True, "children": [
        {"label": "Rural", "share": "65%", "value": "900 Mn", "driver": True},
        {"label": "Urban", "share": "35%", "value": "500 Mn"}]},
    "calculation": [
        {"label": "# votes per EVM per phase",
         "terms": [{"label": "Booth hours", "value": "600 mins"},
                   {"label": "Time per vote", "value": "4 mins"}],
         "op": "/", "result": "150 votes"},
        {"label": "# votes per EVM",
         "terms": [{"label": "Per phase", "value": "150 votes"},
                   {"label": "Phases", "value": "5"}],
         "op": "*", "result": "750 votes"},
        {"label": "# EVMs required",
         "terms": [{"label": "Eligible", "value": "800 Mn"},
                   {"label": "Votes per EVM", "value": "750"}],
         "op": "/", "result": "1.07 Mn"},
    ],
    "observations": ["Implies 62.5% capacity utilisation"],
}


def test_a_guesstimate_gets_its_calculation_chain():
    row = F.generate("g1", "EVMs", "estimate EVMs", "worked", "guesstimate",
                     llm=_FakeLLM(json.dumps(_EVM)))
    assert row and len(row["calculation"]) == 3
    assert row["calculation"][0]["result"] == "150 votes"


def test_segmentation_nodes_keep_the_share_and_the_absolute():
    """"Rural (65%) ~900 Mn" is two numbers, and the share has to stay a number."""
    row = F.generate("g1", "EVMs", "c", "s", "guesstimate", llm=_FakeLLM(json.dumps(_EVM)))
    rural = row["tree"]["children"][0]
    assert rural["share"] == "65%" and rural["value"] == "900 Mn"


def test_a_guesstimate_with_no_chain_is_refused():
    """A tree with nothing under it is half a worked answer, and half is what the
    candidate already had."""
    no_calc = dict(_EVM); no_calc.pop("calculation")
    assert F.generate("g1", "t", "c", "s", "guesstimate",
                      llm=_FakeLLM(json.dumps(no_calc))) is None


def test_a_chain_whose_arithmetic_does_not_close_is_thrown_away():
    """Wrong with certainty. No reviewer catches this as reliably as recomputing it."""
    bad = json.loads(json.dumps(_EVM))
    bad["calculation"][1]["result"] = "9000 votes"      # 150 x 5 is not 9000
    assert F.generate("g1", "t", "c", "s", "guesstimate",
                      llm=_FakeLLM(json.dumps(bad))) is None


def test_one_broken_step_discards_the_whole_draft():
    """A chain is one argument: if the arithmetic failed anywhere, the figures around it
    are not trustworthy either."""
    bad = json.loads(json.dumps(_EVM))
    bad["calculation"][0]["result"] = "7 votes"
    assert F.generate("g1", "t", "c", "s", "guesstimate",
                      llm=_FakeLLM(json.dumps(bad))) is None


def test_a_percentage_in_an_additive_step_is_relative_not_a_magnitude():
    """"1.07 Mn + 10% buffer = 1.2 Mn" is an increase BY ten percent. Reading it as
    adding 0.1 would throw away a correct chain."""
    buf = json.loads(json.dumps(_EVM))
    buf["calculation"].append(
        {"label": "# EVMs with buffer",
         "terms": [{"label": "Required", "value": "1.07 Mn"},
                   {"label": "Buffer", "value": "10%"}],
         "op": "+", "result": "1.2 Mn"})
    row = F.generate("g1", "t", "c", "s", "guesstimate", llm=_FakeLLM(json.dumps(buf)))
    assert row and len(row["calculation"]) == 4


def test_casebook_rounding_is_not_treated_as_an_error():
    """800 Mn / 750 is 1,066,667 and the page writes 1.07 Mn."""
    assert F._step_closes({"terms": [{"value": "800 Mn"}, {"value": "750"}],
                           "op": "/", "result": "1.07 Mn"})


def test_a_case_is_not_forced_to_have_a_chain():
    row = F.generate("c1", "Snacks", "profit down", "worked", "profitability",
                     llm=_FakeLLM(_GOOD))
    assert row and row["calculation"] == []


def test_a_step_missing_its_operator_or_terms_is_dropped_not_stored():
    half = json.loads(json.dumps(_EVM))
    half["calculation"].append({"label": "nonsense", "result": "5"})
    row = F.generate("g1", "t", "c", "s", "guesstimate", llm=_FakeLLM(json.dumps(half)))
    assert row and len(row["calculation"]) == 3
