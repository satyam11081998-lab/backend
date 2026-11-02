"""services/structure_tree.py — the live structure diagram.

Pure functions over (state, control tag), so the whole feature is testable without a
model, a database or a live conversation. That is the point of reporting the tree on the
tag the interviewer already emits rather than extracting it with a second call: the part
that can be wrong is the part that can be tested here.

The adversarial cases are the ones that matter. A model that misbehaves must cost a
branch, never a turn — every malformed tag below has to degrade the picture and return.
"""
import json

from services import structure_tree as S


def _depth(n, k=1):
    return max([_depth(c, k + 1) for c in n.get("children", [])] or [k])


# ---------------------------------------------------------------- the happy path
def test_real_conversation_builds_a_spine():
    """The Chennai pre-owned-cars guesstimate, turn by turn, as it actually ran."""
    st = {}
    for tag in [{"tree": ">population#1.2 crore"},
                {"tree": "population>car owners#40 lakh"},
                {"tree": "car owners>replacement cycle"},
                {"tree": "replacement cycle>cars sold per year"}]:
        st = S.fold_turn(st, tag)
    root = S.to_tree(st, "Pre-owned cars sold annually in Chennai")
    assert _depth(root) == 5                      # synthetic root + the four they stated
    assert "40 lakh" in json.dumps(root)
    assert S.active_id(st) == "cars-sold-per-year"


def test_candidates_own_words_survive():
    st = S.fold_turn({}, {"tree": ">car owners"})
    assert st["structure"][0]["label"] == "Car owners"   # sentence case, not Title Case


def test_figure_refines_an_existing_node():
    st = S.fold_turn(S.fold_turn({}, {"tree": ">revenue"}), {"tree": ">revenue#2 cr"})
    assert len(st["structure"]) == 1 and st["structure"][0]["value"] == "2 cr"


def test_replaying_an_edge_changes_nothing():
    a = S.fold_turn({}, {"tree": "population>owners"})
    b = S.fold_turn(dict(a), {"tree": "population>owners"})
    assert json.dumps(a["structure"]) == json.dumps(b["structure"])


def test_parent_named_before_it_was_introduced_is_promoted_not_dropped():
    """Candidates name a sub-branch before its header constantly. Losing the branch is
    worse than hanging it off the root."""
    st = S.fold_turn({}, {"tree": "costs>fixed costs"})
    assert len(st["structure"]) == 2


# ---------------------------------------------------------------- adversarial
def test_a_turn_that_reports_nothing_is_a_no_op():
    assert S.fold_turn({}, {}) == {}
    assert S.to_tree(S.fold_turn({}, {"mode": "probe"})) is None


def test_garbage_tag_yields_no_tree_and_no_exception():
    assert S.to_tree(S.fold_turn({}, {"tree": "|||>>>###"})) is None


def test_cycle_cannot_recurse_forever():
    """a>b then b>a. Without the ancestor check this hangs to_tree()."""
    root = S.to_tree(S.fold_turn({}, {"tree": "a>b|b>a"}))
    assert root["id"] == "a" and _depth(root) == 2


def test_node_cannot_parent_itself():
    assert json.dumps(S.to_tree(S.fold_turn({}, {"tree": "a>a"}))).count('"id"') == 1


def test_caps_hold_against_a_runaway_model():
    """An unbounded tree is an unbounded SVG inside a live conversation."""
    flood = S.fold_turn({}, {"tree": "|".join(f">n{i}" for i in range(60))})
    assert len(flood["structure"]) <= S.MAX_EDGES_PER_TURN

    big = {}
    for i in range(40):
        big = S.fold_turn(big, {"tree": "|".join(f">n{i}_{j}" for j in range(5))})
    assert len(big["structure"]) == S.MAX_NODES

    deep, prev = S.fold_turn({}, {"tree": ">root"}), "root"
    for i in range(12):
        deep = S.fold_turn(deep, {"tree": f"{prev}>lvl{i}"})
        prev = f"lvl{i}"
    assert _depth(S.to_tree(deep)) == S.MAX_DEPTH

    long_label = S.fold_turn({}, {"tree": ">" + "x" * 200})["structure"][0]["label"]
    assert len(long_label) <= S.MAX_LABEL


def test_single_top_level_node_needs_no_synthetic_root():
    assert S.to_tree(S.fold_turn({}, {"tree": ">revenue"}))["id"] == "revenue"


def test_several_top_level_nodes_get_one():
    assert S.to_tree(S.fold_turn({}, {"tree": ">revenue|>cost"}))["id"] == "__root"


def test_hinglish_and_rupee_labels_survive():
    st = S.fold_turn({}, {"tree": ">kitne log hain#₹2.4 cr"})
    assert "₹" in json.dumps(st, ensure_ascii=False)


def test_payload_is_a_single_sse_line():
    """SSE data lines cannot contain a literal newline. A label with one in it must not be
    able to split the frame and corrupt the stream."""
    st = S.fold_turn({}, {"tree": '>pop"ulation\nx#1.2 crore'})
    body = json.dumps({"root": S.to_tree(st, 'Chennai "pre-owned"'),
                       "active_id": S.active_id(st)}, ensure_ascii=False)
    assert "\n" not in body
    assert json.loads(body)["root"]["label"]


# -------------------------------------------------------- inferred edges (fallback path)
class _FakeLLM:
    """Stands in for the chat client; returns whatever the test wants the model to say."""
    def __init__(self, text):
        self.text = text
        self.chat = self
        self.completions = self

    def create(self, **kw):
        class M:  # noqa: D401
            pass
        m, c, r = M(), M(), M()
        m.content = self.text
        c.message = m
        r.choices = [c]
        return r


def _infer(text, msg="a third own cars, 40 lakh"):
    return S.infer_edges([], msg, {}, llm=_FakeLLM(text))


def test_inferred_edges_are_the_same_grammar_the_tag_uses():
    """One parser, one format, two sources -- so parse_edges and its tests cover both."""
    assert _infer("population>car owners#40 lakh") == "population>car owners#40 lakh"
    assert S.fold_turn({}, {"tree": _infer("population>car owners#40 lakh")})["structure"]


def test_a_turn_that_added_no_branch_infers_nothing():
    for empty in ("none", "None.", "", "   ", "-"):
        assert _infer(empty) == ""


def test_model_wrapping_is_stripped():
    assert _infer("tree=>population#1.2 crore") == ">population#1.2 crore"
    assert _infer("`>population#1.2 crore`") == ">population#1.2 crore"
    assert _infer(">population#1.2 crore\nsome explanation") == ">population#1.2 crore"


def test_unparseable_output_is_discarded_rather_than_stored():
    assert _infer("I think they are estimating the population of Chennai.") != "x"
    assert S.parse_edges(_infer("|||###")) == []


def test_inference_is_skipped_without_a_message():
    assert S.infer_edges([], "", {}, llm=_FakeLLM(">x")) == ""


def test_a_failing_model_degrades_to_no_edges():
    class Boom:
        chat = property(lambda self: (_ for _ in ()).throw(RuntimeError("down")))
    assert S.infer_edges([], "hello", {}, llm=Boom()) == ""


# -------------------------------------------------------- the framework model
from services import case_models as CM  # noqa: E402


def _state(edges, archetype=None):
    st = {}
    for e in edges:
        st = S.fold_turn(st, {"tree": e})
    if archetype:
        st["structure_archetype"] = archetype
    return st


def test_formula_is_rendered_from_the_children_that_exist():
    """Never a stored string: it is derived at read time so it cannot disagree with the
    tree the way a stored formula would the moment a branch is added."""
    st = _state([">profit#70 crore"], "profitability")
    t = S.to_tree(st, "Snacks maker", case_type="profitability", with_gaps=True)
    profit = t["children"][0]
    assert profit["formula"] == "Revenue − Total costs"
    assert profit["role"] == "derived"


def test_gaps_are_one_rung_not_the_skeleton():
    """Drawing the whole archetype dotted turns decomposition into fill-in-the-blanks,
    and decomposition is the skill being taught."""
    st = _state([">profit#70 crore"], "profitability")
    t = S.to_tree(st, "x", case_type="profitability", with_gaps=True)
    profit = t["children"][0]
    assert {c["kind"] for c in profit["children"]} == {"gap"}
    # Revenue's own children (Price, Volume) must NOT appear: they are a rung further
    # down, under a branch the candidate has not opened.
    revenue = next(c for c in profit["children"] if c["label"] == "Revenue")
    assert "children" not in revenue


def test_a_branch_they_opened_earns_its_next_rung():
    st = _state([">profit#70 crore", "profit>revenue#400 crore"], "profitability")
    t = S.to_tree(st, "x", case_type="profitability", with_gaps=True)
    revenue = next(c for c in t["children"][0]["children"] if c["label"] == "Revenue")
    assert {c["label"] for c in revenue.get("children", [])} >= {"Price", "Volume"}


def test_no_gaps_on_an_empty_tree():
    """The first decomposition has to be theirs."""
    assert S.to_tree({}, "x", case_type="profitability", with_gaps=True) is None


def test_gaps_are_never_produced_without_an_archetype():
    st = _state([">profit#70 crore"])          # unclassified
    t = S.to_tree(st, "x", case_type="profitability", with_gaps=True)
    assert "children" not in t["children"][0]


def test_stage_is_derived_from_the_work_not_guessed():
    g = "guesstimate"
    assert CM.stage_of({}, g) == "Clarify"
    assert CM.stage_of(_state([">population#1.5 crore"]), g) == "Classify"
    assert CM.stage_of(_state([">population", "population>drinkers"]), g) == "Decompose"
    assert CM.stage_of(_state([">population#1.5 crore", "population>drinkers"]), g) == "Assume"
    assert CM.stage_of(_state([">population", "population>drinkers#90 lakh"]), g) == "Calculate"
    # Two filled nodes is NOT validating: with no archetype the bar is three, and with
    # one it is however many operands the equation needs. "The root has a figure" used to
    # report Validate on turn two of a top-down chain, where the root is an input.
    assert CM.stage_of(_state([">population#1.5 crore", "population>drinkers#90 lakh"]), g) == "Calculate"
    full = _state([">market size#5000 crore", "market size>customers#17 crore",
                   "market size>penetration#20%", "market size>frequency#4",
                   "market size>price#500"], "market_sizing")
    assert CM.stage_of(full, g) == "Validate"


def test_stage_maps_onto_the_case_list_too():
    c = "profitability"
    assert CM.stage_of({}, c) == "Clarify"
    assert CM.stage_of(_state([">profit#70 crore", "profit>revenue"]), c) == "Assume".replace(
        "Assume", "Prioritise")


def test_the_final_stage_is_never_reached_from_structure_alone():
    """Concluding is committing to an answer, which is the submit, not a branch."""
    for t in ("guesstimate", "profitability"):
        st = _state([">a#1 crore", "a>b#50 lakh"])
        assert CM.stage_of(st, t) != CM.stages_for(t)[-1]


def test_gap_nodes_are_excluded_from_the_stage_decision():
    """A rung the system drew is not work the candidate did."""
    st = _state([">profit#70 crore"], "profitability")
    with_gaps = dict(st)
    with_gaps["structure"] = st["structure"] + CM.gaps_for("profitability", "profitability",
                                                           st["structure"])
    assert CM.stage_of(with_gaps, "profitability") == CM.stage_of(st, "profitability")
