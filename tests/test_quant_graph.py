"""services/quant_graph.py — arithmetic over the candidate's own quantities.

The point of these tests is the two halves of the contract:

  it MUST catch errors regardless of how the candidate phrased them, because phrasing is
  exactly what services/numbers_check.py keys on and exactly where it runs out; and

  it MUST NOT decide what is plausible, because a plausibility threshold per quantity is
  the enumeration this module exists to avoid. A legitimate 70% share has to come back
  unflagged, as a reported relation for the model to judge.
"""
from services import quant_graph as Q
from services import structure_tree as S


def _rels(state, message=""):
    return [r["relation"] for r in Q.implications(state, message)]


def _flags(state, message=""):
    return [r["flag"] for r in Q.implications(state, message)]


# -------------------------------------------------------- the session that motivated it
def test_chennai_error_is_caught_on_the_turn_it_is_made():
    """"upper middle class would be 1/3 that is 40 lacks who ows a car", against a 1.2
    crore population stated a turn earlier. No unit word, a typo in the unit, and no
    vocabulary any regex in numbers_check.py matches on."""
    st = S.fold_turn({}, {"tree": ">population#1.2 crore"})
    rels = _rels(st, "upper middle class would be 1/3 that is 40 lacks who ows a car")
    assert any("33" in r and "Population" in r for r in rels)


def test_the_same_error_is_still_there_once_the_tree_catches_up():
    st = S.fold_turn({}, {"tree": ">population#1.2 crore"})
    st = S.fold_turn(st, {"tree": "population>car owners#40 lakh"})
    assert any("Car owners" in r and "33" in r for r in _rels(st))


# -------------------------------------------------------- impossibility needs no knowledge
def test_crore_million_slip_falls_out_of_a_single_comparison():
    """The whole family of unit-interchange errors, caught without any unit vocabulary:
    once both figures are numbers, 40 million simply exceeds 1.2 crore."""
    st = S.fold_turn({}, {"tree": ">population#1.2 crore"})
    st = S.fold_turn(st, {"tree": "population>car owners#40 million"})
    assert "impossible" in _flags(st)


def test_impossibilities_are_ordered_first():
    st = S.fold_turn({}, {"tree": ">population#1.2 crore"})
    st = S.fold_turn(st, {"tree": "population>adults#85 lakh|population>owners#40 million"})
    assert Q.implications(st)[0]["flag"] == "impossible"


# -------------------------------------------------------- it must not judge
def test_a_legitimate_share_is_reported_but_never_flagged():
    """70% of a population being adults is correct. A module that flagged this would be
    encoding thresholds, which is the enumeration problem wearing a different hat."""
    st = S.fold_turn({}, {"tree": ">population#1.2 crore"})
    st = S.fold_turn(st, {"tree": "population>adults#85 lakh"})
    assert _flags(st) == [None]
    assert "70" in _rels(st)[0]


def test_full_penetration_is_reported_as_100_percent_not_as_an_error():
    st = S.fold_turn({}, {"tree": ">households#3 crore"})
    st = S.fold_turn(st, {"tree": "households>buyers#3 crore"})
    assert _flags(st) == [None] and "100%" in _rels(st)[0]


# -------------------------------------------------------- no false positives
def test_currency_against_a_count_never_manufactures_an_impossibility():
    """Revenue in rupees legitimately exceeds a customer count."""
    st = S.fold_turn({}, {"tree": ">customers#40 lakh"})
    st = S.fold_turn(st, {"tree": "customers>revenue#₹900 crore"})
    assert Q.implications(st) == []


def test_small_detail_figures_are_not_related_to_the_base():
    """A price or a frequency in the same message is not a segment of the population."""
    st = S.fold_turn({}, {"tree": ">population#1.2 crore"})
    assert _rels(st, "say 2 cars per family at 8 lakh rupees each") == []


def test_restating_the_base_is_not_a_split():
    st = S.fold_turn({}, {"tree": ">population#1.2 crore"})
    assert _rels(st, "so we have 1.2 crore people in the metro") == []


def test_percentages_are_not_compared_as_magnitudes():
    st = S.fold_turn({}, {"tree": ">population#1.2 crore"})
    st = S.fold_turn(st, {"tree": "population>share#30%"})
    assert Q.implications(st) == []


# -------------------------------------------------------- degradation
def test_no_figures_no_output():
    assert Q.implications({}, "hello") == []
    assert Q.implications(S.fold_turn({}, {"tree": ">revenue"}), "ok") == []
    assert Q.packet_block(None, "") == {}


def test_output_is_capped():
    st = S.fold_turn({}, {"tree": ">population#10 crore"})
    for i in range(8):
        st = S.fold_turn(st, {"tree": f"population>seg{i}#{i + 1} crore"})
    assert len(Q.implications(st)) <= Q.MAX_RELATIONS


def test_packet_block_separates_the_impossible():
    st = S.fold_turn({}, {"tree": ">population#1.2 crore"})
    st = S.fold_turn(st, {"tree": "population>owners#40 million"})
    block = Q.packet_block(st)
    assert block["arithmetically_impossible"]
    assert len(block["from_their_numbers"]) >= len(block["arithmetically_impossible"])


def test_garbage_state_cannot_raise():
    for bad in ({"structure": "not a list"}, {"structure": [{"id": "x"}]}, {"structure": [None]}):
        assert Q.implications(bad, "40 lakh") == []
