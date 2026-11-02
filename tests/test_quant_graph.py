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


# -------------------------------------------------------- recomputing their chain
def _chain(parent_value, kids, parent="Cups per day"):
    st = {"structure": [{"id": "p", "label": parent, "value": parent_value,
                         "parent": "", "kind": "stated"}]}
    for i, (label, value) in enumerate(kids):
        st["structure"].append({"id": f"k{i}", "label": label, "value": value,
                                "parent": "p", "kind": "stated"})
    return st


def test_a_chain_that_does_not_close_is_caught():
    """Stronger than a ratio: true or false, with no plausibility judgement in it."""
    st = _chain("2.7 crore", [("Drinkers", "90 lakh"), ("Cups per person", "5")])
    rels = Q.arithmetic_relations(st, "guesstimate")
    assert rels and rels[0]["flag"] == "arithmetic"
    assert "4.5" in rels[0]["relation"]


def test_a_chain_that_closes_says_nothing():
    st = _chain("2.7 crore", [("Drinkers", "90 lakh"), ("Cups per person", "3")])
    assert Q.arithmetic_relations(st, "guesstimate") == []


def test_a_percentage_child_is_a_multiplier_not_a_magnitude():
    st = _chain("90 lakh", [("Population", "1.5 crore"), ("Tea drinker share", "60%")])
    assert Q.arithmetic_relations(st, "guesstimate") == []


def test_exam_mode_never_carries_the_corrected_figure():
    """The whole point of Interview mode. The candidate is told their chain does not
    close and has to find where."""
    st = _chain("2.7 crore", [("Drinkers", "90 lakh"), ("Cups per person", "5")])
    exam = Q.arithmetic_relations(st, "guesstimate", exam=True)
    assert exam and all("4.5" not in r["relation"] for r in exam)
    assert "does not come to" in exam[0]["relation"]
    block = Q.packet_block(st, "", case_type="guesstimate", exam=True)
    assert all("4.5" not in line for line in block.get("their_chain_does_not_close", []))


def test_rounding_inside_tolerance_is_not_an_error():
    """These are estimates, rounded on purpose."""
    st = _chain("2.7 crore", [("Drinkers", "90 lakh"), ("Cups per person", "3.05")])
    assert Q.arithmetic_relations(st, "guesstimate") == []


def test_the_operator_comes_from_the_framework_not_a_guess():
    """Profit is Revenue MINUS cost; defaulting to multiply would invent an error."""
    st = _chain("30 crore", [("Revenue", "100 crore"), ("Total costs", "70 crore")],
                parent="Profit")
    assert Q.arithmetic_relations(st, "profitability") == []


def test_a_single_child_is_not_a_chain():
    assert Q.arithmetic_relations(_chain("2.7 crore", [("Drinkers", "90 lakh")]),
                                  "guesstimate") == []


def test_broken_chain_survives_garbage_state():
    for bad in ({"structure": "nope"}, {"structure": [{"id": "x"}]}):
        assert Q.arithmetic_relations(bad, "guesstimate") == []


# -------------------------------------------------------- the judge can tell the modes apart
def test_the_judge_is_told_which_mode_it_is_grading():
    """It could not, and that is how an exam reply once passed for 'aligning with the
    coached behaviour'. The score measured the rubric, not the interviewer."""
    from tools.eval_interviewer_behavior import SCENARIOS, judge_prompt, POLICY_CONTRACT
    s = next(x for x in SCENARIOS if x["id"] == "help_when_stuck_with_work")
    coached, exam = judge_prompt(s, "x", "coached"), judge_prompt(s, "x", "exam")
    assert coached != exam
    assert POLICY_CONTRACT["exam"] in exam and POLICY_CONTRACT["coached"] in coached


def test_exam_judging_is_told_that_withholding_is_correct():
    from tools.eval_interviewer_behavior import POLICY_CONTRACT
    assert "CORRECT here" in POLICY_CONTRACT["exam"]
    assert "do not mark a reply down for withholding" in POLICY_CONTRACT["exam"]


def test_praise_is_allowed_in_one_mode_and_forbidden_in_the_other():
    from tools.eval_interviewer_behavior import POLICY_CONTRACT
    assert "is fine" in POLICY_CONTRACT["coached"]
    assert "praise in any form" in POLICY_CONTRACT["exam"]


def test_the_inverting_scenarios_carry_their_own_exam_rubric():
    """No contract wording turns 'give a directional hint' into correct exam behaviour."""
    from tools.eval_interviewer_behavior import SCENARIOS
    overridden = {x["id"] for x in SCENARIOS if x.get("required_exam")}
    assert {"help_when_stuck_with_work", "explicit_hint_request", "solution_request_late",
            "help_after_good_work", "sanity_check_absent"} <= overridden
    # Asserted here: STRUCTURE only -- the override exists, it differs from the coached
    # one, and it states a prohibition. Whether the prose describes the right behaviour is
    # a human judgement. Two attempts to check it with a regex both failed on ordinary
    # English ("Decline the hint" contains "hint"; "do not hand over the spine" is a
    # negation), and a pattern that cannot read a negation is not a test, it is a test
    # shaped object that will pass on a rubric saying the opposite of what it should.
    for x in SCENARIOS:
        req = x.get("required_exam")
        if req:
            assert req != x["required"], x["id"]
            assert "MUST NOT" in req, x["id"]


def test_an_unknown_policy_falls_back_rather_than_crashing():
    from tools.eval_interviewer_behavior import SCENARIOS, judge_prompt
    s = SCENARIOS[0]
    assert judge_prompt(s, "x", "nonsense") == judge_prompt(s, "x", "coached")


def test_the_arena_call_site_still_works_without_a_policy():
    """routes/arena.py calls judge_prompt with two arguments."""
    from tools.eval_interviewer_behavior import SCENARIOS, judge_prompt
    assert judge_prompt(SCENARIOS[0], "x")
