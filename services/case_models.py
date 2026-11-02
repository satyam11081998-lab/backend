"""The skeleton a case is solved against: its formula, its variables, its stages.

WHY THIS EXISTS

services/structure_tree.py records what the candidate SAID. That is enough to draw their
work back to them and nothing more: it has no model of what the problem REQUIRES, so it
cannot show the formula joining two nodes, cannot mark which branch is still missing, and
cannot tell an assumption they chose from a figure they derived.

This module supplies that model. Each archetype is one equation from the universal
frameworks — Market Size = Customers x Penetration x Frequency x Price, Profit = Revenue -
Cost — expressed as a parent with an OPERATOR and its operands.

TWO DESIGN DECISIONS WORTH THE WORDS

1. The operator, not the formula string. A node carries how its CHILDREN combine ("*",
   "+", "-", "/"), never a rendered "A x B x C". A string drifts from the tree the moment
   the candidate adds a branch; an operator cannot, because the formula is derived from
   whatever children exist at the time. It also makes the arithmetic checkable: with an
   operator and the children's figures, Python can verify the parent the candidate wrote
   instead of merely reporting ratios (see services/quant_graph.py).

2. One rung of gaps, never the skeleton. The obvious move is to draw the whole archetype
   dotted and let the candidate fill it in. That converts a decomposition exercise into a
   fill-in-the-blanks exercise, and decomposition is the skill being taught — Kapur's
   productive-failure work is specifically that guidance arriving before the learner has
   generated their own structure causes lock-in. So gaps_for() returns only the immediate
   children of branches the candidate has ALREADY opened. That is the Whiteboard's own
   promise, "one rung of help at a time", drawn rather than said.

The archetypes are a CLOSED list of about a dozen equations, which is why classifying into
them is safe where enumerating misconceptions was not: a model picking one of twelve named
options is a task small models do reliably, and the list is bounded by arithmetic rather
than by the infinite ways a candidate can be wrong.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# How a parent is produced from its children. Rendered between sibling nodes.
OPS = {"*": "×", "/": "÷", "+": "+", "-": "−"}

# The independent/dependent distinction the frameworks draw, kept to two values because
# only two are honestly knowable from the tree: a node with children is COMPUTED from them
# (dependent); a leaf is something the candidate supplied (independent). Splitting
# "given" from "assumed" would need to know what the case handed them, which is not
# tracked, and a role the system cannot verify is a label it should not print.
ROLE_INPUT, ROLE_DERIVED = "input", "derived"


def _m(label: str, op: str, operands: List[str], note: str = "") -> Dict[str, Any]:
    return {"label": label, "op": op, "operands": operands, "note": note}


# --------------------------------------------------------------------- guesstimate
# Straight from the universal guesstimate framework's formula sheet. Each is the
# MATHEMATICAL RELATIONSHIP that generates the answer, which is the thing the framework
# says to identify at the classify step.
GUESSTIMATE_ARCHETYPES: Dict[str, Dict[str, Any]] = {
    "market_sizing": _m("Market size", "*",
                        ["Potential customers", "Penetration", "Purchase frequency",
                         "Average price"],
                        "Customers x penetration x frequency x price"),
    "population_users": _m("Users", "*",
                           ["Total population", "Eligible share", "Adoption rate"],
                           "Top-down: population narrowed by eligibility then adoption"),
    "consumption": _m("Total consumption", "*",
                      ["Number of consumers", "Units per consumer per day", "Days per year"],
                      "Consumers x consumption rate x occasions"),
    "revenue": _m("Revenue", "*",
                  ["Number of transactions", "Average transaction value"],
                  "Transactions x average ticket"),
    "profit_simple": _m("Profit", "-", ["Revenue", "Total costs"], "Revenue less cost"),
    "operations": _m("Resources required", "/",
                     ["Total workload", "Output per resource"],
                     "Workload divided by output per unit"),
    "infrastructure": _m("Units required", "/",
                         ["Total demand", "Effective capacity per unit"],
                         "Demand divided by effective capacity"),
    "capacity": _m("Units required", "/",
                   ["Required capacity", "Capacity per unit"],
                   "Required capacity divided by capacity per unit"),
    "forecast": _m("Future value", "*",
                   ["Current value", "Growth factor"],
                   "Current base compounded forward"),
    "probability": _m("Expected events", "*",
                      ["Total opportunities", "Probability per opportunity"],
                      "Opportunities x incidence"),
    "bottom_up": _m("Total", "*", ["Number of units", "Output per unit"],
                    "The unit x output identity most guesstimates reduce to"),
}

# --------------------------------------------------------------------- case
CASE_ARCHETYPES: Dict[str, Dict[str, Any]] = {
    "profitability": _m("Profit", "-", ["Revenue", "Total costs"],
                        "Diagnose before cutting: revenue-led or cost-led?"),
    "revenue_growth": _m("Revenue", "*",
                         ["Number of customers", "Purchase frequency",
                          "Average spend per purchase"],
                         "Customers x frequency x spend"),
    "pricing": _m("Profit", "-", ["Contribution", "Fixed costs"],
                  "(P - V) x Q - F: price moves margin and volume together"),
    "market_entry": _m("Entry decision", "+",
                       ["Market attractiveness", "Ability to win", "Entry economics"],
                       "Attractive, winnable, and worth the investment"),
    "product_launch": _m("Incremental profit", "-",
                         ["Incremental revenue", "Incremental costs"],
                         "Net of cannibalisation, not gross"),
    "operations_case": _m("Cost per unit", "/", ["Total cost", "Throughput"],
                          "Find the bottleneck before adding capacity"),
    "market_share": _m("Market share", "/", ["Client sales", "Total market sales"],
                       "Share moves on relative performance, not absolute"),
    "mna": _m("Value created", "-",
              ["Value of combined business", "Standalone values and deal costs"],
              "A good strategic fit can still be a bad price"),
    "investment": _m("Return", "/", ["Net benefit", "Investment"],
                     "Incremental cash flows against required return"),
}

# The child decompositions one level below the headline equation, for the archetypes where
# the frameworks specify them. Keyed by the operand's own label.
SUB_MODELS: Dict[str, Dict[str, Any]] = {
    "Revenue": {"op": "*", "operands": ["Price", "Volume"]},
    "Total costs": {"op": "+", "operands": ["Variable costs", "Fixed costs"]},
    "Incremental revenue": {"op": "*", "operands": ["Price", "Incremental volume"]},
    "Variable costs": {"op": "*", "operands": ["Cost per unit", "Volume"]},
    "Number of transactions": {"op": "*", "operands": ["Customers", "Purchase frequency"]},
    "Potential customers": {"op": "*", "operands": ["Total population", "Eligible share"]},
    "Total workload": {"op": "*", "operands": ["Demand per period", "Periods"]},
}

# --------------------------------------------------------------------- stages
# The process rail. A DIFFERENT object from the decomposition tree: one is where they are,
# the other is what they have built. Drawing them as one chart ruins both.
GUESSTIMATE_STAGES = ["Clarify", "Classify", "Decompose", "Assume",
                      "Calculate", "Validate", "Conclude"]
CASE_STAGES = ["Clarify", "Define success", "Hypothesise", "Structure", "Prioritise",
               "Analyse", "Diagnose", "Solve", "Evaluate", "Recommend"]


def archetypes_for(case_type: str) -> Dict[str, Dict[str, Any]]:
    return GUESSTIMATE_ARCHETYPES if (case_type or "").lower() == "guesstimate" else CASE_ARCHETYPES


def stages_for(case_type: str) -> List[str]:
    return GUESSTIMATE_STAGES if (case_type or "").lower() == "guesstimate" else CASE_STAGES


def model_for(case_type: str, archetype: Optional[str]) -> Optional[Dict[str, Any]]:
    return archetypes_for(case_type).get((archetype or "").strip().lower()) or None


def archetype_names(case_type: str) -> List[str]:
    return sorted(archetypes_for(case_type).keys())


def _slug(label: str) -> str:
    from services.structure_tree import _slug as s
    return s(label)


def gaps_for(case_type: str, archetype: Optional[str],
             nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The NEXT rung only — children of branches the candidate has already opened.

    Returns gap nodes in the same shape as stated ones, so the renderer needs no special
    case beyond drawing kind='gap' dashed. Never returns a gap for a node that exists, and
    never more than one level below what they have actually reached.
    """
    model = model_for(case_type, archetype)
    if not model:
        return []
    have = {n["id"] for n in (nodes or [])}
    by_label = {(n.get("label") or "").strip().lower(): n for n in (nodes or [])}
    out: List[Dict[str, Any]] = []

    def _add(parent_id: str, op: str, operands: List[str]) -> None:
        for label in operands:
            nid = _slug(label)
            if nid and nid not in have and nid not in {g["id"] for g in out}:
                out.append({"id": nid, "label": label, "value": None, "parent": parent_id,
                            "kind": "gap", "role": ROLE_INPUT, "op": None})

    # Rung 1: the headline equation, shown once they have put ANYTHING on the table. With
    # an empty tree nothing is drawn at all -- the first decomposition has to be theirs.
    #
    # The operands hang off the node the equation PRODUCES. When the candidate has already
    # named it ("Profit"), that is their node and the operands are its children. When they
    # have not -- a top-down guesstimate starts at Population, not at "Market size" -- the
    # operands hang off the QUESTION itself (parent ""), never off whatever they happened
    # to mention first. Measured live: anchoring to their first node made Population the
    # parent of Penetration and Average price, which reads as "population x price", an
    # equation nobody wrote and that the archetype does not contain.
    if not nodes:
        return []
    result = by_label.get((model["label"] or "").strip().lower())
    _add(result["id"] if result else "", model["op"], model["operands"])

    # Rung 2: only under an operand they have ALREADY named themselves.
    for label, sub in SUB_MODELS.items():
        node = by_label.get(label.lower())
        if node is not None:
            _add(node["id"], sub["op"], sub["operands"])
    return out


def formula_of(node: Dict[str, Any], children: List[Dict[str, Any]]) -> str:
    """'Customers × Penetration × Price' — rendered from the tree, never stored.

    Derived at read time precisely so it cannot disagree with the children that exist.
    """
    op = OPS.get((node or {}).get("op") or "")
    labels = [c.get("label") or "" for c in (children or []) if c.get("label")]
    if not op or len(labels) < 2:
        return ""
    return f" {op} ".join(labels)


# Which entry of each stage list a phase lands on. An explicit map, because the two lists
# are different lengths and arranged differently -- index arithmetic across both was wrong
# the first time and would be wrong again the next time a stage is added.
_PHASE_INDEX = {
    #  phase:          guesstimate, case
    "orient":          (0, 0),   # Clarify            / Clarify
    "classify":        (1, 1),   # Classify           / Define success
    "decompose":       (2, 3),   # Decompose          / Structure
    "assume":          (3, 4),   # Assume             / Prioritise
    "calculate":       (4, 5),   # Calculate          / Analyse
    "validate":        (5, 6),   # Validate           / Diagnose
}


def stage_of(state: Optional[Dict[str, Any]], case_type: str) -> str:
    """Which step of the universal process the candidate is actually on.

    DERIVED from the tree, not classified by a model. The stages are defined by what
    exists in the work -- a decomposition with no figures in it IS the decompose stage --
    so asking a model to name it would be re-reading something already known, and would be
    wrong some fraction of the time for no gain.

    Never reports the final stage (Conclude / Recommend) from structure alone: that one is
    reached by committing to an answer, which is the submit, not a branch.
    """
    stages = stages_for(case_type)
    col = 0 if (case_type or "").lower() == "guesstimate" else 1

    def at(phase: str) -> str:
        i = _PHASE_INDEX[phase][col]
        return stages[min(i, len(stages) - 1)]

    nodes = [n for n in ((state or {}).get("structure") or [])
             if (n.get("kind") or "stated") != "gap"]
    if not nodes:
        return at("orient")
    if len(nodes) < 2:
        return at("classify")

    def has_value(n: Dict[str, Any]) -> bool:
        return bool((n.get("value") or "").strip())

    parent_ids = {n.get("parent") for n in nodes if n.get("parent")}
    leaves = [n for n in nodes if n["id"] not in parent_ids]

    if not any(has_value(n) for n in nodes):
        return at("decompose")
    if any(not has_value(n) for n in leaves):
        return at("assume")            # still filling in the inputs

    # Validating means the equation is FILLED, not that the top node happens to carry a
    # figure. Measured live: a top-down chain starts at population, so the root is an
    # input, and "root has a value" reported Validate on the second turn of a case that
    # had barely started. Compare against how many operands the archetype actually needs.
    model = model_for(case_type, (state or {}).get("structure_archetype"))
    needed = len(model["operands"]) if model else 3
    filled = sum(1 for n in nodes if has_value(n))
    if filled >= needed:
        return at("validate")
    return at("calculate")


def equation_of(case_type: str, archetype: Optional[str]) -> str:
    """'Market size = Potential customers × Penetration × Purchase frequency × Average price'

    The framework's own equation, printed as a caption above the tree. NOT derived from
    the candidate's children the way a node's formula is: those two differ on purpose. A
    node's formula describes what they have actually built; this describes what the case
    is asking them to build, and it has to stay correct even when their tree is half
    empty or shaped differently.
    """
    model = model_for(case_type, archetype)
    if not model:
        return ""
    op = OPS.get(model["op"], model["op"])
    return f"{model['label']} = " + f" {op} ".join(model["operands"])
