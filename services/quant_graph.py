"""Arithmetic over the candidate's own quantities — without matching on how they said it.

THE PROBLEM THIS SOLVES

services/numbers_check.py is 1700 lines of checks, and every one of them is keyed to
VOCABULARY. _check_households only fires if the candidate writes "household size of 4" or
"4 people per household"; _check_office_week needs the word "week"; the unit-restatement
checks need a unit word to restate. They are good checks, and they are the right tool for
arithmetic inside one sentence, where the candidate has written the equation down.

They cannot cover the space. A candidate who says "upper middle class would be 1/3, that is
40 lakh who own a car" has just claimed one car for every three people in Chennai, roughly
five times the real rate — and walks through every check in that file, because they never
used a word any of the regexes know. The answer is NOT a _check_car_ownership(). The next
candidate says it another way and that one misses too. There is no bottom to that list.

THE MECHANISM

Stop detecting errors by their surface form. Do arithmetic on a labelled graph instead.

The structure tree (services/structure_tree.py) already has the model report each figure
with what it is and where it hangs: population#1.2 crore, population>car owners#40 lakh.
That labelling is the one step a regex cannot do and a language model does effortlessly.
Once it exists, the check is division, and division does not care what words were used.

So the two layers do what each is actually good at:

    the model      names what a number refers to and which branch it sits under
    this module    computes the exact relation between those numbers
    the model      judges whether the resulting relation is plausible in the real world

That last line is the important one. This module DOES NOT decide that 33% car ownership is
wrong. Encoding plausibility would put us straight back to enumerating cases — a threshold
per quantity per country, forever. It computes "Car owners (40 lakh) is 33% of Population
(1.2 crore)" and puts that sentence in the control packet. The model already knows what a
third of a city owning cars would mean. It needs the arithmetic, not the judgement.

Exactly one thing is flagged here, because exactly one thing needs no world knowledge: a
child larger than its parent. A subset cannot exceed its set in any domain, in any country,
under any phrasing — and it is also what a crore/million slip looks like once both figures
are numbers, so that whole family of unit errors falls out of a single comparison with no
unit vocabulary involved at all.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from services.numbers_check import Qty, fmt_number, fmt_pct, parse_quantities

# How many relations reach the packet. The interviewer runs one turn on one thing; a wall of
# arithmetic would bury the one relation that matters and cost tokens on every turn.
MAX_RELATIONS = 4

# A child above its parent by more than this is called impossible rather than rounding. 2%
# absorbs the candidate who says "1.2 crore" and then "1.21 crore" for the same quantity.
OVER_PARENT_TOLERANCE = 1.02


# Qty.cur only captures a LEADING marker ("₹2.4 cr", "$5m"). Indians write the unit after
# the number at least as often -- "8 lakh rupees" -- and without this a car price gets
# related to a population as though it were a share of it. Currency compatibility is a unit
# rule, not a judgement about what is plausible, so it belongs here.
_CURRENCY_AFTER = re.compile(r"\s{0,2}(?:rs\.?|rupees?|inr|dollars?|usd|crore\s+rupees)\b", re.I)


def _currency_of(text: str, q: Qty) -> str:
    """The currency a figure is in, from a leading symbol or a trailing word."""
    if q.cur:
        return q.cur
    m = _CURRENCY_AFTER.match(text or "", q.end)
    if not m:
        return ""
    return "usd" if re.search(r"dollar|usd", m.group(0), re.I) else "inr"


def _value_of(raw: str) -> Optional[Qty]:
    """The single quantity a node's figure stands for, or None.

    A node value is a short fragment the model attached with #, so the first parsed
    quantity is the figure. Percentages are returned too: a node labelled 33% is a share
    the candidate asserted, which is worth relating to what it is a share OF.
    """
    qs = parse_quantities(raw or "")
    return qs[0] if qs else None


def _comparable(a: Qty, b: Qty, a_cur: str = "", b_cur: str = "") -> bool:
    """Whether relating these two figures means anything.

    Same currency (or neither), and neither a bare percentage. Revenue in rupees legitimately
    exceeds a customer count, so comparing across those would manufacture an impossibility --
    and a price quoted beside a population is not a share of it.
    """
    return (a_cur or a.cur or "") == (b_cur or b.cur or "") and not a.pct and not b.pct


def _relation(parent_label: str, p: Qty, child_label: str, c: Qty,
              p_cur: str = "", c_cur: str = "") -> Optional[Dict[str, Any]]:
    if p.value <= 0 or c.value <= 0:
        return None
    if not _comparable(p, c, p_cur, c_cur):
        return None
    if c.value > p.value * OVER_PARENT_TOLERANCE:
        # A multiplier is a plain number, never lakh/crore formatted: "3.33x", not "3.33".
        times = c.value / p.value
        return {
            "flag": "impossible",
            "relation": (f"{child_label} ({c.raw}) is larger than {parent_label} ({p.raw}) "
                         f"— about {fmt_number(times, indian_first=False)}x — "
                         f"though it was given as a part of it"),
        }
    return {
        "flag": None,
        "relation": f"{child_label} ({c.raw}) is {fmt_pct(c.value / p.value)} of {parent_label} ({p.raw})",
    }


def _nodes_by_id(state: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {n["id"]: n for n in ((state or {}).get("structure") or [])}


def tree_relations(state: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every parent→child edge where both carry a figure."""
    by_id = _nodes_by_id(state)
    out: List[Dict[str, Any]] = []
    for n in by_id.values():
        parent = by_id.get(n.get("parent") or "")
        if not parent:
            continue
        pv, cv = parent.get("value") or "", n.get("value") or ""
        p, c = _value_of(pv), _value_of(cv)
        if not p or not c:
            continue
        rel = _relation(parent["label"], p, n["label"], c,
                        _currency_of(pv, p), _currency_of(cv, c))
        if rel:
            out.append(rel)
    return out


def message_relations(state: Optional[Dict[str, Any]], message: str) -> List[Dict[str, Any]]:
    """This turn's figures against the largest quantity already on the tree.

    The tree lags by one turn — the model reports an edge in the same reply it is answering
    with, so a figure the candidate states now only becomes an edge on the next turn. A
    candidate who puts a wrong base on the table should be asked about it NOW, not after the
    interviewer has already built on it, so each quantity in the message is also related to
    the biggest figure established so far, which in a decomposition is the base everything
    else is a share of.
    """
    by_id = _nodes_by_id(state)
    anchors = [(n["label"], q, _currency_of(n.get("value") or "", q)) for n in by_id.values()
               if (q := _value_of(n.get("value") or "")) and not q.pct]
    if not anchors:
        return []
    label, base, base_cur = max(anchors, key=lambda t: t[1].value)

    out: List[Dict[str, Any]] = []
    seen: set = set()
    for q in parse_quantities(message or ""):
        if q.pct or q.value <= 0 or q.value in seen:
            continue
        seen.add(q.value)
        # A figure at or below 1% of the base is detail (a price, a frequency), not a
        # segment of it, and relating the two produces noise rather than a check.
        if q.value < base.value * 0.01:
            continue
        if abs(q.value - base.value) < base.value * 0.02:
            continue                                  # restating the base, not splitting it
        rel = _relation(label, base, "a figure they just gave", q,
                        base_cur, _currency_of(message or "", q))
        if rel:
            out.append(rel)
    return out


def implications(state: Optional[Dict[str, Any]], message: str = "") -> List[Dict[str, Any]]:
    """What the candidate's own numbers entail. Impossibilities first, then the rest.

    Pure arithmetic over figures the candidate supplied. No model call, no network, and no
    judgement about what is plausible — that belongs to the model reading the packet.
    """
    try:
        rels = tree_relations(state) + message_relations(state, message)
    except Exception:  # noqa: BLE001 — a side-channel must never break a turn
        return []
    uniq: Dict[str, Dict[str, Any]] = {}
    for r in rels:
        uniq.setdefault(r["relation"], r)
    ordered = sorted(uniq.values(), key=lambda r: 0 if r["flag"] == "impossible" else 1)
    return ordered[:MAX_RELATIONS]


def packet_block(state: Optional[Dict[str, Any]], message: str = "") -> Dict[str, Any]:
    """The `from_their_numbers` block, or {} when there is no arithmetic to report."""
    rels = implications(state, message)
    if not rels:
        return {}
    block: Dict[str, Any] = {"from_their_numbers": [r["relation"] for r in rels]}
    if any(r["flag"] == "impossible" for r in rels):
        block["arithmetically_impossible"] = [r["relation"] for r in rels
                                              if r["flag"] == "impossible"]
    return block
