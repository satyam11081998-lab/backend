"""Claims ledger + contradiction detection (spec §21–§23).

Contradictions are detected deterministically on comparable slots (team_size, growth_pct,
budget ...) across CV claims and interview statements, plus genuine conflicts the turn
analyzer reports. A contradiction is a reason to CLARIFY, neutrally — never a verdict on
honesty (spec §23). Self-corrections supersede the earlier statement instead of
triggering a clarification.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

CORRECTION = re.compile(r"\b(sorry,? i meant|i meant to say|correction|let me correct|actually,? it was|"
                        r"to be precise|i misspoke|i should clarify)\b", re.IGNORECASE)
DISTINCT_SLOTS = {"team_size", "direct_reports", "growth_pct", "cost_saving_pct", "time_saving_pct", "budget_inr",
                  "budget_usd", "revenue_inr", "revenue_usd", "users", "customers", "accounts", "stores",
                  "projects", "years"}


SLOT_LABELS = {"team_size": "team size", "direct_reports": "number of direct reports", "growth_pct": "growth figure",
               "cost_saving_pct": "cost saving", "time_saving_pct": "time saving", "budget_inr": "budget",
               "budget_usd": "budget", "revenue_inr": "revenue", "revenue_usd": "revenue", "users": "number of users",
               "customers": "number of customers", "accounts": "number of accounts", "stores": "number of stores",
               "projects": "number of projects", "years": "duration"}


def _fmt(v) -> str:
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else f"{f:g}"
    except (TypeError, ValueError):
        return str(v)


def _rel_diff(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return abs(a - b) / max(a, b)


def register(state: dict, new_claims: List[dict], *, exchange_id: str, answer_text: str,
             default_anchor: str = "") -> List[dict]:
    """`default_anchor`: the CV claim the current question is investigating — statements made
    in that exchange are about it unless the analyzer says otherwise."""
    added = []
    correcting = bool(CORRECTION.search(answer_text or ""))
    known = set(state["claims"]) | {c["id"] for c in state["interview_claims"]}
    for nc in new_claims[:6]:
        slot = (nc.get("slot") or "").strip()
        val = nc.get("value")
        if correcting and slot:
            for ic in state["interview_claims"]:
                if ic.get("slots", {}).get(slot) is not None and ic.get("status") != "superseded":
                    ic["status"] = "superseded"
        rel = (nc.get("relates_to") or "").strip()
        anchor = rel if rel in known else default_anchor
        state["counters"]["ic"] += 1
        ic = {"id": f"IC{state['counters']['ic']}", "src": "interview", "text": (nc.get("text") or "")[:240],
              "slots": ({slot: float(val)} if slot in DISTINCT_SLOTS and isinstance(val, (int, float)) else {}),
              "exchange": exchange_id, "anchor": anchor, "high_impact": bool(nc.get("high_impact")),
              "supported": bool(nc.get("supported_in_answer")), "status": "active"}
        state["interview_claims"].append(ic)
        added.append(ic)
    state["interview_claims"] = state["interview_claims"][-40:]
    return added


def _same_subject(a: dict, b: dict) -> Optional[str]:
    """Return the anchor both statements are about, or None if they may be about different things.
    Comparing numbers about different projects/metrics is how a detector invents contradictions."""
    if a.get("src") == "cv" and b.get("src") == "interview":
        return a["id"] if b.get("anchor") == a["id"] else None
    if b.get("src") == "cv" and a.get("src") == "interview":
        return b["id"] if a.get("anchor") == b["id"] else None
    if a.get("src") == "interview" and b.get("src") == "interview":
        if a.get("anchor") and a.get("anchor") == b.get("anchor"):
            return a["anchor"]
        if a.get("anchor") == b["id"]:
            return b["id"]
        if b.get("anchor") == a["id"]:
            return a["id"]
        if a.get("exchange") and a.get("exchange") == b.get("exchange"):
            return f"x:{a['exchange']}"
    return None


def detect(state: dict) -> List[dict]:
    """Deterministic slot comparison between statements about the SAME subject. Each
    (subject, slot) is raised at most once per interview. Returns NEW open contradictions."""
    raised = {c.get("key") for c in state["contradictions"]}
    pool = [c for c in state["claims"].values()] + [c for c in state["interview_claims"] if c.get("status") != "superseded"]
    new: List[dict] = []
    for i, a in enumerate(pool):
        for b in pool[i + 1:]:
            if a["id"] == b["id"]:
                continue
            anchor = _same_subject(a, b)
            if anchor is None:
                continue
            for slot, va in (a.get("slots") or {}).items():
                vb = (b.get("slots") or {}).get(slot)
                if vb is None or slot not in DISTINCT_SLOTS:
                    continue
                d = _rel_diff(float(va), float(vb))
                if d < 0.25:
                    continue
                key = f"{anchor}|{slot}"
                if key in raised:
                    continue
                state["counters"]["k"] += 1
                c = {"id": f"K{state['counters']['k']}", "key": key, "a": a["id"], "b": b["id"], "slot": slot,
                     "a_text": a.get("text", ""), "b_text": b.get("text", ""), "a_value": va, "b_value": vb,
                     "severity": "high" if d >= 0.5 else "medium", "status": "open", "source": "slots"}
                state["contradictions"].append(c)
                raised.add(key)
                new.append(c)
    return new


def from_analyzer(state: dict, conflicts: List[dict], valid_refs: set) -> List[dict]:
    new = []
    existing = {c.get("key") for c in state["contradictions"]}
    for cf in conflicts[:3]:
        ref = (cf.get("memory_ref") or "").strip()
        if ref not in valid_refs or cf.get("severity") == "low":
            continue
        key = f"llm|{ref}"
        if key in existing:
            continue
        state["counters"]["k"] += 1
        c = {"id": f"K{state['counters']['k']}", "key": key, "a": ref, "b": "current", "slot": "",
             "a_text": "", "b_text": "", "description": (cf.get("description") or "")[:240],
             "severity": cf.get("severity", "medium"), "status": "open", "source": "analyzer"}
        state["contradictions"].append(c)
        new.append(c)
    return new


def next_open(state: dict) -> Optional[dict]:
    for c in state["contradictions"]:
        if c["status"] == "open" and c.get("severity") in ("medium", "high"):
            return c
    return None


def describe(c: dict, memory: List[dict]) -> str:
    if c.get("source") == "slots":
        label = SLOT_LABELS.get(c["slot"], c["slot"].replace("_", " "))
        return (f"Earlier the {label} came across as {_fmt(c.get('a_value'))}, and just now as "
                f"{_fmt(c.get('b_value'))}.")
    mem = next((m for m in memory if m["ref"] == c.get("a")), None)
    earlier = mem["summary"] if mem else c.get("a")
    return f"Earlier: {earlier}. Now: {c.get('description', '')}"


def mark(state: dict, cid: str, status: str) -> None:
    for c in state["contradictions"]:
        if c["id"] == cid:
            c["status"] = status


def valid_refs(state: dict) -> set:
    return ({m["ref"] for m in state["memory"]} | set(state["claims"].keys())
            | {c["id"] for c in state["interview_claims"]})


def slots_context(state: dict) -> List[Dict]:
    out = [{"id": c["id"], "text": c.get("text", "")[:160], "slots": c.get("slots", {})}
           for c in list(state["claims"].values())[:8] if c.get("slots")]
    out += [{"id": c["id"], "text": c.get("text", "")[:160], "slots": c.get("slots", {})}
            for c in state["interview_claims"][-8:] if c.get("slots") and c.get("status") != "superseded"]
    return out
