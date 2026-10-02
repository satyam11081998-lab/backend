"""Decision policy — a PURE function: (state, blueprint, intent, analysis, rng) -> Action.

No model call, no I/O. Every turn's decision and its reasons are logged as an event, so an
interview can be replayed and debugged (spec §65).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional

from . import state as S
from .grounding import overlap, same_question
from .modes import BREADTH_SECTIONS
from ..interview_memory import claims as C

PROBE_FOCI = ["specificity", "ownership", "reasoning", "outcome", "quantification", "reflection", "tradeoff",
              "depth", "example"]


@dataclass
class Action:
    type: str
    qid: Optional[str] = None
    focus: Optional[str] = None
    contradiction_id: Optional[str] = None
    probe_text: Optional[str] = None
    close_exchange: bool = False
    new_section: Optional[str] = None
    declined: bool = False
    difficulty_delta: int = 0
    end_reason: Optional[str] = None
    preface: Optional[str] = None
    reasons: List[str] = field(default_factory=list)

    def to_event(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in (None, [], False, 0)}


def rng(session_id: str, turn: int) -> float:
    h = hashlib.sha256(f"{session_id}:{turn}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _current_item(state: dict, bp: dict) -> Optional[dict]:
    cur = state.get("current") or {}
    return S.item(bp, cur.get("qid")) if cur.get("qid") else None


def _ladder_next(state: dict, item: dict) -> Optional[str]:
    claim_ids = (item.get("selection_reason") or {}).get("claim_ids") or []
    tree = item.get("probe_tree") or []
    if not claim_ids or not tree:
        return None
    cl = state["claims"].get(claim_ids[0])
    step = (cl or {}).get("ladder_step", 0)
    return tree[step] if step < len(tree) else None


def repeats_asked(state: dict, bp: dict, it: dict) -> Optional[str]:
    """The qid of an already-asked question that `it` would repeat (same question in other words,
    or the same CV claim from a near-identical angle), else None."""
    claims = set(((it.get("selection_reason") or {}).get("claim_ids")) or [])
    for qid in state.get("asked", []):
        prev = S.item(bp, qid)
        if not prev or prev.get("qid") == it.get("qid"):
            continue
        if same_question(it.get("text", ""), prev.get("text", "")):
            return qid
        prev_claims = set(((prev.get("selection_reason") or {}).get("claim_ids")) or [])
        if claims and claims & prev_claims and overlap(it.get("text", ""), prev.get("text", "")) >= 0.4:
            return qid
    return None


def choose_next(state: dict, bp: dict) -> Action:
    """Pick the next planned item (or move section / start closing). Mutates nothing."""
    order = S.section_ids(bp)
    idx = state["section_idx"]
    level = state["difficulty"]["level"]
    reasons: List[str] = []
    while idx < len(order):
        sid = order[idx]
        if sid == "closing":
            return Action("CLOSE_INVITE", new_section="closing" if sid != state["section_id"] else None,
                          close_exchange=True, reasons=reasons + ["planned sections complete"])
        queue = list(state["queues"].get(sid, []))
        over_budget = (sid == state["section_id"] and
                       S.section_spent_s(state, sid) >= 1.1 * S.section_budget_s(bp, sid) and sid != "intro")
        if over_budget:
            reasons.append(f"section {sid} over budget")
        candidates = []
        if not over_budget:
            for qid in queue:
                it = S.item(bp, qid)
                if it is None:
                    continue
                comps = [c for c in it.get("competency_ids", []) if c in state["coverage"]]
                all_sufficient = comps and all(S.sufficient(state, c) for c in comps)
                critical_untested = any(state["coverage"][c]["importance"] == "critical" and
                                        state["coverage"][c]["asked"] == 0 for c in comps)
                if all_sufficient and not critical_untested and sid not in ("intro", *BREADTH_SECTIONS):
                    continue
                dup = repeats_asked(state, bp, it)
                if dup:
                    reasons.append(f"skipped {qid}: repeats {dup}")
                    continue
                candidates.append(it)
        if candidates:
            # Prefer the item whose difficulty is closest to the current adaptive level, among the next three.
            window = candidates[:3]
            best = min(window, key=lambda it: (abs(int(it.get("difficulty", 3)) - level), window.index(it)))
            return Action("ASK", qid=best["qid"], close_exchange=True,
                          new_section=sid if sid != state["section_id"] else None,
                          reasons=reasons + [f"next item in {sid} (difficulty {best.get('difficulty')} vs level {level})"])
        idx += 1
    return Action("CLOSE_INVITE", close_exchange=True, reasons=reasons + ["no items left"])


def decide(state: dict, bp: dict, *, intent: str, analysis: Optional[dict], session_key: str,
           closing_reply: bool = False) -> Action:
    cur = state.get("current") or {}
    item = _current_item(state, bp)
    in_closing = state["section_id"] == "closing"
    turn = state["counters"]["turns"]

    # ---- closing conversation ------------------------------------------------------------
    if in_closing and state["closing"]["stage"] == "invited":
        return Action("CLOSE_FINAL", close_exchange=True, end_reason="completed_naturally",
                      reasons=["candidate responded to closing invitation"])

    # ---- control intents -----------------------------------------------------------------
    if intent == "end_request":
        return Action("END_EARLY", close_exchange=True, end_reason="ended_early_by_user", reasons=["candidate asked to end"])
    if intent == "break_request":
        return Action("PAUSE", reasons=["candidate asked for a break"])
    if intent == "repeat_request":
        return Action("REPEAT", qid=cur.get("qid"), reasons=["candidate asked to repeat"])
    if intent == "thinking_pause":
        return Action("WAIT", qid=cur.get("qid"), reasons=["candidate asked for a moment"])
    if intent == "clarification_request":
        return Action("CLARIFY_QUESTION", qid=cur.get("qid"), reasons=["candidate asked for clarification"])
    if intent == "meta_question":
        return Action("REDIRECT", qid=cur.get("qid"), preface="meta", reasons=["asked about performance mid-interview"])
    if intent == "off_topic_question":
        return Action("REDIRECT", qid=cur.get("qid"), preface="off_topic", reasons=["unrelated question"])
    if intent == "refusal":
        nxt = choose_next(state, bp)
        nxt.declined = True
        nxt.preface = "ack_refusal"
        nxt.reasons = ["candidate declined; competency marked not tested, not weak"] + nxt.reasons
        return nxt
    if intent == "non_answer":
        # The counter already includes this message: 1 = first non-answer on this question.
        if state["consecutive"]["non_answers"] <= 1 and item is not None:
            return Action("NUDGE", qid=cur.get("qid"), reasons=["very short / empty answer; one nudge"])
        nxt = choose_next(state, bp)
        nxt.reasons = ["second non-answer; moving on"] + nxt.reasons
        return nxt

    # ---- substantive answer --------------------------------------------------------------
    if item is None:
        return choose_next(state, bp)

    remaining = S.remaining_s(state, bp)
    if not in_closing and remaining <= S.closing_budget_s(bp):
        return Action("CLOSE_INVITE", close_exchange=True, new_section="closing", reasons=["time budget reached"])

    probes = int(cur.get("probes", 0))
    max_probes = int(state["difficulty"]["max_probes"])
    if item.get("origin") == "cv_specific":
        max_probes += 1
    sec_ok = (S.section_spent_s(state, state["section_id"]) < 1.15 * S.section_budget_s(bp, state["section_id"])
              or state["section_id"] == "intro")
    a = analysis or {}
    quality = a.get("answer_quality", "adequate")
    gaps = a.get("gaps") or []
    focus = a.get("probe_focus") or "none"

    # 1) contradictions are clarified once, neutrally
    k = C.next_open(state)
    if k is not None and probes < max_probes + 1 and remaining > S.closing_budget_s(bp) + 60:
        return Action("CLARIFY_CONTRADICTION", qid=cur.get("qid"), contradiction_id=k["id"],
                      reasons=[f"contradiction {k['id']} on {k.get('slot') or 'statement'}"])

    if state["section_id"] == "intro":
        return choose_next(state, bp)

    # 2) CV claim ladder
    if item.get("origin") == "cv_specific" and probes < max_probes and sec_ok and quality != "strong":
        step = _ladder_next(state, item)
        if step:
            return Action("PROBE", qid=cur.get("qid"), focus="ownership", probe_text=step,
                          reasons=["CV claim ladder", f"answer {quality}"])

    # 3) evidence-gap probe
    level_name = bp.get("config", {}).get("difficulty", "medium")
    wants_probe = focus != "none" and bool(gaps)
    if level_name == "easy" and quality != "weak":
        wants_probe = False
    if level_name in ("hard", "expert", "grill") and quality == "adequate" and gaps:
        wants_probe = True
        if focus == "none":
            focus = "depth"
    if wants_probe and probes < max_probes and sec_ok:
        tree = item.get("probe_tree") or []
        hint = tree[min(probes, len(tree) - 1)] if tree else None
        return Action("PROBE", qid=cur.get("qid"), focus=focus if focus in PROBE_FOCI else "specificity",
                      probe_text=hint, reasons=[f"gaps: {', '.join(gaps[:3])}", f"answer {quality}"])

    # 4) professional pushback (stress / grill / hard)
    rate = float(state["difficulty"].get("pushback_rate", 0))
    if (not cur.get("challenged") and quality in ("weak", "adequate") and probes < max_probes + 1
            and rate > 0 and rng(session_key, turn) < rate):
        return Action("CHALLENGE", qid=cur.get("qid"), focus=focus if focus in PROBE_FOCI else "reasoning",
                      reasons=[f"pushback (rate {rate:.2f})"])

    # 5) move on, adapting difficulty
    nxt = choose_next(state, bp)
    cons = state["consecutive"]
    if quality == "strong" and cons["strong"] + 1 >= 2 and state["difficulty"]["level"] < min(5, state["difficulty"]["base"] + 1):
        nxt.difficulty_delta = 1
        nxt.reasons.append("two strong answers in a row: raise difficulty")
    elif quality == "weak" and cons["weak"] + 1 >= 2 and state["difficulty"]["level"] > max(1, state["difficulty"]["base"] - 1):
        nxt.difficulty_delta = -1
        nxt.reasons.append("two weak answers in a row: lower difficulty")
    return nxt
