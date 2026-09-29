"""
Interviewer state machine: validated transitions, the assistance ladder, phase
tracking and loop guards. advance() is the ONLY function that mutates BrainState.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from services.interviewer.classify import Signals
from services.interviewer.types import (
    LADDER, MAX_HINT_LEVEL, ActionRecord, BrainState, CandidateState, Decision, Intervention, Phase,
)
from dataclasses import asdict

_PROGRESS_STATES = {CandidateState.PROGRESSING, CandidateState.COMPLETING_STEP, CandidateState.MINOR_ERROR,
                    CandidateState.UNCERTAIN, CandidateState.RECOVERING, CandidateState.TRANSITIONING}
_STUCK_STATES = {CandidateState.STUCK, CandidateState.REPEATEDLY_STUCK, CandidateState.ASKING_FOR_HELP}
_LADDER_VALUES = {i: n for n, i in enumerate(LADDER)}


class TransitionError(ValueError):
    pass


def normalize_state(prev: BrainState, proposed: CandidateState) -> CandidateState:
    """Reject impossible candidate-state transitions by normalising them."""
    if proposed == CandidateState.RECOVERING:
        prior_help = (prev.hint_level > 0 or prev.last_state in {s.value for s in _STUCK_STATES}
                      or prev.last_state in (CandidateState.MATERIAL_ERROR.value, CandidateState.FRUSTRATED.value,
                                             CandidateState.ASKING_FOR_SOLUTION.value))
        if not prior_help:
            return CandidateState.PROGRESSING
    if proposed == CandidateState.REPEATEDLY_STUCK:
        if prev.last_state not in {s.value for s in _STUCK_STATES} and prev.help_requests == 0 and prev.stuck_streak == 0:
            return CandidateState.STUCK
    return proposed


def _phase_from(prev: Phase, sig: Signals, decision: Decision) -> Phase:
    if decision.intervention == Intervention.CLOSE:
        return Phase.CLOSED
    if prev == Phase.CLOSED:
        return Phase.CLOSED
    target = prev
    if decision.intervention == Intervention.OPEN:
        target = Phase.OPENING
    if sig.clarification and prev.rank < Phase.CLARIFYING.rank:
        target = Phase.CLARIFYING
    if sig.structure and prev.rank < Phase.STRUCTURING.rank:
        target = Phase.STRUCTURING
    analytic = bool(sig.arithmetic) or (sig.has_number and not sig.clarification and not sig.numeric_only and sig.word_count >= 6)
    if (analytic or decision.intervention == Intervention.DATA_REVEAL) and target.rank < Phase.ANALYSIS.rank \
            and prev.rank >= Phase.CLARIFYING.rank:
        target = Phase.ANALYSIS
    if analytic and prev == Phase.OPENING:
        target = Phase.ANALYSIS if sig.word_count >= 12 else Phase.CLARIFYING
    if sig.final and target.rank < Phase.SYNTHESIS.rank:
        target = Phase.SYNTHESIS
    if decision.intervention == Intervention.TRANSITION and target.rank < Phase.SYNTHESIS.rank:
        order = list(Phase)
        target = order[target.rank + 1]
    # Monotonic: never move backwards.
    return target if target.rank >= prev.rank else prev


def advance(prev: BrainState, decision: Decision, sig: Signals, *, turn_id: Optional[str] = None,
            spoke_question: bool = False, floor_yield: bool = False) -> BrainState:
    """Fold one decided turn into the persisted state. Pure: returns a new state."""
    st = BrainState.from_dict(prev.to_dict())
    if decision.state == CandidateState.VOICE_PARTIAL:
        return st  # partials never touch state

    cand_state = normalize_state(prev, decision.state)
    st.turns = prev.turns + 1
    st.opened = True
    st.last_state = cand_state.value

    iv = decision.intervention
    # --- assistance ladder -------------------------------------------------
    if iv in _LADDER_VALUES and iv != Intervention.NO_OUTPUT:
        level = _LADDER_VALUES[iv]
        st.hint_level = max(0, min(MAX_HINT_LEVEL, level))
        st.episode_open = True
        st.episode_last_help_turn = st.turns
        st.progress_since_help = 0
        if cand_state in _STUCK_STATES or cand_state == CandidateState.ASKING_FOR_SOLUTION:
            st.help_requests = prev.help_requests + 1 if prev.episode_open else 1
    elif iv == Intervention.REPAIR:
        rung = int(decision.detail.get("rung") or decision.hint_level or prev.hint_level or 1)
        st.hint_level = max(prev.hint_level, min(MAX_HINT_LEVEL, rung))
        st.episode_open = True
        st.episode_last_help_turn = st.turns
        st.progress_since_help = 0
    elif cand_state in _PROGRESS_STATES and prev.episode_open:
        st.progress_since_help = prev.progress_since_help + 1
        st.hint_level = max(0, prev.hint_level - 1)
        if st.progress_since_help >= 2 or st.hint_level == 0:
            st.episode_open = False
            st.hint_level = 0
            st.help_requests = 0

    # --- counters ----------------------------------------------------------
    st.repairs_in_row = prev.repairs_in_row + 1 if iv == Intervention.REPAIR else 0
    if cand_state == CandidateState.FRUSTRATED:
        st.frustration = 2
    else:
        st.frustration = max(0, prev.frustration - 1)
    if cand_state in _STUCK_STATES:
        st.stuck_streak = prev.stuck_streak + 1
    elif cand_state in _PROGRESS_STATES:
        st.stuck_streak = 0

    # --- phase ---------------------------------------------------------------
    st.phase = _phase_from(prev.phase_enum, sig, decision).value
    if iv == Intervention.CLOSE:
        st.closed = True

    # --- history -------------------------------------------------------------
    rec = ActionRecord(i=iv.value, r=decision.reason, t=st.turns, q=bool(spoke_question), fy=bool(floor_yield))
    st.last_actions = (prev.last_actions + [asdict(rec)])[-BrainState.MAX_ACTIONS:]
    if turn_id:
        st.recent_turn_ids = (prev.recent_turn_ids + [turn_id])[-BrainState.MAX_TURN_IDS:]
    validate_state(st)
    return st


def record_output(st: BrainState, text: Optional[str]) -> BrainState:
    """After wording is final: remember the line (repetition guard) and any question asked."""
    if not text:
        return st
    st.recent_lines = (st.recent_lines + [text])[-BrainState.MAX_LINES:]
    qs = extract_questions(text)
    if qs:
        st.asked_questions = (st.asked_questions + qs)[-BrainState.MAX_QUESTIONS:]
    if st.last_actions:
        st.last_actions[-1]["q"] = bool(qs)
    return st


def extract_questions(text: str) -> List[str]:
    parts = re.findall(r"[^.!?]*\?", text or "")
    return [re.sub(r"\s+", " ", p).strip().lower() for p in parts if p.strip()]


def validate_state(st: BrainState) -> None:
    """Invariants that must hold after every transition (asserted in tests and the chaos run)."""
    if not (0 <= st.hint_level <= MAX_HINT_LEVEL):
        raise TransitionError(f"hint_level out of range: {st.hint_level}")
    if not (0 <= st.frustration <= 2):
        raise TransitionError(f"frustration out of range: {st.frustration}")
    if st.turns < 0 or st.repairs_in_row < 0 or st.stuck_streak < 0 or st.help_requests < 0:
        raise TransitionError("negative counter")
    if st.phase not in {p.value for p in Phase}:
        raise TransitionError(f"bad phase {st.phase}")
    if st.closed and st.phase != Phase.CLOSED.value:
        raise TransitionError("closed flag without closed phase")
    if not st.episode_open and st.hint_level != 0:
        raise TransitionError("hint level without an open assistance episode")


def loop_report(st: BrainState) -> Tuple[bool, str]:
    """Detect the loops the brief names. Returns (looping, which)."""
    acts = [a.get("i") for a in st.last_actions]
    if len(acts) >= 4 and len(set(acts[-4:])) == 1 and acts[-1] not in (Intervention.NO_OUTPUT.value,):
        return True, f"same_intervention_x4:{acts[-1]}"
    if st.repairs_in_row >= 3:
        return True, "repair_loop"
    presence = [a for a in st.last_actions[-4:] if a.get("i") in ("ACKNOWLEDGE",) and not a.get("fy")]
    if len(presence) >= 2:
        return True, "presence_metronome"
    qs = st.asked_questions[-4:]
    if len(qs) >= 2 and len(set(qs)) < len(qs):
        return True, "repeated_question"
    return False, ""
