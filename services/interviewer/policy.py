"""
The interviewer's decision: state -> Gate A (substantive) -> Gate B (presence) -> NO_OUTPUT.

decide() is pure given its inputs (the optional assessor is injected), so every
rule here is unit-testable. It decides WHAT happens; presence.py and the
responder decide the words.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from services.interviewer import presence
from services.interviewer.assessor import Assessment
from services.interviewer.classify import Signals
from services.interviewer.numbers import anchor_correction_line, correction_line, scale_slip
from services.interviewer.types import (
    ASSISTANCE_INTERVENTIONS, CONTEXTUAL_PRESENCE, LADDER, MAX_HINT_LEVEL, PRESENCE_INTERVENTIONS, BrainState,
    CandidateState,
    CaseContext, Channel, Decision, Intervention, Phase, TurnInput,
)

# Interventions that may end with (at most) one question. Everything else: zero.
_QUESTION_ALLOWED = {Intervention.TARGETED_PROBE, Intervention.RETHINK_CUE}

_HELP_STATES = {CandidateState.STUCK.value, CandidateState.REPEATEDLY_STUCK.value,
                CandidateState.ASKING_FOR_HELP.value, CandidateState.MATERIAL_ERROR.value,
                CandidateState.FRUSTRATED.value, CandidateState.ASKING_FOR_SOLUTION.value}
_ASSIST_VALUES = {i.value for i in ASSISTANCE_INTERVENTIONS} | {Intervention.DIRECT_CORRECTION.value,
                                                               Intervention.RETHINK_CUE.value,
                                                               Intervention.TARGETED_PROBE.value}
_PRESENCE_VALUES = {i.value for i in PRESENCE_INTERVENTIONS}


@dataclass
class PolicyContext:
    turn: TurnInput
    sig: Signals
    state: BrainState
    case: CaseContext
    transcript_tail: List[Dict[str, str]]
    assess: Optional[Callable[..., Assessment]] = None
    assessor_timeout_s: float = 2.5
    side_assessment: Optional[Assessment] = None   # an assessor call whose verdict did not decide the move


def _episode_active(st: BrainState) -> bool:
    return st.episode_open and (st.turns - st.episode_last_help_turn) <= 4 and st.progress_since_help < 2


def _next_rung(st: BrainState, sig: Signals, *, frustrated: bool = False) -> int:
    base = st.hint_level if _episode_active(st) else 0
    step = 2 if sig.insist else 1
    rung = base + step
    # Oscillation guard: same help intervention in 3 of the last 4 actions -> force the next rung.
    recent = [a.get("i") for a in st.recent(4)]
    if base and recent.count(LADDER[min(base, MAX_HINT_LEVEL)].value) >= 3:
        rung = base + 1
    if frustrated:
        rung = max(rung, 2)
    return max(1, min(MAX_HINT_LEVEL, rung))


def _recent_candidate_texts(tail: List[Dict[str, str]]) -> List[str]:
    return [(t.get("content") or "") for t in tail if t.get("role") == "user"][-3:]


def _last_interviewer_line(tail: List[Dict[str, str]]) -> str:
    for t in reversed(tail):
        if t.get("role") == "assistant" and (t.get("content") or "").strip():
            return t["content"].strip()
    return ""


def _questions_in_a_row(st: BrainState) -> int:
    n = 0
    for a in reversed(st.last_actions):
        if a.get("i") == Intervention.NO_OUTPUT.value:
            continue
        if a.get("q"):
            n += 1
        else:
            break
    return n


def _recovering(st: BrainState, sig: Signals) -> bool:
    if not sig.recovery_language:
        return False
    last = st.last_action()
    return bool(
        (last and last.get("i") in _ASSIST_VALUES)
        or st.last_state in _HELP_STATES
        or (_episode_active(st) and st.hint_level > 0)
    )


def _self_corrected_after(raw: str, expression: str) -> bool:
    idx = raw.find(expression)
    if idx < 0:
        return False
    after = raw[idx + len(expression):].lower()
    return bool(re.search(r"\b(no wait|wait|sorry|oops|i mean|actually|correction|scratch that|my bad|no,)\b", after))


def _decision(intervention: Intervention, state: CandidateState, reason: str, *, fixed: Optional[str] = None,
              detail: Optional[Dict[str, Any]] = None, max_q: int = 0, hint_level: int = 0) -> Decision:
    d = Decision(intervention=intervention, state=state, reason=reason, max_questions=max_q,
                 detail=detail or {}, fixed_text=fixed, hint_level=hint_level)
    d.needs_model = fixed is None and (d.lane.value == "SUBSTANTIVE" or intervention in CONTEXTUAL_PRESENCE)
    return d


def _contextual(intervention: Intervention, state: CandidateState, reason: str, *, fallback: str, turn_type: str,
                may_say_correct: bool = False, verified_claims: Optional[List[str]] = None,
                extra: Optional[Dict[str, Any]] = None) -> Decision:
    """A presence beat the model words from the candidate's own content. `fallback` is the plain
    deterministic hand-back used only if the model fails (it carries no case content).
    `may_say_correct` is True only when a check actually vetted the step - otherwise the line
    must not say or imply that the work is right."""
    detail: Dict[str, Any] = {"fallback": fallback, "turn_type": turn_type, "may_say_correct": bool(may_say_correct),
                              "verified_claims": list(verified_claims or [])}
    detail.update(extra or {})
    return _decision(intervention, state, reason, detail=detail)


def _verified_arithmetic(sig: Signals) -> List[str]:
    """Explicit calculations in this turn that the deterministic checker found within tolerance."""
    if not sig.arithmetic or any(f.severity != "ok" for f in sig.arithmetic):
        return []
    return [f.expression for f in sig.arithmetic]


def _last_structure_turn(tail: List[Dict[str, str]]) -> str:
    from services.interviewer.classify import extract as _extract
    for t in reversed(tail):
        if t.get("role") == "user":
            c = (t.get("content") or "").strip()
            if c and _extract(c).structure:
                return c[:600]
    return ""


def _ladder_decision(ctx: PolicyContext, state: CandidateState, reason: str, *, frustrated: bool = False) -> Decision:
    st, sig = ctx.state, ctx.sig
    rung = _next_rung(st, sig, frustrated=frustrated)
    intervention = LADDER[rung]
    detail = {"rung": rung, "insist": sig.insist, "has_work": st.phase_enum.rank >= Phase.STRUCTURING.rank,
              "step_solution": rung == MAX_HINT_LEVEL}
    if intervention == Intervention.DELIVER_SOLUTION:
        detail["level"] = "step"
    return _decision(intervention, state, reason, detail=detail, hint_level=rung)


def decide(ctx: PolicyContext) -> Decision:
    d = _decide(ctx)
    if ctx.side_assessment is not None and "assessment" not in (d.detail or {}):
        d.detail = dict(d.detail or {}, assessment=_a_meta(ctx.side_assessment))   # telemetry / cost only
    return d


def _decide(ctx: PolicyContext) -> Decision:
    turn, sig, st, case = ctx.turn, ctx.sig, ctx.state, ctx.case
    voice_like = turn.channel in (Channel.VOICE, Channel.STT)
    recent_lines = st.recent_lines

    # 0. A partial transcript is never a turn.
    if turn.is_partial:
        return _decision(Intervention.NO_OUTPUT, CandidateState.VOICE_PARTIAL, "voice_partial")

    # Filler / transcription noise ("thank you.", "you", "ok") - nothing to respond to.
    if sig.asr_noise:
        if sig.norm in ("hello", "hi", "hey", "hello?") and st.opened:
            return _decision(Intervention.HAND_BACK, CandidateState.PROGRESSING, "checking_presence",
                             fixed=presence.hand_back("here", turn.channel, recent_lines))
        return _decision(Intervention.NO_OUTPUT, CandidateState.PROGRESSING, "filler_only")

    # ---------------- Gate A: substantive need ----------------
    if sig.meta:
        kind = "identity" if sig.identity_question else (
            "model" if re.search(r"\b(model|llm)\b", sig.norm) else (
                "rubric" if re.search(r"\b(rubric|answer key|score|scoring|evaluator)\b", sig.norm) else "injection"))
        return _decision(Intervention.DEFLECT, CandidateState.META, f"meta_{kind}",
                         fixed=presence.deflect_line(kind, recent_lines))

    if sig.unintelligible:
        if voice_like and sig.word_count < 3:
            return _decision(Intervention.NO_OUTPUT, CandidateState.UNINTELLIGIBLE, "asr_garbage_short")
        return _decision(Intervention.ANSWER_DIRECT, CandidateState.UNINTELLIGIBLE, "unintelligible",
                         fixed=presence.unintelligible_line(turn.channel, recent_lines), max_q=1)

    if sig.frustration:
        if st.repairs_in_row >= 2:
            return _decision(Intervention.DELIVER_SOLUTION, CandidateState.FRUSTRATED, "repair_loop_breaker",
                             detail={"level": "step", "frustrated": True}, hint_level=MAX_HINT_LEVEL)
        rung = _next_rung(st, sig, frustrated=True)
        cause = ("questions" if _questions_in_a_row(st) >= 1 or re.search(r"question|circles|asking", sig.norm)
                 else "stuck" if (st.stuck_streak or _episode_active(st)) else "unclear_task")
        return _decision(Intervention.REPAIR, CandidateState.FRUSTRATED, f"frustration_{cause}",
                         detail={"cause": cause, "rung": rung, "help": LADDER[rung].value,
                                 "also_help": bool(sig.help or sig.solution or sig.stuck_soft)},
                         hint_level=rung)

    if sig.solution:
        level = sig.solution_level
        return _decision(Intervention.DELIVER_SOLUTION, CandidateState.ASKING_FOR_SOLUTION,
                         f"solution_requested_{level}", detail={"level": level}, hint_level=MAX_HINT_LEVEL)

    if sig.help or (sig.stuck_soft and not sig.has_number) or (sig.insist and _episode_active(st)):
        if sig.help or sig.insist:
            state = (CandidateState.REPEATEDLY_STUCK if _episode_active(st) and st.help_requests >= 1
                     else CandidateState.ASKING_FOR_HELP)
            reason = "help_insist" if sig.insist else "help_requested"
        else:
            state = CandidateState.REPEATEDLY_STUCK if _episode_active(st) else CandidateState.STUCK
            reason = "stuck_signal"
        return _ladder_decision(ctx, state, reason)

    if sig.ux_question:
        return _decision(Intervention.ANSWER_DIRECT, CandidateState.ASKING_CLARIFICATION, "product_question",
                         fixed=presence.ux_line(recent_lines))

    if sig.assumption_check:
        anchor = sig.material_anchor
        if anchor is not None:
            return _decision(Intervention.DIRECT_CORRECTION, CandidateState.MATERIAL_ERROR, "anchor_error_in_assumption",
                             fixed=anchor_correction_line(anchor), detail={"anchor": anchor.anchor.key})
        ok_anchor = next((a for a in sig.anchors if a.severity == "ok"), None)
        if ok_anchor is not None:
            return _decision(Intervention.VALIDATE, CandidateState.ASKING_CLARIFICATION, "assumption_anchor_ok",
                             fixed=presence.validate("anchor_ok", recent_lines, ok_anchor.raw))
        if turn.clarifications_exhausted:
            return _decision(Intervention.ANSWER_DIRECT, CandidateState.ASKING_CLARIFICATION,
                             "clarifications_exhausted", fixed=presence.clarifications_spent_line(recent_lines))
        return _decision(Intervention.ANSWER_DIRECT, CandidateState.ASKING_CLARIFICATION, "assumption_check",
                         detail={"task": "assumption"})

    if sig.clarification:
        if turn.clarifications_exhausted and not sig.repeat_request:
            return _decision(Intervention.ANSWER_DIRECT, CandidateState.ASKING_CLARIFICATION,
                             "clarifications_exhausted", fixed=presence.clarifications_spent_line(recent_lines))
        if sig.data_request:
            return _decision(Intervention.DATA_REVEAL, CandidateState.ASKING_CLARIFICATION, "data_requested")
        if sig.repeat_request:
            return _decision(Intervention.ANSWER_DIRECT, CandidateState.ASKING_CLARIFICATION, "repeat_requested",
                             detail={"task": "repeat", "last_line": _last_interviewer_line(ctx.transcript_tail)})
        if sig.why_question:
            return _decision(Intervention.ANSWER_DIRECT, CandidateState.ASKING_CLARIFICATION, "why_question",
                             detail={"task": "why", "last_line": _last_interviewer_line(ctx.transcript_tail)})
        return _decision(Intervention.ANSWER_DIRECT, CandidateState.ASKING_CLARIFICATION, "clarification")

    anchor = sig.material_anchor
    if anchor is not None and not _self_corrected_after(sig.raw, anchor.raw):
        return _decision(Intervention.DIRECT_CORRECTION, CandidateState.MATERIAL_ERROR, "anchor_error",
                         fixed=anchor_correction_line(anchor), detail={"anchor": anchor.anchor.key})

    arith = sig.material_arithmetic
    if arith is not None and not _self_corrected_after(sig.raw, arith.expression):
        return _decision(Intervention.DIRECT_CORRECTION, CandidateState.MATERIAL_ERROR, "arithmetic_error",
                         fixed=correction_line(arith),
                         detail={"ratio": round(arith.ratio, 3), "op": arith.op_word})

    # A bare result that is ~10^k away from a simple combination of the candidate's own recent
    # numbers and matches none of them: ask the assessor (with the conversation) whether it is a
    # real slip. Only its "material" verdict corrects; timeout / failure / "fine" -> carry on.
    if not sig.question and not sig.final and not sig.transition:
        slip = scale_slip(sig.raw, _recent_candidate_texts(ctx.transcript_tail))
        if slip is not None:
            a = _run_assessor(ctx)
            ctx.side_assessment = a
            if a is not None and a.ok and a.material:
                return _decision(Intervention.DIRECT_CORRECTION, CandidateState.MATERIAL_ERROR, f"assessor_{a.kind}",
                                 detail={"note": a.note, "kind": a.kind, "trigger": "scale_slip",
                                         "factor": round(slip.factor, 2), "assessment": _a_meta(a)})

    if sig.transition:
        return _decision(Intervention.TRANSITION, CandidateState.TRANSITIONING, "transition_requested",
                         detail={"phase": st.phase})

    if sig.final:
        return _decision(Intervention.CLOSE, CandidateState.FINAL_RECOMMENDATION, "final_recommendation",
                         fixed=presence.close_line(case.is_guesstimate, recent_lines))

    if sig.opening and not st.opened:
        return _decision(Intervention.OPEN, CandidateState.TRANSITIONING, "session_open",
                         fixed=presence.open_line(turn.channel, case.is_guesstimate, recent_lines))
    if sig.opening and st.opened and sig.word_count <= 3:
        return _decision(Intervention.HAND_BACK, CandidateState.PROGRESSING, "checking_presence",
                         fixed=presence.hand_back("here", turn.channel, recent_lines))

    # ---------------- not substantive by explicit intent ----------------
    if sig.thinking and not sig.question:
        return _decision(Intervention.NO_OUTPUT, CandidateState.PROGRESSING, "thinking_aloud")

    if _recovering(st, sig):
        if sig.question:
            return _decision(Intervention.ANSWER_DIRECT, CandidateState.RECOVERING, "recovery_check",
                             detail={"task": "confirm_understanding"})
        if voice_like and sig.word_count <= 4:
            return _decision(Intervention.HAND_BACK, CandidateState.RECOVERING, "recovered_short",
                             fixed=presence.hand_back("recovered", turn.channel, recent_lines))
        return _decision(Intervention.NO_OUTPUT, CandidateState.RECOVERING, "recovered_step_back")

    # Step-completing analytic turns: the only place the optional assessor runs.
    wants_review = (sig.validation_request
                    or (sig.structure and (sig.completion or sig.floor_yield or sig.word_count >= 25))
                    or (sig.hypothesis and sig.floor_yield))
    if wants_review:
        a = _run_assessor(ctx)
        if a is not None and a.ok and a.material:
            if a.kind == "missing_branch":
                max_q = 0 if _questions_in_a_row(st) >= 2 or st.frustration else 1
                return _decision(Intervention.TARGETED_PROBE, CandidateState.MATERIAL_ERROR, "assessor_missing_branch",
                                 detail={"note": a.note, "kind": a.kind, "assessment": _a_meta(a)}, max_q=max_q)
            return _decision(Intervention.DIRECT_CORRECTION, CandidateState.MATERIAL_ERROR, f"assessor_{a.kind}",
                             detail={"note": a.note, "kind": a.kind, "assessment": _a_meta(a)})
        meta = {"assessment": _a_meta(a)} if a is not None else {}
        vetted = bool(a is not None and a.ok and not a.material)
        if sig.validation_request:
            # "Is my approach okay?" - say it works only if the assessor actually checked it.
            d = _contextual(Intervention.ACKNOWLEDGE_AND_CONTINUE, CandidateState.COMPLETING_STEP,
                            "validation_ok" if vetted else "validation_unassessed",
                            fallback=presence.validate("approach_ok" if vetted else "unassessed", recent_lines),
                            turn_type="approach_check", may_say_correct=vetted, extra=meta)
            return d
        if sig.structure:
            if sig.floor_yield or sig.completion or turn.channel != Channel.TEXT:
                return _contextual(Intervention.REFLECT_PROGRESS, CandidateState.COMPLETING_STEP,
                                   "structure_complete" if vetted else "structure_unassessed",
                                   fallback=presence.hand_back("structure" if vetted else "structure_unchecked",
                                                               turn.channel, recent_lines),
                                   turn_type="structure", may_say_correct=vetted, extra=meta)
            return _decision(Intervention.NO_OUTPUT, CandidateState.COMPLETING_STEP, "structure_in_progress", detail=meta)
        return _contextual(Intervention.ACKNOWLEDGE_AND_CONTINUE, CandidateState.COMPLETING_STEP, "hypothesis_floor_yield",
                           fallback=presence.hand_back("proceed", turn.channel, recent_lines),
                           turn_type="hypothesis", may_say_correct=False, extra=meta)

    # A causal claim about the business (not about the candidate's own process) in a case:
    # release the data that tests it. Guesstimates have no hidden data to release.
    if sig.hypothesis and not case.is_guesstimate and not sig.structure and not sig.question:
        return _decision(Intervention.DATA_REVEAL, CandidateState.PROGRESSING, "hypothesis_needs_data",
                         detail={"task": "test_hypothesis"})

    if sig.floor_yield:
        if sig.has_number and sig.question:
            arith_ok = any(f.severity == "ok" for f in sig.arithmetic)
            return _decision(Intervention.VALIDATE, CandidateState.UNCERTAIN, "number_check",
                             fixed=presence.validate("arithmetic_ok" if arith_ok else "assumption", recent_lines))
        return _decision(Intervention.HAND_BACK, CandidateState.UNCERTAIN, "floor_yield",
                         fixed=presence.hand_back("proceed", turn.channel, recent_lines))

    if sig.question and sig.has_number and sig.hedged:
        return _decision(Intervention.VALIDATE, CandidateState.UNCERTAIN, "hedged_number_check",
                         fixed=presence.validate("assumption", recent_lines))

    if sig.minor_arithmetic is not None:
        return _decision(Intervention.NO_OUTPUT, CandidateState.MINOR_ERROR, "minor_arithmetic_let_stand")

    if sig.hedged:
        return _decision(Intervention.NO_OUTPUT, CandidateState.UNCERTAIN, "hedged_progressing")

    # ---------------- Gate B: presence ----------------
    last = st.last_action()
    last_was_presence = bool(last and last.get("i") in _PRESENCE_VALUES and not last.get("fy"))
    recent_presence = any(a.get("i") in _PRESENCE_VALUES for a in st.recent(2))

    # A stage of the candidate's OWN plan is finished ("so that's the urban side"): a human
    # interviewer often marks it and points to the next part of that same plan. Only when their
    # structure is on record, and never twice in a row.
    if (sig.stage_done and not sig.question and st.frustration == 0 and not last_was_presence
            and not recent_presence and not sig.next_part_named):
        plan_text = _last_structure_turn(ctx.transcript_tail)
        if plan_text:
            return _contextual(Intervention.ACKNOWLEDGE_AND_ORIENT, CandidateState.COMPLETING_STEP, "stage_done",
                               fallback=presence.hand_back("continue", turn.channel, recent_lines),
                               turn_type="stage_done", may_say_correct=False,
                               verified_claims=_verified_arithmetic(sig), extra={"candidate_plan": plan_text})

    long_completed_step = (sig.word_count >= 25 and not sig.question and sig.raw.rstrip().endswith((".", "!"))
                           and (sig.has_number or sig.structure or sig.hypothesis))
    if (turn.channel == Channel.VOICE and long_completed_step and not last_was_presence and st.frustration == 0
            and not recent_presence):
        # In voice a long finished step with no reaction feels like a dropped line. The beat names
        # what they did (model-worded); it may call the work right only where a check verified it.
        verified = _verified_arithmetic(sig)
        return _contextual(Intervention.ACKNOWLEDGE_AND_CONTINUE, CandidateState.COMPLETING_STEP, "long_step_presence",
                           fallback=presence.acknowledge(recent_lines), turn_type="step",
                           may_say_correct=bool(verified), verified_claims=verified)

    if sig.question:
        # A question we could not classify as a request, clarification or floor-yield. Answering
        # it is safer than ignoring it, and the model is told to answer, not interrogate.
        return _decision(Intervention.ANSWER_DIRECT, CandidateState.ASKING_CLARIFICATION, "unclassified_question")

    return _decision(Intervention.NO_OUTPUT, CandidateState.PROGRESSING, "progressing")


def _a_meta(a: Optional[Assessment]) -> Dict[str, Any]:
    if a is None:
        return {}
    return {"ok": a.ok, "material": a.material, "kind": a.kind, "error_type": a.error_type,
            "latency_ms": a.latency_ms, "model": a.model, "provider": a.provider,
            "tokens_in": a.tokens_in, "tokens_out": a.tokens_out}


def _run_assessor(ctx: PolicyContext) -> Optional[Assessment]:
    if ctx.assess is None or ctx.state.frustration:
        return None
    try:
        return ctx.assess(case_type=ctx.case.case_type, case_content=ctx.case.content,
                          transcript_tail=ctx.transcript_tail, text=ctx.sig.raw,
                          timeout_s=ctx.assessor_timeout_s)
    except Exception as e:  # noqa: BLE001 - the assessor must never break a turn
        return Assessment(material=False, ok=False, error_type=f"assessor_error:{type(e).__name__}")


def apply_question_rules(d: Decision, st: BrainState) -> Decision:
    """Question discipline applied after the decision: 0 by default, 1 only where allowed,
    none while frustrated or after two interviewer questions in a row."""
    if d.intervention in _QUESTION_ALLOWED:
        d.max_questions = 1
    elif d.fixed_text is None:
        d.max_questions = 0
    if st.frustration or _questions_in_a_row(st) >= 2:
        if d.fixed_text is None:
            d.max_questions = 0
    return d
