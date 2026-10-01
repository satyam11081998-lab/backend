"""
The one interviewer brain, as the routes use it.

    plan = decide_turn(...)                  # fast: deterministic (+ optional assessor)
    if plan.lane is NO_OUTPUT:        nothing is said
    elif plan.text is not None:       a deterministic line (presence / fixed)
    else:                             word it: word_complete(plan) (voice) or word_stream(plan) (text/stt)
    state = finalize(plan, final_text, error=None)   # state dict to persist + telemetry line

The same function decides for TEXT, STT and VOICE. Channel changes only the
rendering constraints (length, spoken style, whether an optional voice-only
acknowledgement is allowed), never the decision rules.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

from services.interviewer import classify, policy, responder, state_machine, telemetry
from services.interviewer.assessor import Assessment, make_llm_assessor
from services.interviewer.providers import CallMeta, InterviewerLLM, json_completer
from services.interviewer.types import (
    CONTEXTUAL_PRESENCE, BrainState, CaseContext, Channel, Decision, InterviewerError, Lane, TurnInput,
)

STATE_KEY = "brain"

# Test seam: tests install a factory returning a scripted provider. Production leaves it None.
LLM_FACTORY: Optional[Callable[["TurnPlan"], Any]] = None


@dataclass
class TurnPlan:
    turn: TurnInput
    case: CaseContext
    decision: Decision
    state_before: BrainState
    state_after: BrainState
    transcript: List[Dict[str, str]]
    signals: classify.Signals
    attempt_id: str = ""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    text: Optional[str] = None                       # deterministic line, or final generated text
    t_start: float = field(default_factory=time.perf_counter)
    decision_ms: int = 0
    metas: List[CallMeta] = field(default_factory=list)
    assessment: Optional[Dict[str, Any]] = None
    violations: List[str] = field(default_factory=list)
    llm: Any = None
    soft_error: Optional[str] = None                 # contextual beat fell back to its plain hand-back

    @property
    def lane(self) -> Lane:
        return self.decision.lane

    @property
    def needs_model(self) -> bool:
        return self.decision.needs_model


def load_state(session_state: Optional[Dict[str, Any]]) -> BrainState:
    return BrainState.from_dict((session_state or {}).get(STATE_KEY))


def merge_state(session_state: Optional[Dict[str, Any]], brain_state: Dict[str, Any]) -> Dict[str, Any]:
    """Return session_state with the brain's namespaced key replaced (V11's keys untouched)."""
    out = dict(session_state or {})
    out[STATE_KEY] = brain_state
    return out


def _assessor_timeout(channel: Channel) -> float:
    return 1.2 if channel == Channel.VOICE else 2.5


def default_assessor(user_id: Optional[str]) -> Callable[..., Assessment]:
    return make_llm_assessor(json_completer(user_id))


def decide_turn(*, turn: TurnInput, case: CaseContext, transcript: List[Dict[str, str]],
                session_state: Optional[Dict[str, Any]], attempt_id: str = "", user_id: Optional[str] = None,
                session_id: Optional[str] = None,
                assess: Optional[Callable[..., Assessment]] = None,
                llm: Any = None) -> TurnPlan:
    t0 = time.perf_counter()
    prior = load_state(session_state)
    sig = classify.extract(turn.text, is_partial=turn.is_partial)
    ctx = policy.PolicyContext(turn=turn, sig=sig, state=prior, case=case, transcript_tail=transcript[-12:],
                               assess=assess, assessor_timeout_s=_assessor_timeout(turn.channel))
    decision = policy.apply_question_rules(policy.decide(ctx), prior)
    after = state_machine.advance(prior, decision, sig, turn_id=turn.turn_id,
                                  floor_yield=bool(sig.floor_yield or sig.question))
    plan = TurnPlan(turn=turn, case=case, decision=decision, state_before=prior, state_after=after,
                    transcript=transcript, signals=sig, attempt_id=attempt_id, user_id=user_id,
                    session_id=session_id, text=decision.fixed_text, t_start=t0, llm=llm)
    plan.assessment = (decision.detail or {}).get("assessment")
    plan.decision_ms = int((time.perf_counter() - t0) * 1000)
    return plan


def _llm_for(plan: TurnPlan) -> Any:
    if plan.llm is not None:
        return plan.llm
    if LLM_FACTORY is not None:
        return LLM_FACTORY(plan)
    endpoint = "/attempts/voice-decision" if plan.turn.channel == Channel.VOICE else "/attempts/messages"
    return InterviewerLLM(user_id=plan.user_id, endpoint=endpoint, channel=plan.turn.channel.value)


def word_complete(plan: TurnPlan) -> str:
    """Full validated line (regenerates once if needed). Raises InterviewerError - except for a
    contextual presence beat, which falls back to its plain deterministic hand-back (it carries no
    case content, so the fallback invents nothing) and records the failure in telemetry."""
    if not plan.needs_model:
        return plan.text or ""
    contextual = plan.decision.intervention in CONTEXTUAL_PRESENCE
    try:
        text, metas, res = responder.generate_complete(_llm_for(plan), plan.decision, plan.case, plan.state_before,
                                                       plan.turn.channel, plan.transcript, plan.turn.text,
                                                       metas_out=plan.metas)
    except InterviewerError as e:
        fallback = (plan.decision.detail or {}).get("fallback")
        if not contextual or not fallback:
            raise
        plan.soft_error = getattr(e, "error_type", None) or type(e).__name__
        plan.violations = ["fallback"]
        plan.text = fallback
        return fallback
    plan.violations = list(res.violations)
    plan.text = text
    return text


def word_stream(plan: TurnPlan) -> Iterator[str]:
    """Text/STT path: validated sentences as they complete. Raises InterviewerError."""
    if not plan.needs_model:
        if plan.text:
            yield plan.text
        return
    if plan.decision.max_questions > 0 or plan.decision.intervention in CONTEXTUAL_PRESENCE:
        # Generated whole: a question-allowed move so a repeated or generic question can be
        # regenerated once, and a one-sentence contextual beat so it can be checked for being
        # generic / unreferenced / unverified before the candidate sees any of it.
        yield word_complete(plan)
        return
    meta = CallMeta()
    plan.metas.append(meta)
    out: Dict[str, Any] = {}
    parts: List[str] = []
    for chunk in responder.generate_stream(_llm_for(plan), plan.decision, plan.case, plan.state_before,
                                           plan.turn.channel, plan.transcript, plan.turn.text, meta, out):
        parts.append(chunk)
        yield chunk
    res = out.get("validation")
    plan.violations = list(res.violations) if res else []
    plan.text = "".join(parts).strip()


def finalize(plan: TurnPlan, final_text: Optional[str], *, error: Optional[BaseException] = None,
             duplicate: bool = False, response_start_ms: Optional[int] = None) -> Dict[str, Any]:
    """Record the spoken/written line in state, emit the telemetry line, return the state dict."""
    # A turn whose line could not be produced did not happen for the candidate: keep the prior
    # state (the hint was never given, so the ladder must not move).
    st = plan.state_before if error is not None else plan.state_after
    if final_text and plan.lane != Lane.NO_OUTPUT and not error:
        st = state_machine.record_output(st, final_text)
    d = plan.decision
    meta = plan.metas[-1] if plan.metas else None
    a = plan.assessment or {}
    error_type = None
    if error is not None:
        error_type = getattr(error, "error_type", None) or type(error).__name__
    elif plan.soft_error:
        error_type = f"{plan.soft_error}:fallback"
    elif a.get("error_type"):
        error_type = a.get("error_type")
    telemetry.emit_turn({
        "session_id": plan.session_id,
        "attempt_id": plan.attempt_id,
        "turn_id": plan.turn.turn_id,
        "channel": plan.turn.channel.value,
        "turn_complete": not plan.turn.is_partial,
        "interviewer_state": d.state.value,
        "interviewer_lane": d.lane.value,
        "interviewer_mode": d.intervention.value,
        "decision_reason": d.reason,
        "provider": (meta.provider if meta else None) or a.get("provider"),
        "model": (meta.model if meta else None) or a.get("model"),
        "llm_called": bool(plan.metas),
        "llm_calls_n": len(plan.metas),
        "assessor_called": bool(a),
        "decision_timestamp": telemetry.now_ms(),
        "response_start_timestamp": response_start_ms,
        "error_type": error_type,
        "decision_ms": plan.decision_ms,
        "generation_ms": sum(m.latency_ms for m in plan.metas) or None,
        "first_token_ms": meta.first_token_ms if meta else None,
        "tokens_in": sum(int(m.tokens_in or 0) for m in plan.metas) + int(a.get("tokens_in") or 0),
        "tokens_out": sum(int(m.tokens_out or 0) for m in plan.metas) + int(a.get("tokens_out") or 0),
        "hint_level": st.hint_level,
        "phase": st.phase,
        "duplicate": duplicate or None,
        "violations": ",".join(plan.violations) or None,
    })
    return st.to_dict()


def is_error(e: BaseException) -> bool:
    return isinstance(e, InterviewerError)
