"""
Turns a SUBSTANTIVE decision that needs case-aware wording into validated text.

  * voice (complete):  generate -> validate -> if invalid, ONE regeneration with a
                       corrective note -> validate -> else EmptyModelOutput.
  * text/stt (stream): tokens pass through SentenceGate; only validated sentences
                       are emitted. If nothing survives -> EmptyModelOutput.

It never substitutes canned content for a failed generation.
"""
from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Protocol, Tuple

from services.interviewer import prompting
from services.interviewer.providers import CallMeta
from services.interviewer.types import CONTEXTUAL_PRESENCE, BrainState, CaseContext, Channel, Decision, EmptyModelOutput
from services.interviewer.validate import (
    SentenceFilter, SentenceGate, ValidationResult, contextual_violations, validate_text,
)

_REGEN_NOTE = ("Your previous draft broke the rules ({why}). Rewrite it: carry out the response_function directly, "
               "no praise, no refusal, no generic or repeated questions, within the length and question limits.")
_REGEN_HINTS = {
    "generic_ack": "it was a stock acknowledgement - name what the candidate actually did",
    "unreferenced": "it did not refer to anything the candidate said",
    "unverified_claim": "it said or implied their work is right, which nothing has checked",
}


class LLM(Protocol):
    def complete(self, messages: List[Dict[str, str]], *, max_tokens: int) -> Tuple[str, CallMeta]: ...
    def stream(self, messages: List[Dict[str, str]], *, max_tokens: int, meta: CallMeta) -> Iterator[str]: ...


def _may_say_correct(decision: Decision) -> bool:
    if decision.intervention in CONTEXTUAL_PRESENCE:
        return bool((decision.detail or {}).get("may_say_correct"))
    return True


def _filter(decision: Decision, channel: Channel, state: BrainState) -> SentenceFilter:
    return SentenceFilter(max_questions=decision.max_questions,
                          max_sentences=prompting.max_sentences(decision, channel),
                          recent_lines=state.recent_lines, recent_questions=state.asked_questions,
                          may_say_correct=_may_say_correct(decision))


_SERIOUS = {"banned_phrase", "leak", "question_budget", "generic_question", "repeated_question"}
# For a contextual beat these mean "not the line we asked for" -> regenerate, then fall back.
_SERIOUS_CONTEXTUAL = _SERIOUS | {"generic_ack", "unreferenced"}


def _check(raw: str, decision: Decision, channel: Channel, state: BrainState, candidate_text: str) -> ValidationResult:
    res = validate_text(raw, max_questions=decision.max_questions, max_sentences=prompting.max_sentences(decision, channel),
                        recent_lines=state.recent_lines, recent_questions=state.asked_questions,
                        may_say_correct=_may_say_correct(decision))
    if decision.intervention in CONTEXTUAL_PRESENCE and not res.empty:
        res.violations = list(res.violations) + contextual_violations(res.text, candidate_text)
    return res


def generate_complete(llm: LLM, decision: Decision, case: CaseContext, state: BrainState, channel: Channel,
                      transcript: List[Dict[str, str]], candidate_text: str,
                      metas_out: Optional[List[CallMeta]] = None) -> Tuple[str, List[CallMeta], ValidationResult]:
    """Generate -> validate -> at most one regeneration. Every provider call is appended to
    `metas_out` as it happens, so calls are accounted for even when this raises."""
    msgs = prompting.build_messages(decision, case, state, channel, transcript, candidate_text)
    mt = prompting.max_tokens(decision, channel)
    metas: List[CallMeta] = metas_out if metas_out is not None else []
    serious_set = _SERIOUS_CONTEXTUAL if decision.intervention in CONTEXTUAL_PRESENCE else _SERIOUS
    raw, meta = llm.complete(msgs, max_tokens=mt)
    metas.append(meta)
    res = _check(raw, decision, channel, state, candidate_text)
    serious = [v for v in res.violations if v in serious_set]
    if res.empty or serious:
        why = ", ".join(_REGEN_HINTS.get(v, v) for v in (serious or ["empty"]))
        retry = msgs + [{"role": "assistant", "content": raw or "(empty)"},
                        {"role": "system", "content": _REGEN_NOTE.format(why=why)}]
        raw2, meta2 = llm.complete(retry, max_tokens=mt)
        metas.append(meta2)
        res2 = _check(raw2, decision, channel, state, candidate_text)
        serious2 = [v for v in res2.violations if v in serious_set]
        if not res2.empty and (not serious2 or decision.intervention not in CONTEXTUAL_PRESENCE):
            res2.violations = sorted(set(res.violations) | set(res2.violations) | {"regenerated"})
            return res2.text, metas, res2
        if not res.empty and decision.intervention not in CONTEXTUAL_PRESENCE:
            return res.text, metas, res
        raise EmptyModelOutput(f"no valid interviewer line after regeneration ({why})")
    return res.text, metas, res


def generate_stream(llm: LLM, decision: Decision, case: CaseContext, state: BrainState, channel: Channel,
                    transcript: List[Dict[str, str]], candidate_text: str, meta: CallMeta,
                    result_out: Optional[Dict] = None) -> Iterator[str]:
    """Yields validated sentences (each with a trailing space except the last)."""
    msgs = prompting.build_messages(decision, case, state, channel, transcript, candidate_text)
    f = _filter(decision, channel, state)
    gate = SentenceGate(f)
    emitted = 0
    for tok in llm.stream(msgs, max_tokens=prompting.max_tokens(decision, channel), meta=meta):
        for sent in gate.feed(tok):
            yield (" " if emitted else "") + sent
            emitted += 1
    for sent in gate.flush():
        yield (" " if emitted else "") + sent
        emitted += 1
    res = f.result()
    if result_out is not None:
        result_out["validation"] = res
    if emitted == 0:
        raise EmptyModelOutput("model output empty or fully rejected by validation")
