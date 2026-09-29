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
from services.interviewer.types import BrainState, CaseContext, Channel, Decision, EmptyModelOutput
from services.interviewer.validate import SentenceFilter, SentenceGate, ValidationResult, validate_text

_REGEN_NOTE = ("Your previous draft broke the rules ({why}). Rewrite it: carry out the move directly, no praise, "
               "no refusal, no generic or repeated questions, within the length and question limits.")


class LLM(Protocol):
    def complete(self, messages: List[Dict[str, str]], *, max_tokens: int) -> Tuple[str, CallMeta]: ...
    def stream(self, messages: List[Dict[str, str]], *, max_tokens: int, meta: CallMeta) -> Iterator[str]: ...


def _filter(decision: Decision, channel: Channel, state: BrainState) -> SentenceFilter:
    return SentenceFilter(max_questions=decision.max_questions,
                          max_sentences=prompting.max_sentences(decision, channel),
                          recent_lines=state.recent_lines, recent_questions=state.asked_questions)


_SERIOUS = {"banned_phrase", "leak", "question_budget", "generic_question", "repeated_question"}


def generate_complete(llm: LLM, decision: Decision, case: CaseContext, state: BrainState, channel: Channel,
                      transcript: List[Dict[str, str]], candidate_text: str) -> Tuple[str, List[CallMeta], ValidationResult]:
    msgs = prompting.build_messages(decision, case, state, channel, transcript, candidate_text)
    mt = prompting.max_tokens(decision, channel)
    metas: List[CallMeta] = []
    raw, meta = llm.complete(msgs, max_tokens=mt)
    metas.append(meta)
    res = validate_text(raw, max_questions=decision.max_questions, max_sentences=prompting.max_sentences(decision, channel),
                        recent_lines=state.recent_lines, recent_questions=state.asked_questions)
    serious = [v for v in res.violations if v in _SERIOUS]
    if res.empty or serious:
        why = ", ".join(serious or ["empty"])
        retry = msgs + [{"role": "assistant", "content": raw or "(empty)"},
                        {"role": "system", "content": _REGEN_NOTE.format(why=why)}]
        raw2, meta2 = llm.complete(retry, max_tokens=mt)
        metas.append(meta2)
        res2 = validate_text(raw2, max_questions=decision.max_questions,
                             max_sentences=prompting.max_sentences(decision, channel),
                             recent_lines=state.recent_lines, recent_questions=state.asked_questions)
        if not res2.empty:
            res2.violations = sorted(set(res.violations) | set(res2.violations) | {"regenerated"})
            return res2.text, metas, res2
        if not res.empty:
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
