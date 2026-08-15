"""
Interview Engine — runs one interviewer turn against OpenAI.

Used by /attempts/{id}/messages to generate the assistant's live reply
when a candidate sends a new message. Streaming-capable: callers can
iterate over `stream_interviewer_reply(...)` to forward tokens via SSE.
"""

import os
import json
import time
from typing import Iterable, Dict, List, Generator, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

from services.ai_usage import log_ai_usage

from prompts.interview_prompts import (
    build_interviewer_messages,
    CONVERSATION_SCORING_SYSTEM_PROMPT,
    build_conversation_scoring_user_prompt,
)

load_dotenv()

_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not _OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment")

_client = OpenAI(api_key=_OPENAI_API_KEY)

# Mini is fine for the interviewer — replies are 1-3 sentences and the
# heavy lifting (final scoring) still uses gpt-4o.
INTERVIEWER_MODEL = "gpt-4o-mini"
SCORING_MODEL = "gpt-4o"

# Live-turn sampling. Was 0.4, which — combined with a prompt that literally
# contained the words "say 'Let's assume X'" — made the interviewer open EVERY
# reply with that exact phrase. A whole 7-question transcript read as one
# sentence repeated with different nouns. 0.75 buys phrasing variety without
# loosening the behavioural rules (those live in the system prompt, and the
# 1-3 sentence cap plus max_tokens still bound the reply).
# Scoring deliberately stays at its own low temperature — do NOT reuse this.
INTERVIEWER_TEMPERATURE = 0.75

# Repetition pressure at the sampler level, belt-and-braces with the prompt's
# "never open two consecutive replies the same way" rule. Small values only:
# these penalise token reuse, and the interviewer legitimately needs to repeat
# domain nouns (revenue, market, segment) turn after turn.
INTERVIEWER_FREQUENCY_PENALTY = 0.35
INTERVIEWER_PRESENCE_PENALTY = 0.25


class InterviewEngineError(Exception):
    pass


# -----------------------------------------------------------------------------
# Live turn (streaming)
# -----------------------------------------------------------------------------

def stream_interviewer_reply(
    case_content: str,
    case_type: str,
    transcript: Iterable[Dict[str, str]],
    new_user_message: str,
    user_id: Optional[str] = None,
    clarifications_exhausted: bool = False,
) -> Generator[str, None, None]:
    """Yield text chunks as the interviewer responds.

    Intended to be wrapped in an SSE StreamingResponse. Each yielded chunk
    is a partial string suitable for client-side concatenation.

    `clarifications_exhausted` makes the interviewer decline the clarification
    and redirect, rather than the caller returning no reply at all.
    """
    messages = build_interviewer_messages(
        case_content=case_content,
        case_type=case_type,
        transcript=transcript,
        new_user_message=new_user_message,
        clarifications_exhausted=clarifications_exhausted,
    )
    try:
        t0 = time.time()
        stream = _client.chat.completions.create(
            model=INTERVIEWER_MODEL,
            messages=messages,
            temperature=INTERVIEWER_TEMPERATURE,
            frequency_penalty=INTERVIEWER_FREQUENCY_PENALTY,
            presence_penalty=INTERVIEWER_PRESENCE_PENALTY,
            max_tokens=180,   # cap — interviewer replies must stay short
            stream=True,
            stream_options={"include_usage": True},  # final chunk carries token usage
        )
    except Exception as e:
        raise InterviewEngineError(f"OpenAI streaming call failed: {e}")

    class _U:  # tiny shim so log_ai_usage can read .usage off a response-like object
        usage = None
        id = None

    final = _U()
    try:
        for chunk in stream:
            if getattr(chunk, "usage", None):
                final.usage = chunk.usage
                final.id = getattr(chunk, "id", None)
            try:
                delta = chunk.choices[0].delta
                token = getattr(delta, "content", None)
            except (AttributeError, IndexError):
                token = None
            if token:
                yield token
    except Exception as e:
        raise InterviewEngineError(f"Stream interrupted: {e}")
    finally:
        log_ai_usage(user_id=user_id, endpoint="/attempts/messages", model=INTERVIEWER_MODEL,
                     response=final, latency_ms=int((time.time() - t0) * 1000))


def complete_interviewer_reply(
    case_content: str,
    case_type: str,
    transcript: Iterable[Dict[str, str]],
    new_user_message: str,
    clarifications_exhausted: bool = False,
) -> str:
    """Non-streaming variant — used when SSE is not available."""
    messages = build_interviewer_messages(
        case_content=case_content,
        case_type=case_type,
        transcript=transcript,
        new_user_message=new_user_message,
        clarifications_exhausted=clarifications_exhausted,
    )
    try:
        resp = _client.chat.completions.create(
            model=INTERVIEWER_MODEL,
            messages=messages,
            temperature=INTERVIEWER_TEMPERATURE,
            frequency_penalty=INTERVIEWER_FREQUENCY_PENALTY,
            presence_penalty=INTERVIEWER_PRESENCE_PENALTY,
            max_tokens=180,
        )
    except Exception as e:
        raise InterviewEngineError(f"OpenAI call failed: {e}")
    return (resp.choices[0].message.content or "").strip()


# -----------------------------------------------------------------------------
# Final scoring (at submit)
# -----------------------------------------------------------------------------
# Two independent paths:
#   case_type == 'guesstimate' -> reuse the existing 5-dim rubric + arithmetic
#                                 backstop in services/ai_scorer.score_guesstimate_answer
#   anything else              -> general conversation analysis (case rubric TBD)

def _flatten_for_legacy_scorer(
    transcript: Iterable[Dict[str, str]],
    final_recommendation: str,
) -> str:
    """Collapse the conversation + recommendation into the single answer_text
    string that the legacy guesstimate scorer expects. The structured format
    mirrors what a candidate would have typed into the old textarea, with the
    interviewer's turns folded in as light prompts so segmentation reasoning
    stays readable to the scorer.
    """
    lines: List[str] = []
    for t in transcript:
        role = (t.get("role") or "user").upper()
        kind = t.get("kind") or "text"
        content = (t.get("content") or "").strip()
        if not content:
            continue
        # Voice is collapsed to text for SCORING (owner decision 2026-08-13): a
        # spoken attempt must be judged on the same document as a typed one, or
        # talk mode quietly becomes a different exam. image/file stay tagged —
        # those genuinely change what the turn means (a chart was uploaded).
        display_kind = "text" if kind == "voice" else kind
        prefix = role if display_kind == "text" else f"{role} ({display_kind})"
        lines.append(f"[{prefix}] {content}")
    lines.append("")
    lines.append(f"[FINAL] {final_recommendation.strip()}")
    return "\n".join(lines)


def _score_case_conversation(
    case_content: str,
    case_type: str,
    transcript: Iterable[Dict[str, str]],
    final_recommendation: str,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """General conversation analysis for CASES (not guesstimates).
    The formal case rubric is being developed separately; until it lands this
    returns a holistic { score, breakdown:{overall}, strengths, improvements,
    summary } dict that the legacy submissions table can store unchanged.
    """
    user_prompt = build_conversation_scoring_user_prompt(
        case_content=case_content,
        case_type=case_type,
        transcript=transcript,
        final_recommendation=final_recommendation,
    )
    try:
        t0 = time.time()
        resp = _client.chat.completions.create(
            model=SCORING_MODEL,
            messages=[
                {"role": "system", "content": CONVERSATION_SCORING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=2500,
            response_format={"type": "json_object"},
        )
        log_ai_usage(user_id=user_id, endpoint="/attempts/submit", model=SCORING_MODEL,
                     response=resp, latency_ms=int((time.time() - t0) * 1000))
    except Exception as e:
        raise InterviewEngineError(f"Scoring call failed: {e}")

    raw = resp.choices[0].message.content or ""
    try:
        feedback = json.loads(raw)
    except json.JSONDecodeError as e:
        raise InterviewEngineError(f"Scorer returned invalid JSON: {e}. Raw: {raw[:200]}")

    required = {"score", "breakdown", "strengths", "improvements", "summary"}
    missing = required - set(feedback.keys())
    if missing:
        raise InterviewEngineError(f"Scorer missing keys: {missing}")
    try:
        feedback["score"] = max(0, min(100, int(feedback["score"])))
    except (TypeError, ValueError):
        raise InterviewEngineError("Scorer returned non-integer score")
    # Ensure breakdown is a dict — required by the SubmitResponse contract.
    if not isinstance(feedback.get("breakdown"), dict):
        feedback["breakdown"] = {"overall": feedback["score"]}
    feedback.setdefault("rubric", "case")
    return feedback


def score_conversation(
    case_content: str,
    case_type: str,
    transcript: Iterable[Dict[str, str]],
    final_recommendation: str,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Top-level scorer. Branches on case_type:

    - 'guesstimate' -> hands off to the existing
      `services.ai_scorer.score_guesstimate_answer`, feeding it a flattened
      transcript so the 5-dim rubric + deterministic arithmetic backstop run
      unchanged. The session conversation is treated as one long answer.
    - any other type -> general conversation analysis (the formal case rubric
      is being developed separately and will replace the prompt body in place).

    Returns the same dict shape in both branches:
      { score, breakdown, strengths, improvements, summary, rubric, backstop? }
    """
    is_guesstimate = (case_type or "").lower() == "guesstimate"
    if is_guesstimate:
        # Lazy import — avoids a hard dep on ai_scorer at module load and lets
        # this file be unit-tested in isolation.
        from services.ai_scorer import score_guesstimate_answer, AIScoringError
        flat = _flatten_for_legacy_scorer(transcript, final_recommendation)
        try:
            return score_guesstimate_answer(case_content=case_content, user_answer=flat, user_id=user_id)
        except AIScoringError as e:
            raise InterviewEngineError(f"Guesstimate scoring failed: {e}")

    return _score_case_conversation(
        case_content=case_content,
        case_type=case_type,
        transcript=transcript,
        final_recommendation=final_recommendation,
        user_id=user_id,
    )


# -----------------------------------------------------------------------------
# Clarification classifier
# -----------------------------------------------------------------------------
# Lives in services/clarification_counter.py — a pure string heuristic with no
# OpenAI dependency, moved out so it can be tested without an API key (this
# module builds a client at import time and raises without one). Re-exported
# here so every existing importer keeps working unchanged.
from services.clarification_counter import count_clarifications  # noqa: E402,F401
