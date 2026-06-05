"""
Interview Engine — runs one interviewer turn against OpenAI.

Used by /attempts/{id}/messages to generate the assistant's live reply
when a candidate sends a new message. Streaming-capable: callers can
iterate over `stream_interviewer_reply(...)` to forward tokens via SSE.
"""

import os
import json
from typing import Iterable, Dict, List, Generator, Any
from openai import OpenAI
from dotenv import load_dotenv

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
) -> Generator[str, None, None]:
    """Yield text chunks as the interviewer responds.

    Intended to be wrapped in an SSE StreamingResponse. Each yielded chunk
    is a partial string suitable for client-side concatenation.
    """
    messages = build_interviewer_messages(
        case_content=case_content,
        case_type=case_type,
        transcript=transcript,
        new_user_message=new_user_message,
    )
    try:
        stream = _client.chat.completions.create(
            model=INTERVIEWER_MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=180,   # cap — interviewer replies must stay short
            stream=True,
        )
    except Exception as e:
        raise InterviewEngineError(f"OpenAI streaming call failed: {e}")

    try:
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta
                token = getattr(delta, "content", None)
            except (AttributeError, IndexError):
                token = None
            if token:
                yield token
    except Exception as e:
        raise InterviewEngineError(f"Stream interrupted: {e}")


def complete_interviewer_reply(
    case_content: str,
    case_type: str,
    transcript: Iterable[Dict[str, str]],
    new_user_message: str,
) -> str:
    """Non-streaming variant — used when SSE is not available."""
    messages = build_interviewer_messages(
        case_content=case_content,
        case_type=case_type,
        transcript=transcript,
        new_user_message=new_user_message,
    )
    try:
        resp = _client.chat.completions.create(
            model=INTERVIEWER_MODEL,
            messages=messages,
            temperature=0.4,
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
        prefix = role if kind == "text" else f"{role} ({kind})"
        lines.append(f"[{prefix}] {content}")
    lines.append("")
    lines.append(f"[FINAL] {final_recommendation.strip()}")
    return "\n".join(lines)


def _score_case_conversation(
    case_content: str,
    case_type: str,
    transcript: Iterable[Dict[str, str]],
    final_recommendation: str,
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
            return score_guesstimate_answer(case_content=case_content, user_answer=flat)
        except AIScoringError as e:
            raise InterviewEngineError(f"Guesstimate scoring failed: {e}")

    return _score_case_conversation(
        case_content=case_content,
        case_type=case_type,
        transcript=transcript,
        final_recommendation=final_recommendation,
    )


# -----------------------------------------------------------------------------
# Clarification classifier
# -----------------------------------------------------------------------------

def count_clarifications(text: str) -> int:
    """Heuristic — counts how many clarification questions are in the turn.

    Counts question marks. If none, but it opens with an interrogative phrase,
    counts as 1. Prevents users from packing 5 questions into a single message
    and only consuming 1 quota point.
    """
    if not text:
        return 0
    
    q_count = text.count("?")
    if q_count > 0:
        return q_count
        
    stripped = text.strip().lower()
    openers = (
        "what ", "why ", "how ", "when ", "where ", "who ",
        "is ", "are ", "do ", "does ", "did ", "can ", "could ",
        "should ", "would ", "may ", "might ",
    )
    first = stripped.split("\n", 1)[0]
    if any(first.startswith(o) for o in openers):
        return 1
        
    return 0
