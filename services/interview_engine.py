"""
Interview Engine — runs one interviewer turn against OpenAI.

Implements V11 Contextual Response Generation Layer.
Pipeline: Signals -> Response Function (Python) -> Control Packet (JSON) -> Natural Language (LLM).
Retains 0-token Fast Lane for true SILENCE and simple local acks.
"""

import os
import json
import time
from typing import Iterable, Dict, List, Generator, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

from services.ai_usage import log_ai_usage
from services.ai_providers import resolve_llm, openai_client

from prompts.interview_prompts import (
    build_interviewer_messages,
    CONVERSATION_SCORING_SYSTEM_PROMPT,
    build_conversation_scoring_user_prompt,
)
from prompts.interview_prompts_v2 import build_adaptive_interviewer_messages
from services.session_signals import compute_signals
from services.interviewer_mode import ALLOW_QUESTIONS, build_mode_block
from services.interviewer_decision import (
    StreamTagStripper, parse_control_tag, enforce_mode, 
    determine_response_function, build_interviewer_control_packet, get_fast_lane_event
)

load_dotenv()

_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not _OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment")

_client = OpenAI(api_key=_OPENAI_API_KEY)

INTERVIEWER_MODEL = "gpt-4o-mini"
SCORING_MODEL = "gpt-4o"

INTERVIEWER_TEMPERATURE = 0.75
INTERVIEWER_FREQUENCY_PENALTY = 0.35
INTERVIEWER_PRESENCE_PENALTY = 0.25


class InterviewEngineError(Exception):
    pass

def _adaptive_enabled() -> bool:
    return os.getenv("ADAPTIVE_INTERVIEWER", "").strip().lower() in ("1", "true", "yes", "on")

def _teaching_policy(explicit=None) -> str:
    p = (explicit or os.getenv("INTERVIEWER_TEACHING_POLICY", "coached") or "").strip().lower()
    return p if p in ("exam", "coached") else "coached"

def _resolve_adaptive_llm():
    model = (os.getenv("INTERVIEWER_ADAPTIVE_MODEL", "gpt-4o-mini") or "gpt-4o-mini").strip()
    return openai_client() or _client, model, "openai"

def _build_adaptive_messages(case_content, case_type, transcript, new_user_message,
                             clarifications_exhausted, response_function, control_packet, policy):
    # The mode block now contains the rich JSON payload and strict LLM generation instructions
    block = build_mode_block(response_function, control_packet)
    
    return build_adaptive_interviewer_messages(
        case_content=case_content,
        case_type=case_type,
        transcript=transcript,
        new_user_message=new_user_message,
        teaching_policy=policy,
        signals_block=block,  # Passed as the primary driving instruction
        clarifications_exhausted=clarifications_exhausted,
    )


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
    teaching_policy: Optional[str] = None,
    prior_state: Optional[dict] = None,
    control_out: Optional[dict] = None,
    channel: str = "text",
    is_voice_partial: bool = False
) -> Generator[str, None, None]:
    adaptive = _adaptive_enabled()
    
    if adaptive:
        tlist = list(transcript)
        policy = _teaching_policy(teaching_policy)
        
        # 1. State Estimation
        signals = compute_signals(tlist, new_user_message, policy, prior_state=prior_state, channel=channel, is_voice_partial=is_voice_partial)
        
        # 2. Determine Intent (Policy Layer)
        response_function = determine_response_function(signals)
        
        # 3. Fast Lane Intercept (0 LLM Tokens)
        if response_function == "NO_OUTPUT":
            t0 = time.time()
            if control_out is not None:
                control_out["tag"] = {"mode": "NO_OUTPUT", "intervention": "silence"}
                control_out["mode"] = "NO_OUTPUT"
            yield ""
            actual_ms = int((time.time() - t0) * 1000)
            class _FastUsage:
                usage = None
                id = None
            log_ai_usage(user_id=user_id, endpoint="/attempts/messages/fastlane_silence", model="local", response=_FastUsage(), latency_ms=max(1, actual_ms))
            return
            
        # Fast Lane Intercept for short, non-substantive affirmations
        if response_function in ["SHORT_ACK", "HAND_BACK"] and not signals.get("is_substantive_reasoning"):
            t0 = time.time()
            fast_event_dict = get_fast_lane_event(response_function, signals)
            if control_out is not None:
                control_out["tag"] = {"mode": response_function, "intervention": "fast_lane"}
                control_out["mode"] = response_function
                
            if channel == "text":
                yield fast_event_dict["data"]["text"]
            else:
                yield json.dumps(fast_event_dict)
                
            actual_ms = int((time.time() - t0) * 1000)
            class _FastUsageBeat:
                usage = None
                id = None
            log_ai_usage(user_id=user_id, endpoint="/attempts/messages/fastlane_event", model="local", response=_FastUsageBeat(), latency_ms=max(1, actual_ms))
            return
            
        # 4. Contextual Generation (Deep Lane)
        control_packet = build_interviewer_control_packet(signals, response_function)
        allow_questions = control_packet["interviewer_control"]["interviewer_decision"]["question_allowed"]
        
        messages = _build_adaptive_messages(
            case_content, case_type, tlist, new_user_message, clarifications_exhausted,
            response_function, control_packet, policy
        )
        cli, model, provider = _resolve_adaptive_llm()
    else:
        messages = build_interviewer_messages(
            case_content=case_content, case_type=case_type, transcript=transcript,
            new_user_message=new_user_message, clarifications_exhausted=clarifications_exhausted
        )
        cli, model, provider = resolve_llm("interviewer")

    def _complete(c, m):
        return c.chat.completions.create(
            model=m,
            messages=messages,
            temperature=INTERVIEWER_TEMPERATURE,
            frequency_penalty=INTERVIEWER_FREQUENCY_PENALTY,
            presence_penalty=INTERVIEWER_PRESENCE_PENALTY,
            max_tokens=180,
        )

    def _open_stream(c, m):
        return c.chat.completions.create(
            model=m,
            messages=messages,
            temperature=INTERVIEWER_TEMPERATURE,
            frequency_penalty=INTERVIEWER_FREQUENCY_PENALTY,
            presence_penalty=INTERVIEWER_PRESENCE_PENALTY,
            max_tokens=180,
            stream=True,
            stream_options={"include_usage": True},
        )

    t0 = time.time()
    
    # 5. Pre-Emission Safety Validation
    if adaptive and not allow_questions:
        try:
            resp = _complete(cli, model)
        except Exception as e:
            if provider != "openai":
                cli, model = openai_client(), INTERVIEWER_MODEL
                resp = _complete(cli, model)
            else:
                raise InterviewEngineError(f"OpenAI call failed: {e}")
        
        raw_text = (resp.choices[0].message.content or "").strip()
        tag, clean_text = parse_control_tag(raw_text)
        final_text = enforce_mode(clean_text, response_function, allow_questions, policy)
        
        if control_out is not None:
            control_out["tag"] = tag or {}
            control_out["mode"] = response_function
            
        yield final_text
        log_ai_usage(user_id=user_id, endpoint="/attempts/messages", model=model,
                     response=resp, latency_ms=int((time.time() - t0) * 1000))
        return

    # STANDARD STREAMING
    try:
        stream = _open_stream(cli, model)
    except Exception as e:
        if provider != "openai":
            cli, model = openai_client(), INTERVIEWER_MODEL
            try:
                stream = _open_stream(cli, model)
            except Exception as e2:
                raise InterviewEngineError(f"OpenAI streaming call failed: {e2}")
        else:
            raise InterviewEngineError(f"OpenAI streaming call failed: {e}")

    class _U:
        usage = None
        id = None

    final = _U()
    stripper = StreamTagStripper() if adaptive else None
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
                if stripper is not None:
                    for _out in stripper.feed(token):
                        yield _out
                else:
                    yield token
        if stripper is not None:
            for _out in stripper.flush():
                yield _out
            if control_out is not None:
                control_out["tag"] = getattr(stripper, "tag", {}) or {}
                control_out["mode"] = response_function if 'response_function' in locals() else "UNKNOWN"
    except Exception as e:
        raise InterviewEngineError(f"Stream interrupted: {e}")
    finally:
        log_ai_usage(user_id=user_id, endpoint="/attempts/messages", model=model,
                     response=final, latency_ms=int((time.time() - t0) * 1000))


def complete_interviewer_reply(
    case_content: str,
    case_type: str,
    transcript: Iterable[Dict[str, str]],
    new_user_message: str,
    clarifications_exhausted: bool = False,
    teaching_policy: Optional[str] = None,
    prior_state: Optional[dict] = None,
    control_out: Optional[dict] = None,
    channel: str = "text",
    is_voice_partial: bool = False
) -> str:
    adaptive = _adaptive_enabled()
    transcript = list(transcript)
    
    if adaptive:
        policy = _teaching_policy(teaching_policy)
        signals = compute_signals(transcript, new_user_message, policy, prior_state=prior_state, channel=channel, is_voice_partial=is_voice_partial)
        
        response_function = determine_response_function(signals)
        
        if response_function == "NO_OUTPUT":
            if control_out is not None:
                control_out["tag"] = {"mode": "NO_OUTPUT", "intervention": "silence"}
                control_out["mode"] = "NO_OUTPUT"
            return ""
            
        if response_function in ["SHORT_ACK", "HAND_BACK"] and not signals.get("is_substantive_reasoning"):
            fast_event = get_fast_lane_event(response_function, signals)
            if control_out is not None:
                control_out["tag"] = {"mode": response_function, "intervention": "fast_lane"}
                control_out["mode"] = response_function
            return fast_event["data"]["text"] if channel == "text" else json.dumps(fast_event)
            
        control_packet = build_interviewer_control_packet(signals, response_function)
        allow_questions = control_packet["interviewer_control"]["interviewer_decision"]["question_allowed"]
        
        messages = _build_adaptive_messages(
            case_content, case_type, transcript, new_user_message, clarifications_exhausted,
            response_function, control_packet, policy
        )
        cli, model, provider = _resolve_adaptive_llm()
    else:
        messages = build_interviewer_messages(
            case_content=case_content, case_type=case_type, transcript=transcript,
            new_user_message=new_user_message, clarifications_exhausted=clarifications_exhausted
        )
        cli, model, provider = resolve_llm("interviewer")

    def _complete(c, m):
        return c.chat.completions.create(
            model=m,
            messages=messages,
            temperature=INTERVIEWER_TEMPERATURE,
            frequency_penalty=INTERVIEWER_FREQUENCY_PENALTY,
            presence_penalty=INTERVIEWER_PRESENCE_PENALTY,
            max_tokens=180,
        )

    try:
        resp = _complete(cli, model)
    except Exception as e:
        if provider != "openai":
            try:
                resp = _complete(openai_client(), INTERVIEWER_MODEL)
            except Exception as e2:
                raise InterviewEngineError(f"OpenAI call failed: {e2}")
        else:
            raise InterviewEngineError(f"OpenAI call failed: {e}")
            
    text = (resp.choices[0].message.content or "").strip()
    if adaptive:
        tag, text = parse_control_tag(text)
        text = enforce_mode(text, response_function, allow_questions, policy)
        if control_out is not None:
            control_out["tag"] = tag or {}
            control_out["mode"] = response_function
    return text


# -----------------------------------------------------------------------------
# Final scoring (at submit)
# -----------------------------------------------------------------------------

def _flatten_for_legacy_scorer(
    transcript: Iterable[Dict[str, str]],
    final_recommendation: str,
) -> str:
    lines: List[str] = []
    for t in transcript:
        role = (t.get("role") or "user").upper()
        kind = t.get("kind") or "text"
        content = (t.get("content") or "").strip()
        if not content:
            continue
        display_kind = "text" if kind == "voice" else kind
        prefix = role if display_kind == "text" else f"{role} ({display_kind})"
        lines.append(f"[{prefix}] {content}")
    lines.append("")
    lines.append(f"[FINAL] {final_recommendation.strip()}")
    return "\n".join(lines)


def _transcript_body_text(transcript: Iterable[Dict[str, str]]) -> str:
    parts: List[str] = []
    for t in transcript:
        if (t.get("role") or "user") != "user":
            continue
        if (t.get("kind") or "") == "recommendation":
            continue
        c = (t.get("content") or "").strip()
        if c:
            parts.append(c)
    return "\n".join(parts)


def _rec_is_weak(recommendation: str) -> bool:
    from services.answer_validity import deterministic_verdict
    r = (recommendation or "").strip()
    if not r:
        return True
    return deterministic_verdict(r) is not None


def _score_case_conversation(
    case_content: str,
    case_type: str,
    transcript: Iterable[Dict[str, str]],
    final_recommendation: str,
    user_id: Optional[str] = None,
    case_id: Optional[str] = None,
) -> Dict[str, Any]:
    from services.answer_validity import screen_answer
    from services.ai_scorer import _enforce_case, _rejection_case, _is_hard_reject

    body = _transcript_body_text(transcript)
    rec = (final_recommendation or "").strip()
    body_validity = screen_answer(case_content, case_type, body, user_id)

    if _is_hard_reject(body_validity):
        if rec and not _rec_is_weak(rec):
            rec_validity = screen_answer(case_content, case_type, rec, user_id)
            if _is_hard_reject(rec_validity):
                return _rejection_case(case_type, body_validity)
            validity = rec_validity
        else:
            return _rejection_case(case_type, body_validity)
    else:
        validity = body_validity

    recommendation_missing = _rec_is_weak(rec)

    user_prompt = build_conversation_scoring_user_prompt(
        case_content=case_content,
        case_type=case_type,
        transcript=transcript,
        final_recommendation=final_recommendation,
        thin=validity["verdict"] in ("thin", "off_topic"),
        recommendation_missing=recommendation_missing,
    )

    try:
        from services.exemplar_bank import build_exemplar_reference_block
        ref = build_exemplar_reference_block(case_id=case_id, case_content=case_content)
        if ref:
            user_prompt = user_prompt + "\n\n" + ref
    except Exception:
        pass

    feedback = None
    last_err = None
    for _attempt in range(2):
        try:
            t0 = time.time()
            resp = _client.chat.completions.create(
                model=SCORING_MODEL,
                messages=[
                    {"role": "system", "content": CONVERSATION_SCORING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=8000,
                response_format={"type": "json_object"},
            )
            log_ai_usage(user_id=user_id, endpoint="/attempts/submit", model=SCORING_MODEL,
                         response=resp, latency_ms=int((time.time() - t0) * 1000))
        except Exception as e:
            raise InterviewEngineError(f"Scoring call failed: {e}")

        raw = resp.choices[0].message.content or ""
        try:
            feedback = json.loads(raw)
            break
        except json.JSONDecodeError as e:
            last_err = e
            
    if feedback is None:
        raise InterviewEngineError(f"Scorer returned invalid JSON after retry: {last_err}. Raw: {raw[:200]}")

    required = {"score", "breakdown", "strengths", "improvements", "summary"}
    missing = required - set(feedback.keys())
    if missing:
        raise InterviewEngineError(f"Scorer missing keys: {missing}")
    if not isinstance(feedback.get("breakdown"), dict):
        raise InterviewEngineError("Scorer 'breakdown' is not an object")

    return _enforce_case(feedback, validity)


def score_conversation(
    case_content: str,
    case_type: str,
    transcript: Iterable[Dict[str, str]],
    final_recommendation: str,
    user_id: Optional[str] = None,
    case_id: Optional[str] = None,
) -> Dict[str, Any]:
    is_guesstimate = (case_type or "").lower() == "guesstimate"
    if is_guesstimate:
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
        case_id=case_id,
    )

from services.clarification_counter import count_clarifications  # noqa: E402,F401
