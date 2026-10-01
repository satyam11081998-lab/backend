"""
Interview Engine — runs one interviewer turn against OpenAI.

Implements V10 MECE Architecture: True Semantic Events, Dual Gates, and Safe UX Fallbacks.
V12: function first, language second. Signals -> gates -> RESPONSE FUNCTION
(services/interviewer_decision.decide_response) -> JSON INTERVIEWER CONTROL PACKET
-> the model words the turn. The 0-token fast lane is kept only for silence and
for turns that carried almost nothing; substantive candidate turns are always
worded from their own content, validated, and fall back to a plain hand-back
(never an error) if the model cannot produce a usable line.
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
from services.session_signals import compute_signals, needs_contextual_assessment, build_signal_block
from services.interviewer_mode import get_function_instruction, ALLOW_QUESTIONS, build_mode_block
from services.interviewer_decision import (
    StreamTagStripper, parse_control_tag, sanitize_reply, enforce_mode, 
    assess_context_with_llm, evaluate_intervention_gate, get_fast_lane_event,
    decide_response, build_interviewer_control_packet, validate_contextual_line,
    CONTEXTUAL_PRESENCE, scrub_control_leak, StreamLeakGuard,
)
from services.learning_model import evaluate_intervention_outcome, build_learning_block
from prompts.voice_renderer import strip_say_label

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
                             clarifications_exhausted, signals, mode, instruction, allow_questions,
                             policy, prior_state, outcome, packet=None, function=None):
    # V12: the control packet carries the deterministic read of the learner (it
    # replaces the SESSION SIGNALS lines); the learner-profile block is unchanged.
    head = (build_mode_block(mode, instruction, allow_questions, packet, function) if packet
            else build_signal_block(signals) + "\n\n" + build_mode_block(mode, instruction, allow_questions))
    block = (head
             + "\n\n" + build_learning_block(
                 (prior_state or {}).get("profile"), signals, outcome))
    # Voice lines saved before the label fix may start with "SAY:"; the model
    # must never see (and copy) that protocol label.
    transcript = [dict(t, content=strip_say_label(t.get("content") or "")) if t.get("role") == "assistant" else t
                  for t in transcript]
    return build_adaptive_interviewer_messages(
        case_content=case_content,
        case_type=case_type,
        transcript=transcript,
        new_user_message=new_user_message,
        teaching_policy=policy,
        signals_block=block,
        clarifications_exhausted=clarifications_exhausted,
    )


CONTEXTUAL_MAX_TOKENS = 90
CONTEXTUAL_TEMPERATURE = 0.7


class _NoUsage:
    usage = None
    id = None


def _recent_lines(transcript) -> List[str]:
    return [(t.get("content") or "").strip() for t in transcript
            if t.get("role") == "assistant" and (t.get("content") or "").strip()][-3:]


def _plan_adaptive_turn(case_content, case_type, tlist, new_user_message, clarifications_exhausted,
                        teaching_policy, prior_state, channel, is_voice_partial) -> Dict[str, Any]:
    """Signals -> gates -> response function -> (for model turns) control packet + messages."""
    policy = _teaching_policy(teaching_policy)
    signals = compute_signals(tlist, new_user_message, policy, prior_state=prior_state,
                              channel=channel, is_voice_partial=is_voice_partial)
    if needs_contextual_assessment(signals):
        signals.update(assess_context_with_llm(tlist, new_user_message, case_content))
    decision = decide_response(signals)
    plan = {"policy": policy, "signals": signals, "decision": decision,
            "mode": decision["mode"], "function": decision["function"], "reason": decision["reason"]}
    if decision["render"] != "model":
        return plan
    fn, mode = decision["function"], decision["mode"]
    allow_questions = ALLOW_QUESTIONS.get(fn, ALLOW_QUESTIONS.get(mode, False))
    instruction = get_function_instruction(fn, mode, policy, new_user_message, signals)
    packet = build_interviewer_control_packet(signals, decision, candidate_text=new_user_message,
                                              recent_lines=_recent_lines(tlist),
                                              allow_questions=allow_questions)
    outcome = evaluate_intervention_outcome(prior_state, signals)
    plan.update({
        "allow_questions": allow_questions,
        "packet": packet,
        "messages": _build_adaptive_messages(
            case_content, case_type, tlist, new_user_message, clarifications_exhausted,
            signals, mode, instruction, allow_questions, policy, prior_state, outcome,
            packet=packet, function=fn),
    })
    return plan


def _set_control(control_out, plan, tag, render):
    if control_out is None:
        return
    control_out["tag"] = tag
    control_out["mode"] = plan["mode"] if render != "silent" else "NO_OUTPUT"
    control_out["function"] = plan["function"]
    control_out["lane"] = plan["decision"]["lane"]
    control_out["render"] = render
    control_out["reason"] = plan["reason"]


def _contextual_line(plan, new_user_message, channel, user_id) -> Dict[str, Any]:
    """Word a contextual presence beat from the control packet.

    One model call; one regeneration if the line is unusable (leaked metadata,
    empty after the question/claim filter, or picks up nothing the candidate
    said); then a plain fast-lane hand-back. A presence beat never becomes an
    error and never becomes a stock line while the model can do better.
    """
    cli, model, provider = _resolve_adaptive_llm()
    messages = list(plan["messages"])
    max_sent = plan["packet"]["interviewer_control"]["response_generation"]["target_sentence_count"]
    tag: Dict[str, str] = {}
    best = ""
    problems: List[str] = []
    for attempt in range(2):
        t0 = time.time()
        try:
            resp = cli.chat.completions.create(
                model=model, messages=messages, temperature=CONTEXTUAL_TEMPERATURE,
                max_tokens=CONTEXTUAL_MAX_TOKENS,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[interviewer] contextual_generation_failed function={plan['function']} "
                  f"error={type(e).__name__}")
            break
        log_ai_usage(user_id=user_id, endpoint="/attempts/messages/contextual", model=model,
                     response=resp, latency_ms=int((time.time() - t0) * 1000))
        raw = (resp.choices[0].message.content or "").strip()
        tag, clean = parse_control_tag(raw)
        line, problems = validate_contextual_line(clean, new_user_message, channel, max_sent)
        if line and not problems:
            return {"text": line, "tag": tag or {}, "fallback": False, "problems": []}
        if line and "leak" not in problems and not best and len(line.split()) > 3:
            best = line  # usable words, just not specific enough: kept over a stock line
        messages = messages + [
            {"role": "assistant", "content": raw[:400]},
            {"role": "system", "content": (
                "That line cannot be used (" + ", ".join(problems or ["empty"]) + "). Write the "
                "interviewer's line again: plain words only, no question, pick up something "
                "specific the candidate just said, do not say their work is correct.")},
        ]
    if best:
        print(f"[interviewer] contextual_line_accepted_with {problems} function={plan['function']}")
        return {"text": best, "tag": tag or {}, "fallback": False, "problems": problems}
    ev = get_fast_lane_event("HAND_BACK", plan["signals"])
    print(f"[interviewer] contextual_fallback function={plan['function']} problems={problems}")
    return {"text": ev["data"]["text"], "tag": {"mode": "HAND_BACK", "intervention": "fast_lane"},
            "fallback": True, "problems": problems}


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
        plan = _plan_adaptive_turn(case_content, case_type, tlist, new_user_message,
                                   clarifications_exhausted, teaching_policy, prior_state,
                                   channel, is_voice_partial)
        policy, signals = plan["policy"], plan["signals"]
        mode, reason, function = plan["mode"], plan["reason"], plan["function"]
        render = plan["decision"]["render"]

        # FAST LANE: SILENCE (0 tokens)
        if render == "silent":
            t0 = time.time()
            _set_control(control_out, plan, {"mode": "NO_OUTPUT", "intervention": "silence"}, "silent")
            yield ""
            log_ai_usage(user_id=user_id, endpoint="/attempts/messages/fastlane_silence", model="local",
                         response=_NoUsage(), latency_ms=max(1, int((time.time() - t0) * 1000)))
            return

        # FAST LANE: a minimal turn gets a minimal beat (0 tokens). Only SHORT_ACK /
        # HAND_BACK reach here, and only when the candidate's turn was not substantive.
        if render == "fast":
            t0 = time.time()
            fast_event_dict = get_fast_lane_event(function, signals)
            _set_control(control_out, plan, {"mode": function, "intervention": "fast_lane"}, "fast")
            # Text: plain line (the SSE UI renders text). Voice: the typed event.
            if channel == "text":
                yield fast_event_dict["data"]["text"]
            else:
                yield json.dumps(fast_event_dict)
            log_ai_usage(user_id=user_id, endpoint="/attempts/messages/fastlane_event", model="local",
                         response=_NoUsage(), latency_ms=max(1, int((time.time() - t0) * 1000)))
            return

        # CONTEXTUAL PRESENCE: the model words the beat from the control packet.
        if function in CONTEXTUAL_PRESENCE:
            out = _contextual_line(plan, new_user_message, channel, user_id)
            _set_control(control_out, plan, out["tag"], "fast" if out["fallback"] else "model")
            if control_out is not None and out["fallback"]:
                control_out["fallback"] = True
            yield out["text"]
            return

        # DEEP LANE (substantive functions): same generation path as before, now
        # briefed by the control packet.
        allow_questions = plan["allow_questions"]
        messages = plan["messages"]
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
    
    # 7. Pre-Emission Safety Validation for Deep Lane Zero-Question Modes
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
        final_text = enforce_mode(scrub_control_leak(clean_text), mode, allow_questions, policy)
        
        if control_out is not None:
            _set_control(control_out, plan, tag or {}, "model")
            
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
    guard = StreamLeakGuard() if adaptive else None
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
                        yield from guard.feed(_out)
                else:
                    yield token
        if stripper is not None:
            for _out in stripper.flush():
                yield from guard.feed(_out)
            held = guard.flush(fallback=enforce_mode("", mode, False, policy))
            if held:
                yield held
            if control_out is not None:
                _set_control(control_out, plan, getattr(stripper, "tag", {}) or {}, "model")
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
        plan = _plan_adaptive_turn(case_content, case_type, transcript, new_user_message,
                                   clarifications_exhausted, teaching_policy, prior_state,
                                   channel, is_voice_partial)
        policy, signals = plan["policy"], plan["signals"]
        mode, reason, function = plan["mode"], plan["reason"], plan["function"]
        render = plan["decision"]["render"]

        if render == "silent":
            _set_control(control_out, plan, {"mode": "NO_OUTPUT", "intervention": "silence"}, "silent")
            return ""

        if render == "fast":
            fast_event = get_fast_lane_event(function, signals)
            _set_control(control_out, plan, {"mode": function, "intervention": "fast_lane"}, "fast")
            return fast_event["data"]["text"] if channel == "text" else json.dumps(fast_event)

        if function in CONTEXTUAL_PRESENCE:
            out = _contextual_line(plan, new_user_message, channel, None)
            _set_control(control_out, plan, out["tag"], "fast" if out["fallback"] else "model")
            if control_out is not None and out["fallback"]:
                control_out["fallback"] = True
            return out["text"]

        allow_questions = plan["allow_questions"]
        messages = plan["messages"]
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
        text = enforce_mode(scrub_control_leak(text), mode, allow_questions, policy)
        if control_out is not None:
            _set_control(control_out, plan, tag or {}, "model")
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
