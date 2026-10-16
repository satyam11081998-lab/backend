"""Interviewer arena: run ONE turn of the real interviewer against a chosen
prompt variant, in complete isolation from live user data.

WHY THIS EXISTS
The interviewer is the product. Judging a prompt change by reading a couple of
replies is how you ship a regression, so this runs two arms side by side on the
same case and the same candidate turn, and reports both a subjective reply and
objective per-reply signals.

WHAT IS SHARED WITH PRODUCTION (deliberately)
Everything except the prompt layer. Each arm runs
services/interview_engine.stream_interviewer_reply -- the SAME generator the live text
route consumes -- and joins it exactly as that route does. Session signals, the
intervention gate, the response-function decision, the control packet, the presence
lane, tag stripping, the leak guard and mode enforcement are all that code path. Only
`messages_builder` (and the pinned model) differ per arm.

Why the streaming path and not complete_interviewer_reply (v1 of this module used
that): the two post-process differently. In the question-allowed functions (OPEN,
TARGETED_QUESTION, PROBE, RETHINK_CUE ...) production streams the reply straight to
the candidate with NO sanitize_reply, so a stacked second question or a praise opener
reaches the user. complete_interviewer_reply sanitises every reply, so the arena
showed a cleaner interviewer than users get -- on exactly the turns worth judging.
tests/test_arena_isolation.py now pins arena text == production text on such a turn.

WHAT IS ISOLATED (deliberately)
  * no attempts / attempt_messages row is created or touched
  * no clarification quota is spent
  * no user_skill_profile / learner profile is persisted -- the learning functions
    are pure, and the arena keeps each arm's state in its own dict
  * each arm carries its OWN prior_state. Sharing one would let arm A's hint level
    and repetition memory drive arm B's reply, and the comparison would be void.

WHAT IS NOT ISOLATED (deliberately)
  ai_usage_log. Arena turns cost real money, and the daily kill switch reads that
  table, so every arm's spend is logged under endpoint /admin/arena/turn. Silent
  arena spend would be a worse bug than a noisy log.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from prompts.variants import registry
from services.ai_providers import openai_client
from services.ai_usage import log_ai_usage
from services.interview_engine import (
    InterviewEngineError,
    stream_interviewer_reply,
)
from services.interviewer_decision import (
    detect_violations,
    parse_control_tag,
    update_session_state,
)
from services.session_signals import compute_signals

ARENA_ENDPOINT = "/admin/arena/turn"

# Pinning both arms to ONE explicit model is the whole point of this control: it is
# the only way to tell a prompt difference from provider variance. "default" follows
# the live admin toggle and is therefore the least diagnostic choice.
MODEL_MODES: Dict[str, Dict[str, Any]] = {
    "default": {"label": "Live admin toggle (not diagnostic)", "model": None, "provider": None},
    "openai-mini": {"label": "OpenAI gpt-4o-mini (production interviewer)", "model": "gpt-4o-mini", "provider": "openai"},
    "openai-4o": {"label": "OpenAI gpt-4o (headroom check)", "model": "gpt-4o", "provider": "openai"},
}
DEFAULT_MODEL_MODE = "openai-mini"


def model_modes() -> List[Dict[str, str]]:
    return [{"id": k, "label": v["label"]} for k, v in MODEL_MODES.items()]


def _resolve_llm_override(model_mode: str) -> Optional[Tuple[Any, str, str]]:
    spec = MODEL_MODES.get(model_mode or DEFAULT_MODEL_MODE)
    if not spec or not spec["model"]:
        return None  # engine resolves from the live admin toggle
    client = openai_client()
    if client is None:
        return None
    return (client, spec["model"], spec["provider"])


def _reply_metrics(text: str) -> Dict[str, int]:
    """Cheap objective shape of a reply. Verbosity bias is real -- a judge who can
    see length is less likely to mistake 'longer' for 'better'."""
    t = (text or "").strip()
    sentences = [s for s in t.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    return {
        "chars": len(t),
        "words": len(t.split()),
        "sentences": len(sentences),
        "questions": t.count("?"),
    }


def _is_silence(control: Dict[str, Any], text: str) -> bool:
    """V11 can decide to say nothing. That is a real behaviour and must be shown as
    a deliberate silence, not rendered as an empty bubble or an error."""
    if (text or "").strip():
        return False
    tag = control.get("tag") or {}
    return (
        str(control.get("mode") or "").upper() == "NO_OUTPUT"
        or str(tag.get("mode") or "").upper() == "NO_OUTPUT"
        or str(tag.get("intervention") or "").lower() == "silence"
    )


def run_turn(
    *,
    case_content: str,
    case_type: str,
    variant_id: str,
    transcript: List[Dict[str, str]],
    new_user_message: str,
    prior_state: Optional[dict] = None,
    teaching_policy: str = "coached",
    model_mode: str = DEFAULT_MODEL_MODE,
    clarifications_exhausted: bool = False,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """One interviewer turn for ONE arm, on the production path. Never raises: an
    arm that fails returns an error row so the other arm's reply is still usable."""
    builder = registry.get_builder(variant_id)  # KeyError -> 400 at the route
    llm_override = _resolve_llm_override(model_mode)
    control: Dict[str, Any] = {}
    usage: Dict[str, Any] = {}
    t0 = time.perf_counter()
    error = None
    text = ""
    try:
        # Consumed exactly as the live text route consumes it: every chunk, joined,
        # stripped. usage_out makes the engine hand its model calls (and the raw model
        # text) back here instead of logging them as production traffic.
        text = "".join(stream_interviewer_reply(
            case_content=case_content,
            case_type=case_type,
            transcript=transcript,
            new_user_message=new_user_message,
            user_id=user_id,
            clarifications_exhausted=clarifications_exhausted,
            teaching_policy=teaching_policy,
            prior_state=prior_state,
            control_out=control,
            channel="text",
            messages_builder=builder,
            llm_override=llm_override,
            usage_out=usage,
        )).strip()
    except InterviewEngineError as e:
        error = f"interviewer failed: {e}"
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    latency_ms = int((time.perf_counter() - t0) * 1000)

    # Every model call the turn made, with real token counts, under the arena's own
    # endpoint so the daily kill switch sees it without it counting as production
    # interviewer traffic. The deterministic lanes record a zero-cost "local" call;
    # nothing was spent, so nothing is logged.
    spent_calls = [c for c in (usage.get("calls") or []) if c.get("model") != "local"]
    for call in spent_calls:
        log_ai_usage(
            user_id=user_id,
            endpoint=ARENA_ENDPOINT,
            model=call.get("model") or "",
            response=call.get("response"),
            latency_ms=call.get("latency_ms"),
            success=error is None,
            meta={"arena_variant": variant_id, "model_mode": model_mode,
                  "engine_endpoint": call.get("endpoint")},
        )

    silent = _is_silence(control, text)
    tag = control.get("tag") or {}
    # CRITICAL for interpreting a comparison: V11 decides many turns in a
    # DETERMINISTIC lane -- "silent" (say nothing) or "fast" (a canned beat like
    # "Go ahead."). Those lanes never build a prompt, so on such a turn BOTH arms
    # emit byte-identical text and the variant had no effect whatsoever. Reading
    # that as "the prompts are equivalent" is the easiest wrong conclusion this tool
    # could invite, so every arm reports whether its prompt was consulted.
    prompt_consulted = str(control.get("render") or "") == "model"

    # What the MODEL said, before production's post-processing (tag strip, leak
    # guard, enforce_mode in the no-question functions, the presence-line filter).
    # Where that post-processing rewrote the reply, the visible difference between
    # arms is not the prompt's doing, so both texts are shown.
    raw_reply = ""
    if usage.get("raw_text"):
        try:
            _, raw_reply = parse_control_tag(str(usage["raw_text"]).strip())
            raw_reply = (raw_reply or "").strip()
        except Exception:  # noqa: BLE001
            raw_reply = ""
    # Next state for THIS arm only, folded the way production folds it after a turn
    # (compute_signals + update_session_state, both pure). Nothing is written.
    next_state = prior_state or {}
    if not error and (tag or control.get("function")):
        try:
            sig = compute_signals(transcript, new_user_message, teaching_policy, prior_state)
            next_state = update_session_state(prior_state, tag, sig, function=control.get("function"))
        except Exception as e:  # noqa: BLE001
            error = error or f"state fold skipped: {type(e).__name__}"

    return {
        "variant_id": variant_id,
        "reply": text,
        "silent": silent,
        "error": error,
        "prompt_consulted": prompt_consulted,
        "raw_reply": raw_reply,
        "enforced": bool(raw_reply and raw_reply != (text or "").strip()),
        "control": {
            "mode": control.get("mode"),
            "function": control.get("function"),
            "render": control.get("render"),
            "reason": control.get("reason"),
            "intervention": tag.get("intervention"),
            "materiality": tag.get("materiality"),
            "skill": tag.get("skill"),
            "error_kind": tag.get("error"),
            "hint": tag.get("hint"),
            "fallback": bool(control.get("fallback")),
        },
        # Deterministic guardrail breaches, measured on the text the candidate would
        # actually receive: objective, taste-free, and the fastest way to see a
        # regression the eye would miss.
        "violations": detect_violations(text, teaching_policy, tag) if text else [],
        "metrics": _reply_metrics(text),
        "latency_ms": latency_ms,
        # None when no model ran (a deterministic lane), rather than naming a model
        # that never answered.
        "model": usage.get("model") if spent_calls else None,
        "provider": usage.get("provider") if spent_calls else None,
        "next_state": next_state,
    }
