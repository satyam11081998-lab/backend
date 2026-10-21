"""
Control-tag parsing, streaming strip, Contextual Assessor, and Intervention Gate.
Implements V10.2: Safely routing imperatives to Deep Lane and cleanly falling back 
to contextual PROBEs or HAND_BACKs.
V12: the gates now feed a policy layer that picks a conversational RESPONSE
FUNCTION; the model words it from a JSON INTERVIEWER CONTROL PACKET (see
decide_response / build_interviewer_control_packet / validate_contextual_line).
"""
from __future__ import annotations

import os
import re
import json
import hashlib
from typing import Dict, Generator, Tuple, Any, List, Optional

from services.learning_model import evaluate_intervention_outcome, update_learning_profile
from services.ai_providers import openai_client

_TAG_RE = re.compile(r"^\s*<<(.*?)>>\s*", re.DOTALL)
_DEC = "\u0001"  


_ABBR_DOT = re.compile(r"(?i)\b(?:rs|approx|etc|vs|avg|mr|mrs|ms|dr|st|est|incl|yr|yrs|sq|ft|nos?|fig|govt|dept|"
                       r"min|max)\.(?=\s+\S)|\b(?:e\.g|i\.e)\.|(?<=[a-z0-9])\.(?=[a-z]{2,}\b)|\.{2,}")
_LIST_DOT = re.compile(r"(?m)^(\s*\d{1,2})\.(?=\s)")


def _protect(t: str) -> str:
    t = _LIST_DOT.sub(lambda m: m.group(1) + _DEC, t)
    return _ABBR_DOT.sub(lambda m: m.group(0).replace(".", _DEC), t)


def _sentences(text: str, keep_punct: bool = True):
    t = re.sub(r"(\d)\.(\d)", r"\1" + _DEC + r"\2", text or "")
    if keep_punct:
        # V12.1: "Rs. 300", "e.g.", "app.mece.in", "..." and a "1." list marker do not end a sentence
        t = _protect(t)
        parts = re.findall(r"[^.!?]*[.!?]+(?=\s|$)|[^.!?]*[.!?]+(?=[^.!?\s])|[^.!?]+$", t)
    else:
        parts = re.split(r"[.!?]+", t)
    return [p.replace(_DEC, ".").strip() for p in parts if p.strip()]


def parse_control_tag(text: str) -> Tuple[Dict[str, str], str]:
    if not text:
        return {}, ""
    m = _TAG_RE.match(text)
    if not m:
        return {}, text.strip()
    tag: Dict[str, str] = {}
    for part in re.split(r"[;,]", m.group(1)):
        if "=" in part:
            k, v = part.split("=", 1)
            tag[k.strip().lower()] = v.strip().lower()
    cleaned = text[m.end():].lstrip("\n").strip()
    return (tag, cleaned) if cleaned else (tag, text.strip())


class StreamTagStripper:
    def __init__(self, budget: int = 240):
        self.buf = ""
        self.resolved = False
        self.budget = budget
        self._strip_leading = False
        self.tag = {}

    def feed(self, token: str) -> Generator[str, None, None]:
        if self.resolved:
            if token:
                if self._strip_leading:
                    token = token.lstrip()
                    if not token:
                        return
                    self._strip_leading = False
                yield token
            return
        self.buf += token or ""
        stripped = self.buf.lstrip()
        if not stripped:
            return
        if stripped.startswith("<<"):
            end = self.buf.find(">>")
            if end != -1:
                self.tag = parse_control_tag(self.buf[:end + 2])[0]
                rest = self.buf[end + 2:].lstrip("\n").lstrip()
                self.resolved, self.buf = True, ""
                if rest:
                    yield rest
                else:
                    self._strip_leading = True
            elif len(self.buf) > self.budget:
                out, self.buf, self.resolved = self.buf, "", True
                if out:
                    yield out
            return
        if len(stripped) >= 2 or len(self.buf) > 8:
            out, self.buf, self.resolved = self.buf, "", True
            if out:
                yield out

    def flush(self) -> Generator[str, None, None]:
        if not self.buf:
            return
        out, self.buf, self.resolved = self.buf, "", True
        m = _TAG_RE.match(out)
        if m:
            out = out[m.end():].lstrip()
        if out:
            yield out


# --- STATE ESTIMATION (Contextual Assessor) ----------------------------------

_ASSESSOR_PROMPT = """You are a state assessor for a consulting case interview.
Evaluate the candidate's last message in the context of the transcript.
Output ONLY a JSON object with these exact keys and values from the enums below:
{
  "intent": "answering" | "clarification" | "asking_for_help" | "asking_for_solution" | "hypothesis" | "meta" | "wants_to_stop" | "working" | "unknown",
  "progress": "progressing" | "recovering" | "stuck" | "repeatedly_stuck" | "stalled" | "completed_step" | "unclear",
  "materiality": "none" | "minor" | "material" | "critical" | "unknown",
  "case_phase": "opening" | "clarification" | "structuring" | "analysis" | "quantification" | "synthesis" | "closing" | "unknown"
}
"""

def assess_context_with_llm(transcript: list, new_message: str, case_content: str) -> Dict[str, Any]:
    cli = openai_client()
    if not cli:
        return {}
        
    history_lines = []
    for t in transcript[-4:]:
        role = t.get('role', 'user')
        content = t.get('content', '')
        if content:
            history_lines.append(f"{role.upper()}: {content}")
    history = "\n".join(history_lines)
    
    case_ctx = (case_content or "")[:500]
    user_prompt = f"CASE CONTEXT: {case_ctx}...\n\nRECENT TRANSCRIPT:\n{history}\n\nCANDIDATE CURRENT TURN:\n{new_message}"
    
    try:
        resp = cli.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _ASSESSOR_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=150,
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        return {}


# --- INTERVENTION GATE (V10 TWO-GATE SYSTEM) --------------------------------

def _wants_teaching(signals: Dict[str, Any]) -> bool:
    """Teach the step (name the approach, show the skeleton, do step one) rather
    than drop one more hint: explicit "teach me / walk me through", a blank "i dont
    know", repeated stuckness, high frustration, a weak learner, or a help request
    before any work is on the table."""
    if signals.get("wants_teaching"):
        return True
    if re.search(r"\bhints?\b", signals.get("new_message_norm") or ""):
        return False  # they asked for a hint: the hint ladder escalates it
    if signals.get("latest_work"):
        # Arena adversarial run (Oct 21): "help na i'm not getting this at all" right after
        # "0.95S - 2 = 0.80(S-2) then S comes to 2.67 cr" got a generic profit tree. With
        # their own working on the table, help is a hint ON that working; teaching from
        # scratch only once a hint on it has already failed.
        helped = any(f in ("MICRO_HINT", "TEACH_STEP", "REPAIR_AND_RESET")
                     for f in (signals.get("recent_functions") or [])[-2:])
        return bool(helped and signals.get("help_streak", 0) >= 2)
    if signals.get("stuck_blank"):
        return True
    return bool(signals.get("help_streak", 0) >= 2
                or signals.get("frustration_explicit")
                or signals.get("learner_level") == "weak"
                or (signals.get("help_requested") and not signals.get("work_so_far")))


def _evaluate_v121_requests(signals: Dict[str, Any]) -> Tuple[bool, str, str]:
    """V12.1 (15-day review): every one of these got a canned "Okay." / "Sure,
    continue." / "Let's move on to the next phase." in real sessions. Each now has
    its own function, ahead of the generic routes."""
    finished = signals.get("case_finished")
    if signals.get("wants_to_exit"):
        if finished or signals.get("is_final_recommendation"):
            return True, "CLOSE", "session_end"
        return True, "WRAP_UP_EARLY", "wants_to_exit"
    if signals.get("courtesy_only") and finished:
        return True, "CLOSE", "courtesy_after_close"
    if signals.get("asks_session_status"):
        return True, "SESSION_STATUS", "session_status"
    if signals.get("wants_case_restate"):
        return True, "RESTATE_CASE", "case_restate"
    if signals.get("asks_how_it_works") and signals.get("early_in_case"):
        return True, "EXPLAIN_FORMAT", "how_it_works"
    # Arena adversarial run (Oct 21): these three were caught by help / repair first.
    if signals.get("asks_product_help"):
        return True, "PRODUCT_HELP", "product_question"
    if signals.get("asks_why_asked"):
        return True, "EXPLAIN_WHY", "why_this_question"
    if signals.get("asks_for_space"):
        return True, "HOLD_SPACE", "candidate_wants_space"
    if signals.get("complains_unanswered"):
        return True, "ANSWER_DIRECT", "unanswered_questions"
    if signals.get("math_issue"):
        return True, "CORRECT_MATERIAL", "math_check"
    li = signals.get("logic_issue") or {}
    if li:
        if li.get("kind") == "extreme_assumption":
            return True, "RETHINK_CUE", "extreme_assumption"
        return True, "CORRECT_MATERIAL", "logic_check"
    if (signals.get("answers_interviewer_number") and signals.get("hedged_self_estimate")
            and not signals.get("final_declared")):
        # "maybe 20 lakh? not very sure" answers your question; the "?" is doubt, not a question
        return True, "FOLLOW_UP", "hedged_answer"
    if signals.get("final_declared"):
        if signals.get("final_needs_check"):
            # an implausible or unchecked final gets ONE sanity check before the close
            return True, "RETHINK_CUE", "final_sanity_check"
        return True, "FINAL_REVIEW", "final_answer"
    if signals.get("concept_asked"):
        return True, "EXPLAIN_CONCEPT", "concept_question"
    if signals.get("wants_next_step"):
        return True, "TRANSITION", "next_step"
    return False, "", ""


def _evaluate_substantive_gate(signals: Dict[str, Any]) -> Tuple[bool, str, str]:
    if signals.get("is_session_open"):
        if signals.get("asks_how_it_works"):
            return True, "EXPLAIN_FORMAT", "how_it_works"  # "How it work" as the very first message
        if signals.get("wants_teaching") or signals.get("stuck_blank"):
            return True, "TEACH", "teach_from_the_start"   # "teach me" / "i dont know" before anything
        return True, "OPEN", "session_start"
    hit, mode, reason = _evaluate_v121_requests(signals)
    if hit:
        return True, mode, reason
    if signals.get("is_session_close"): return True, "CLOSE", "session_end"
    if signals.get("wants_to_advance") or signals.get("intent") == "wants_to_stop":
        return True, "TRANSITION", "candidate_transition"
        
    if signals.get("solution_requested") or signals.get("intent") == "asking_for_solution":
        return True, "DELIVER_SOLUTION", "solution_requested"
    if (signals.get("help_requested") or signals.get("intent") == "asking_for_help"
            or signals.get("wants_teaching") or signals.get("asks_how_it_works")):
        if _wants_teaching(signals) or signals.get("asks_how_it_works"):
            return True, "TEACH", "teach_step"
        return True, "HINT", "help_requested"
    if signals.get("needs_structure"):
        return True, "TEACH", "structure_coaching"
    if signals.get("looks_garbage"): return True, "NOISE", "noise"
    if signals.get("is_meta") or signals.get("intent") == "meta": return True, "DEFLECT_META", "meta"

    if signals.get("error_materiality") == "material" or signals.get("materiality") in ["material", "critical"]:
        return True, "CORRECT_MATERIAL", "material_error"
    if signals.get("states_final_estimate") or signals.get("confident_unsupported_claim"):
        return True, "RETHINK_CUE", "check_suspicious_claim"
    if signals.get("frustration_explicit") or signals.get("progress") in ["stuck", "repeatedly_stuck"]:
        return True, "REPAIR", "frustration_or_stuck"

    # V12.1: "clarification" from a stray directive word ("...how they are able to PROVIDE
    # such costs...") is not a question; only a turn that asks the interviewer something is.
    clarifies = ((signals.get("intent") == "clarification"
                  and (signals.get("asks_interviewer", True) or signals.get("asks_to_proceed")))
                 # a question the intent read missed ("competitors ka market share kitna hai",
                 # "I would like to know the number of stores", "And the margins")
                 or (signals.get("asks_interviewer") and not signals.get("tag_question_only")
                     and signals.get("intent") in ("answering", "hypothesis", "greeting")))
    if signals.get("why_this_question") or signals.get("product_ux_question") or clarifies:
        return True, "ANSWER_DIRECT", "clarification"
    if (signals.get("asks_owned_fact") or signals.get("is_scope_question")) and signals.get("asks_interviewer", True):
        return True, "DATA_REVEAL", "fact_requested"

    if signals.get("intent") == "hypothesis":
        if signals.get("progress") != "progressing":
            return True, "DATA_REVEAL", "hypothesis_needs_data"

    return False, "", ""

def _evaluate_presence_gate(signals: Dict[str, Any]) -> Tuple[bool, str, str]:
    # "give me a minute" / "ek minute soch raha hu": the floor stays with them (V12.1 review)
    if signals.get("candidate_mid_thought") and not signals.get("is_substantive_reasoning"):
        return True, "HAND_BACK", "candidate_mid_thought"
    if signals.get("last_action_was_presence"):
        return False, "", "recently_acknowledged"
        
    if signals.get("is_affirmation_only"):
        return True, "HAND_BACK", "affirmation_received"
        
    if signals.get("hedged_self_estimate") or "?" in signals.get("new_message_norm", ""):
        return True, "HAND_BACK", "return_floor_explicitly"
        
    if signals.get("candidate_working") or signals.get("wants_space"):
        return True, "HAND_BACK", "candidate_needs_space"
        
    if signals.get("just_recovered"):
        return True, "VALIDATE", "candidate_recovered"
        
    if signals.get("has_work"):
        return True, "ACKNOWLEDGE", "conversational_presence"
        
    return False, "", ""

def evaluate_intervention_gate(signals: Dict[str, Any]) -> Tuple[str, str, str]:
    if not signals:
        return "SUBSTANTIVE", "PROBE", "no_signals"

    channel = signals.get("channel", "text")
    is_voice_partial = signals.get("is_voice_partial", False)

    if is_voice_partial:
        return "SILENCE", "NO_OUTPUT", "voice_partial"

    is_substantive, sub_mode, sub_reason = _evaluate_substantive_gate(signals)
    if is_substantive:
        return "SUBSTANTIVE", sub_mode, sub_reason

    is_presence, pres_mode, pres_reason = _evaluate_presence_gate(signals)
    if is_presence:
        return "PRESENCE", pres_mode, pres_reason

    # Text Fallback: Prevents Blank Bubbles on un-recognized states
    if channel == "text":
        if signals.get("has_work") or signals.get("candidate_working"):
            return "PRESENCE", "ACKNOWLEDGE", "text_fallback_prevent_blank"
        else:
            # The candidate said something short that wasn't an affirmation or a known directive.
            # E.g. "I don't know." -> It's best to gently probe instead of saying "Got it".
            return "SUBSTANTIVE", "PROBE", "text_fallback_no_work"

    return "SILENCE", "NO_OUTPUT", "no_presence_needed_for_voice"


# --- V12 POLICY LAYER: STATE -> RESPONSE FUNCTION ---------------------------
#
# Function first, language second. The two gates above still decide WHETHER the
# interviewer intervenes and every business rule they encode (help is honoured,
# solutions are delivered, material errors are corrected, frustration is
# repaired, meta is deflected, voice partials are silent). This layer decides
# WHAT CONVERSATIONAL FUNCTION the turn needs. It never writes the sentence:
#   - NO_OUTPUT                      -> nothing (fast lane)
#   - SHORT_ACK / HAND_BACK          -> one of a few fixed words, ONLY when the
#                                      candidate's turn itself carried almost
#                                      nothing (fast lane, no model call)
#   - every other function           -> worded by the model from the JSON
#                                      INTERVIEWER CONTROL PACKET below.
# A substantive candidate turn never reaches the fixed-phrase lane.

RESPONSE_FUNCTIONS = (
    "NO_OUTPUT", "SHORT_ACK", "HAND_BACK", "ACKNOWLEDGE_AND_CONTINUE",
    "ACKNOWLEDGE_AND_ORIENT", "REFLECT_PROGRESS", "VALIDATE_AND_HAND_BACK",
    "TARGETED_QUESTION", "SANITY_CHECK", "MICRO_HINT", "CORRECT_AND_CONTINUE",
    "REPAIR_AND_RESET", "ANSWER_DIRECT", "DATA_REVEAL", "DELIVER_SOLUTION",
    "TRANSITION", "OPEN", "CLOSE", "DEFLECT_META", "NOISE", "CONTINUE_AS_AGREED",
    # V12.1
    "RESTATE_CASE", "EXPLAIN_FORMAT", "EXPLAIN_CONCEPT", "TEACH_STEP", "WRAP_UP_EARLY",
    "FINAL_REVIEW", "SESSION_STATUS", "FOLLOW_UP_ANSWER",
    "PRODUCT_HELP", "EXPLAIN_WHY",
)
FAST_FUNCTIONS = frozenset({"SHORT_ACK", "HAND_BACK"})
CONTEXTUAL_PRESENCE = frozenset({
    "ACKNOWLEDGE_AND_CONTINUE", "ACKNOWLEDGE_AND_ORIENT", "REFLECT_PROGRESS",
    "VALIDATE_AND_HAND_BACK",
})

# Gate mode (unchanged V10.2 vocabulary) -> the function it now names.
_SUBSTANTIVE_FUNCTION = {
    "OPEN": "OPEN",
    "CLOSE": "CLOSE",
    "TRANSITION": "TRANSITION",
    "DELIVER_SOLUTION": "DELIVER_SOLUTION",
    "HINT": "MICRO_HINT",
    "NOISE": "NOISE",
    "DEFLECT_META": "DEFLECT_META",
    "CORRECT_MATERIAL": "CORRECT_AND_CONTINUE",
    "RETHINK_CUE": "SANITY_CHECK",
    "REPAIR": "REPAIR_AND_RESET",
    "ANSWER_DIRECT": "ANSWER_DIRECT",
    "DATA_REVEAL": "DATA_REVEAL",
    "PROBE": "TARGETED_QUESTION",
    # V12.1
    "RESTATE_CASE": "RESTATE_CASE",
    "EXPLAIN_FORMAT": "EXPLAIN_FORMAT",
    "EXPLAIN_CONCEPT": "EXPLAIN_CONCEPT",
    "TEACH": "TEACH_STEP",
    "WRAP_UP_EARLY": "WRAP_UP_EARLY",
    "FINAL_REVIEW": "FINAL_REVIEW",
    "SESSION_STATUS": "SESSION_STATUS",
    "FOLLOW_UP": "FOLLOW_UP_ANSWER",
    "PRODUCT_HELP": "PRODUCT_HELP",
    "EXPLAIN_WHY": "EXPLAIN_WHY",
    "HOLD_SPACE": "HAND_BACK",   # fixed line: "Fair enough, it's all yours." (no model, no question)
}

_CONTEXTUAL_BY_TURN = {
    "strategic_insight": "REFLECT_PROGRESS",
    "structure": "REFLECT_PROGRESS",
    "hypothesis": "REFLECT_PROGRESS",
    "calculation_result": "ACKNOWLEDGE_AND_ORIENT",
    "plan": "ACKNOWLEDGE_AND_CONTINUE",
    "calculation": "ACKNOWLEDGE_AND_CONTINUE",
    "estimate": "ACKNOWLEDGE_AND_CONTINUE",
}
# When the same contextual function was used on the previous turn, rotate.
_ROTATE = {
    "REFLECT_PROGRESS": "ACKNOWLEDGE_AND_CONTINUE",
    "ACKNOWLEDGE_AND_CONTINUE": "REFLECT_PROGRESS",
    "ACKNOWLEDGE_AND_ORIENT": "ACKNOWLEDGE_AND_CONTINUE",
    "VALIDATE_AND_HAND_BACK": "ACKNOWLEDGE_AND_CONTINUE",
}


def _contextual_function(signals: Dict[str, Any], gate_mode: str) -> str:
    if gate_mode == "VALIDATE" or signals.get("just_recovered"):
        fn = "VALIDATE_AND_HAND_BACK"
    else:
        fn = _CONTEXTUAL_BY_TURN.get(signals.get("turn_type") or "", "ACKNOWLEDGE_AND_CONTINUE")
    if fn == signals.get("last_function"):
        fn = _ROTATE.get(fn, fn)
    return fn


_OWN_WORK_RE = re.compile(r"^(so |okay |ok |now )?(i'll|i will|i'd|i would|let me|i'm going to|im going to)\b")


def _announces_own_work(norm: str) -> bool:
    """A short "I'll split households first" announces work to come: hand the floor back."""
    return bool(_OWN_WORK_RE.search(norm))


def _decision(lane: str, mode: str, function: str, reason: str) -> Dict[str, Any]:
    render = ("silent" if function == "NO_OUTPUT"
              else "fast" if function in FAST_FUNCTIONS else "model")
    return {"lane": lane, "mode": mode, "function": function, "reason": reason, "render": render}


def contextual_presence_enabled() -> bool:
    """Kill switch: INTERVIEWER_CONTEXTUAL_PRESENCE=off sends the model-worded
    presence beats back to the fixed short lines (pre-V12 behaviour, 0 tokens).
    Read per turn, so flipping it on Render needs a restart, not a deploy."""
    return os.getenv("INTERVIEWER_CONTEXTUAL_PRESENCE", "on").strip().lower() not in ("off", "0", "false", "no")


def _fast_ok_in_text(sig: Dict[str, Any]) -> bool:
    """In chat, a fixed word ("Okay.", "Go ahead.") is right only when the turn
    itself carried nothing to answer."""
    norm = sig.get("new_message_norm") or ""
    if sig.get("is_affirmation_only") or sig.get("courtesy_only"):
        return True
    if sig.get("candidate_mid_thought") or sig.get("wants_space") or sig.get("asks_for_space"):
        return True
    if sig.get("intent") == "greeting":
        return True
    # "i'll go with the demand approach" -> "Go ahead."; a longer plan gets its content picked up
    return _announces_own_work(norm) and not re.search(r"\d", norm) and len(norm.split()) <= 10


def _never_canned(d: Dict[str, Any], sig: Dict[str, Any]) -> Dict[str, Any]:
    """V12.1 invariant (15-day review): a question or request to the interviewer is
    always answered by the model, never by the fixed-phrase lane or a presence beat
    that strips every question; and in chat, a short answer to the interviewer's own
    question, or any turn with content, is responded to in its own terms."""
    fn = d["function"]
    if d["render"] == "silent":
        return d
    # "can you stop asking questions and let me think" is a request for space, not a question
    asks = sig.get("asks_interviewer") and not sig.get("tag_question_only") and not sig.get("asks_for_space")
    if fn in FAST_FUNCTIONS or fn in CONTEXTUAL_PRESENCE:
        if asks and not (fn in CONTEXTUAL_PRESENCE and d["reason"] == "floor_yield_after_work"):
            return _decision("SUBSTANTIVE", "ANSWER_DIRECT", "ANSWER_DIRECT", d["reason"] + "|question_answered")
    if sig.get("channel", "text") == "text" and contextual_presence_enabled():
        if fn in FAST_FUNCTIONS and not _fast_ok_in_text(sig):
            if sig.get("last_assistant_asked"):
                return _decision("SUBSTANTIVE", "FOLLOW_UP", "FOLLOW_UP_ANSWER", d["reason"] + "|answer_to_question")
            return _decision("PRESENCE", d["mode"], "ACKNOWLEDGE_AND_CONTINUE", d["reason"] + "|not_canned")
        if (fn in CONTEXTUAL_PRESENCE and sig.get("last_assistant_asked")
                and (sig.get("word_count") or 0) < 12 and not sig.get("is_affirmation_only")):
            return _decision("SUBSTANTIVE", "FOLLOW_UP", "FOLLOW_UP_ANSWER", d["reason"] + "|answer_to_question")
    return d


def decide_response(signals: Dict[str, Any]) -> Dict[str, Any]:
    d = _never_canned(_decide_response(signals), signals or {})
    if d["function"] in CONTEXTUAL_PRESENCE and not contextual_presence_enabled():
        fn = "HAND_BACK" if d["function"] == "VALIDATE_AND_HAND_BACK" else "SHORT_ACK"
        return _decision(d["lane"], d["mode"], fn, d["reason"] + "|contextual_off")
    return d


def _decide_response(signals: Dict[str, Any]) -> Dict[str, Any]:
    """The policy layer: candidate state -> {lane, mode, function, reason, render}.

    `mode` is the V10.2 gate mode (kept for telemetry, the learner-state fold and
    existing callers); `function` is what the interviewer must accomplish.
    """
    lane, mode, reason = evaluate_intervention_gate(signals or {})
    sig = signals or {}

    # "Shall we proceed with that?" -> "Yes, go ahead.": the interviewer carries on
    # with what it proposed (model-worded). Help, solution, meta, close etc. above
    # in the gate still win; a voice partial is still silent.
    if (sig.get("is_affirmation_only") and sig.get("last_assistant_asked")
            and not sig.get("is_voice_partial") and lane != "SUBSTANTIVE"):
        return _decision("SUBSTANTIVE", "TRANSITION", "CONTINUE_AS_AGREED", "go_ahead_after_question")

    if lane == "SILENCE":
        return _decision(lane, mode, "NO_OUTPUT", reason)

    if lane == "SUBSTANTIVE":
        # A candidate who lays out real work and then hands the floor back
        # ("... shall I proceed?") is not asking a clarification: they need the
        # work acknowledged and the floor returned. Anything that is a real
        # question (facts, scope, why, product) keeps its V10.2 route.
        if (reason == "clarification" and sig.get("asks_to_proceed")
                and sig.get("is_substantive_reasoning")
                and not (sig.get("asks_owned_fact") or sig.get("why_this_question")
                         or sig.get("product_ux_question"))):
            fn = _contextual_function(sig, "ACKNOWLEDGE")
            return _decision("PRESENCE", "ACKNOWLEDGE", fn, "floor_yield_after_work")
        return _decision(lane, mode, _SUBSTANTIVE_FUNCTION.get(mode, mode), reason)

    # PRESENCE: the gate decided a presence beat is due. Pick the function.
    if sig.get("candidate_mid_thought"):
        # Still calculating ("let me think", "hold on", trailing "..."): in voice
        # any line interrupts them; in text a short turn simply gets the floor back
        # (a substantive one still gets its content acknowledged, below).
        if sig.get("channel") == "voice":
            return _decision("SILENCE", "NO_OUTPUT", "NO_OUTPUT", "candidate_mid_thought")
        if not sig.get("is_substantive_reasoning"):
            return _decision(lane, mode, "HAND_BACK", "candidate_mid_thought")
    if sig.get("is_substantive_reasoning"):
        return _decision(lane, mode, _contextual_function(sig, mode), reason)
    if mode == "VALIDATE":
        # Recovery after a correction: acknowledge the specific fix (model-worded),
        # never a stock "That works." that certifies it.
        return _decision(lane, mode, "VALIDATE_AND_HAND_BACK", reason)
    if (mode == "HAND_BACK" or sig.get("turn_type") in ("plan", "affirmation")
            or _announces_own_work(sig.get("new_message_norm") or "")):
        return _decision(lane, mode, "HAND_BACK", reason)
    return _decision(lane, mode, "SHORT_ACK", reason)


def determine_response_function(signals: Dict[str, Any]) -> str:
    return decide_response(signals)["function"]


# --- V12 INTERVIEWER CONTROL PACKET -------------------------------------------

_OBJECTIVE = {
    "NO_OUTPUT": "stay silent; the candidate holds the floor",
    "SHORT_ACK": "register that you heard a minimal turn",
    "HAND_BACK": "return the floor so the candidate keeps working",
    "ACKNOWLEDGE_AND_CONTINUE": "recognise the specific progress the candidate just made and let them continue",
    "ACKNOWLEDGE_AND_ORIENT": "recognise the step the candidate just completed and point them to the next part of their own plan",
    "REFLECT_PROGRESS": "reflect back the distinction or progress the candidate just made, then let them carry it forward",
    "VALIDATE_AND_HAND_BACK": "acknowledge the adjustment the candidate just made and hand the floor back",
    "TARGETED_QUESTION": "move the case forward with one targeted question",
    "SANITY_CHECK": "prompt the candidate to sanity-check a number or claim themselves",
    "MICRO_HINT": "give the next rung of help tied to the candidate's own work",
    "CORRECT_AND_CONTINUE": "name the one material issue, then let them continue",
    "REPAIR_AND_RESET": "reset the frustrated or stuck candidate with a simpler framing",
    "ANSWER_DIRECT": "answer the candidate's question directly, then hand back",
    "DATA_REVEAL": "give only the case data requested",
    "DELIVER_SOLUTION": "give the approach the candidate asked for, concisely",
    "TRANSITION": "move cleanly to the next stage",
    "OPEN": "open the case",
    "CLOSE": "close the case",
    "DEFLECT_META": "stay in role and return to the case",
    "NOISE": "ask the candidate to restate a garbled turn",
    "CONTINUE_AS_AGREED": "the candidate agreed to what you just proposed or asked: carry on with exactly that",
    "RESTATE_CASE": "give the case back to a candidate who cannot see it, then say where to start",
    "EXPLAIN_FORMAT": "explain how this interview works and what to do first",
    "EXPLAIN_CONCEPT": "explain the concept they asked about and show it on their own case",
    "TEACH_STEP": "teach the next step: name the approach, show the skeleton, do the first step with them",
    "WRAP_UP_EARLY": "let a candidate who wants to stop leave without pressure, and show the way back in",
    "FINAL_REVIEW": "close on their final answer: sanity-check it, one strength, one improvement, point to Submit",
    "SESSION_STATUS": "tell them honestly where they stand and what is left; point to Submit for the score",
    "FOLLOW_UP_ANSWER": "react to the short answer they just gave to your question, then the next step",
    "PRODUCT_HELP": "answer the question about the app plainly, then hand back to the case",
    "EXPLAIN_WHY": "say in one line why your question matters for this case, then hand it back",
}
_STRONG = {"MICRO_HINT", "CORRECT_AND_CONTINUE", "REPAIR_AND_RESET", "DELIVER_SOLUTION",
           "TEACH_STEP", "EXPLAIN_CONCEPT"}
_MODERATE = {"TARGETED_QUESTION", "SANITY_CHECK", "ANSWER_DIRECT", "DATA_REVEAL",
             "TRANSITION", "OPEN", "CLOSE", "DEFLECT_META", "NOISE", "RESTATE_CASE",
             "EXPLAIN_FORMAT", "WRAP_UP_EARLY", "FINAL_REVIEW", "SESSION_STATUS", "FOLLOW_UP_ANSWER",
             "PRODUCT_HELP", "EXPLAIN_WHY"}
_NEW_FACTS_OK = {"ANSWER_DIRECT", "DATA_REVEAL", "OPEN", "DELIVER_SOLUTION", "MICRO_HINT",
                 "REPAIR_AND_RESET", "CORRECT_AND_CONTINUE", "TEACH_STEP", "EXPLAIN_CONCEPT",
                 "EXPLAIN_FORMAT", "FOLLOW_UP_ANSWER", "SESSION_STATUS"}
_USES_LATEST_WORK = frozenset({"MICRO_HINT", "REPAIR_AND_RESET", "TEACH_STEP", "ANSWER_DIRECT"})
# Functions that may lay a structure out on separate short lines (text channel only).
VISUAL_FUNCTIONS = frozenset({"TEACH_STEP", "EXPLAIN_FORMAT", "EXPLAIN_CONCEPT", "DELIVER_SOLUTION",
                              "RESTATE_CASE"})


def _intervention_level(fn: str) -> str:
    if fn == "NO_OUTPUT":
        return "none"
    if fn in FAST_FUNCTIONS:
        return "minimal"
    if fn in CONTEXTUAL_PRESENCE:
        return "light"
    if fn in _STRONG:
        return "strong"
    return "moderate"


def _reasoning_stage(sig: Dict[str, Any]) -> str:
    phase = sig.get("case_phase")
    if phase and phase != "unknown":
        return phase
    tt = sig.get("turn_type")
    if sig.get("is_session_open"):
        return "opening"
    if sig.get("is_session_close"):
        return "closing"
    if tt in ("structure", "plan"):
        return "structuring"
    if tt in ("calculation", "calculation_result", "estimate"):
        return "quantification"
    if tt in ("strategic_insight", "hypothesis"):
        return "analysis"
    return "unknown"


def _progress_state(sig: Dict[str, Any]) -> str:
    if sig.get("progress") and sig.get("progress") != "unclear":
        return sig["progress"]
    if sig.get("just_recovered"):
        return "recovering"
    if sig.get("turns_without_progress", 0) >= 2:
        return "stuck"
    if sig.get("step_completed"):
        return "completed_step"
    if sig.get("has_work"):
        return "progressing"
    return "unclear"


def _gist(text: str, limit: int = 180) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    first = _sentences(t)[0] if _sentences(t) else t
    first = first if len(first) <= limit else first[:limit].rsplit(" ", 1)[0] + "..."
    return first


# Sentences of prose; a structure laid out on arrow lines does not count. Never above 4:
# a reply over 5 sentences is flagged too long (detect_violations) and is a wall of text
# on a phone (arena adversarial run, Oct 21: three teaching replies failed on length alone).
_SENTENCES_TEXT = {"TEACH_STEP": 4, "EXPLAIN_FORMAT": 4, "EXPLAIN_CONCEPT": 4, "RESTATE_CASE": 4,
                   "FINAL_REVIEW": 4, "SESSION_STATUS": 4, "WRAP_UP_EARLY": 3, "DELIVER_SOLUTION": 4,
                   "ANSWER_DIRECT": 3, "FOLLOW_UP_ANSWER": 2, "PRODUCT_HELP": 2, "EXPLAIN_WHY": 2,
                   "SANITY_CHECK": 2, "CORRECT_AND_CONTINUE": 2, "MICRO_HINT": 2}
_SENTENCES_VOICE = {"TEACH_STEP": 3, "EXPLAIN_FORMAT": 3, "EXPLAIN_CONCEPT": 2, "RESTATE_CASE": 3,
                    "FINAL_REVIEW": 2, "SESSION_STATUS": 2, "WRAP_UP_EARLY": 2, "DELIVER_SOLUTION": 2,
                    "PRODUCT_HELP": 2, "EXPLAIN_WHY": 2}


def _sentence_target(fn: str, voice: bool, contextual: bool) -> int:
    if voice:
        return _SENTENCES_VOICE.get(fn, 1)
    return _SENTENCES_TEXT.get(fn, 2 if contextual else 3)


def _case_context(sig: Dict[str, Any], fn: str) -> Dict[str, Any]:
    """Facts the turn may lean on, computed in code: never a line to copy."""
    ctx: Dict[str, Any] = {}
    facts = list(sig.get("reference_facts") or [])
    if facts:
        ctx["reference_facts"] = facts
    fc = sig.get("final_context") or {}
    if fn in ("FINAL_REVIEW", "SESSION_STATUS", "EXPLAIN_CONCEPT", "SANITY_CHECK"):
        for k in ("final_answer", "earlier_results", "implied_check", "implied_verdict"):
            if fc.get(k):
                ctx[k] = fc[k]
        if fn == "SANITY_CHECK" and fc.get("no_base"):
            ctx["no_base_for_check"] = True
    if fn in ("FINAL_REVIEW", "SESSION_STATUS") and sig.get("reference_answer"):
        ctx["reference_answer"] = sig["reference_answer"]
    if fn == "ANSWER_DIRECT" and sig.get("complains_unanswered") and sig.get("unanswered_questions"):
        ctx["unanswered_questions"] = list(sig["unanswered_questions"])
    if sig.get("concept_asked") and fn in ("EXPLAIN_CONCEPT", "TEACH_STEP"):
        ctx["concept_asked"] = sig["concept_asked"]
    if fn == "TEACH_STEP":
        ctx["teach_focus"] = sig.get("teach_focus") or "next_step"
        if sig.get("candidate_ideas"):
            ctx["candidate_ideas"] = list(sig["candidate_ideas"])
        if sig.get("already_tried"):
            # the pointer they already got and are still stuck after: switch method
            ctx["already_tried"] = sig["already_tried"]
    if fn == "PRODUCT_HELP":
        ctx["product_facts"] = list(PRODUCT_FACTS)
    if fn == "EXPLAIN_WHY" and sig.get("why_target"):
        ctx["your_question_they_ask_about"] = sig["why_target"]
    if fn == "WRAP_UP_EARLY" or fn == "SESSION_STATUS" or fn == "FINAL_REVIEW" or fn == "CLOSE":
        ctx["how_to_finish"] = ("tap Submit (top right of the chat) to end and see the score, feedback and "
                                "the full worked answer")
    if fn in ("RESTATE_CASE", "OPEN", "EXPLAIN_FORMAT"):
        ctx["where_the_case_is"] = ("the brief is on screen; on a phone it sits behind the menu icon "
                                    "at the top left of the chat")
    return ctx


def exam_safe_issue(issue: Dict[str, Any]) -> Optional[str]:
    """The slip without the corrected figure (exam policy: they must find it themselves)."""
    if not issue:
        return None
    if "expected" not in issue:
        return issue.get("message")
    expr, stated = issue.get("expr") or "that step", issue.get("stated") or "that figure"
    if issue.get("kind") == "equation":
        return f"{stated} does not solve {expr} - that step needs rechecking"
    return f"{expr} does not come to {stated} - that step needs rechecking"


# What the interviewer may say about the app (PRODUCT_HELP). Only what the product
# really does; anything else is "after you tap Submit, the results page has it".
PRODUCT_FACTS = (
    "Submit is at the top right of the chat; tapping it ends the case",
    "after Submit, the results page shows the score, feedback on the attempt and the full worked answer",
    "on a phone, the case brief is behind the menu icon at the top left of the chat",
    "the candidate can stop whenever they like and Submit what they have so far",
)


def build_interviewer_control_packet(signals: Dict[str, Any], decision: Any,
                                     candidate_text: str = "",
                                     recent_lines: Optional[List[str]] = None,
                                     allow_questions: Optional[bool] = None) -> Dict[str, Any]:
    """The decision brief the model words a turn from. Plain data: no sentence to
    copy, no fallback line, no hidden case solution."""
    if isinstance(decision, str):
        decision = {"function": decision, "lane": "", "mode": decision, "reason": ""}
    sig = signals or {}
    fn = decision["function"]
    voice = sig.get("channel") == "voice"
    contextual = fn in CONTEXTUAL_PRESENCE
    q_ok = bool(allow_questions) if allow_questions is not None else False
    alerts = []
    if sig.get("repair_due"):
        alerts.append("your recent moves are not working: stop probing, change strategy")
    if sig.get("interviewer_repeating"):
        alerts.append("you have repeated yourself: do not reuse a previous line")
    if sig.get("turns_without_progress", 0) >= 2:
        alerts.append(f"no progress for {sig['turns_without_progress']} turns")
    if sig.get("questions_suppressed"):
        alerts.append("the candidate asked you to stop asking questions: do not ask one this turn, keep it short")
    li = sig.get("logic_issue") or {}
    exam = (sig.get("teaching_policy") or "coached") == "exam"
    mi = sig.get("math_issue") or {}
    # exam policy: the packet names the step that is off, never the corrected figure
    issue_text = (exam_safe_issue(mi) if exam else mi.get("message")) if mi else li.get("message")
    material = fn == "CORRECT_AND_CONTINUE"
    work_check = dict(sig.get("latest_work_check") or {})
    if exam and work_check.get("issue"):
        work_check["issue"] = work_check.get("exam") or "a step in it is off and needs rechecking"
    may_confirm = bool((sig.get("math_verified") and not sig.get("math_issue"))
                       or (work_check.get("verified") and fn in ("MICRO_HINT", "ANSWER_DIRECT",
                                                                 "REPAIR_AND_RESET", "TEACH_STEP")))
    return {"interviewer_control": {
        "response_function": fn,
        "conversational_objective": _OBJECTIVE.get(fn, "move the case forward"),
        "candidate_state": {
            "candidate_turn_type": sig.get("turn_type", "statement"),
            "candidate_turn_summary": _gist(candidate_text) if candidate_text else "",
            "numbers_stated": list(sig.get("numbers_stated") or []),
            "reasoning_stage": _reasoning_stage(sig),
            "progress_state": _progress_state(sig),
            "reasoning_quality": ("material_issue" if material
                                  else "tentative" if sig.get("candidate_confidence") == "low"
                                  else "not_verified"),
            "candidate_confidence": sig.get("candidate_confidence", "moderate"),
            "candidate_is_working": bool(sig.get("has_work") or sig.get("candidate_working")),
            "candidate_needs_space": bool(sig.get("wants_space") or sig.get("candidate_mid_thought")),
            "explicit_help_request": bool(sig.get("help_requested")),
            "explicit_solution_request": bool(sig.get("solution_requested")),
            "material_error": material,
            "frustration_state": sig.get("frustration", "none"),
            # V12.1: computed by services/numbers_check (Python did the arithmetic)
            "arithmetic_check": ("issue" if sig.get("math_issue")
                                 else "verified" if sig.get("math_verified") else "not_checked"),
            "detected_issue": issue_text,
            "issue_kind": (sig.get("math_issue") or {}).get("kind") or li.get("kind"),
            "verified_steps": list(sig.get("math_verified") or [])[:3],
            # their own latest working, checked in code, for a hint ON it (help after work)
            "latest_work": sig.get("latest_work") if fn in _USES_LATEST_WORK else None,
            "latest_work_check": ((work_check.get("verified") and "verified: " + work_check["verified"])
                                  or (work_check.get("issue") and "issue: " + work_check["issue"])
                                  or None) if fn in _USES_LATEST_WORK else None,
        },
        "interviewer_decision": {
            "lane": decision.get("lane", ""),
            "intervention_level": _intervention_level(fn),
            "question_allowed": q_ok,
            "hint_allowed": fn in ("MICRO_HINT", "REPAIR_AND_RESET", "DELIVER_SOLUTION", "TEACH_STEP",
                                   "EXPLAIN_CONCEPT", "EXPLAIN_FORMAT"),
            "correction_allowed": fn == "CORRECT_AND_CONTINUE",
            "solution_allowed": fn == "DELIVER_SOLUTION",
            "silence_allowed": fn == "NO_OUTPUT",
            # True only when Python verified the arithmetic in this turn: the model may
            # then say the MATHS checks out (never that an assumption is right).
            "may_confirm_correctness": may_confirm,
            "new_case_facts_allowed": fn in _NEW_FACTS_OK,
            "visual_structure_allowed": fn in VISUAL_FUNCTIONS and not voice,
        },
        "response_generation": {
            "medium": "spoken" if voice else "chat",
            "tone": "calm_human_interviewer",
            "target_length": ("brief" if (contextual or voice) else
                              "a few short lines" if fn in VISUAL_FUNCTIONS else "short"),
            "target_sentence_count": _sentence_target(fn, voice, contextual),
            "content_specificity": "required" if contextual else "normal",
            "must_reference_candidate_content": contextual,
            "generic_acknowledgement_only": False,
            "do_not_overpraise": True,
            # Teaching turns do ONE step with the candidate; everything else leaves the
            # reasoning with them.
            "do_not_take_over_candidate_reasoning": fn not in ("TEACH_STEP", "EXPLAIN_CONCEPT", "DELIVER_SOLUTION"),
        },
        "case_context": _case_context(sig, fn),
        "conversation_memory": {
            "recent_response_functions": list(sig.get("recent_functions") or []),
            "recent_interviewer_lines": [l for l in (recent_lines or [])][-3:],
            "repetition_avoidance": True,
            "candidate_plan": sig.get("candidate_plan") if fn == "ACKNOWLEDGE_AND_ORIENT" else None,
        },
        "alerts": alerts,
    }}


# --- V12 LAYER C: CONTEXTUAL LINE VALIDATION ----------------------------------

_CLAIM_RE = re.compile(
    r"(?i)\b(that'?s|that is|this is|you'?re|you are|it'?s|it is|looks|sounds|seems)\s+"
    r"(exactly\s+|absolutely\s+|completely\s+|totally\s+|quite\s+)?"
    r"(correct|right|accurate|spot[- ]on|on track|on the right track)\b"
    r"|\bcorrect(ly)?\b|\byou'?ve got it\b|\bthat checks out\b")
_LEAK_RE = re.compile(
    r"[{}]|<<|>>|response[_ ]function|control[_ ]packet|interviewer[_ ]control|"
    r"may[_ ]confirm|question[_ ]allowed|candidate[_ ]turn[_ ]type|"
    r"\b(ACKNOWLEDGE_AND_CONTINUE|ACKNOWLEDGE_AND_ORIENT|REFLECT_PROGRESS|VALIDATE_AND_HAND_BACK)\b")
_STOP = frozenset("""
a an the and or but so then that this these those there their they them you your yours we our
i i'm im me my it its it's is are was were be been being to of in on at for from with by as
into about over under just only also very really quite some any all each other than more most
first next now let lets let's will would can could should shall may might do does did done have
has had get got go going keep carry take make makes made think okay ok right alright sure yes
yeah well good great nice step steps part parts thing things way point
""".split())


def _content_keys(text: str) -> set:
    t = (text or "").lower().replace("’", "'")
    keys = set()
    for tok in re.findall(r"[a-z][a-z'\-]+|\d[\d,.]*", t):
        tok = tok.strip("'.,-")
        if not tok:
            continue
        if tok[0].isdigit():
            keys.add(tok.replace(",", "").rstrip("."))
        elif len(tok) >= 4 and tok not in _STOP:
            keys.add(tok[:5])
    return keys


def references_candidate(line: str, candidate_text: str) -> bool:
    """True when the line picks up something the candidate actually said (a number
    or a content word, compared on a 5-letter stem)."""
    return bool(_content_keys(line) & _content_keys(candidate_text))


def validate_contextual_line(text: str, candidate_text: str, channel: str = "text",
                             max_sentences: int = 2, may_confirm: bool = False) -> Tuple[str, List[str]]:
    """Clean a model-worded presence line. Returns (line, problems).

    Problems: "leak" (control metadata in the text), "empty", "unreferenced"
    (nothing the candidate said is picked up). Questions, solicitations, gush and
    unverified correctness claims are removed, not just flagged: no check verified
    the candidate's work, so the line may not certify it.
    """
    t = (text or "").strip()
    if _LEAK_RE.search(t):
        return "", ["leak"]
    t = sanitize_reply(t)
    t = strip_solicitations(strip_all_questions(t)).strip()
    kept = [s for s in _sentences(t, keep_punct=True) if may_confirm or not _CLAIM_RE.search(s)]
    n = 1 if channel == "voice" else max(1, max_sentences)
    t = " ".join(kept[:n]).strip()
    if not t:
        return "", ["empty"]
    problems = [] if references_candidate(t, candidate_text) else ["unreferenced"]
    return t, problems


_JSON_TEXT_KEYS = ("reply", "text", "response", "line", "say", "message", "output", "interviewer")


def scrub_control_leak(text: str) -> str:
    """Deep-lane guard: the prompt now carries a JSON packet, so a model may answer
    in JSON or echo a field name. Recover the spoken line from a JSON reply, drop
    any sentence that carries control metadata; return "" if nothing is left."""
    t = (text or "").strip()
    if not t or not _LEAK_RE.search(t):
        return t
    body = re.sub(r"^```(?:json)?|```$", "", t).strip()
    if body.startswith("{"):
        try:
            obj = json.loads(body)
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            for k in _JSON_TEXT_KEYS:
                v = obj.get(k)
                if isinstance(v, str) and v.strip() and not _LEAK_RE.search(v):
                    return v.strip()
        return ""
    kept = [x for x in _sentences(t, keep_punct=True) if not _LEAK_RE.search(x)]
    return " ".join(kept).strip()


class StreamLeakGuard:
    """Streams normally, unless the reply opens like JSON / a code fence: then it
    holds the whole reply and releases only the scrubbed line at the end."""

    def __init__(self):
        self.decided = False
        self.hold = False
        self.buf = ""

    def feed(self, piece: str):
        if self.decided and not self.hold:
            yield piece
            return
        self.buf += piece or ""
        head = self.buf.lstrip()
        if not head:
            return
        if not self.decided:
            if head.startswith("{") or head.startswith("`"):
                self.decided, self.hold = True, True
                return
            self.decided = True
            out, self.buf = self.buf, ""
            yield out

    def flush(self, fallback: str = "") -> str:
        if not self.hold:
            return ""
        return scrub_control_leak(self.buf) or fallback


# --- FAST LANE SEMANTIC EVENT GENERATOR --------------------------------------
# Only for turns that carried almost nothing (an affirmation, a bare number, a
# short "let me work it out"). Never used for substantive reasoning.

_FAST_PHRASES = {
    "SHORT_ACK": [
        {"code": "ACK_01", "text": "Got it."},
        {"code": "ACK_02", "text": "Alright."},
        {"code": "ACK_03", "text": "Okay."},
        {"code": "ACK_04", "text": "Right."}
    ],
    "HAND_BACK": [
        {"code": "HB_01", "text": "Go ahead."},
        {"code": "HB_02", "text": "Sure, continue."},
        {"code": "HB_03", "text": "Take it from there."},
        {"code": "HB_04", "text": "Alright, continue."},
        {"code": "HB_05", "text": "Makes sense. Go ahead."}
    ],
    # V12.1 (arena adversarial run): "let me think for a second" got "Alright, continue."
    "THINK": [
        {"code": "TH_01", "text": "Take your time."},
        {"code": "TH_02", "text": "Sure, take your time."},
        {"code": "TH_03", "text": "No rush. Take a moment."},
    ],
    # "you're asking too many questions, just let me solve" -> back off, no question.
    "SPACE": [
        {"code": "SP_01", "text": "Fair enough. Over to you."},
        {"code": "SP_02", "text": "Fair point. It's all yours."},
        {"code": "SP_03", "text": "Got it. Go ahead and solve it your way."},
    ],
}
_FAST_ALIASES = {"ACKNOWLEDGE": "SHORT_ACK", "VALIDATE": "HAND_BACK"}

def get_fast_lane_event(mode: str, signals: Dict[str, Any]) -> Dict[str, Any]:
    key = _FAST_ALIASES.get(mode, mode)
    if key in ("HAND_BACK", "SHORT_ACK"):
        if signals.get("asks_for_space"):
            key = "SPACE"
        elif signals.get("candidate_mid_thought") and not signals.get("is_substantive_reasoning"):
            key = "THINK"
    options = _FAST_PHRASES.get(key, _FAST_PHRASES["SHORT_ACK"])
    
    msg = signals.get("new_message_norm", "default")
    hash_val = int(hashlib.md5(msg.encode('utf-8')).hexdigest(), 16)
    
    recent_phrases = signals.get("recent_assistant_turns", [])
    recent_norm = [re.sub(r"[^\w\s]", "", t.lower().strip()) for t in recent_phrases]
    
    valid_options = []
    for opt in options:
        opt_norm = re.sub(r"[^\w\s]", "", opt["text"].lower().strip())
        if opt_norm not in recent_norm:
            valid_options.append(opt)
            
    if not valid_options:
        valid_options = options
        
    selection = valid_options[hash_val % len(valid_options)]
    
    return {
        "event_type": "interviewer_presence",
        "data": {
            "mode": mode,
            "code": selection["code"],
            "text": selection["text"]
        }
    }


# --- state update + guardrail telemetry --------------------------------------

_HINT_INTERVENTIONS = {"micro_hint", "reframe", "analogy", "decompose", "demonstrate", "reveal"}
_HINT_FLOOR = {"reframe": 1, "analogy": 1, "micro_hint": 2, "decompose": 3, "demonstrate": 4, "reveal": 5}

_BANNED_PHRASES = (
    "isn't specified", "isnt specified", "not specified", "isn't in the prompt",
    "not in the prompt", "isn't provided", "not provided", "isn't given", "not given",
    "i can't give you that", "i cannot give you that", "i won't provide", "i wont provide",
    "i cannot provide", "that detail isn't", "information isn't provided",
    # V12.1 (15-day review): the same refusal in other words
    "isn't explicitly stated", "not explicitly stated", "isn't stated", "not stated in",
    "aren't detailed", "not detailed in", "isn't detailed", "don't have specific data",
    "do not have specific data", "don't have an exhibit", "don't have data", "do not have data",
    "no exhibit", "i can't provide", "can't provide an exact", "cannot provide an exact",
    "not available in the case", "the case doesn't specify",
    "the case does not specify", "isn't mentioned", "not mentioned in the", "you'll need to assume",
    "you can make reasonable assumptions", "make reasonable assumptions about",
)


def looks_like_refusal(text: str) -> bool:
    """The interviewer told the candidate the data isn't available (a hard failure:
    the interviewer owns the facts). A sentence that also GIVES a figure ("The
    population isn't stated in the brief, so take 2 crore") is answering, not refusing."""
    for sent in _sentences((text or "").lower().replace("\u2019", "'"), keep_punct=True):
        if any(p in sent for p in _BANNED_PHRASES) and not re.search(r"\d", sent):
            return True
    return False
_RUBBER_STAMP = (
    "solid structure", "great job", "well done", "impressive", "well-structured",
    "well structured", "comprehensive", "excellent job",
)

def _infer_learner_level(prior_level, signals):
    tw = signals.get("turns_without_progress", 0)
    if tw >= 3:
        return "weak" if prior_level in ("developing", "weak") else "developing"
    if signals.get("has_work") and signals.get("frustration") == "none" and tw == 0:
        return "strong" if prior_level in ("unknown", "developing") else prior_level
    return prior_level if prior_level else "unknown"

def update_session_state(prior_state, tag, signals, function=None):
    ps = dict(prior_state or {})
    tag = tag or {}
    intervention = (tag.get("intervention") or "").strip().lower()
    prior_hint = int(ps.get("hint_level", 0) or 0)
    try:
        tag_hint = int(tag["hint"]) if tag.get("hint") not in (None, "") else None
    except (TypeError, ValueError, KeyError):
        tag_hint = None

    progressed = not (signals.get("help_requested") or signals.get("looks_garbage")
                      or signals.get("turns_without_progress", 0) >= 1)
    if intervention in _HINT_INTERVENTIONS:
        rung = tag_hint if tag_hint is not None else max(_HINT_FLOOR.get(intervention, 1), prior_hint + 1)
        hint_level = max(prior_hint, rung)
    elif progressed and prior_hint > 0:
        hint_level = prior_hint - 1
    else:
        hint_level = prior_hint
    hint_level = max(0, min(5, hint_level))

    ps.update({
        "mode": (tag.get("mode") or ps.get("mode") or "interviewer"),
        "hint_level": hint_level,
        "last_intervention": intervention or ps.get("last_intervention"),
        "repairs_done": int(ps.get("repairs_done", 0) or 0) + (1 if intervention == "repair" else 0),
        "frustration": signals.get("frustration", "none"),
        "consecutive_probes": signals.get("recent_probes", 0),
        "updated_turn": signals.get("candidate_turn_count", ps.get("updated_turn", 0)),
        "learner_level": _infer_learner_level(ps.get("learner_level", "unknown"), signals),
    })
    if function:
        # V12: the response function is the presence cool-down and the
        # repetition memory for the next turn (model-worded lines have no fixed text).
        recent = [f for f in (ps.get("recent_functions") or []) if isinstance(f, str)]
        ps["recent_functions"] = (recent + [function])[-4:]
        ps["last_function"] = function
        ps["function_turn"] = signals.get("candidate_turn_count", 0)
    outcome = evaluate_intervention_outcome(prior_state, signals)
    ps["last_intervention_effect"] = outcome
    ps["profile"] = update_learning_profile(
        ps.get("profile"), tag, outcome,
        hint_level=int((prior_state or {}).get("hint_level", 0) or 0),
        signals=signals, prior_modality=(prior_state or {}).get("last_intervention"),
    )
    return ps

def detect_violations(reply_text, policy="coached", tag=None):
    import re
    t = (reply_text or "").lower()
    out = []
    if any(p in t for p in _BANNED_PHRASES):
        out.append("banned_isnt_specified")
    if any(p in t for p in _RUBBER_STAMP):
        out.append("possible_rubber_stamp")
    sents = _sentences(reply_text, keep_punct=False)
    if len(sents) > 5:
        out.append("too_long")
    if (reply_text or "").count("?") >= 2:
        out.append("multiple_questions")
    interv = ((tag or {}).get("intervention") or "").strip().lower()
    looks_full_solution = (len(re.findall(r"\d", reply_text or "")) >= 4 and len(sents) >= 5
                           and ("=" in (reply_text or "") or "the answer is" in t or "the total is" in t))
    if interv in ("continue", "probe") and looks_full_solution:
        out.append("solved_when_declared_continue")
    if policy == "exam" and looks_full_solution:
        out.append("revealed_full_solution_in_exam")
    return out

_GUSH = (
    "great job", "great work", "great instinct", "great answer", "well done",
    "impressive", "brilliant", "awesome", "amazing", "excellent", "excellent job",
    "fantastic", "outstanding", "superb", "spot on", "nailed it", "perfect",
    "flawless", "masterful", "genius", "incredible", "love it", "really impressive",
    "solid structure", "well structured", "well-structured", "comprehensive",
    "you're on the right track", "youre on the right track", "phenomenal", "stellar",
    "you nailed", "beautifully done", "top notch", "top-notch", "love that", "love this",
)

# V12.1: praise-only sentences seen in real sessions ("That's a solid estimate for the
# annual number of laptops sold in Mumbai." for a 3x-wrong number; "Great start with
# the population figure."). Removed only when the sentence is short and carries no
# number, so a sentence with substance is never dropped for its opening word.
_PRAISE_RE = re.compile(
    r"\b(great|good|solid|excellent|nice|strong|fantastic|wonderful|impressive|brilliant|thoughtful|"
    r"clear|smart|perfect|thorough|well[- ]grounded|robust|insightful)\s+(start|starting point|breakdown|"
    r"approach|estimate|calculation|analysis|job|work|thinking|structure|segmentation|plan|catch|"
    r"observation|insight|summary|recap|progress|framework|choice|way|attempt|reasoning|logic|set of numbers|"
    r"first step|step|point|question|idea|thought|perspective|consideration)\b|\byou'?ve done a (?:thorough|great|solid|good|excellent)\b|"
    r"\bthat'?s (?:a )?(?:great|excellent|fantastic|brilliant) (?:question|point)\b", re.I)
_DANGLING_RE = re.compile(r"^(?:but|and|however|though|although|yet)[,\s]+", re.I)


def _is_gush(sentence: str) -> bool:
    s = sentence.lower().replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")
    if any(g in s for g in _GUSH):
        return True
    return bool(_PRAISE_RE.search(s)) and len(s.split()) <= 16 and not re.search(r"\d", s)


def is_praise_only(text: str) -> bool:
    """Every sentence is praise (nothing a candidate can use)."""
    sents = [x for x in _sentences(text or "", keep_punct=True) if x.strip()]
    return bool(sents) and all(_is_gush(x) and not _PIVOT_RE.search(x) for x in sents)


def _capitalize(t: str) -> str:
    for i, ch in enumerate(t or ""):
        if ch.isalpha():
            return t[:i] + ch.upper() + t[i + 1:]
        if not ch.isspace() and ch not in "\"'(":
            break
    return t


def _undangle(t: str) -> str:
    """'but there's a material issue...' after a removed opener -> 'There's a material issue...'.
    Only ever called on text that FOLLOWS something that was removed."""
    t2 = _DANGLING_RE.sub("", (t or "").lstrip())
    return _capitalize(t2) if t2 else t


def _filter_lines(text: str, keep) -> str:
    """Apply a per-sentence filter line by line, so a structure laid out on separate
    lines keeps its line breaks."""
    out_lines = []
    for line in (text or "").split("\n"):
        if not line.strip():
            if out_lines and out_lines[-1] != "":
                out_lines.append("")
            continue
        kept = [x for x in _sentences(line, keep_punct=True) if keep(x)]
        if kept:
            out_lines.append(" ".join(kept).strip())
    while out_lines and out_lines[-1] == "":
        out_lines.pop()
    return "\n".join(out_lines).strip()


_PIVOT_RE = re.compile(r"\b(but|however|though|although|yet|except)\b", re.I)


def trim_praise_opener(sentence: str) -> str:
    """'Great start, but there's a material issue...' -> 'There's a material issue...';
    a praise-only sentence -> ''; anything else unchanged."""
    m = re.match(r"^\s*([^,;:\u2014]{0,60}?)(?:[,;:\u2014]|\s-\s)\s*(\S.*)$", sentence or "", re.DOTALL)
    if m and _is_gush(m.group(1) + "."):
        rest = _undangle(m.group(2))
        if _is_gush(rest) or (len(rest.split()) <= 4 and rest.rstrip().endswith("!")):
            return ""
        return _capitalize(rest)
    if _is_gush(sentence or "") and not _PIVOT_RE.search(sentence or ""):
        return ""
    return sentence


def sanitize_reply(text: str, policy: str = "coached") -> str:
    t = (text or "").strip()
    if not t:
        return t
    first = True
    trimmed_head = False
    out_lines: List[str] = []
    for line in t.split("\n"):
        if not line.strip():
            if out_lines and out_lines[-1] != "":
                out_lines.append("")
            continue
        kept = []
        for x in _sentences(line, keep_punct=True):
            if first:
                first = False
                y = trim_praise_opener(x)
                trimmed_head = y != x
                if not y:
                    continue
                x = y
            elif _is_gush(x) and not _PIVOT_RE.search(x):
                continue
            kept.append(x)
        if kept:
            out_lines.append(" ".join(kept).strip())
    while out_lines and out_lines[-1] == "":
        out_lines.pop()
    cleaned = "\n".join(out_lines).strip()
    if cleaned:
        # Repair the start only if the opening was cut: "So far, you have..." stays.
        t = _undangle(cleaned) if trimmed_head else cleaned
    if t.count("?") >= 2:
        t = t[: t.find("?") + 1].strip()
    return t


class StreamReplyGuard:
    """V12.1: a question-allowed reply still streams, with the two repairs only the
    non-streamed lane used to get: a praise opener is dropped ("Great point, love
    that!" / "Great start, but ..."), and a stacked second question is cut (ONE
    question per reply). The first sentence is held until it ends, so the cost is
    the first sentence's latency, not the whole reply's."""

    def __init__(self):
        self.head = True
        self.buf = ""
        self.undangle_pending = False
        self.q_seen = False
        self.held = ""
        self.emitted = ""

    def _emit(self, text: str):
        if not text:
            return
        if self.undangle_pending:
            self.buf += text
            if len(self.buf.strip()) < 14:
                return
            text, self.buf, self.undangle_pending = _undangle(self.buf.lstrip()), "", False
        if self.q_seen:
            self.held += text
            return
        i = text.find("?")
        if i == -1:
            self.emitted += text
            yield text
            return
        self.q_seen = True
        self.held = text[i + 1:]
        self.emitted += text[:i + 1]
        yield text[:i + 1]

    def feed(self, piece: str):
        if not piece:
            return
        if not self.head:
            yield from self._emit(piece)
            return
        self.buf += piece
        m = re.search(r"[.!?](?=\s)|\n", self.buf)
        if not m and len(self.buf) < 220:
            return
        yield from self._release_head(m.end() if m else len(self.buf))

    def _release_head(self, end: int):
        first, rest = self.buf[:end], self.buf[end:]
        self.head, self.buf = False, ""
        kept = trim_praise_opener(first)
        if kept and kept.strip() == first.strip():
            yield from self._emit(first + rest)
            return
        if kept:
            # keep the reply's own spacing (a line break after the first sentence survives)
            joined = kept + rest if (not rest or rest[:1].isspace() or kept.endswith((" ", "\n"))) else kept + " " + rest
            yield from self._emit(joined)
            return
        self.undangle_pending = True
        yield from self._emit(rest.lstrip())

    def flush(self) -> str:
        out = []
        if self.head and self.buf:
            out += list(self._release_head(len(self.buf)))
        if self.undangle_pending and self.buf:
            self.undangle_pending = False
            t, self.buf = _undangle(self.buf.lstrip()), ""
            out += list(self._emit(t))
        if self.held.strip():
            # what followed the first question, up to the next one, with its own line breaks
            # ("Which segment first?\n- Students\n- Professionals" keeps the options)
            lines_out, stop = [], False
            for line in self.held.split("\n"):
                if stop:
                    break
                keep = []
                for x in _sentences(line, keep_punct=True):
                    if x.rstrip().endswith("?"):
                        stop = True
                        break
                    keep.append(x)
                lines_out.append(" ".join(keep))
            tail = "\n".join(lines_out).rstrip()
            if tail.strip():
                tail = tail if tail[:1] in (" ", "\n") else " " + tail
                self.emitted += tail
                out.append(tail)
        return "".join(out)

# --- V12.1 (arena adversarial run): reply shape -------------------------------
# Three teaching replies failed on length alone: numbered lists ("1. Total market size")
# count every item as a sentence, and the model ran to 6-7 sentences. A list becomes
# short lines (numbered -> arrow steps, bullets -> dots) and prose is capped; the final
# question, if any, is always kept.
_LIST_LINE = re.compile(r"^\s*(?:(?P<num>\d{1,2})[.)]|(?P<bul>[-*\u2022]))\s+(?P<item>\S.*?)\s*$")
_SKELETON_LINE = re.compile(r"^\s*(?:\u2192|->|=|\u00b7|\u2022)")


def _split_inline_lists(line: str) -> str:
    """'First, look at: 1. Total market 2. Segments 3. Needs' -> one item per line. Only a
    real run 1., 2., 3. counts ('costs rose by 5. Then volume fell 10. So...' is prose)."""
    marks = [m for m in re.finditer(r"(?:^|(?<=\s))(\d{1,2})\.\s+(?=[A-Za-z(])", line)]
    run = []
    for m in marks:
        if int(m.group(1)) == len(run) + 1:
            run.append(m)
    if len(run) < 2:
        return line
    out, last = [], 0
    for m in run:
        out.append(line[last:m.start()].rstrip())
        last = m.start()
    out.append(line[last:])
    return "\n".join(x for x in out if x.strip())


def shape_lists(text: str) -> str:
    """Numbered items -> '→ step' lines (the first stays plain), bullets -> '· item'.
    Short items lose their full stop, so a list is never counted as prose sentences."""
    if not text or not re.search(r"(?m)^\s*(?:\d{1,2}[.)]|[-*\u2022])\s+|\s\d{1,2}\.\s+[A-Za-z(]", text):
        return text
    if re.search(r"(?m)^\s*2\.\s+\S", text):
        # a list whose "1." was written at the end of a sentence line
        text = re.sub(r"(?<=[.!?:])[ \t]+(?=1\.\s+[A-Za-z(])", "\n", text)
    lines = "\n".join(_split_inline_lists(l) for l in text.split("\n")).split("\n")
    out: List[str] = []
    prev_num = False
    for line in lines:
        m = _LIST_LINE.match(line)
        if not m:
            out.append(line.rstrip())
            prev_num = False
            continue
        item = re.sub(r"\*\*(.+?)\*\*", r"\1", m.group("item")).strip()
        if len(item.split()) <= 14:
            item = item.rstrip(" .;")
        if m.group("num"):
            out.append(("\u2192 " + item) if prev_num else item)
            prev_num = True
        else:
            out.append("\u00b7 " + item)
            prev_num = False
    return "\n".join(out).strip()


def _prose_units(text: str) -> List[Tuple[int, int, str, bool]]:
    """(line_no, idx, sentence, is_prose) for every sentence. An arrow/dot line's first
    piece and a line with no sentence punctuation are structure; prose written after an
    arrow item on the same line ("-> Population. Use Rs 300 per ticket.") is prose."""
    units = []
    for ln, line in enumerate((text or "").split("\n")):
        if not line.strip():
            units.append((ln, 0, "", False))  # a blank line keeps the layout
            continue
        if not re.search(r"[.!?](?:\s|$)", line):
            units.append((ln, 0, line, False))
            continue
        parts = _sentences(line, keep_punct=True)
        skel = bool(_SKELETON_LINE.match(line))
        for i, x in enumerate(parts):
            if skel and i == 0:
                units.append((ln, i, re.sub(r"\.\s*$", "", x) if len(parts) > 1 else x, False))
            else:
                units.append((ln, i, x, True))
    return units


def cap_sentences(text: str, limit: int = 5) -> str:
    """At most `limit` sentences as detect_violations counts them. Drops prose from the
    middle: the first sentence (the point), the question (there is at most one) and the
    last sentence (the hand-back) stay, and so does any structure on its own lines."""
    t = (text or "").strip()
    if not t or len(_sentences(t, keep_punct=False)) <= limit:
        return t
    units = _prose_units(t)
    prose = [k for k, u in enumerate(units) if u[3]]
    keep_always = {prose[0], prose[-1]} if prose else set()
    keep_always |= {k for k in prose if units[k][2].rstrip().endswith("?")}
    droppable = [k for k in prose if k not in keep_always]
    drop: set = set()

    def build():
        lines: Dict[int, List[str]] = {}
        for k, (ln, _i, x, _p) in enumerate(units):
            if k not in drop:
                lines.setdefault(ln, []).append(x)
        return _join_lines(lines)
    cand = build()
    # drop from the end of the middle backwards: the latest filler goes first
    for k in reversed(droppable):
        drop.add(k)
        cand = build()
        if len(_sentences(cand, keep_punct=False)) <= limit:
            return cand
    return cand


def _join_lines(lines: Dict[int, List[str]]) -> str:
    out = "\n".join(" ".join(v).strip() for _ln, v in sorted(lines.items()))
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def tidy_abbreviations(text: str) -> str:
    """'Rs. 300' -> 'Rs 300', 'approx. 20' -> 'approx 20': the same words, and a sentence
    counter no longer sees a sentence end inside a figure."""
    t = re.sub(r"\b([Rr]s|RS)\.\s?(?=\d)", r"\1 ", text or "")
    return re.sub(r"(?i)\b(approx)\.(?=\s)", r"\1", t)


def tidy_skeleton(text: str) -> str:
    """'= total consumption.' -> '= total consumption': an arrow line is not a sentence."""
    return re.sub(r"(?m)^(\s*(?:\u2192|->|=)[^\n]*?)\.[ \t]*$", r"\1", text or "")


class StreamShapeGuard:
    """The streamed lane's version of shape_lists + cap_sentences. Text goes out a sentence
    (or a line) at a time. A list line is held until it is complete and goes out as an
    arrow / dot line. Prose stops at `budget` sentences (counted the way detect_violations
    counts them); at the end, the last held question, if no question went out yet, is
    still sent, so a reply is never more than budget + 1 sentences. A sentence is sent or
    held as a whole, never cut."""

    _BOUNDARY = re.compile(r"\n|[.!?]+(?=\s)")
    _ABBR_END = re.compile(r"(?i)(?:\b(?:rs|approx|etc|vs|avg|mr|mrs|ms|dr|st|est|incl|yr|yrs|sq|ft|nos?|fig|govt|"
                           r"dept|min|max)|\be\.g|\bi\.e|\.\.)\.$")
    _MAYBE_MARK = re.compile(r"^[ \t]*(?:\d{1,2}[.)]?|[-*\u2022])?$")
    _MARK = re.compile(r"^[ \t]*(?:\d{1,2}[.)]|[-*\u2022])[ \t]")
    _LONG = 240

    def __init__(self, budget: int = 4):
        self.budget = budget
        self.buf = ""
        self.count = 0
        self.held: List[str] = []
        self.q_out = False
        self.line_start = True
        self.prev_num = False
        self.cur: Optional[str] = None  # a sentence in progress: "out" or "held"
        self.pending_item = False
        self.emitted = ""

    def _list_line(self, line: str) -> str:
        m = _LIST_LINE.match(line.rstrip("\n"))
        nl = "\n" if line.endswith("\n") else ""
        if not m:
            return line
        item = re.sub(r"\*\*(.+?)\*\*", r"\1", m.group("item")).strip()
        if len(item.split()) <= 14:
            item = item.rstrip(" .;")
        if m.group("num"):
            out = (("\u2192 " + item) if self.prev_num else item) + nl
            self.prev_num = True
        else:
            out = "\u00b7 " + item + nl
            self.prev_num = False
        return out

    def _out(self, chunk: str):
        self.emitted += chunk
        if "?" in chunk:
            self.q_out = True
        yield chunk

    @staticmethod
    def _n(body: str) -> int:
        return max(1, len(_sentences(body, keep_punct=False)))

    def _take(self, chunk: str, is_list: bool = False, partial: bool = False):
        if not chunk:
            return
        chunk = tidy_abbreviations(chunk)
        if self.cur is not None:
            # the rest of a sentence already sent (or already held)
            if self.cur == "out":
                yield from self._out(chunk)
            else:
                self.held[-1] += chunk
            self.line_start = chunk.endswith("\n")
            if not partial:
                self.cur = None
            return
        at_start = self.line_start
        self.line_start = chunk.endswith("\n")
        body = chunk.strip()
        if not at_start and not is_list and re.fullmatch(r"\d{1,2}\.", body):
            # "...the share. 1. Price per unit.\n2. ..." -> the item goes on its own line
            self.pending_item, self.prev_num = True, False
            self.line_start = True
            yield from self._out("\n")
            return
        if self.pending_item and body and not is_list:
            self.pending_item = False
            if len(body.split()) <= 14 and "?" not in body:
                self.prev_num = True
                yield from self._out(body.rstrip(" .;") + ("\n" if chunk.endswith("\n") else ""))
                return
        if is_list:
            chunk = self._list_line(chunk)
            body = chunk.strip()
            parts = _sentences(chunk.rstrip("\n"), keep_punct=True)
            if len(parts) > 1 or "?" in body:
                # prose glued onto a list item ("2. Total market. What share would you take?"):
                # the item is structure, the rest is counted like any prose
                nl = "\n" if chunk.endswith("\n") else ""
                head, rest = parts[0], parts[1:]
                if "?" in head:
                    rest, head = [head] + rest, ""
                if head:
                    yield from self._out(head.rstrip(" .;") + " ")
                self.line_start = False
                for x in rest:
                    yield from self._take(x + " ")
                if nl:
                    yield from self._take("\n")
                return
        elif at_start and body:
            self.prev_num = False
        skeleton = at_start and bool(_SKELETON_LINE.match(chunk))
        if skeleton and chunk.rstrip().endswith("."):
            chunk = re.sub(r"\.(\s*)$", r"\1", chunk)  # "= total consumption." is not a sentence
        structure = (is_list or skeleton or not body
                     or (at_start and chunk.endswith("\n") and not re.search(r"[.!?]\s*$", body)))
        if self.held and body:
            self.held.append(chunk)  # once prose is held, everything after it waits
            if partial:
                self.cur = "held"
            return
        if structure and not partial:
            yield from self._out(chunk)
            return
        n = self._n(body)
        if self.count + n > self.budget:
            self.held.append(chunk)
            if partial:
                self.cur = "held"
            return
        self.count += n
        if partial:
            self.cur = "out"
        yield from self._out(chunk)

    def _boundary(self):
        for m in self._BOUNDARY.finditer(self.buf):
            if m.group(0) != "\n" and self._ABBR_END.search(self.buf[:m.end()]):
                continue  # "Rs. 300", "e.g. the", "approx. 20"
            return m
        return None

    def feed(self, piece: str):
        if not piece:
            return
        self.buf += piece
        while self.buf:
            if self.line_start and self.cur is None:
                if self._MAYBE_MARK.match(self.buf):
                    return  # "1" / "1." / "-" so far: wait to see whether a list line starts
                if self._MARK.match(self.buf):
                    nl = self.buf.find("\n")
                    if nl == -1:
                        if len(self.buf) > 4 * self._LONG:
                            chunk, self.buf = self.buf, ""
                            yield from self._take(chunk, is_list=True)
                        return  # hold the list line until it ends
                    line, self.buf = self.buf[:nl + 1], self.buf[nl + 1:]
                    yield from self._take(line, is_list=True)
                    continue
            m = self._boundary()
            if not m:
                if len(self.buf) > self._LONG:
                    cut = self.buf.rfind(" ", 0, self._LONG)
                    cut = cut + 1 if cut > 0 else self._LONG
                    chunk, self.buf = self.buf[:cut], self.buf[cut:]
                    yield from self._take(chunk, partial=True)
                    continue
                return
            end = m.end()
            while end < len(self.buf) and self.buf[end] in " \t":
                end += 1
            if end >= len(self.buf) and self.buf[m.end() - 1:m.end()] != "\n":
                return  # wait: the next token may continue the spacing or the line
            chunk, self.buf = self.buf[:end], self.buf[end:]
            yield from self._take(chunk)

    def flush(self) -> str:
        out = []
        if self.buf:
            chunk, self.buf = self.buf, ""
            out += list(self._take(chunk, is_list=bool(self.line_start and self.cur is None
                                                      and self._MARK.match(chunk))))
        self.cur = None
        if self.held and not self.q_out:
            qs = [x for h in self.held for x in _sentences(h, keep_punct=True) if x.rstrip().endswith("?")]
            if qs:
                tail = qs[-1].strip()
                sep = "" if (not self.emitted or self.emitted.endswith((" ", "\n"))) else " "
                self.emitted += sep + tail
                out.append(sep + tail)
        self.held = []
        return "".join(out)


_SOLICIT_RE = re.compile(
    r"(?i)\b(please\s+)?(share|tell me|give me|walk me|provide|explain|describe|outline|"
    r"let me know|talk me through|elaborate)\b[^.!?]*\b"
    r"(thoughts?|final|recommendations?|insights?|feedback|answer|reasoning|next step)\b")

_MODE_FALLBACK = {
    "CLOSE": "Good — that's a reasonable place to close the case.",
    "DELIVER_SOLUTION": "In short: work the cost side first, then tie it back to the profit impact.",
    "DATA_REVEAL": "Carry on with your next step.",
    "CORRECT_MATERIAL": "Check the denominator.",
    "HINT": "Consider a different structural approach here.",
    "REPAIR": "Let's step back and look at the broader picture.",
    "ANSWER_DIRECT": "That is the standard assumption.",
    "TRANSITION": "Let's move on to the next phase.",
    "WRAP_UP_EARLY": "No problem at all - you can stop here. Tap Submit at the top right to see feedback on what you've done so far, or come back to this case any time.",
    "FINAL_REVIEW": "That's your final answer noted. Tap Submit at the top right to see your score and the full worked answer.",
    "SESSION_STATUS": "There's no pass or fail in the room - tap Submit at the top right and you'll get your score, feedback and the full worked answer.",
    "RESTATE_CASE": "The full brief is on screen - on a phone, tap the menu at the top left to open it. Start with a clarifying question or two, then lay out your structure.",
    "TEACH": "Let's take it one piece at a time: put a number on the first driver and build from there.",
    "FOLLOW_UP": "Take that into your next step.",
    "EXPLAIN_FORMAT": "You lead and I play the interviewer: ask a clarifying question or two, lay out your structure, work the numbers step by step, then give a final answer. Tap Submit at the end for your score and the worked answer.",
    "PRODUCT_HELP": "Tap Submit at the top right when you're ready: the results page shows your score, feedback and the full worked answer. Carry on whenever you like.",
    "EXPLAIN_WHY": "It's the link between that piece and what the case is asking you to explain. Go ahead with it.",
    "HOLD_SPACE": "Fair enough. Over to you.",
}

def strip_all_questions(text: str) -> str:
    return _filter_lines(text, lambda x: not x.rstrip().endswith("?"))

def strip_solicitations(text: str) -> str:
    return _filter_lines(text, lambda x: not _SOLICIT_RE.search(x))

def enforce_mode(text: str, mode: str, allow_questions: bool = True, policy: str = "coached") -> str:
    t = sanitize_reply(tidy_skeleton(shape_lists(tidy_abbreviations((text or "")[:3000]))), policy)
    if not allow_questions:
        t = strip_solicitations(strip_all_questions(t)).strip()
        if not t:
            t = _MODE_FALLBACK.get(mode, "Okay, continue.")
    return cap_sentences(t.strip(), 5)
