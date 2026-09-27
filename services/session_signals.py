"""
Deterministic session signals for the adaptive MECE interviewer.

V11 goals:
- Treat each candidate turn with its own intent/state.
- Protect short but valid numerical work (e.g. "1 crore", "50%", "3").
- Avoid substring-driven false positives for stops/greetings.
- Use conservative semantic hypothesis detection.
- Track explicit interviewer presence events separately when metadata exists.
- Keep learning/profile state out of current-turn decision logic.
"""
from __future__ import annotations

import difflib
import re
from typing import Any, Dict, Iterable, List


_HELP = (
    "help", "hint", "stuck", "i am stuck", "i'm stuck", "im stuck",
    "not getting", "not able to", "unable to", "don't get", "dont get",
    "how do i", "how to start", "how should i", "where do i start",
    "where to start", "what should i do", "what do i do", "confused",
    "i am lost", "i'm lost", "no idea", "idk", "i don't know", "i dont know", "no clue",
    "not sure how", "guide me",
    "give me direction", "point me", "don't understand", "dont understand",
)

_SOLUTION = (
    "give me the answer", "what is the answer", "what's the answer",
    "tell me the answer", "just tell me", "show me the answer",
    "show me the approach", "show me the solution", "give me the approach",
    "give me the solution", "provide me with the approach", "provide the approach",
    "correct approach", "the correct approach", "solve it for me",
    "solve this for me", "what is the correct", "show the correct",
    "show me how", "what is the answer you", "you tell me first",
    "give me the final", "tell me the final", "just the final answer",
    "the final answer now", "what's the final answer", "whats the final answer",
    "give me the answer",
)

# Only use explicit stop/advance phrases. In particular, do NOT use a raw
# substring such as "leave" because it occurs naturally in case reasoning.
# Stop/advance detection is intentionally narrow. Avoid broad phrases such as
# "I don't want to..." because they occur naturally inside case reasoning.
_STOP_EXACT = {
    "quit", "skip", "give up", "leave", "leave it", "move on", "next question",
    "forget it", "i'm done", "im done", "i am done", "i'm finished",
    "im finished", "i am finished", "wrap this up",
}

_SESSION_END_EXACT = {
    "quit", "give up", "i'm done", "im done", "i am done",
    "i'm finished", "im finished", "i am finished", "wrap this up",
    "let's wrap", "lets wrap", "we can wrap",
}

_FRUSTRATION = (
    "irritating", "irritated", "frustrat", "annoying", "annoyed",
    "beating around the bush", "too many questions", "waste of time",
    "guide or leave", "you are not agreeing", "you're not helping",
    "not helping", "going in circles", "round and round", "same question",
    "stop asking",
)

_META = (
    "are you a bot", "are you an ai", "are you ai", "are you human",
    "which model", "what model are you", "what model is this",
    "chatgpt", "chat gpt", "i am the admin", "i am admin",
    "ignore the", "ignore your", "ignore all", "reveal your",
    "system prompt", "your instructions",
)

_INTERROGATIVE = {
    "what", "why", "how", "when", "where", "who", "which", "is", "are",
    "do", "does", "did", "can", "could", "should", "would", "may", "might",
    "will",
}

# Exact-only greetings. "so" is intentionally NOT a greeting.
_GREETING = {
    "hi", "hii", "hello", "hey", "ok", "okay", "start",
    "let's start", "lets start", "good morning", "good evening", "yo",
}

_MID_THOUGHT = (
    "let me think", "let me redo", "let me recompute", "let me recalculate",
    "no wait", "wait no", "hold on", "give me a sec", "give me a second",
    "one sec", "let me re-do", "scratch that", "let me redo the",
)

_WANTS_SPACE = (
    "let me solve", "let me work", "let me try", "just let me", "let me do it",
    "stop asking", "too many questions", "let me finish", "give me a moment",
)

_WANTS_ADVANCE = (
    "move on", "next part", "next question", "skip this", "skip it",
    "skip this part", "skip this question", "let's move", "lets move",
    "moving on", "shift gears", "next topic", "next section",
)

_PRODUCT_UX = (
    "where do i see", "where will i see", "my results", "results page",
    "results after", "where do i find", "how do i submit", "where is the",
    "what happens after", "after this", "after we finish", "see my score",
)

_WHY_THIS = (
    "why are you asking", "why do you ask", "why this question",
    "why that question", "what's the point of", "whats the point of",
    "why do you want to know", "why are we", "why does that matter",
)

_OWNED_FACT = (
    "population", "growth rate", "market growth", "competitors", "how many players",
    "market size", "gdp", "per capita", "how big is the market", "what's the price",
    "whats the price", "what is the price", "how many people", "what is the market",
)

_CONFIDENT = (
    "obviously", "definitely", "everyone in", "everyone buys", "always",
    "100%", "hundred percent", "surely", "no doubt", "clearly", "must be 100",
    "almost everyone", "nearly everyone", "basically everyone", "everyone uses",
    "everyone drives", "everyone has", "literally everyone",
)

_RECOVERY = (
    "oh right", "ohh", "oh i see", "i see so", "i see, so", "got it, so",
    "okay so i just", "so i just", "makes sense so", "ah so", "oh so",
)

# Keep closing signals explicit enough that "in summary" in the middle of
# analysis does not accidentally terminate the interview.
_RECOMMEND = (
    "my recommendation", "i'd recommend", "id recommend", "i would recommend",
    "i recommend", "my final recommendation",
)

_POST_CLOSE = (
    "how did i do", "how'd i do", "any tips", "any feedback", "how was i",
    "what's my score", "whats my score", "how did that go", "any advice",
    "any pointers",
)

_FINAL_ESTIMATE = (
    "final answer", "final number", "so my answer", "my answer is",
    "thats my number", "that's my number", "final figure",
)

_PLANNING = (
    "i need to think about", "let me first", "i'll start with", "step one",
    "i need to figure out how many",
)

_WORK_MARKERS = (
    "i'll ", "i'd ", "i will ", "let me ", "i want to", "i'm going to", "im going to",
    "instead of", "rather than", "i'll go", "i'll use", "i'll take", "i'll split",
    "i'll size", "i would", "i'd go", "i'd start",
)

_HEDGE = (
    "maybe", "i think", "not sure", "not fully sure", "not totally", "feels about",
    "feels right", "seems okay", "seems right", "i'd guess", "around", "roughly",
    "probably", "kind of", "sort of", "somewhere near", "ish",
)

_KICKOFF = (
    "start", "begin", "let's go", "lets go", "let's do this", "lets do this",
    "ready", "shall we", "kick off", "kickoff", "go ahead",
)

# Conservative causal markers. Generic "if we" / "if they" are deliberately
# excluded because they often describe ordinary planning rather than hypotheses.
_HYPOTHESIS_MARKERS = (
    "because", "due to", "driven by", "explains", "therefore",
    "which suggests", "my hypothesis", "implies", "caused by",
)
_HYPOTHESIS_QUESTION_MARKERS = (
    "could it be", "could this be", "might it be", "is it possible",
    "could the", "might the", "do you think", "does that suggest",
)

_ANALYTICAL_TERMS = (
    "revenue", "sales", "profit", "margin", "price", "volume", "demand", "supply",
    "cost", "customer", "customers", "consumer", "market", "market share",
    "competitor", "capacity", "units", "growth", "decline", "declined", "fell",
    "increased", "decreased", "conversion", "penetration", "distribution",
)

_YIELD_FLOOR = (
    "that's my answer", "thats my answer", "that's my estimate", "thats my estimate",
    "that's my number", "thats my number", "my final answer is", "i'll move to",
    "i will move to", "moving to the next", "i'll go to the next",
)

_PRESENCE_TEXT_MATCHES = {
    "got it.", "alright.", "okay.", "right.", "understood.",
    "go ahead.", "makes sense. go ahead.", "sure, continue.",
    "take it from there.", "alright, continue.", "that works.", "that makes sense.",
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _has_any(text: str, needles: Iterable[str]) -> bool:
    t = _norm(text)
    for needle in needles:
        n = _norm(needle)
        if not n:
            continue
        # Word-boundary style matching prevents "leave" matching "leaving".
        if re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", t):
            return True
    return False


def _looks_garbage(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    # Numerical work containing an explicit arithmetic operator is valid even when
    # its alphabetic vowel ratio is low (e.g. "1.4 billion / 3 = 0.46 billion").
    if re.search(r"\d", t) and re.search(r"[=+*/×÷%-]", t):
        return False
    if re.search(r"(.)\1{4,}", t):
        return True
    letters = re.sub(r"[^a-zA-Z]", "", t)
    if len(letters) >= 6:
        vowels = len(re.findall(r"[aeiou]", letters.lower()))
        if vowels / max(1, len(letters)) < 0.12:
            return True
    if len(letters) >= 10 and len(set(letters.lower())) <= 5:
        return True
    if " " not in t and len(letters) >= 8 and re.search(r"[bcdfghjklmnpqrstvwxz]{5,}", letters.lower()):
        return True
    return False


def _similar(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() > 0.9


def _is_hypothesis_turn(t: str, is_question: bool) -> bool:
    if len(t.split()) < 5:
        return False
    analytical = _has_any(t, _ANALYTICAL_TERMS)
    if not analytical:
        return False
    if _has_any(t, _HYPOTHESIS_MARKERS):
        return True
    # Questions can also be hypotheses when they explicitly propose a causal or
    # explanatory possibility. Ordinary "why/what" clarifications remain questions.
    return is_question and _has_any(t, _HYPOTHESIS_QUESTION_MARKERS)


def _is_stop_intent(t: str) -> bool:
    if not t:
        return False
    if t in _STOP_EXACT:
        return True
    # Allow a small family of explicit transition requests without treating
    # ordinary case reasoning as a stop command.
    transition_patterns = (
        r"^(?:please\s+)?skip(?:\s+(?:this|it|this question|this part))$",
        r"^(?:please\s+)?move\s+on(?:\s+to\s+(?:the\s+)?next(?:\s+question|\s+part|\s+section))?$",
        r"^(?:please\s+)?let(?:'|’)s\s+move\s+on$",
    )
    return any(re.fullmatch(pattern, t) for pattern in transition_patterns)


def _is_session_end_intent(t: str) -> bool:
    if not t:
        return False
    if t in _SESSION_END_EXACT:
        return True
    terminal_patterns = (
        r"^i(?:'|’)m\s+(?:all\s+)?done$",
        r"^i\s+am\s+(?:all\s+)?done$",
        r"^i(?:'|’)m\s+(?:all\s+)?finished$",
        r"^i\s+am\s+(?:all\s+)?finished$",
        r"^i(?:\s+really)?\s+(?:do\s+not|don’t|don't)\s+want\s+to\s+continue$",
    )
    return any(re.fullmatch(pattern, t) for pattern in terminal_patterns)


def detect_intent(text: str) -> Dict[str, Any]:
    t = _norm(text)
    first = t.split(" ", 1)[0] if t else ""
    is_question = ("?" in (text or "")) or first in _INTERROGATIVE

    flags = {
        "help_requested": _has_any(t, _HELP),
        "solution_requested": _has_any(t, _SOLUTION),
        "skip_or_stop": _is_stop_intent(t),
        "frustration": _has_any(t, _FRUSTRATION),
        "is_meta": _has_any(t, _META),
        "looks_garbage": _looks_garbage(text),
        "is_question": is_question,
        "is_greeting_only": t in _GREETING,
        "is_hypothesis": _is_hypothesis_turn(t, is_question),
    }

    if flags["is_meta"]:
        primary = "meta"
    elif flags["solution_requested"]:
        primary = "asking_for_solution"
    elif flags["skip_or_stop"]:
        primary = "wants_to_stop"
    elif flags["help_requested"]:
        primary = "asking_for_help"
    elif flags["is_greeting_only"]:
        primary = "greeting"
    elif flags["is_hypothesis"]:
        primary = "hypothesis"
    elif flags["is_question"]:
        primary = "clarification"
    else:
        primary = "answering"

    flags["intent"] = primary
    return flags


def _candidate_turns(transcript: Iterable[Dict[str, Any]]) -> List[str]:
    return [
        _norm(t.get("content"))
        for t in transcript
        if (t.get("role") or "user") == "user" and (t.get("content") or "").strip()
    ]


def _assistant_turns(transcript: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        t for t in transcript
        if t.get("role") == "assistant" and (t.get("content") or "").strip()
    ]


def _assistant_was_presence(turn: Dict[str, Any]) -> bool:
    """Prefer explicit event metadata; parse a serialized semantic event as a fallback."""
    event_type = str(turn.get("event_type") or "").lower()
    kind = str(turn.get("kind") or turn.get("message_type") or "").lower()
    event = turn.get("event") or turn.get("semantic_event") or {}

    if isinstance(event, dict):
        event_type = str(event.get("event_type") or event.get("type") or event_type).lower()
        kind = str(event.get("kind") or kind).lower()

    if event_type in {"interviewer_presence", "presence", "interviewer_presence_event"}:
        return True
    if kind in {"presence", "interviewer_presence", "semantic_event"}:
        return True

    content_raw = turn.get("content") or ""
    content = _norm(content_raw)
    if content in _PRESENCE_TEXT_MATCHES:
        return True
    if content.startswith("{") and "interviewer_presence" in content:
        try:
            parsed = __import__("json").loads(content)
            return str(parsed.get("event_type") or parsed.get("type") or "").lower() == "interviewer_presence"
        except Exception:
            return False
    return False


def _stuckish_text(turn_text: str) -> bool:
    """Classify one historical candidate turn using that turn's own intent."""
    flags = detect_intent(turn_text)
    return bool(
        flags.get("help_requested")
        or flags.get("solution_requested")
        or flags.get("looks_garbage")
        or flags.get("frustration")
    )


def _error_materiality(t: str) -> str:
    """Conservative lexical detector for clearly material reasoning errors."""
    t = _norm(t)

    def _bad_conversion(pattern: str, factor: float) -> bool:
        for m in re.finditer(pattern, t, flags=re.I):
            left = float(m.group(1).replace(",", ""))
            right = float(m.group(2).replace(",", ""))
            if abs(right - left * factor) > max(1e-9, abs(left * factor) * 0.02):
                return True
        return False

    litre_to_ml = _bad_conversion(
        r"(\d+(?:\.\d+)?)\s*(?:litre|liter|l)\s*(?:=|is)\s*(\d+(?:\.\d+)?)\s*ml\b",
        1000.0,
    )
    kg_to_g = _bad_conversion(
        r"(\d+(?:\.\d+)?)\s*kg\s*(?:=|is)\s*(\d+(?:\.\d+)?)\s*g\b",
        1000.0,
    )

    negation_guard = re.search(r"\b(?:not|don't|dont|do not|won't|wont|never)\b", t)
    double_count_positive = (
        ("add" in t and "whole" in t and "population" in t)
        or "add all of them" in t
        or "add them all" in t
        or "double count" in t
        or "double-count" in t
    )
    double_count = bool(double_count_positive and not negation_guard)
    wrong_denom = (
        "market share" in t
        and bool(re.search(
            r"market\s+share.{0,140}\b(?:divid(?:ed|ing)|divide)\b.{0,80}\b(?:our own|own)\s+sales\b",
            t,
        ))
    )
    mece_overlap_positive = (
        ("overlap" in t or "overlapping" in t or "double counted" in t
         or "double-counted" in t)
        and ("segment" in t or "categorize" in t or "bucket" in t or "population" in t)
    )
    mece_overlap = bool(mece_overlap_positive and not negation_guard)

    if litre_to_ml or kg_to_g or double_count or wrong_denom or mece_overlap:
        return "material"

    hedged = _has_any(
        t,
        ("roughly", "about", "call it", "approx", "i'll round", "ill round",
         "round to", "ballpark", "give or take"),
    )
    if hedged:
        return "trivial"
    return "none"


def compute_signals(
    transcript: Iterable[Dict[str, Any]],
    new_user_message: str,
    teaching_policy: str = "coached",
    prior_state=None,
    channel: str = "text",
    is_voice_partial: bool = False,
) -> Dict[str, Any]:
    transcript = list(transcript or [])
    cand = _candidate_turns(transcript)
    asst_turns = _assistant_turns(transcript)
    asst_contents = [_norm(t.get("content")) for t in asst_turns]
    new_norm = _norm(new_user_message)
    intent = detect_intent(new_user_message)

    channel = channel if channel in {"text", "voice"} else "text"

    has_work = (
        any(bool(re.search(r"\d", c)) or len(c) > 60 for c in (cand + [new_norm]))
        or _has_any(new_norm, _WORK_MARKERS)
        or intent["is_hypothesis"]
    )

    recent_probes = sum(1 for a in asst_contents[-3:] if a.endswith("?"))
    interviewer_repeating = (
        len(asst_contents) >= 2
        and (_similar(asst_contents[-1], asst_contents[-2]) or asst_contents[-1] in asst_contents[:-1])
    )
    candidate_repeating = any(_similar(new_norm, c) for c in cand[-4:])

    # Explicit semantic event history is preferred. Compatibility fallback is exact text.
    recent_assistant_turns = asst_contents[-4:]
    recent_presence_events = [
        _norm(t.get("content")) for t in asst_turns[-6:] if _assistant_was_presence(t)
    ][-4:]
    recent_substantive_events = [
        _norm(t.get("content")) for t in asst_turns[-6:] if not _assistant_was_presence(t)
    ][-4:]
    last_action_was_presence = bool(asst_turns and _assistant_was_presence(asst_turns[-1]))

    # IMPORTANT: every historical candidate turn is classified using its OWN intent.
    candidate_sequence = cand + [new_norm]
    turns_without_progress = 0
    for c in reversed(candidate_sequence):
        if _stuckish_text(c):
            turns_without_progress += 1
        else:
            break

    if intent["frustration"]:
        frustration = "high"
    elif (intent["help_requested"] or intent["solution_requested"]) and turns_without_progress >= 2:
        frustration = "high"
    elif candidate_repeating and intent["help_requested"]:
        frustration = "high"
    elif intent["help_requested"] or turns_without_progress >= 1:
        frustration = "mild"
    else:
        frustration = "none"

    repair_due = (
        (recent_probes >= 2 and (
            intent["help_requested"]
            or turns_without_progress >= 2
            or candidate_repeating
            or interviewer_repeating
        ))
        or frustration == "high"
    )

    ends_ellipsis = new_norm.endswith("...") or "..." in new_norm
    candidate_mid_thought = _has_any(new_norm, _MID_THOUGHT) or ends_ellipsis
    wants_space = _has_any(new_norm, _WANTS_SPACE)
    wants_to_advance = _has_any(new_norm, _WANTS_ADVANCE)
    candidate_working = (
        (_has_any(new_norm, _PLANNING) or _has_any(new_norm, _WORK_MARKERS))
        and not intent["help_requested"]
        and not intent["solution_requested"]
    )
    product_ux_question = _has_any(new_norm, _PRODUCT_UX)
    why_this_question = _has_any(new_norm, _WHY_THIS)

    asks_owned_fact = (
        intent["is_question"]
        and _has_any(new_norm, _OWNED_FACT)
        and not intent["help_requested"]
    )

    is_session_open = (
        (intent["intent"] == "greeting" and len(cand) == 0)
        or (
            len(cand) == 0
            and len(new_norm) <= 24
            and _has_any(new_norm, _KICKOFF)
            and not intent["is_question"]
            and not intent["solution_requested"]
            and not intent["help_requested"]
        )
    )

    is_final_recommendation = _has_any(new_norm, _RECOMMEND)
    is_post_close = _has_any(new_norm, _POST_CLOSE)
    wants_to_end = _is_session_end_intent(new_norm) and not wants_to_advance
    is_session_close = (
        (is_final_recommendation or is_post_close or wants_to_end)
        and not is_session_open
    )

    has_number = bool(re.search(r"\d", new_norm))
    defended = (
        " because" in (" " + new_norm)
        or " since " in (" " + new_norm)
        or "reason" in new_norm
    )
    confident_unsupported_claim = (
        _has_any(new_norm, _CONFIDENT) and has_number and not defended
    )

    states_final_estimate = _has_any(new_norm, _FINAL_ESTIMATE) and has_number
    hedged_self_estimate = has_number and _has_any(new_norm, _HEDGE)
    just_recovered = _has_any(new_norm, _RECOVERY) and len(asst_contents) >= 1
    explicit_floor_yield = _has_any(new_norm, _YIELD_FLOOR)
    error_materiality = _error_materiality(new_norm)

    # Deterministic fallback progress state. The contextual assessor may refine it.
    if turns_without_progress >= 2:
        progress = "repeatedly_stuck"
    elif turns_without_progress == 1:
        progress = "stalled"
    elif has_work:
        progress = "progressing"
    else:
        progress = "unclear"

    return {
        "intent": intent["intent"],
        "teaching_policy": teaching_policy,
        "channel": channel,
        "is_voice_partial": bool(is_voice_partial),
        "hint_level": int((prior_state or {}).get("hint_level", 0) or 0),
        "learner_level": (prior_state or {}).get("learner_level", "unknown"),
        "repairs_done": int((prior_state or {}).get("repairs_done", 0) or 0),
        "help_requested": intent["help_requested"],
        "solution_requested": intent["solution_requested"],
        "skip_or_stop": intent["skip_or_stop"],
        "is_meta": intent["is_meta"],
        "is_scope_question": intent["intent"] == "clarification",
        "looks_garbage": intent["looks_garbage"],
        "has_work": has_work,
        "recent_probes": recent_probes,
        "interviewer_repeating": interviewer_repeating,
        "candidate_repeating": candidate_repeating,
        "recent_assistant_turns": recent_assistant_turns,
        "recent_presence_events": recent_presence_events,
        "recent_substantive_events": recent_substantive_events,
        "last_action_was_presence": last_action_was_presence,
        "turns_without_progress": turns_without_progress,
        "frustration": frustration,
        "frustration_explicit": intent["frustration"],
        "repair_due": repair_due,
        "candidate_turn_count": len(cand) + 1,
        "is_session_open": is_session_open,
        "is_session_close": is_session_close,
        "candidate_mid_thought": candidate_mid_thought,
        "wants_space": wants_space,
        "wants_to_advance": wants_to_advance,
        "candidate_working": candidate_working,
        "product_ux_question": product_ux_question,
        "why_this_question": why_this_question,
        "asks_owned_fact": asks_owned_fact,
        "confident_unsupported_claim": confident_unsupported_claim,
        "states_final_estimate": states_final_estimate,
        "hedged_self_estimate": hedged_self_estimate,
        "just_recovered": just_recovered,
        "explicit_floor_yield": explicit_floor_yield,
        "error_materiality": error_materiality,
        "progress": progress,
        "new_message_norm": new_norm,
    }


def needs_contextual_assessment(sig: Dict[str, Any]) -> bool:
    """Use the assessor only where deterministic signals genuinely need context."""
    if not sig:
        return False
    if sig.get("is_voice_partial"):
        return False
    if sig.get("help_requested") or sig.get("solution_requested"):
        return False
    if sig.get("is_meta") or sig.get("looks_garbage") or sig.get("is_session_open") or sig.get("is_session_close"):
        return False
    if sig.get("error_materiality") == "material":
        return False
    if sig.get("wants_to_advance") or sig.get("asks_owned_fact"):
        return False

    # Generic clarifications are directly answerable and should not incur an LLM assessor call.
    # Hypotheses can require contextual evaluation of whether the interviewer should reveal data.
    return bool(sig.get("intent") == "hypothesis")


def build_signal_block(signals: Dict[str, Any]) -> str:
    lines = [f"- intent this turn: {signals.get('intent', 'unknown')}"]
    lines.append(f"- case phase: {signals.get('case_phase', 'unknown')}")
    lines.append(f"- progress status: {signals.get('progress', 'unknown')}")
    lines.append(
        "- learner has real work on the table already"
        if signals.get("has_work")
        else "- learner has put little/nothing concrete down yet"
    )
    if signals.get("repair_due"):
        lines.append("- YOUR RECENT MOVES ARE NOT WORKING -> repair is due: stop probing, change strategy.")
    if signals.get("turns_without_progress", 0) >= 2:
        lines.append(f"- stuck for {signals['turns_without_progress']} turns with no progress.")
    if signals.get("frustration", "none") != "none":
        lines.append(f"- frustration: {signals['frustration']} -> change what you DO.")
    if signals.get("interviewer_repeating"):
        lines.append("- you have repeated yourself -> do NOT reuse a previous line; find a fresh move.")
    if signals.get("looks_garbage"):
        lines.append("- last message looks like noise/typo -> ask them to restate briefly.")

    block = (
        "SESSION SIGNALS (contextual read of the learner right now; use them, never quote them):\n"
        + "\n".join(lines)
    )
    block += f"\nTEACHING POLICY: {signals.get('teaching_policy', 'coached')}"
    return block
