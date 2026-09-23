# ============================================================================
# MECE PREP COPILOT - ISOLATED COPY. Do NOT sync with the original.
# Copied services/session_signals.py on 2026-09-23 for the role/company-aware Prep Copilot (v2).
# Tweak freely here; the LIVE cases/guesstimates engine is the ORIGINAL and is
# never imported from this package. See .brain/handoffs/ANTIGRAVITY_HANDOFF_prep-copilot-v2.md
# ============================================================================
"""
Deterministic session signals for the adaptive interviewer.
Separates explicit intents from ambiguous domain activity to route safely to the gate.
"""
from __future__ import annotations

import re
import difflib
from typing import Any, Dict, Iterable, List

_HELP = (
    "help", "hint", "i am stuck", "i'm stuck", "im stuck", " stuck",
    "not getting", "not able to", "unable to", "don't get", "dont get",
    "how do i", "how to start", "how should i", "where do i start",
    "where to start", "what should i do", "what do i do", "confused",
    "i am lost", "i'm lost", "no idea", "not sure how", "guide me",
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
    "give me the answer", "just give me the",
)
_SKIP_STOP = (
    "leave it", "skip", "move on", "next question", "forget it",
    "i don't want to", "i dont want to", "i don't want", "dont want",
    "not interested", "let's move", "lets move", "i'm done", "im done",
    "give up", "leave", "quit",
)
_FRUSTRATION = (
    "irritating", "irritated", "frustrat", "annoying", "annoyed",
    "beating around the bush", "going too into deep", "too deep",
    "too many questions", "erratic", "waste of time", "useless",
    "guide or leave", "not even saying", "you are not agreeing",
    "you're not helping", "not helping", "going in circles",
    "round and round", "same question",
)
_META = (
    "are you a bot", "are you an ai", "are you ai", "are you human",
    "which model", "what model", "chatgpt", "chat gpt", "claude", "gemini",
    "gemina", "chargipity", "i am the admin", "i am admin", "ignore the",
    "ignore your", "ignore all", "reveal your", "system prompt",
    "your instructions",
)
_INTERROGATIVE = (
    "what", "why", "how", "when", "where", "who", "which", "is", "are",
    "do", "does", "did", "can", "could", "should", "would", "may", "might",
    "will",
)
_GREETING = ("hi", "hii", "hello", "hey", "ok", "okay", "start", "let's start",
             "lets start", "so", "good morning", "good evening", "yo")

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
    "move on", "next part", "next question", "let's move", "lets move",
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
    "whats the price", "how many people", "what is the market",
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
_END_SESSION = (
    "i'm done", "im done", "i am done", "leave it", "i don't want to continue",
    "dont want to continue", "don't want to continue", "i give up", "give up",
    "quit", "let's wrap", "lets wrap", "we can wrap", "i'm finished", "im finished",
)
_RECOMMEND = (
    "my recommendation", "i'd recommend", "id recommend", "i would recommend",
    "i recommend", "overall i'd", "overall i would", "so overall i", "in summary",
    "to summarize", "to sum up", "my final recommendation", "overall my",
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


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _has_any(text: str, needles) -> bool:
    return any(n in text for n in needles)


def _looks_garbage(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
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


def detect_intent(text: str) -> Dict[str, Any]:
    t = _norm(text)
    flags = {
        "help_requested": _has_any(t, _HELP),
        "solution_requested": _has_any(t, _SOLUTION),
        "skip_or_stop": _has_any(t, _SKIP_STOP),
        "frustration": _has_any(t, _FRUSTRATION),
        "is_meta": _has_any(t, _META),
        "looks_garbage": _looks_garbage(text),
        "is_question": ("?" in (text or "")) or (bool(t) and t.split(" ", 1)[0] in _INTERROGATIVE),
        "is_greeting_only": t in _GREETING,
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
    elif flags["is_question"]:
        primary = "clarification"
    else:
        primary = "answering"
    flags["intent"] = primary
    return flags


def _candidate_turns(transcript: Iterable[Dict[str, str]]) -> List[str]:
    return [_norm(t.get("content")) for t in transcript
            if (t.get("role") or "user") == "user" and (t.get("content") or "").strip()]


def _assistant_turns(transcript: Iterable[Dict[str, str]]) -> List[str]:
    return [_norm(t.get("content")) for t in transcript
            if t.get("role") == "assistant" and (t.get("content") or "").strip()]


def _stuckish(turn_norm: str) -> bool:
    if not turn_norm or len(turn_norm) < 12:
        return True
    f = detect_intent(turn_norm)
    return bool(f["help_requested"] or f["solution_requested"]
                or f["looks_garbage"] or f["frustration"])


def _error_materiality(t: str) -> str:
    unit_scale = ("ml" in t) and ("litre" in t or "liter" in t)
    double_count = (("add" in t and "whole" in t and "population" in t)
                    or "add all of them" in t or "add them all" in t)
    wrong_denom = ("market share" in t) and ("own sales" in t or "divided by our own" in t)
    mece_overlap = (("split them into" in t or "segment" in t or "categorize" in t
                     or "break them into" in t or "buckets" in t)
                    and ("buy a lot" in t or "heavy" in t or "people who" in t))
    if unit_scale or double_count or wrong_denom or mece_overlap:
        return "material"
    hedged = _has_any(t, ("roughly", "about", "call it", "approx", "i'll round",
                          "ill round", "round to", "ballpark", "give or take"))
    if hedged:
        return "minor"
    return "none"


def compute_signals(transcript: Iterable[Dict[str, str]], new_user_message: str,
                    teaching_policy: str = "coached", prior_state=None) -> Dict[str, Any]:
    transcript = list(transcript or [])
    cand = _candidate_turns(transcript)
    asst = _assistant_turns(transcript)
    new_norm = _norm(new_user_message)
    intent = detect_intent(new_user_message)

    has_work = (any((len(c) > 60 or re.search(r"\d", c)) for c in (cand + [new_norm]))
                or _has_any(new_norm, _WORK_MARKERS))
    recent_probes = sum(1 for a in asst[-3:] if a.endswith("?"))
    interviewer_repeating = len(asst) >= 2 and (_similar(asst[-1], asst[-2]) or asst[-1] in asst[:-1])
    candidate_repeating = any(_similar(new_norm, c) for c in cand[-4:])

    tw = 0
    for c in reversed(cand + [new_norm]):
        if _stuckish(c):
            tw += 1
        else:
            break

    if intent["frustration"]:
        frustration = "high"
    elif (intent["help_requested"] or intent["solution_requested"]) and tw >= 2:
        frustration = "high"
    elif candidate_repeating and intent["help_requested"]:
        frustration = "high"
    elif intent["help_requested"] or tw >= 1:
        frustration = "mild"
    else:
        frustration = "none"

    repair_due = (recent_probes >= 2 and (intent["help_requested"] or tw >= 2
                  or candidate_repeating or interviewer_repeating)) or frustration == "high"

    ends_ellipsis = new_norm.endswith("...") or "..." in new_norm
    candidate_mid_thought = _has_any(new_norm, _MID_THOUGHT) or ends_ellipsis
    wants_space = _has_any(new_norm, _WANTS_SPACE)
    wants_to_advance = _has_any(new_norm, _WANTS_ADVANCE)
    candidate_working = (_has_any(new_norm, _PLANNING)
                         and not intent["help_requested"]
                         and not intent["solution_requested"])
    product_ux_question = _has_any(new_norm, _PRODUCT_UX)
    why_this_question = _has_any(new_norm, _WHY_THIS)
    
    # Needs explicit grammatical question to trigger data reveal, preventing intent conflation
    asks_owned_fact = intent["is_question"] and _has_any(new_norm, _OWNED_FACT) and not intent["help_requested"]

    is_session_open = (
        (intent["intent"] == "greeting")
        or (len(asst) == 0 and len(cand) == 0)
        or (len(cand) == 0 and len(new_norm) <= 24 and _has_any(new_norm, _KICKOFF)
            and not intent["is_question"] and not intent["solution_requested"]
            and not intent["help_requested"]))
    is_final_recommendation = _has_any(new_norm, _RECOMMEND)
    is_post_close = _has_any(new_norm, _POST_CLOSE)
    wants_to_end = _has_any(new_norm, _END_SESSION) and not wants_to_advance
    is_session_close = ((is_final_recommendation or is_post_close or wants_to_end)
                        and not is_session_open)

    has_number = bool(re.search(r"\d", new_norm))
    defended = (" because" in (" " + new_norm)) or (" since " in new_norm) or ("reason" in new_norm)
    confident_unsupported_claim = _has_any(new_norm, _CONFIDENT) and has_number and not defended
    
    # Must explicitly state finality to trigger sanity check, sparing intermediate calculations
    states_final_estimate = _has_any(new_norm, _FINAL_ESTIMATE) and has_number
    
    hedged_self_estimate = has_number and _has_any(new_norm, _HEDGE)
    just_recovered = _has_any(new_norm, _RECOVERY) and len(asst) >= 1
    error_materiality = _error_materiality(new_norm)

    return {
        "intent": intent["intent"],
        "teaching_policy": teaching_policy,
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
        "turns_without_progress": tw,
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
        "error_materiality": error_materiality,
    }

def needs_contextual_assessment(sig: Dict[str, Any]) -> bool:
    """Determines if the deterministic layer lacks confidence for an active domain turn."""
    # Obvious deterministic overrides that bypass assessor
    if sig.get("help_requested") or sig.get("solution_requested"):
        return False
    if sig.get("is_meta") or sig.get("looks_garbage") or sig.get("is_session_open") or sig.get("is_session_close"):
        return False
    if sig.get("error_materiality") == "material":
        return False
    
    # Active reasoning/analysis turns require contextual assessment (e.g., hypotheses, math)
    if sig.get("has_work") or sig.get("asks_owned_fact") or sig.get("candidate_working"):
        return True
        
    return False

def build_signal_block(signals: Dict[str, Any]) -> str:
    lines = [f"- intent this turn: {signals.get('intent', 'unknown')}"]
    lines.append(f"- case phase: {signals.get('case_phase', 'unknown')}")
    lines.append(f"- progress status: {signals.get('progress', 'unknown')}")
    
    lines.append("- learner has real work on the table already" if signals.get("has_work") else "- learner has put little/nothing concrete down yet")
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
    
    block = ("SESSION SIGNALS (contextual read of the learner right now; use them, never quote them):\n"
             + "\n".join(lines))
    block += f"\nTEACHING POLICY: {signals.get('teaching_policy', 'coached')}"
    return block
