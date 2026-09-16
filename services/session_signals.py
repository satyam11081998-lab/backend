"""
Deterministic session signals for the adaptive interviewer (Phase 1).

Pure Python. NO model call, NO API key (so it is unit-testable in isolation).
Reads the transcript + the new candidate message and produces:
  (a) a machine dict of signals, and
  (b) a compact human-readable block injected into the interviewer prompt.

These are SIGNALS, not scripted responses. We never map a phrase to a reply
(that is the brittle approach the redesign rejects). We tell the model what
situation it is in -- stuck vs progressing, asked-for-help vs answering,
frustrated vs calm, looping vs fresh -- and the model chooses the move.
"""
from __future__ import annotations

import re
import difflib
from typing import Any, Dict, Iterable, List

# --- keyword families (lowercased substring match) ---------------------------
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


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _has_any(text: str, needles) -> bool:
    return any(n in text for n in needles)


def _looks_garbage(text: str) -> bool:
    """Keyboard mash / ASR noise: letters but no real word structure."""
    t = (text or "").strip()
    if not t:
        return True
    if re.search(r"(.)\1{4,}", t):            # "bhhhhhh", "kloooool"
        return True
    letters = re.sub(r"[^a-zA-Z]", "", t)
    if len(letters) >= 6:
        vowels = len(re.findall(r"[aeiou]", letters.lower()))
        if vowels / max(1, len(letters)) < 0.12:   # almost no vowels -> mash
            return True
    if len(letters) >= 10 and len(set(letters.lower())) <= 5:   # "yibuybuyb..." — few unique letters
        return True
    return False


def _similar(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() > 0.9


def detect_intent(text: str) -> Dict[str, Any]:
    """Primary intent + flags for ONE candidate message. Signals, not scripts."""
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
        primary = "wants_solution"
    elif flags["skip_or_stop"]:
        primary = "wants_to_stop"
    elif flags["help_requested"]:
        primary = "asking_for_help"
    elif flags["is_greeting_only"]:
        primary = "greeting"
    elif flags["is_question"]:
        primary = "scope_question"
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


def compute_signals(transcript: Iterable[Dict[str, str]], new_user_message: str,
                    teaching_policy: str = "coached") -> Dict[str, Any]:
    transcript = list(transcript or [])
    cand = _candidate_turns(transcript)
    asst = _assistant_turns(transcript)
    new_norm = _norm(new_user_message)
    intent = detect_intent(new_user_message)

    has_work = any((len(c) > 60 or re.search(r"\d", c)) for c in cand)
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

    return {
        "intent": intent["intent"],
        "teaching_policy": teaching_policy,
        "help_requested": intent["help_requested"],
        "solution_requested": intent["solution_requested"],
        "skip_or_stop": intent["skip_or_stop"],
        "is_meta": intent["is_meta"],
        "is_scope_question": intent["intent"] == "scope_question",
        "looks_garbage": intent["looks_garbage"],
        "has_work": has_work,
        "recent_probes": recent_probes,
        "interviewer_repeating": interviewer_repeating,
        "candidate_repeating": candidate_repeating,
        "turns_without_progress": tw,
        "frustration": frustration,
        "repair_due": repair_due,
        "candidate_turn_count": len(cand) + 1,
    }


def build_signal_block(signals: Dict[str, Any]) -> str:
    lines = [f"- intent this turn: {signals['intent']}"]
    lines.append("- learner has real work on the table already" if signals["has_work"]
                 else "- learner has put little/nothing concrete down yet")
    if signals["repair_due"]:
        lines.append("- YOUR RECENT MOVES ARE NOT WORKING -> repair is due: stop probing, change "
                     "strategy (reframe / simple analogy / a partial step), restore momentum")
    if signals["turns_without_progress"] >= 2:
        lines.append(f"- stuck for {signals['turns_without_progress']} turns with no progress")
    if signals["frustration"] != "none":
        lines.append(f"- frustration: {signals['frustration']} -> change what you DO; never say "
                     "'I understand your frustration'")
    if signals["interviewer_repeating"]:
        lines.append("- you have repeated yourself -> do NOT reuse a previous line; find a fresh move")
    if signals["candidate_repeating"]:
        lines.append("- learner is repeating themselves -> your last move did not land; try a different tack")
    if signals["is_scope_question"]:
        lines.append("- scope/fact question -> answer with a specific number, own the facts")
    if signals["looks_garbage"]:
        lines.append("- last message looks like noise/typo -> ask them to restate briefly; do not analyse it")
    block = ("SESSION SIGNALS (deterministic read of the learner right now; use them, never quote them):\n"
             + "\n".join(lines))
    block += f"\nTEACHING POLICY: {signals['teaching_policy']}"
    return block
