"""
Model-led realtime voice: the server side that runs ALONGSIDE the conversation.

The realtime speech model talks to the candidate directly (see
prompts/voice_interviewer_playbook.py). This module is everything the server
still owns, none of it in the path of a reply:

  * voice_interviewer_mode()  -- who gets the model-led interviewer (switch).
  * coach_notes()             -- after each candidate turn, a short deterministic
                                 read of the learner (stuck, frustrated, asked for
                                 help / the answer, interviewer over-questioning,
                                 clarifications used up) turned into notes that
                                 refresh the model's instructions.
  * hint()                    -- the hint ladder: cue -> hint -> framework or
                                 analogy -> one partial step. Generated from the
                                 case, never containing the final answer.
  * answer_request()          -- the answer rule: the first ask gets a framework
                                 and "the full worked answer is on your results
                                 page"; once they insist, the worked answer, with
                                 an honest note that their results will show it.

The case solution never goes into the speech model's instructions; it leaves the
server only through answer_request(), and only when the rule allows it.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from services.session_signals import compute_signals

# ---------------------------------------------------------------- the switch

def voice_interviewer_mode(user_id: Optional[str], email: Optional[str]) -> str:
    """'model_led' (default) or 'renderer' (the V11/V12 decide-every-turn flow).

    VOICE_INTERVIEWER = model_led (default) | renderer | allowlist
    VOICE_INTERVIEWER_ALLOWLIST = comma-separated emails or user ids (allowlist mode:
    listed users get model_led, everyone else renderer)
    """
    mode = (os.getenv("VOICE_INTERVIEWER", "model_led") or "model_led").strip().lower()
    if mode == "model_led":
        return "model_led"
    if mode == "allowlist":
        allowed = {a.strip().lower() for a in os.getenv("VOICE_INTERVIEWER_ALLOWLIST", "").split(",") if a.strip()}
        if (user_id and user_id.lower() in allowed) or (email and email.lower() in allowed):
            return "model_led"
    return "renderer"


def answer_after_asks() -> int:
    """How many times the candidate must ask before the worked answer is given.
    2 = the first ask gets a framework, asking again (insisting once) gets the answer."""
    try:
        return max(1, int(os.getenv("VOICE_ANSWER_AFTER_ASKS", "2")))
    except ValueError:
        return 2


# ---------------------------------------------------------------- state helpers

def voice_state(session_state: Optional[dict]) -> Dict[str, Any]:
    v = dict((session_state or {}).get("voice") or {})
    v.setdefault("hint_level", 0)
    v.setdefault("answer_requests", 0)
    v.setdefault("answer_revealed", False)
    v.setdefault("coach_notes", [])
    return v


def with_voice_state(session_state: Optional[dict], voice: Dict[str, Any]) -> Dict[str, Any]:
    ss = dict(session_state or {})
    ss["voice"] = voice
    # Keep the learner model's ladder in step (the debrief reads hint_level).
    ss["hint_level"] = max(int(ss.get("hint_level", 0) or 0), int(voice.get("hint_level", 0) or 0))
    return ss


def _split_last_user(transcript: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], str]:
    for i in range(len(transcript) - 1, -1, -1):
        if (transcript[i].get("role") or "user") == "user" and (transcript[i].get("content") or "").strip():
            return transcript[:i], transcript[i]["content"]
    return transcript, ""


# ---------------------------------------------------------------- coach notes

NOTE_ANSWER_GIVEN = ("You have already given them the worked answer. Help them wrap up: ask for their final "
                     "recommendation or summary, then tell them they can end and submit.")
NOTE_NO_CLARIFICATIONS = ("Their clarification questions are used up: don't give new figures; ask them to make a "
                          "reasonable assumption and carry on.")
NOTE_ASKED_ANSWER = "They just asked for the answer: if you haven't already, call answer_request and follow it."
NOTE_FRUSTRATED = ("They sound frustrated or stuck in a loop: ask no questions for now, be warm, simplify, and give "
                   "them something concrete to hold on to (get_hint).")
NOTE_ASKED_HELP = "They asked for help: if you haven't already, call get_hint and give it in your own words."
NOTE_STUCK = "They've made no progress for a few turns: offer a small cue (get_hint) rather than another question."
NOTE_TOO_MANY_QUESTIONS = ("Your recent turns were mostly questions. Next turn: no question - react to what they "
                           "just did and let them work.")
NOTE_PROGRESSING = ("They're making real progress: at most a short, specific acknowledgement of what they just did, "
                    "then let them continue. Don't quiz them.")
NOTE_MID_THOUGHT = "They're mid-thought: stay out of the way and let them finish."


def coach_notes(transcript: List[Dict[str, str]], attempt: Dict[str, Any],
                session_state: Optional[dict] = None, policy: str = "coached") -> Tuple[List[str], Dict[str, Any]]:
    """Notes for the model's next turns, from the deterministic learner read.

    `transcript` is the saved history INCLUDING the candidate's latest turn.
    Returns (notes, signals). At most three notes, most important first.
    """
    v = voice_state(session_state)
    history, last_user = _split_last_user(transcript)
    sig = compute_signals(history, last_user, policy, prior_state=session_state, channel="voice") if last_user else {}
    asst = [(t.get("content") or "").strip() for t in transcript if t.get("role") == "assistant"]
    recent_questions = sum(1 for a in asst[-3:] if a.endswith("?"))
    quota = int(attempt.get("clarification_quota") or 0)
    used = int(attempt.get("clarification_used") or 0)

    notes: List[str] = []
    if v.get("answer_revealed"):
        notes.append(NOTE_ANSWER_GIVEN)
    if quota and used >= quota:
        notes.append(NOTE_NO_CLARIFICATIONS)
    if sig.get("solution_requested") and not v.get("answer_revealed"):
        notes.append(NOTE_ASKED_ANSWER)
    if sig.get("frustration") == "high" or sig.get("repair_due"):
        notes.append(NOTE_FRUSTRATED)
    elif sig.get("help_requested"):
        notes.append(NOTE_ASKED_HELP)
    elif sig.get("turns_without_progress", 0) >= 2:
        notes.append(NOTE_STUCK)
    if recent_questions >= 2 and NOTE_FRUSTRATED not in notes:
        notes.append(NOTE_TOO_MANY_QUESTIONS)
    if not notes:
        if sig.get("candidate_mid_thought"):
            notes.append(NOTE_MID_THOUGHT)
        elif sig.get("is_substantive_reasoning"):
            notes.append(NOTE_PROGRESSING)
    return notes[:3], sig


# ---------------------------------------------------------------- hint ladder

LEVELS = {
    1: ("a cue", "Point them at what to look at next. No method, no numbers. One short sentence."),
    2: ("a hint", "Name the missing step or driver they need next, still without numbers from the solution."),
    3: ("a framework or analogy", "Give a simple framework or an everyday analogy that preserves the logic of the "
                                  "next step, so they can see the way forward."),
    4: ("one partial step", "Work ONE step with them (set it up and do the first part), then hand it back for "
                            "them to finish."),
}

_HINT_SYSTEM = """You write what a case interviewer should say to help a stuck candidate in a live spoken practice interview.
Give exactly the requested level of help, tied to where THIS candidate is in the conversation.
Rules: 1-2 short spoken sentences. No question at the end. No praise. Never state the final answer or the final
number of the case, and never walk through the whole solution. Plain words, no markdown."""

_FRAMEWORK_SYSTEM = """A candidate in a live spoken case-interview practice asked for the answer. Instead of the answer,
write a way of thinking they can use: a simple framework or an everyday analogy for how to approach THIS case from where
they are. 1-2 short spoken sentences. Never state the final answer or any number from the model solution. No question."""

_GENERIC = {
    1: "Look at which piece of the problem you haven't sized yet, and start there.",
    2: "Break it into the next driver you need, before putting any numbers on it.",
    3: "Think of it like a funnel: start from the whole population, and narrow it step by step to the people who matter here.",
    4: "Let's do the first step together: take the total base, apply the first filter, and you take it from there.",
}
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _final_numbers(solution: str) -> List[str]:
    """Numbers in the last two sentences of the model solution (its final result)."""
    sents = [s for s in re.split(r"(?<=[.!?])\s+", (solution or "").strip()) if s.strip()]
    tail = " ".join(sents[-2:])
    out = []
    for n in _NUM_RE.findall(tail):
        v = n.replace(",", "")
        try:
            if float(v) >= 10 or "." in v:
                out.append(v)
        except ValueError:
            continue
    return out


def strip_final_answer(text: str, solution: str) -> str:
    """Drop any sentence that carries a number from the solution's final result."""
    finals = set(_final_numbers(solution))
    if not finals:
        return (text or "").strip()
    kept = []
    for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()):
        nums = {n.replace(",", "") for n in _NUM_RE.findall(s)}
        if not (nums & finals):
            kept.append(s)
    return " ".join(kept).strip()


def _transcript_tail(transcript: List[Dict[str, str]], n: int = 10) -> str:
    lines = []
    for t in transcript[-n:]:
        c = (t.get("content") or "").strip()
        if c:
            lines.append(f"{'INTERVIEWER' if t.get('role') == 'assistant' else 'CANDIDATE'}: {c[:600]}")
    return "\n".join(lines)


def _generate(system: str, user: str, user_id: Optional[str], llm=None) -> str:
    from services.ai_providers import openai_client
    from services.ai_usage import log_ai_usage
    cli = llm or openai_client()
    if cli is None:
        return ""
    model = os.getenv("VOICE_COACH_MODEL", "gpt-4o-mini")
    t0 = time.time()
    try:
        resp = cli.chat.completions.create(
            model=model, temperature=0.4, max_tokens=120, timeout=8,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
    except Exception as e:  # noqa: BLE001
        print(f"[voice_coach] generation failed: {type(e).__name__}")
        return ""
    try:
        log_ai_usage(user_id=user_id, endpoint="/attempts/voice-tool", model=model, response=resp,
                     latency_ms=int((time.time() - t0) * 1000))
    except Exception:  # noqa: BLE001
        pass
    return (resp.choices[0].message.content or "").strip()


def _case_brief(case: Dict[str, Any], case_content: str) -> str:
    return (f"CASE ({case.get('type') or 'case'}):\n{case_content}\n\n"
            f"STORED HINT: {case.get('hint') or '-'}\n\n"
            f"MODEL SOLUTION (for grounding only - never reveal its final result):\n{case.get('solution') or '-'}")


def hint(case: Dict[str, Any], case_content: str, transcript: List[Dict[str, str]],
         session_state: Optional[dict], reason: str = "", where_stuck: str = "",
         user_id: Optional[str] = None, llm=None) -> Tuple[str, Dict[str, Any]]:
    """Next rung of the ladder. Returns (tool_output_for_the_model, new_voice_state)."""
    v = voice_state(session_state)
    level = min(4, int(v.get("hint_level", 0) or 0) + 1)
    if reason == "frustrated":
        level = max(level, 3)  # frustration skips the gentle cue: give something concrete
    name, how = LEVELS[level]
    user = (_case_brief(case, case_content)
            + f"\n\nCONVERSATION SO FAR:\n{_transcript_tail(transcript)}\n\n"
            + f"THEY ARE STUCK ON: {where_stuck or '(see the conversation)'}\n"
            + f"GIVE: {name}. {how}")
    text = strip_final_answer(_generate(_HINT_SYSTEM, user, user_id, llm), case.get("solution") or "")
    if not text:
        text = (case.get("hint") if level >= 3 and case.get("hint") else None) or _GENERIC[level]
    v["hint_level"] = level
    output = (f"Hint to give ({name}): {text}\n"
              "Say it in your own words in one or two sentences, then hand the floor back. Don't add a question.")
    return output, v


def answer_request(case: Dict[str, Any], case_content: str, transcript: List[Dict[str, str]],
                   session_state: Optional[dict], user_id: Optional[str] = None,
                   llm=None) -> Tuple[str, Dict[str, Any], bool]:
    """The answer rule. Returns (tool_output_for_the_model, new_voice_state, answer_given)."""
    v = voice_state(session_state)
    v["answer_requests"] = int(v.get("answer_requests", 0) or 0) + 1
    solution = (case.get("solution") or "").strip()
    if v["answer_requests"] < answer_after_asks() and not v.get("answer_revealed"):
        user = (_case_brief(case, case_content)
                + f"\n\nCONVERSATION SO FAR:\n{_transcript_tail(transcript)}")
        way = strip_final_answer(_generate(_FRAMEWORK_SYSTEM, user, user_id, llm), solution)
        if not way:
            way = case.get("hint") or _GENERIC[3]
        output = ("Do NOT give the answer yet. Say, in your own words: you'd recommend thinking along these lines "
                  "rather than you handing over the answer, because working it out is what builds the skill - and "
                  "that the full worked answer will be on their results page when they finish. Then offer this way "
                  f"of thinking: {way}\nKeep it to two or three sentences and hand back.")
        return output, v, False
    v["answer_revealed"] = True
    caveat = ("First say briefly, without lecturing: since they're taking the answer from you here, their results "
              "will show they got this part from the interviewer, so they won't be able to tell from them whether "
              "they could have worked it out themselves.")
    if solution:
        output = (f"{caveat} Then give the worked answer, spoken and concise (at most four sentences): {solution}\n"
                  "After that, ask whether they want to try the recommendation or wrap up.")
    else:
        output = (f"{caveat} This case has no stored model answer: walk them through a sensible approach and a "
                  "reasonable final estimate yourself, in at most four spoken sentences.")
    return output, v, True
