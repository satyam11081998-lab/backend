"""
Deterministic interviewer lines: presence beats and the handful of fixed
substantive lines that never need a model (open, close, deflect, product/UX
answers, clarification-quota decline, unintelligible input).

Wording is picked from the CONTEXT of the turn - what kind of floor-yield it was,
what was just said, which line was used recently - never from a turn counter,
a hash or a rotation. The first candidate in each list is the most natural one;
later ones exist only so the interviewer does not say the identical line twice
in a row.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional

from services.interviewer.types import Channel


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


def pick(candidates: List[str], recent_lines: Iterable[str]) -> str:
    recent = {_norm(x) for x in list(recent_lines)[-3:]}
    for c in candidates:
        if _norm(c) not in recent:
            return c
    return candidates[0]


# --- presence --------------------------------------------------------------
def hand_back(trigger: str, channel: Channel, recent: Iterable[str]) -> str:
    """The candidate handed the floor back; hand it straight back to them."""
    if trigger == "proceed":          # "shall I proceed / continue / go ahead?"
        options = ["Yes, go ahead.", "Sure, carry on.", "Yes, please continue."]
    elif trigger == "recovered":      # bare "got it" / "oh right" after help
        options = ["Go ahead.", "Carry on.", "Over to you."]
    elif trigger == "structure":      # structure laid out, no material gap
        options = (["Okay. Take it from there."] if channel != Channel.TEXT
                   else ["Okay, that works as a structure. Take it from there.", "Okay. Take it from there."])
    elif trigger == "structure_unchecked":   # structure laid out, nothing has checked it: no verdict
        options = ["Okay. Take it from there.", "Okay, over to you."]
    elif trigger == "here":           # "hello? are you there?"
        options = ["I'm here. Go ahead.", "Still here. Carry on."]
    else:
        options = ["Go ahead.", "Carry on.", "Please continue."]
    return pick(options, recent)


def validate(kind: str, recent: Iterable[str], value: Optional[str] = None) -> str:
    if kind == "anchor_ok" and value:
        options = [f"Yes, {value} works. Use it.", f"{value} is fine. Go with it."]
    elif kind == "assumption":
        options = ["That's a reasonable assumption. Go with it.", "That works. Use it."]
    elif kind == "arithmetic_ok":
        options = ["Yes. Carry on.", "That holds. Keep going."]
    elif kind == "approach_ok":
        options = ["That works. Go ahead.", "Yes, that holds together. Carry on."]
    elif kind == "unassessed":       # asked "is this right?" but nothing checked it: no verdict
        options = ["Take it through to a number and we'll see where it lands.", "Carry it through and keep going."]
    else:
        options = ["That works. Carry on.", "Fine. Keep going."]
    return pick(options, recent)


def acknowledge(recent: Iterable[str]) -> str:
    return pick(["Okay.", "Right.", "Mm-hm."], recent)


# --- fixed substantive lines -------------------------------------------------
def open_line(channel: Channel, is_guesstimate: bool, recent: Iterable[str]) -> str:
    if channel == Channel.TEXT:
        base = ("Hi. The prompt is on your screen. Take a moment with it, then tell me how you'd go about "
                "the estimate." if is_guesstimate else
                "Hi. The case is on your screen. Take a moment with it, then walk me through how you'd approach it.")
    else:
        base = ("Hi. Take a moment with the prompt, then tell me how you'd go about the estimate."
                if is_guesstimate else "Hi. Take a moment with the case, then walk me through how you'd approach it.")
    return pick([base], recent)


def close_line(is_guesstimate: bool, recent: Iterable[str]) -> str:
    if is_guesstimate:
        options = ["Okay, noted. If that's your final estimate, submit it and the debrief will walk through the numbers.",
                   "Noted. Submit it when you're ready and you'll get the full breakdown."]
    else:
        options = ["Okay, noted. If that's your final recommendation, submit it and you'll get the full debrief.",
                   "Noted. Submit it when you're ready and you'll get the full breakdown."]
    return pick(options, recent)


def deflect_line(kind: str, recent: Iterable[str]) -> str:
    if kind == "identity":
        options = ["I'm the AI interviewer for this practice case. Carry on from where you were.",
                   "I'm the AI interviewer running this practice session. Let's keep going with the case."]
    elif kind == "model":
        options = ["I'm the AI interviewer for this practice session; how it's built isn't part of the case. "
                   "Carry on from where you were."]
    elif kind == "rubric":
        options = ["Scoring happens after you submit, so let's keep the focus on the case. Carry on from where you were."]
    else:
        options = ["I'll keep us on the case. Carry on from where you were.",
                   "Let's stay with the case. Pick up from your last step."]
    return pick(options, recent)


def ux_line(recent: Iterable[str]) -> str:
    return pick(["When you're done, give me your final answer and hit submit. Your score and detailed feedback "
                 "show up on the results page right after."], recent)


def clarifications_spent_line(recent: Iterable[str]) -> str:
    return pick(["Make a reasonable assumption you can defend, state it, and carry on.",
                 "Go with an assumption you can defend and keep moving."], recent)


def unintelligible_line(channel: Channel, recent: Iterable[str]) -> str:
    if channel == Channel.TEXT:
        return pick(["Sorry, I didn't follow that. Could you put it another way?"], recent)
    return pick(["Sorry, I didn't catch that. Could you say it again?"], recent)
