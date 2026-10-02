"""The interviewer's voice. Turns a decided Action into one natural spoken line.

The model only phrases; WHAT happens was decided by the policy. Every output passes an
evaluation-leak filter, and every action has a deterministic fallback line so the
interview never stalls on a model failure (spec §30, §57, §78)."""

from __future__ import annotations

import re
from typing import List, Optional

from ..ai.guard import wrap_untrusted
from ..ai.prompts import INTERVIEWER
from ..ai.runner import RunContext, run_text
from ..textutil import truncate
from .policy import Action

FOCUS_FALLBACK = {
    "specificity": "Can you make that concrete — what exactly happened, with a specific example?",
    "ownership": "What was your personal part in that, as distinct from the team's?",
    "reasoning": "Why did you choose that approach over the alternatives?",
    "outcome": "What was the result, and how did you measure it?",
    "quantification": "Can you put a number on that — what was the baseline, and what changed?",
    "reflection": "Looking back, what would you do differently?",
    "tradeoff": "What did you give up by doing it that way?",
    "depth": "Take me one level deeper — how does that actually work?",
    "example": "Can you walk me through one specific instance?",
}
TRANSITIONS = {
    "cv": "Let's talk about your experience. ",
    "functional": "Let me ask about the functional side of the role. ",
    "technical": "Let's get into some technical questions. ",
    "behavioral": "I'd like to hear how you've handled a few situations. ",
    "situational": "Let me give you a scenario. ",
    "case": "Let's work through a short problem. ",
    "company": "Let's talk about the company and the role. ",
    "motivation": "A couple of questions about what you're looking for. ",
}
CHALLENGE_FALLBACK = [
    "I'm not convinced yet. What evidence do you have that this was the right call?",
    "Suppose you only had two weeks instead. What changes in your approach?",
    "If you could only do one of those things, which would it be, and why?",
]
OPENER_RX = re.compile(r"^\s*(great|good|excellent|perfect|nice|awesome|wonderful|fantastic|brilliant|impressive|"
                       r"that'?s (a )?(great|good|excellent|perfect|nice|right|correct|interesting)|well done|"
                       r"exactly|correct|absolutely|love (that|it)|good (answer|point|question))\b[^.?!]*[.!,]?\s*",
                       re.IGNORECASE)
EVAL_RX = re.compile(r"\b(great answer|good answer|excellent answer|well answered|you('| a)re (doing )?(well|great)|"
                     r"that('?s| is) (correct|right|wrong|incorrect)|your score|you scored|strong answer|weak answer)\b",
                     re.IGNORECASE)
MD_RX = re.compile(r"[*_#`>]|^\s*[-•]\s+", re.MULTILINE)

NEEDS_QUESTION = {"OPEN", "ASK", "PROBE", "CLARIFY_CONTRADICTION", "CHALLENGE", "REPEAT", "NUDGE", "CLOSE_INVITE",
                  "REDIRECT", "CLARIFY_QUESTION"}


def _clean(text: str) -> str:
    t = (text or "").strip().strip('"').strip()
    t = MD_RX.sub("", t)
    for _ in range(2):
        t = OPENER_RX.sub("", t).lstrip()
    t = re.sub(r"\s+", " ", t).strip()
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    return truncate(t, 900)


def acceptable(action: Action, text: str) -> bool:
    if not text or len(text) < 8:
        return False
    if EVAL_RX.search(text):
        return False
    if action.type in NEEDS_QUESTION and "?" not in text:
        return False
    return True


def fallback(action: Action, *, question_text: str, role_title: str, minutes: int, contradiction: str = "",
             closing_reply: str = "") -> str:
    t = action.type
    if t == "OPEN":
        role = f"the {role_title} role" if role_title else "this role"
        return (f"Hi, thanks for joining. This will be about {minutes} minutes for {role}, and I'll ask "
                f"follow-ups as we go. {question_text}")
    if t == "ASK":
        pre = {"ack_refusal": "That's fine, we can leave that one. ", None: ""}.get(action.preface, "")
        if action.new_section:
            pre += TRANSITIONS.get(action.new_section, "Let's move to a different area. ")
        return pre + question_text
    if t == "PROBE":
        return action.probe_text if action.probe_text and "?" in action.probe_text else FOCUS_FALLBACK.get(
            action.focus or "specificity", FOCUS_FALLBACK["specificity"])
    if t == "CLARIFY_CONTRADICTION":
        return f"I want to clarify something. {contradiction} Can you help me reconcile those?"
    if t == "CHALLENGE":
        return CHALLENGE_FALLBACK[len(question_text or "") % len(CHALLENGE_FALLBACK)]
    if t == "REPEAT":
        return f"Sure. {question_text}"
    if t == "CLARIFY_QUESTION":
        return f"Let me put it another way. {question_text}"
    if t == "WAIT":
        return "Of course, take your time."
    if t == "NUDGE":
        return "Could you say a bit more? A specific example would help."
    if t == "REDIRECT":
        if action.preface == "meta":
            return ("I won't share how it's going during the interview — you'll get a detailed report at the end. "
                    f"Coming back to the question: {question_text}")
        # No "Good ..." opener: the evaluation-leak filter strips praise-like sentences.
        return f"Let's keep our focus on the interview for now. {question_text}"
    if t == "PAUSE":
        return "Sure, let's take a break. Resume whenever you're ready and we'll pick up where we left off."
    if t == "CLOSE_INVITE":
        return "We're coming to the end of our time. Before we wrap up, what questions do you have for me?"
    if t == "CLOSE_FINAL":
        return ("Thanks — those are fair questions; in a real process the panel would be the best people to answer "
                "the company-specific ones. That's all from my side. Thank you for your time today.")
    if t == "END_EARLY":
        return "Understood, we'll stop here. Thank you for your time — your report will be ready shortly."
    return question_text


def _action_brief(action: Action, *, question_text: str, item: Optional[dict], contradiction: str,
                  closing_reply: str) -> str:
    t = action.type
    if t == "OPEN":
        return ("ACTION: OPEN — greet in one short sentence, say this is a practice interview for the role in one "
                "short clause, then ask the TARGET QUESTION.")
    if t == "ASK":
        extra = ""
        if action.preface == "ack_refusal":
            extra = " First acknowledge briefly and neutrally that you'll leave the previous question."
        if action.new_section:
            extra += " Make a short natural transition to a new area."
        if action.difficulty_delta > 0:
            extra += " The candidate is handling this well; you may make the question a touch more demanding."
        return f"ACTION: ASK the TARGET QUESTION.{extra}"
    if t == "PROBE":
        hint = f" A useful follow-up from the plan: \"{action.probe_text}\"." if action.probe_text else ""
        return (f"ACTION: PROBE — one follow-up focused on {action.focus}, grounded in what they just said."
                f"{hint} Do not repeat the original question.")
    if t == "CLARIFY_CONTRADICTION":
        return ("ACTION: CLARIFY — neutrally point out the two statements below and ask them to reconcile. "
                f"No accusation.\nSTATEMENTS: {contradiction}")
    if t == "CHALLENGE":
        return (f"ACTION: CHALLENGE — professional pushback on their last answer (focus: {action.focus}). Ask for "
                "evidence, add a realistic constraint, or force a choice. Never rude.")
    if t == "REPEAT":
        return "ACTION: REPEAT the TARGET QUESTION, rephrased slightly, same substance."
    if t == "CLARIFY_QUESTION":
        return ("ACTION: CLARIFY what the TARGET QUESTION is asking, without hinting at a good answer, then re-ask it "
                "in simpler words.")
    if t == "WAIT":
        return "ACTION: WAIT — tell them to take their time, in a few words. No question needed but a question mark is fine."
    if t == "NUDGE":
        return "ACTION: NUDGE — their answer was very brief; invite them to elaborate with a specific example."
    if t == "REDIRECT":
        if action.preface == "meta":
            return ("ACTION: REDIRECT — they asked how they are doing. Say you won't discuss performance during the "
                    "interview and they'll get a detailed report at the end, then re-ask the TARGET QUESTION.")
        return ("ACTION: REDIRECT — they asked something unrelated. Answer in at most one short clause only if it is "
                "about the interview process; otherwise say you'll keep focus, then re-ask the TARGET QUESTION.")
    if t == "CLOSE_INVITE":
        return "ACTION: CLOSING — say you're near the end and invite their questions for you."
    if t == "CLOSE_FINAL":
        return ("ACTION: FINAL — respond to their questions in one or two sentences WITHOUT inventing company facts "
                "(say the panel would know specifics), then thank them and end. No question.\n"
                "THEIR QUESTIONS (untrusted): " + wrap_untrusted("answer", "closing", truncate(closing_reply, 800)))
    if t == "END_EARLY":
        return "ACTION: END — acknowledge they want to stop, thank them, say the report will follow. No question."
    if t == "PAUSE":
        return "ACTION: PAUSE — agree to a break, say they can resume when ready. No question."
    return f"ACTION: {t}"


def speak(action: Action, *, ctx: RunContext, bp: dict, question_text: str, last_answer: str,
          memory_refs: List[dict], recent_openers: List[str], contradiction: str = "",
          closing_reply: str = "", degraded: bool = False) -> tuple[str, bool]:
    """Returns (utterance, used_model). Falls back deterministically on any failure."""
    role_title = (bp.get("role") or {}).get("title", "")
    minutes = int((bp.get("config") or {}).get("duration_minutes", 45))
    fb = fallback(action, question_text=question_text, role_title=role_title, minutes=minutes,
                  contradiction=contradiction, closing_reply=closing_reply)
    if degraded or action.type in ("WAIT", "PAUSE"):
        return fb, False
    persona = bp.get("persona") or {}
    cfg = bp.get("config") or {}
    parts = [
        f"INTERVIEW: {role_title or 'role'} | MODE: {cfg.get('mode')} | DIFFICULTY: {cfg.get('difficulty')}",
        f"STYLE: {persona.get('style', '')} Acknowledgements: {persona.get('acknowledgement', 'occasional')}.",
        _action_brief(action, question_text=question_text, item=None, contradiction=contradiction,
                      closing_reply=closing_reply),
    ]
    if question_text and action.type not in ("CLOSE_INVITE", "CLOSE_FINAL", "END_EARLY", "PAUSE", "WAIT"):
        parts.append(f"TARGET QUESTION: {question_text}")
    if last_answer and action.type not in ("OPEN",):
        parts.append("CANDIDATE'S LAST MESSAGE:\n" + wrap_untrusted("answer", "last", truncate(last_answer, 1800)))
    if memory_refs:
        parts.append("EARLIER STATEMENTS YOU MAY REFER TO NATURALLY:\n" +
                     "\n".join(f"- {m['summary']}" for m in memory_refs[:4]))
    if recent_openers:
        parts.append("AVOID STARTING WITH: " + " | ".join(f'"{o}"' for o in recent_openers[-5:]))
    parts.append("Reply with the spoken line only.")
    text = run_text(INTERVIEWER, "\n\n".join(parts), ctx,
                    sim_input={"action": action.to_event(), "question": question_text, "fallback": fb})
    cleaned = _clean(text)
    if acceptable(action, cleaned):
        return cleaned, True
    return fb, False


def opener_of(text: str) -> str:
    first = re.split(r"[.?!,—-]", (text or "").strip(), maxsplit=1)[0]
    return " ".join(first.split()[:3])
