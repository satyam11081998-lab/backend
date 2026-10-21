"""
Modality Router and Instruction generation.
V10.2: Tightened DELIVER_SOLUTION to prevent massive info dumps when requested.
V12: instructions per RESPONSE FUNCTION, and build_mode_block carries the JSON
INTERVIEWER CONTROL PACKET the model words the turn from. The instructions say
what the turn must ACCOMPLISH; none of them is a line to copy.
"""
from __future__ import annotations
import json
from typing import Any, Dict, Optional, Tuple

ALLOW_QUESTIONS: Dict[str, bool] = {
    "OPEN": True,
    "CLOSE": False,
    "DATA_REVEAL": False,
    "TARGETED_PROBE": True,
    "RETHINK_CUE": True,
    # V12.1 (arena adversarial run): a correction points at the step and asks ONE
    # targeted question so the candidate fixes it ("How many ml make a litre?").
    "CORRECT_MATERIAL": True,
    "HINT": False,
    "STRUCTURAL_HINT": False,
    "REPAIR": False,
    "ANSWER_DIRECT": False,
    "DELIVER_SOLUTION": False,
    "TRANSITION": False,
    "DEFLECT_META": False,
    "NOISE": True,
    "PROBE": True,
    # Legacy fallbacks
    "ACK_ADVANCE": False,
    "CHALLENGE_CLAIM": True,
    "SANITY_CHECK": True,
    "RELEASE": False,
    # V12 response functions
    "NO_OUTPUT": False,
    "SHORT_ACK": False,
    "HAND_BACK": False,
    "ACKNOWLEDGE_AND_CONTINUE": False,
    "ACKNOWLEDGE_AND_ORIENT": False,
    "REFLECT_PROGRESS": False,
    "VALIDATE_AND_HAND_BACK": False,
    "TARGETED_QUESTION": True,
    "MICRO_HINT": False,
    "CORRECT_AND_CONTINUE": True,
    "REPAIR_AND_RESET": False,
    "CONTINUE_AS_AGREED": True,
    # V12.1
    "RESTATE_CASE": False,
    "EXPLAIN_FORMAT": False,
    "EXPLAIN_CONCEPT": False,
    "TEACH_STEP": True,
    "TEACH": True,
    "WRAP_UP_EARLY": False,
    "FINAL_REVIEW": False,
    "SESSION_STATUS": False,
    "FOLLOW_UP_ANSWER": True,
    "FOLLOW_UP": True,
    "PRODUCT_HELP": False,
    "EXPLAIN_WHY": False,
    "HOLD_SPACE": False,
}

Q_BUDGET: Dict[str, int] = {k: (1 if v else 0) for k, v in ALLOW_QUESTIONS.items()}


def get_modality_instruction(mode: str, policy: str, new_message: str, signals: Dict[str, Any] = None) -> str:
    """Returns the explicit generation boundary for Deep Lane LLM modalities."""
    if signals is None:
        signals = {}

    if mode in ["NO_OUTPUT", "PRESENCE", "HAND_BACK", "ACKNOWLEDGE", "VALIDATE"]:
        return "Generate nothing. Event handled by Fast Lane."
                
    if mode == "DATA_REVEAL":
        return ("Provide ONLY the specific case data they asked for, as a specific realistic number stated as "
                "fact (use case_context.reference_facts for populations). Never say the data isn't available, "
                "given or in the case. Do not explain the implication of the data. Do not add a framework. Stop.")
                
    if mode == "TARGETED_PROBE":
        return ("Drill into a specific missing branch or driver in their structure. "
                "Ask ONE narrowly targeted question (e.g., 'What about the supply side?'). "
                "Do not ask generic questions like 'What else?' or 'Why?'.")
                
    if mode == "RETHINK_CUE":
        return ("Make the candidate sanity-check this number or claim themselves; do not fix it for them. Point "
                "at the concrete thing: candidate_state.detected_issue, if present, names the assumption (for "
                "example everyone buying it every day); case_context.implied_check, if present, is the exact "
                "implication (and implied_verdict, if present, says it looks high). Then ask ONE short, specific "
                "question that makes them test it against something real: a group who would not do it (infants, "
                "people with no shop nearby), or the size of the place (shape: 'A small town has about 50,000 to "
                "2 lakh people - does 90,000 orders a day fit that?'). If no_base_for_check is true, ask how many "
                "people or households that number needs, given the place. Never a bare 'Does that number look "
                "right to you?', never give the correct number, never call it plausible. Two sentences at most.")
                
    if mode == "CORRECT_MATERIAL":
        return ("There is a real slip in the candidate's work. candidate_state.detected_issue, when present, was "
                "computed in code and is exact. Point at that exact step in their own words and name the kind of "
                "slip (a unit, a double count, a figure that contradicts something already established, an "
                "arithmetic step). Then ask ONE short, targeted question that lets them fix it themselves - for "
                "example 'How many ml make a litre?', 'Aren't the families and singles already inside the city "
                "population?', 'If volumes are flat, can fewer units explain the drop?'. If your recent lines "
                "already pointed at this same step, give the corrected figure from detected_issue instead and "
                "let them carry on. Never compute a corrected figure yourself: use only figures in detected_issue. "
                "With no detected_issue, name the one material issue precisely (the unit, the denominator, the "
                "overlap). No praise first, no lecture, do not redo the rest of their calculation. Two sentences.")
                
    if mode == "HINT" or mode == "STRUCTURAL_HINT":
        return ("Provide ONE concise directional hint tied to their own work. If candidate_state.latest_work is "
                "present, the hint is ON that working: first say in a few words whether it holds "
                "(latest_work_check 'verified' -> the maths holds, say so plainly; 'issue' -> name the step that "
                "is off), then point to the very next move with it (what to do with the result, or which piece "
                "to check). If they complained that you are not telling them what is wrong, answer that first. "
                "One or two sentences. Do not restart the case or lay out a generic framework. Do not append a "
                "probing question. Do not ask what they want to do next.")
                
    if mode == "TRANSITION":
        return ("Move the candidate cleanly to the next analytical stage. "
                "State the transition and hand back. Stop.")
                
    if mode == "REPAIR":
        return ("The candidate is frustrated or stuck. Change strategy. "
                "Restate the objective simply or clarify the task. "
                "Do not interrogate them further. Stop.")
                
    if mode == "ANSWER_DIRECT":
        return ("Answer the candidate's question FIRST, directly and concretely. If it asks for a fact or a "
                "number, state a specific, realistic figure as fact, consistent with the case and anything said "
                "so far - you own the facts; never say it isn't given, stated, specified, detailed or available, "
                "and never tell them to 'make reasonable assumptions' instead. If case_context.unanswered_questions "
                "is present, they are telling you that you skipped those: answer each one now, one short line each. "
                "If they ask whether their maths is right and arithmetic_check is 'verified', say yes, the maths "
                "checks out; if it is 'issue', say which step is off. Give a figure in a shape they can use: for "
                "competitors, how many main players and roughly what share they hold (e.g. 'three big players "
                "with about 70% between them, the rest fragmented'); for a market, its size and growth; for a "
                "population, the number and what it covers. Then hand back in a few words. Do not solve "
                "the next analytical step for them, and never deflect with 'let's focus on structuring first'. Stop.")
                
    if mode == "DELIVER_SOLUTION":
        return ("The candidate asked for the solution or the approach. Give the approach for THIS case: name it "
                "in plain words (e.g. 'top-down from population', 'profit = revenue - cost, then drill into the "
                "side that moved'), and if visual_structure_allowed is true lay the skeleton out on 3-5 short "
                "lines joined by arrows (→) using this case's terms. Then do the first step with them. Do NOT "
                "work through to the final answer - mention that the full worked answer is on their results "
                "page after they submit. At most three short sentences besides the arrow lines; no numbered "
                "lists.")
                
    if mode == "OPEN":
        return ("Kick the case off. If their first message asks you something, answer it FIRST, directly, as a "
                "specific fact (you own the facts). Otherwise set the prompt in <=2 short lines (if they asked for "
                "the case or the problem, give it to them properly: the client, the situation and exactly what to "
                "answer). In chat, add a short note that on a phone the full case is under the menu at the top "
                "left. Then ask ONE clean opening question.")
        
    if mode == "CLOSE":
        return ("Acknowledge their recommendation or thanks briefly and end the case: tell them to tap Submit "
                "(top right) to see their score, feedback and the full worked answer. Do not reopen any threads. "
                "If you already said this, say it again in different, shorter words. Stop.")
        
    if mode == "DEFLECT_META":
        return "Deflect the identity/meta probe in role. Redirect to the case. Stop."
        
    if mode == "NOISE":
        return "The message is garbled. Ask them to restate it briefly."
        
    return "Ask ONE advancing question that deepens the case. Push for structure or specifics."


# V12: the four model-worded presence functions. Each says what the line must DO
# with the candidate's own content; the model chooses the words.
FUNCTION_INSTRUCTIONS: Dict[str, str] = {
    "ACKNOWLEDGE_AND_CONTINUE": (
        "The candidate just did real work. In a few words, recognise the SPECIFIC thing they did "
        "(pick it out of their message: the step, the split, the number they reached) and hand the "
        "floor back so they continue. Do not grade it, do not repeat their arithmetic, do not add "
        "analysis of your own. No question."),
    "REFLECT_PROGRESS": (
        "The candidate just made a distinction or laid out a plan. In one short, neutral sentence say what "
        "it separates or what they are about to check - no evaluative words (smart, good, insightful, "
        "effective, 'zeroing in'). Then name the concrete next step INSIDE their approach as an instruction: "
        "what to compute, compare or check next (e.g. 'Compare raw material cost per unit now against last "
        "year.'). Do not add branches they did not mention and do not grade it. No question, and never end "
        "on a vague 'take it forward from here'."),
    "ACKNOWLEDGE_AND_ORIENT": (
        "The candidate just completed a step. Name the step they completed (not their arithmetic), "
        "then point them to the next part of THEIR OWN plan (conversation_memory.candidate_plan, if "
        "present). If they have not named a next part, just tell them to take it into the next step. "
        "Never introduce a new driver, number or method. No question."),
    "VALIDATE_AND_HAND_BACK": (
        "The candidate just adjusted their reasoning. Acknowledge the specific adjustment in a few "
        "words and hand the floor back. Do not certify the result as correct. No question."),
    "CONTINUE_AS_AGREED": (
        "Your last line asked or proposed something and the candidate just said yes / go ahead. Carry "
        "on with exactly what you proposed (set out the context, give the data you offered, or move to "
        "the step you named) in one to three short sentences. Do not ask them to confirm again. At most "
        "one question, and only if it hands the next move to them."),
    # ---- V12.1 (15-day review) ------------------------------------------------
    "RESTATE_CASE": (
        "The candidate cannot see the case or wants it again (on a phone the brief sits behind the menu at "
        "the top left). Give it back in 2-3 short lines using ONLY facts from the case prompt: who the "
        "client is, what is happening (with the key numbers), and exactly what they must answer. In chat, "
        "add one short line that on a phone they can open the full brief from the menu at the top left. "
        "Then one line on where to begin (a clarifying question or two, then their structure). No question."),
    "EXPLAIN_FORMAT": (
        "The candidate is asking how this works or what to ask. In 3-5 short lines: you are the interviewer "
        "and they lead; the flow for THIS case type (case: clarify -> structure -> analyse -> recommend; "
        "guesstimate: clarify scope -> pick an approach, e.g. top-down from population or bottom-up from "
        "supply -> calculate step by step -> sanity-check -> final number). If they asked what clarifying "
        "questions to ask, give 2-3 example questions specific to THIS case. Say that at the end they tap "
        "Submit for a score and the full worked answer. Finish by inviting their first move, without a "
        "question mark."),
    "EXPLAIN_CONCEPT": (
        "The candidate asked what a concept means or how to do it (case_context.concept_asked). Explain it in "
        "one plain sentence, then show it on THEIR case in one or two lines using their own numbers - for a "
        "sanity check, use case_context.implied_check when it is there (never invent a figure they did not "
        "produce) and say whether it looks plausible. Then hand back in a few words. No question."),
    "TEACH_STEP": (
        "The candidate is stuck, lost or asked to be taught. TEACH the step; do not quiz them. "
        "1) Name the approach for THIS case in plain words ('Let's do it top-down: population, then the "
        "share who buy, then how often'; 'Profit is revenue minus cost - let's find which side moved'). "
        "If case_context.teach_focus is 'structure', group THE CANDIDATE'S OWN ideas "
        "(case_context.candidate_ideas) into 2-4 clear buckets and name the gap. "
        "If case_context.already_tried is present, they already got that pointer and are still stuck: do "
        "NOT repeat it or start from the same anchor. Switch method: build it bottom-up from ONE thing they "
        "can picture (one household, one shop, one day), put a number on it together, then show how it "
        "scales up to the whole market. "
        "If candidate_state.latest_work is present, build on THEIR working instead of starting over. "
        "2) If visual_structure_allowed is true, lay the skeleton out on 3-4 short lines joined by arrows "
        "(→), filled with this case's terms and any numbers already on the table (use "
        "case_context.reference_facts for a starting figure); no numbered lists, no bullets, no markdown. "
        "3) Do the first step WITH them in one line, then hand them the very next step; you may end with one "
        "short question about that step. At most three short sentences besides the arrow lines. Warm and "
        "brief: no 'great question', no lecture, no apology."),
    "WRAP_UP_EARLY": (
        "The candidate wants to stop, maybe because they feel unprepared. No guilt, no pressure, no questions. "
        "In 2-3 short sentences: it is completely fine to stop here; they can tap Submit (top right) to get "
        "feedback on what they have done so far, or simply come back to this case later; and if they would "
        "rather learn than leave, they can say 'teach me' and you will walk them through this case step by "
        "step. Never mention a low score."),
    "FINAL_REVIEW": (
        "The candidate has given their final answer (case_context.final_answer). In 2-4 short sentences: "
        "say their final number back in their words; give ONE concrete sanity check - use "
        "case_context.implied_check if present (e.g. 'that's about 0.4 per person a year') and say plainly "
        "whether it looks plausible against that check (plausible / looks high / looks low - never 'correct'); "
        "if case_context.implied_verdict is present the number does not survive that check: say so plainly "
        "in those plain numbers and name what to recheck - never call it plausible; "
        "if case_context.earlier_results show the final does not match their own working, say that instead; "
        "name one thing they did well and one to sharpen, both specific; then tell them to tap Submit (top "
        "right) to see their score and the full worked answer. If case_context.reference_answer is present "
        "you may say in one line roughly how their number compares with it (in range, or about how many times "
        "higher/lower), without walking through the model answer. No question, no 'great job'."),
    "SESSION_STATUS": (
        "The candidate asks whether they are done, whether they passed, how close they are, or for feedback. "
        "Be honest and specific; there is no pass/fail in the room - the score is on the results page after "
        "they submit. If case_context.final_answer is present: yes, they can wrap up; one specific strength "
        "and one specific thing to sharpen; if case_context.reference_answer is present, say in one line how "
        "their number compares (in range, or roughly how far off); then tell them to tap Submit (top right) "
        "for the full score and worked answer. If there is no final answer yet: say in one line what is "
        "left (for example: land a final number and sanity-check it) and that they can Submit whenever they "
        "like. No question."),
    "FOLLOW_UP_ANSWER": (
        "The candidate just answered the question in your last line, briefly. React to THEIR answer in a few "
        "words, never with a bare 'Okay' or 'Got it'. If arithmetic_check is 'verified' you may say the maths "
        "checks out; if detected_issue is present, point to it. If they gave a workable figure but sound "
        "unsure (candidate_confidence 'low'), reassure them lightly that it works as a number to carry "
        "forward - do NOT ask them to justify, break down or refine it. If the answer does not fit what you "
        "asked (a percentage where you asked for a total, a method where you asked for a number), say so "
        "kindly and re-ask more simply with a concrete nudge. Then name the next step in a few words. At most "
        "one question, and only about the next step."),
    "PRODUCT_HELP": (
        "The candidate asked about the app itself (where the results are, how to submit or end). Answer it "
        "plainly in one or two short sentences using only case_context.product_facts; if those do not cover "
        "it, say the results page after Submit has everything about this attempt. Then hand back to the case "
        "in a few words. No question, no case content."),
    "EXPLAIN_WHY": (
        "The candidate asked why you asked case_context.your_question_they_ask_about. In ONE plain sentence, "
        "give the link from that topic to what this case must answer (e.g. 'The value chain shows which stage "
        "the extra cost is sitting in, and that is what explains the loss.'). Do not answer it for them and "
        "do not add a framework. Then hand it back in a few words as an instruction ('Walk me through the "
        "stages.'), with no question mark."),
}

# V12 function -> the V10.2 instruction it inherits (unchanged behaviour).
_FUNCTION_TO_MODE = {
    "MICRO_HINT": "HINT",
    "CORRECT_AND_CONTINUE": "CORRECT_MATERIAL",
    "SANITY_CHECK": "RETHINK_CUE",
    "REPAIR_AND_RESET": "REPAIR",
    "TARGETED_QUESTION": "PROBE",
}


_VOICE_NOTES = {
    "TEACH_STEP": (" SPOKEN: two or three short sentences - name the approach, say the chain of steps in "
                   "words, give them the first step. No lists, no arrows."),
    "RESTATE_CASE": " SPOKEN: two or three sentences; mention that on a phone the brief is under the menu at the top left.",
    "EXPLAIN_FORMAT": " SPOKEN: two or three sentences, no lists.",
    "FINAL_REVIEW": " SPOKEN: two short sentences - the sanity check, then that they can end and submit for the score and worked answer.",
    "SESSION_STATUS": " SPOKEN: two short sentences.",
    "WRAP_UP_EARLY": " SPOKEN: two short sentences.",
    "PRODUCT_HELP": " SPOKEN: one or two short sentences.",
    "EXPLAIN_WHY": " SPOKEN: one sentence of why, then hand it back.",
}
_EXAM_NOTES = {
    "TEACH_STEP": (" TEACHING POLICY exam: name the approach and the skeleton, but do not put numbers in it "
                   "or do the first step for them."),
    "CORRECT_AND_CONTINUE": (" TEACHING POLICY exam: point at the exact step (what they wrote) and ask them to "
                             "recheck it; do NOT give the corrected figure, even on a second pointer."),
    "FINAL_REVIEW": " TEACHING POLICY exam: no reference answer; the sanity check and the Submit pointer only.",
    "MICRO_HINT": " TEACHING POLICY exam: say which step to recheck; never give the corrected figure.",
}


def get_function_instruction(function: str, mode: str, policy: str, new_message: str,
                             signals: Optional[Dict[str, Any]] = None) -> str:
    if function in FUNCTION_INSTRUCTIONS:
        text = FUNCTION_INSTRUCTIONS[function]
    else:
        text = get_modality_instruction(_FUNCTION_TO_MODE.get(function, mode), policy, new_message, signals)
    if (signals or {}).get("channel") == "voice":
        text += _VOICE_NOTES.get(function, "")
    if policy == "exam":
        text += _EXAM_NOTES.get(function, "")
    return text


_PACKET_RULES = (
    "HOW TO USE THE PACKET: the policy layer already chose the response_function; fulfil it in "
    "your own natural words, grounded in what the candidate actually said. Never quote the packet, "
    "never output JSON or field names. Do not reuse a line from "
    "conversation_memory.recent_interviewer_lines. Ask a question only if question_allowed is true. "
    "Never say the candidate's work is correct, right or accurate unless may_confirm_correctness is "
    "true. Stay within target_sentence_count sentences. If must_reference_candidate_content is true, "
    "the line must pick up something specific from the candidate's message - a stock "
    "acknowledgement on its own (\"Got it.\", \"Right.\", \"Okay.\") is not acceptable. "
    # V12.1
    "Numbers in candidate_state (detected_issue, verified_steps) and case_context (implied_check, "
    "reference_facts, final_answer, earlier_results) were computed in code and are exact: use them and "
    "never contradict them; never claim their arithmetic is off when arithmetic_check is 'verified'; never "
    "invent a figure for them to sanity-check. If visual_structure_allowed is true you may lay a structure "
    "out on separate short lines joined by arrows (→); otherwise plain sentences only. Never use numbered "
    "lists ('1.', '2.'). Never tell the "
    "candidate a fact is not given or not available: you own the facts."
)


def build_mode_block(mode: str, instruction: str, allow_questions: bool,
                     control_packet: Optional[Dict[str, Any]] = None,
                     function: Optional[str] = None,
                     packet_rules: Optional[str] = None) -> str:
    # packet_rules: the admin arena passes a frozen variant's rules here; None (every
    # production caller) means this module's _PACKET_RULES, unchanged.
    q_str = "YES (max 1)" if allow_questions else "NO (do not ask follow-ups)"
    head = f"YOUR MOVE THIS TURN: {mode}"
    if function and function != mode:
        head += f" -> RESPONSE FUNCTION: {function}"
    block = (f"{head}\n"
             f"QUESTIONS ALLOWED: {q_str}\n"
             f"{instruction}")
    if control_packet:
        block += ("\n\nINTERVIEWER CONTROL PACKET (decision brief for this turn):\n"
                  + json.dumps(control_packet, ensure_ascii=False)
                  + "\n" + (_PACKET_RULES if packet_rules is None else packet_rules))
    return block

# -----------------------------------------------------------------------------
# LEGACY COMPATIBILITY
# -----------------------------------------------------------------------------
def select_mode(sig: Dict[str, Any], policy: str = "coached",
                new_message: str = "") -> Tuple[str, str, int]:
    from services.interviewer_decision import evaluate_intervention_gate
    _lane, mode, _reason = evaluate_intervention_gate(sig)
    instruction = get_modality_instruction(mode, policy, new_message, sig)
    budget = Q_BUDGET.get(mode, 1)
    return mode, instruction, budget
