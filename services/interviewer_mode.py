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
    "CORRECT_MATERIAL": False,
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
    "CORRECT_AND_CONTINUE": False,
    "REPAIR_AND_RESET": False,
}

Q_BUDGET: Dict[str, int] = {k: (1 if v else 0) for k, v in ALLOW_QUESTIONS.items()}


def get_modality_instruction(mode: str, policy: str, new_message: str, signals: Dict[str, Any] = None) -> str:
    """Returns the explicit generation boundary for Deep Lane LLM modalities."""
    if signals is None:
        signals = {}

    if mode in ["NO_OUTPUT", "PRESENCE", "HAND_BACK", "ACKNOWLEDGE", "VALIDATE"]:
        return "Generate nothing. Event handled by Fast Lane."
                
    if mode == "DATA_REVEAL":
        return ("Provide ONLY the specific case data required to test their hypothesis or answer their fact request. "
                "Do not explain the implication of the data. Do not add a framework. Stop.")
                
    if mode == "TARGETED_PROBE":
        return ("Drill into a specific missing branch or driver in their structure. "
                "Ask ONE narrowly targeted question (e.g., 'What about the supply side?'). "
                "Do not ask generic questions like 'What else?' or 'Why?'.")
                
    if mode == "RETHINK_CUE":
        return ("Trigger self-correction on a suspicious estimate. "
                "Ask a lightweight challenge (e.g., 'Does that number look right to you?' or 'Walk me through that assumption'). "
                "Do not give the correct number.")
                
    if mode == "CORRECT_MATERIAL":
        return ("Identify the material issue (e.g., 'Check your units' or 'Check the denominator'). "
                "Do not automatically solve the full calculation for them. Stop.")
                
    if mode == "HINT" or mode == "STRUCTURAL_HINT":
        return ("Provide ONE concise directional hint tied to their current work. Stop. "
                "Do not append a probing question. Do not ask what they want to do next.")
                
    if mode == "TRANSITION":
        return ("Move the candidate cleanly to the next analytical stage. "
                "State the transition and hand back. Stop.")
                
    if mode == "REPAIR":
        return ("The candidate is frustrated or stuck. Change strategy. "
                "Restate the objective simply or clarify the task. "
                "Do not interrogate them further. Stop.")
                
    if mode == "ANSWER_DIRECT":
        return ("Answer the clarification or logistics question directly and concisely. "
                "Do not solve the next analytical step for them. Stop.")
                
    if mode == "DELIVER_SOLUTION":
        return ("The candidate asked for the solution/approach. Provide ONLY the immediate next step, hint, or structural spine. "
                "DO NOT break down the entire step-by-step final solution. Give them a direction to work with and stop. Be extremely concise.")
                
    if mode == "OPEN":
        return "Kick the case off. Set the prompt in <=2 short lines, then ask ONE clean opening question."
        
    if mode == "CLOSE":
        return "Acknowledge their recommendation briefly. End the case. Do not reopen any threads. Stop."
        
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
        "The candidate just made a distinction or laid out a structure. Reflect back, in your own "
        "words, what it separates or what angle it adds, then let them carry it forward. Do not add "
        "branches they did not mention and do not grade it. No question."),
    "ACKNOWLEDGE_AND_ORIENT": (
        "The candidate just completed a step. Name the step they completed (not their arithmetic), "
        "then point them to the next part of THEIR OWN plan (conversation_memory.candidate_plan, if "
        "present). If they have not named a next part, just tell them to take it into the next step. "
        "Never introduce a new driver, number or method. No question."),
    "VALIDATE_AND_HAND_BACK": (
        "The candidate just adjusted their reasoning. Acknowledge the specific adjustment in a few "
        "words and hand the floor back. Do not certify the result as correct. No question."),
}

# V12 function -> the V10.2 instruction it inherits (unchanged behaviour).
_FUNCTION_TO_MODE = {
    "MICRO_HINT": "HINT",
    "CORRECT_AND_CONTINUE": "CORRECT_MATERIAL",
    "SANITY_CHECK": "RETHINK_CUE",
    "REPAIR_AND_RESET": "REPAIR",
    "TARGETED_QUESTION": "PROBE",
}


def get_function_instruction(function: str, mode: str, policy: str, new_message: str,
                             signals: Optional[Dict[str, Any]] = None) -> str:
    if function in FUNCTION_INSTRUCTIONS:
        return FUNCTION_INSTRUCTIONS[function]
    return get_modality_instruction(_FUNCTION_TO_MODE.get(function, mode), policy, new_message, signals)


_PACKET_RULES = (
    "HOW TO USE THE PACKET: the policy layer already chose the response_function; fulfil it in "
    "your own natural words, grounded in what the candidate actually said. Never quote the packet, "
    "never output JSON or field names. Do not reuse a line from "
    "conversation_memory.recent_interviewer_lines. Ask a question only if question_allowed is true. "
    "Never say the candidate's work is correct, right or accurate unless may_confirm_correctness is "
    "true. Stay within target_sentence_count sentences. If must_reference_candidate_content is true, "
    "the line must pick up something specific from the candidate's message - a stock "
    "acknowledgement on its own (\"Got it.\", \"Right.\", \"Okay.\") is not acceptable."
)


def build_mode_block(mode: str, instruction: str, allow_questions: bool,
                     control_packet: Optional[Dict[str, Any]] = None,
                     function: Optional[str] = None) -> str:
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
                  + "\n" + _PACKET_RULES)
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
