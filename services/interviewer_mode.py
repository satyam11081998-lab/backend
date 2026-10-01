"""
V11 Modality Router.
Replaces the hardcoded phrase instructions with the dynamic Interviewer Control Packet architecture.
"""
from __future__ import annotations
import json
from typing import Any, Dict, Tuple

ALLOW_QUESTIONS: Dict[str, bool] = {
    "OPEN": True,
    "CLOSE": False,
    "NO_OUTPUT": False,
    "SHORT_ACK": False,
    "HAND_BACK": False,
    "ACKNOWLEDGE_AND_CONTINUE": False,
    "ACKNOWLEDGE_AND_ORIENT": False,
    "REFLECT_PROGRESS": False,
    "DATA_REVEAL": False,
    "TARGETED_QUESTION": True,
    "NEW_DIRECTION": True,
    "SANITY_CHECK": True,
    "VALIDATE_AND_HAND_BACK": False,
    "MICRO_HINT": False,
    "CORRECT_AND_CONTINUE": False,
    "REPAIR_AND_RESET": False,
    "ANSWER_DIRECT": False,
    "DELIVER_SOLUTION": False,
    "TRANSITION": False,
    "DEFLECT_META": False,
    "NOISE": True,
    
    # Legacy fallbacks
    "ACK_ADVANCE": False,
    "CHALLENGE_CLAIM": True,
    "PROBE": True,
    "RELEASE": False,
}

# Retained for legacy external callers
Q_BUDGET: Dict[str, int] = {k: (1 if v else 0) for k, v in ALLOW_QUESTIONS.items()}


def build_mode_block(response_function: str, control_packet: Dict[str, Any]) -> str:
    """
    Transforms the JSON Control Packet into a strict behavioral brief for the Contextual Generator (LLM).
    """
    decision = control_packet.get("interviewer_control", {}).get("interviewer_decision", {})
    q_allowed = decision.get("question_allowed", False)
    q_str = "YES (max 1)" if q_allowed else "NO (do not ask follow-ups. Ensure your response ends with a period, not a question mark.)"

    prompt = f"""You are the Contextual Response Generation Layer for an expert consulting case interviewer.
Read the following INTERVIEWER CONTROL PACKET (JSON) to understand the conversational objective for this turn.
Do NOT output JSON. Output ONLY the natural human speech for the interviewer.

INTERVIEWER CONTROL PACKET:
{json.dumps(control_packet, indent=2)}

BEHAVIORAL RULES:
1. You must fulfill the "response_function" specified in the packet.
2. If the function is REFLECT_PROGRESS or ACKNOWLEDGE_AND_CONTINUE, briefly acknowledge what the candidate actually did using natural context from their message, then return the floor. Do NOT use robotic generic phrases like "Got it" or "Right" for substantive work. Provide context-aware validation.
3. Do NOT ask a question unless QUESTIONS ALLOWED is YES.
4. Do NOT over-praise ("Great job", "Excellent"). Be calm, professional, and economical with words.
5. Do NOT claim reasoning is correct unless the packet explicitly flags the quality as positive.

QUESTIONS ALLOWED: {q_str}
"""
    return prompt


# -----------------------------------------------------------------------------
# LEGACY COMPATIBILITY
# -----------------------------------------------------------------------------
def select_mode(sig: Dict[str, Any], policy: str = "coached",
                new_message: str = "") -> Tuple[str, str, int]:
    from services.interviewer_decision import determine_response_function, build_interviewer_control_packet
    response_func = determine_response_function(sig)
    packet = build_interviewer_control_packet(sig, response_func)
    instruction = build_mode_block(response_func, packet)
    budget = Q_BUDGET.get(response_func, 1)
    return response_func, instruction, budget
