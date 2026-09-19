"""
Modality Router and Instruction generation.
Includes the LISTENING_BEAT modality and explicit anti-repetition guardrails.
Maintains backward compatibility for legacy tests via Q_BUDGET and select_mode proxies.
"""
from __future__ import annotations
from typing import Any, Dict, Tuple

ALLOW_QUESTIONS: Dict[str, bool] = {
    "OPEN": True,
    "CLOSE": False,
    "NO_INTERVENTION": False,
    "LISTENING_BEAT": False,
    "HAND_BACK": False,
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
}

Q_BUDGET: Dict[str, int] = {k: (1 if v else 0) for k, v in ALLOW_QUESTIONS.items()}


def get_modality_instruction(mode: str, policy: str, new_message: str, signals: Dict[str, Any] = None) -> str:
    """Returns the explicit generation boundary for the given modality."""
    if signals is None:
        signals = {}
        
    recent_phrases = signals.get("recent_assistant_turns", [])
    recent_str = " | ".join(recent_phrases) if recent_phrases else "none"

    if mode == "NO_INTERVENTION":
        return "Generate nothing. Stay silent."
        
    if mode == "LISTENING_BEAT":
        return (f"You are actively listening to the candidate. Output a 1-3 word conversational acknowledgement "
                f"(e.g., 'Got it', 'Alright', 'Okay', 'Makes sense'). "
                f"DO NOT ask a question. DO NOT summarize. "
                f"CRITICAL: Do NOT use these exact phrases you recently used: [{recent_str}]. Stop.")
        
    if mode == "HAND_BACK":
        return (f"Acknowledge concisely and return cognitive ownership. "
                f"Say something like 'Go ahead', 'Proceed', or 'Take it from there'. "
                f"Do not summarize. Do not ask what they will do next. "
                f"CRITICAL: Do NOT use these exact phrases you recently used: [{recent_str}]. Stop.")
                
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
        return ("Answer the clarification or logistics question directly. "
                "If it's a scope question, give a defensible answer. Stop.")
                
    if mode == "DELIVER_SOLUTION":
        return ("Provide the necessary solution, recommendation, or structural spine. "
                "Be concise. Do not deflect with another question. Stop.")
                
    if mode == "OPEN":
        return "Kick the case off. Set the prompt in <=2 short lines, then ask ONE clean opening question."
        
    if mode == "CLOSE":
        return "Acknowledge their recommendation briefly. End the case. Do not reopen any threads. Stop."
        
    if mode == "DEFLECT_META":
        return "Deflect the identity/meta probe in role. Redirect to the case. Stop."
        
    if mode == "NOISE":
        return "The message is garbled. Ask them to restate it briefly."
        
    return "Ask ONE advancing question that deepens the case. Push for structure or specifics."


def build_mode_block(mode: str, instruction: str, allow_questions: bool) -> str:
    q_str = "YES (max 1)" if allow_questions else "NO (do not ask follow-ups)"
    return (f"YOUR MOVE THIS TURN: {mode}\n"
            f"QUESTIONS ALLOWED: {q_str}\n"
            f"{instruction}")


# -----------------------------------------------------------------------------
# LEGACY COMPATIBILITY
# -----------------------------------------------------------------------------
def select_mode(sig: Dict[str, Any], policy: str = "coached",
                new_message: str = "") -> Tuple[str, str, int]:
    from services.interviewer_decision import evaluate_intervention_gate
    _intervene, mode, _reason = evaluate_intervention_gate(sig)
    instruction = get_modality_instruction(mode, policy, new_message, sig)
    budget = Q_BUDGET.get(mode, 1)
    return mode, instruction, budget
