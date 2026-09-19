"""
Modality Router and Instruction generation.
Implements specific boolean ALLOW_QUESTIONS to remove the Q_BUDGET interrogation incentive.
"""
from __future__ import annotations
from typing import Any, Dict

ALLOW_QUESTIONS: Dict[str, bool] = {
    "OPEN": True,
    "CLOSE": False,
    "NO_INTERVENTION": False,
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
}


def get_modality_instruction(mode: str, policy: str, new_message: str) -> str:
    """Returns the explicit generation boundary for the given modality."""
    if mode == "NO_INTERVENTION":
        return "Generate nothing. Stay silent."
        
    if mode == "HAND_BACK":
        return ("Acknowledge concisely. Return cognitive ownership. "
                "Say 'Go ahead', 'Proceed', or 'Makes sense'. "
                "Do not summarize. Do not ask what they will do next. Stop.")
                
    if mode == "DATA_REVEAL":
        return ("Provide ONLY the specific case data required to test their hypothesis or answer their fact request. "
                "Do not explain the implication of the data. Do not add a framework. Stop.")
                
    if mode == "TARGETED_PROBE":
        return ("Drill into a specific missing branch or driver in their structure. "
                "Ask ONE narrowly targeted question (e.g., 'What about the supply side?'). "
                "Do not ask generic questions like 'What else?'.")
                
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
        
    # Default fallback
    return "Ask ONE advancing question that deepens the case. Push for structure or specifics."


def build_mode_block(mode: str, instruction: str, allow_questions: bool) -> str:
    q_str = "YES (max 1)" if allow_questions else "NO (do not ask follow-ups)"
    return (f"YOUR MOVE THIS TURN: {mode}\n"
            f"QUESTIONS ALLOWED: {q_str}\n"
            f"{instruction}")
