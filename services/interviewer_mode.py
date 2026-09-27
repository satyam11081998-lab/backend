"""
Instruction generation for the adaptive MECE interviewer.

V11 keeps questions as an explicit capability of the selected substantive mode.
Fast-lane presence events never invoke the LLM.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple


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
    "ACK_ADVANCE": False,
    "CHALLENGE_CLAIM": True,
    "SANITY_CHECK": True,
    "RELEASE": False,
}

Q_BUDGET: Dict[str, int] = {k: (1 if v else 0) for k, v in ALLOW_QUESTIONS.items()}


def get_modality_instruction(
    mode: str,
    policy: str,
    new_message: str,
    signals: Dict[str, Any] = None,
) -> str:
    """Return the generation boundary for one substantive interviewer move."""
    if signals is None:
        signals = {}

    if mode in {"NO_OUTPUT", "PRESENCE", "HAND_BACK", "ACKNOWLEDGE", "VALIDATE"}:
        return "Generate nothing. Event handled by the Fast Lane."

    if mode == "DATA_REVEAL":
        return (
            "Provide ONLY the specific case data required to test their hypothesis or answer their fact request. "
            "Do not explain the implication of the data. Do not add a framework. Stop."
        )

    if mode == "TARGETED_PROBE":
        return (
            "Drill into one specific missing branch or driver. Ask ONE narrowly targeted question. "
            "Do not ask generic questions like 'What else?' or 'Why?'."
        )

    if mode == "RETHINK_CUE":
        return (
            "Trigger self-correction on a suspicious estimate. Ask one lightweight challenge. "
            "Do not give the correct number unless the candidate explicitly asks for the solution."
        )

    if mode == "CORRECT_MATERIAL":
        return (
            "Identify the material issue directly. Use the smallest correction that restores valid reasoning. "
            "Do not automatically solve the full calculation. Stop."
        )

    if mode in {"HINT", "STRUCTURAL_HINT"}:
        return (
            "Provide ONE concise directional hint tied to their current work. Stop. "
            "Do not append a probing question or ask what they want to do next."
        )

    if mode == "TRANSITION":
        return (
            "Move the candidate cleanly to the next analytical stage. State the transition and hand back. Stop."
        )

    if mode == "REPAIR":
        return (
            "The candidate is frustrated or stuck. Change strategy. Restate the objective simply or clarify the task. "
            "Do not interrogate them further. Stop."
        )

    if mode == "ANSWER_DIRECT":
        return (
            "Answer the clarification or logistics question directly. Give only the information needed to continue. Stop."
        )

    if mode == "DELIVER_SOLUTION":
        return (
            "Provide the necessary solution, recommendation, or structural spine. Be concise. "
            "Do not deflect with another question. Stop."
        )

    if mode == "OPEN":
        return "Kick the case off. Set the prompt in <=2 short lines, then ask ONE clean opening question."

    if mode == "CLOSE":
        return "Acknowledge their recommendation briefly. End the case. Do not reopen any threads. Stop."

    if mode == "DEFLECT_META":
        return "Deflect the identity/meta probe in role. Redirect to the case. Stop."

    if mode == "NOISE":
        return "The message is garbled. Ask them to restate it briefly."

    if mode == "PROBE":
        return ("Ask ONE advancing question tied to the most important unresolved branch. "
                "Do not ask a generic breadth question.")

    return "Ask ONE advancing question tied to the most important unresolved branch."


def build_mode_block(mode: str, instruction: str, allow_questions: bool) -> str:
    q_str = "YES (max 1)" if allow_questions else "NO (do not ask follow-ups)"
    return (
        f"YOUR MOVE THIS TURN: {mode}\n"
        f"QUESTIONS ALLOWED: {q_str}\n"
        f"{instruction}"
    )


# Legacy compatibility

def select_mode(
    sig: Dict[str, Any],
    policy: str = "coached",
    new_message: str = "",
) -> Tuple[str, str, int]:
    from services.interviewer_decision import evaluate_intervention_gate

    _lane, mode, _reason = evaluate_intervention_gate(sig)
    instruction = get_modality_instruction(mode, policy, new_message, sig)
    budget = Q_BUDGET.get(mode, 1)
    return mode, instruction, budget
