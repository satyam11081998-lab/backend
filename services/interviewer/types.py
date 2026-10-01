"""
Core types for the unified interviewer brain.

Everything here is plain data: no provider, database or web framework imports,
so the whole decision path can be exercised by tests with nothing installed but
the standard library.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class Channel(str, Enum):
    TEXT = "text"      # typed (or dictated then reviewed) chat
    STT = "stt"        # talk mode: speech -> transcript -> brain -> spoken reply
    VOICE = "voice"    # full realtime voice (OpenAI Realtime / Gemini Live)

    @classmethod
    def parse(cls, value: Any) -> "Channel":
        v = str(value or "").strip().lower()
        for c in cls:
            if c.value == v:
                return c
        return cls.TEXT


class CandidateState(str, Enum):
    PROGRESSING = "PROGRESSING"
    MINOR_ERROR = "MINOR_ERROR"
    MATERIAL_ERROR = "MATERIAL_ERROR"
    UNCERTAIN = "UNCERTAIN"
    STUCK = "STUCK"
    REPEATEDLY_STUCK = "REPEATEDLY_STUCK"
    FRUSTRATED = "FRUSTRATED"
    ASKING_CLARIFICATION = "ASKING_CLARIFICATION"
    ASKING_FOR_HELP = "ASKING_FOR_HELP"
    ASKING_FOR_SOLUTION = "ASKING_FOR_SOLUTION"
    RECOVERING = "RECOVERING"
    TRANSITIONING = "TRANSITIONING"
    COMPLETING_STEP = "COMPLETING_STEP"
    FINAL_RECOMMENDATION = "FINAL_RECOMMENDATION"
    VOICE_PARTIAL = "VOICE_PARTIAL"
    # Auxiliary states the adversarial suites need (the brief lists a minimum set).
    META = "META"                      # identity / injection / rubric requests
    UNINTELLIGIBLE = "UNINTELLIGIBLE"  # garbled text or ASR noise


class Intervention(str, Enum):
    NO_OUTPUT = "NO_OUTPUT"
    ACKNOWLEDGE = "ACKNOWLEDGE"
    HAND_BACK = "HAND_BACK"
    VALIDATE = "VALIDATE"
    DATA_REVEAL = "DATA_REVEAL"
    DIRECT_CORRECTION = "DIRECT_CORRECTION"
    MICRO_HINT = "MICRO_HINT"
    TARGETED_HINT = "TARGETED_HINT"
    STRUCTURAL_HINT = "STRUCTURAL_HINT"
    DEMONSTRATION = "DEMONSTRATION"
    REPAIR = "REPAIR"
    ANSWER_DIRECT = "ANSWER_DIRECT"
    TRANSITION = "TRANSITION"
    RETHINK_CUE = "RETHINK_CUE"
    DELIVER_SOLUTION = "DELIVER_SOLUTION"
    TARGETED_PROBE = "TARGETED_PROBE"
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    DEFLECT = "DEFLECT"
    # Contextual presence: a short beat that returns the floor, worded by the model from what the
    # candidate actually said (never a stock "Right." for substantive work). Fall back to a plain
    # deterministic hand-back if the model fails - they carry no case content to fabricate.
    ACKNOWLEDGE_AND_CONTINUE = "ACKNOWLEDGE_AND_CONTINUE"   # name what they just did, hand back
    REFLECT_PROGRESS = "REFLECT_PROGRESS"                   # reflect the shape of their structure/approach
    ACKNOWLEDGE_AND_ORIENT = "ACKNOWLEDGE_AND_ORIENT"       # a stage of THEIR plan is done; name the next part of it


class Lane(str, Enum):
    NO_OUTPUT = "NO_OUTPUT"
    PRESENCE = "PRESENCE"
    SUBSTANTIVE = "SUBSTANTIVE"


CONTEXTUAL_PRESENCE = frozenset({Intervention.ACKNOWLEDGE_AND_CONTINUE, Intervention.REFLECT_PROGRESS,
                                 Intervention.ACKNOWLEDGE_AND_ORIENT})
PRESENCE_INTERVENTIONS = frozenset({Intervention.ACKNOWLEDGE, Intervention.HAND_BACK, Intervention.VALIDATE}) | CONTEXTUAL_PRESENCE

# Assistance ladder, in order. Index == hint level.
LADDER: List[Intervention] = [
    Intervention.NO_OUTPUT,          # 0 (no assistance)
    Intervention.MICRO_HINT,         # 1
    Intervention.TARGETED_HINT,      # 2
    Intervention.STRUCTURAL_HINT,    # 3
    Intervention.DEMONSTRATION,      # 4
    Intervention.DELIVER_SOLUTION,   # 5
]
MAX_HINT_LEVEL = len(LADDER) - 1

ASSISTANCE_INTERVENTIONS = frozenset(LADDER[1:]) | {Intervention.REPAIR}


def lane_of(intervention: Intervention) -> Lane:
    if intervention == Intervention.NO_OUTPUT:
        return Lane.NO_OUTPUT
    if intervention in PRESENCE_INTERVENTIONS:
        return Lane.PRESENCE
    return Lane.SUBSTANTIVE


class Phase(str, Enum):
    OPENING = "opening"
    CLARIFYING = "clarifying"
    STRUCTURING = "structuring"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    CLOSED = "closed"

    @property
    def rank(self) -> int:
        return _PHASE_ORDER.index(self)


_PHASE_ORDER = [Phase.OPENING, Phase.CLARIFYING, Phase.STRUCTURING, Phase.ANALYSIS, Phase.SYNTHESIS, Phase.CLOSED]


@dataclass
class TurnInput:
    """One candidate turn as the routes hand it to the brain."""
    text: str
    channel: Channel = Channel.TEXT
    is_partial: bool = False
    turn_id: Optional[str] = None
    # C9: set by the route when this turn asks for information and the quota is spent.
    clarifications_exhausted: bool = False


@dataclass
class CaseContext:
    case_type: str = ""
    content: str = ""                   # llm_case_content(case) - what models may see
    teaching_policy: str = "coached"    # "coached" | "exam"
    title: str = ""

    @property
    def is_guesstimate(self) -> bool:
        return (self.case_type or "").strip().lower() == "guesstimate"


@dataclass
class Decision:
    """WHAT the interviewer does this turn. Wording is decided afterwards."""
    intervention: Intervention
    state: CandidateState
    reason: str
    lane: Lane = Lane.NO_OUTPUT
    max_questions: int = 0
    # Free-form, internal-only context for the responder (never shown to the candidate).
    detail: Dict[str, Any] = field(default_factory=dict)
    needs_model: bool = False
    # Deterministic line (presence, deflection, open/close, numeric correction). None when a
    # model must word it or when the lane is NO_OUTPUT.
    fixed_text: Optional[str] = None
    hint_level: int = 0

    def __post_init__(self):
        self.lane = lane_of(self.intervention)

    def to_public(self) -> Dict[str, Any]:
        return {"lane": self.lane.value, "intervention": self.intervention.value,
                "state": self.state.value, "reason": self.reason}


@dataclass
class ActionRecord:
    i: str          # intervention value
    r: str          # reason
    t: int          # candidate turn number
    q: bool = False  # did the interviewer's line end in a question
    fy: bool = False  # was it an answer to an explicit floor-yield


@dataclass
class BrainState:
    """Persisted per attempt under attempts.session_state['brain'] (JSON)."""
    v: int = 1
    turns: int = 0
    phase: str = Phase.OPENING.value
    last_state: str = CandidateState.PROGRESSING.value
    hint_level: int = 0
    episode_open: bool = False
    episode_last_help_turn: int = -99
    help_requests: int = 0
    stuck_streak: int = 0
    frustration: int = 0
    repairs_in_row: int = 0
    progress_since_help: int = 0
    last_actions: List[Dict[str, Any]] = field(default_factory=list)
    asked_questions: List[str] = field(default_factory=list)
    recent_lines: List[str] = field(default_factory=list)
    recent_turn_ids: List[str] = field(default_factory=list)
    closed: bool = False
    opened: bool = False

    MAX_ACTIONS = 8
    MAX_QUESTIONS = 6
    MAX_LINES = 6
    MAX_TURN_IDS = 32

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "BrainState":
        s = cls()
        if not isinstance(data, dict):
            return s
        for k, default in asdict(cls()).items():
            if k in data:
                val = data[k]
                if isinstance(default, bool):
                    val = bool(val)
                elif isinstance(default, int):
                    try:
                        val = int(val)
                    except (TypeError, ValueError):
                        val = default
                elif isinstance(default, list):
                    val = list(val) if isinstance(val, list) else []
                elif isinstance(default, str):
                    val = str(val) if val is not None else default
                setattr(s, k, val)
        # Clamp anything a hand-edited or corrupted row could carry.
        s.hint_level = max(0, min(MAX_HINT_LEVEL, s.hint_level))
        s.frustration = max(0, min(2, s.frustration))
        if s.phase not in {p.value for p in Phase}:
            s.phase = Phase.OPENING.value
        if s.last_state not in {c.value for c in CandidateState}:
            s.last_state = CandidateState.PROGRESSING.value
        return s

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["last_actions"] = d["last_actions"][-self.MAX_ACTIONS:]
        d["asked_questions"] = d["asked_questions"][-self.MAX_QUESTIONS:]
        d["recent_lines"] = d["recent_lines"][-self.MAX_LINES:]
        d["recent_turn_ids"] = d["recent_turn_ids"][-self.MAX_TURN_IDS:]
        return d

    # Convenience readers ---------------------------------------------------
    @property
    def phase_enum(self) -> Phase:
        try:
            return Phase(self.phase)
        except ValueError:
            return Phase.OPENING

    def recent(self, n: int) -> List[Dict[str, Any]]:
        return self.last_actions[-n:]

    def last_action(self) -> Optional[Dict[str, Any]]:
        return self.last_actions[-1] if self.last_actions else None


class InterviewerError(Exception):
    """Base error. `error_type` feeds telemetry; never shown raw to candidates."""
    error_type = "INTERVIEWER_ERROR"


class ProviderError(InterviewerError):
    error_type = "PROVIDER_ERROR"


class ProviderTimeout(ProviderError):
    error_type = "PROVIDER_TIMEOUT"


class EmptyModelOutput(InterviewerError):
    """The model returned nothing (or nothing that survived validation).
    Distinct from NO_OUTPUT (a decision) and from provider errors."""
    error_type = "EMPTY_MODEL_OUTPUT"


class InvalidModelOutput(InterviewerError):
    error_type = "INVALID_MODEL_OUTPUT"
