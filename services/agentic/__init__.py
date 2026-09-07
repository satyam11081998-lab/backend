"""MECE agentic-AI layer: a planner that orchestrates domain specialists.

Only light, dependency-free pieces are imported here so the package is always
importable. The live (OpenAI/Supabase-backed) pieces live in `.live` and are
imported lazily by the route — that keeps a missing key or SDK from ever breaking
import, and keeps this unit-testable without any external dependency.

Public API (light):
    orchestrate(mission, planner, data, mode=...) -> OrchestrationResult
    make_sim_planner(), SampleData          # always-works demo
    catalog()                                # specialist roster

Live API (import from services.agentic.live):
    make_live_planner(user_id, data), SupabaseData
"""

from .orchestrator import orchestrate, catalog
from .simulation import make_sim_planner, SampleData
from .schemas import OrchestrationResult, OrchestrationStep, SpecialistResult
from .specialists import SPECIALISTS, DIMENSIONS

__all__ = [
    "orchestrate", "catalog",
    "make_sim_planner", "SampleData",
    "OrchestrationResult", "OrchestrationStep", "SpecialistResult",
    "SPECIALISTS", "DIMENSIONS",
]
