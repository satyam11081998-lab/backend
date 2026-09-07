"""MECE Prep Copilot — the per-user agentic-AI layer.

This is the USER-FACING counterpart to services/agentic (which is platform/admin
facing). Where the admin orchestrator reasons about the whole cohort, this one
reasons about ONE Pro candidate: it reads their graded history, understands where
they want to go, and plans a personalised prep path by deploying a team of
domain specialists.

Vocabulary (the distinction that matters for the demo):
  * an "AI agent" = ONE autonomous, tool-using entity.
  * "agentic AI"  = a SYSTEM that shows agency by PLANNING and ORCHESTRATING
                    several specialists across DIFFERENT DOMAINS, deciding at
                    runtime which to deploy, reading their results, looping, and
                    terminating on its own judgement.

Only light, dependency-free pieces are imported here so the package is always
importable. The live (OpenAI/Supabase-backed) pieces live in `.live` and are
imported lazily by the route — a missing key or SDK can never break import.

Public (light) API:
    orchestrate(mission, planner, data, mode=...) -> CoachResult
    make_sim_planner(), SampleCandidate, SAMPLE_CANDIDATES   # always-works demo
    catalog()                                                # specialist roster

Live API (import from services.coach.live):
    make_live_planner(user_id, data, context), SupabaseUserData
"""

from .orchestrator import orchestrate, catalog
from .simulation import make_sim_planner, SampleCandidate, SAMPLE_CANDIDATES
from .schemas import CoachResult, CoachStep, SpecialistResult
from .specialists import SPECIALISTS, DIMENSIONS

__all__ = [
    "orchestrate", "catalog",
    "make_sim_planner", "SampleCandidate", "SAMPLE_CANDIDATES",
    "CoachResult", "CoachStep", "SpecialistResult",
    "SPECIALISTS", "DIMENSIONS",
]
