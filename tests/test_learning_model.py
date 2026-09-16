"""Unit tests for the learning-intelligence engine (Phase 4). Stdlib only:
    python -m tests.test_learning_model
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.learning_model import (  # noqa: E402
    evaluate_intervention_outcome, update_learning_profile, best_modality,
    weak_skills, recommend_next_drill, build_learning_block, build_debrief,
)

_fail = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        _fail.append(name)


# ---- intervention outcome (point 8) ----
check("outcome worked (recovered)",
      evaluate_intervention_outcome({"last_intervention": "micro_hint"},
                                    {"help_requested": False, "turns_without_progress": 0,
                                     "has_work": True, "frustration": "none"}) == "worked")
check("outcome failed (still stuck)",
      evaluate_intervention_outcome({"last_intervention": "micro_hint"},
                                    {"help_requested": True, "turns_without_progress": 2,
                                     "frustration": "high"}) == "failed")
check("outcome na (not a help move)",
      evaluate_intervention_outcome({"last_intervention": "probe"}, {"help_requested": True}) == "na")

# ---- accumulation + independence/fading (point 17) ----
prof = update_learning_profile(None, {"skill": "capacity_bridge", "error": "capacity_bridge", "modality": "analogy"},
                               "worked", hint_level=2, signals={})
check("error tallied", prof["errors"].get("capacity_bridge") == 1)
check("modality success tallied", prof["modalities"]["analogy"]["worked"] == 1)
check("independence rung recorded (after_h2)", prof["independence"]["capacity_bridge"].get("after_h2") == 1)

prof = update_learning_profile(prof, {"skill": "segmentation"}, "na",
                               hint_level=0, signals={"has_work": True, "help_requested": False})
check("independent success recorded", prof["independence"]["segmentation"].get("independent") == 1)

# ---- best modality + weak skills ----
mp = {"modalities": {"analogy": {"worked": 2, "failed": 0}, "direct_hint": {"worked": 0, "failed": 2}}}
check("best modality = analogy", best_modality(mp) == "analogy")
check("weak skills ranked", weak_skills({"errors": {"unit": 3, "calculation": 1}}) == ["unit", "calculation"])

# ---- recommendation (points 14, 28) ----
rec = recommend_next_drill({"errors": {"capacity_bridge": 2}})
check("recommend focuses weakest", rec and rec["focus_error"] == "capacity_bridge")
check("recommend carries a reusable rule", rec and "bridge" in rec["remember"].lower())

# ---- learning block (points 8, 9, 16) ----
lb_fail = build_learning_block(mp, {}, "failed")
check("block always states minimum-assistance", "MINIMUM ASSISTANCE" in lb_fail)
check("block on fail -> switch modality", "did NOT land" in lb_fail and "analogy" in lb_fail)
lb_ok = build_learning_block(mp, {}, "worked")
check("block on success -> fade", "fade" in lb_ok.lower())

# ---- debrief (points 27, 28) ----
dp = {"independence": {"segmentation": {"independent": 2}}, "errors": {"unit": 2}}
db = build_debrief(dp)
check("debrief did_well cites strength", "segmentation" in db["did_well"])
check("debrief watch_this cites recurring error", "unit" in db["watch_this"].lower())
check("debrief remember is a reusable rule", "unit" in db["remember"].lower())
check("debrief carries a next drill", db["next_drill"] and db["next_drill"]["focus_error"] == "unit")

print()
if _fail:
    print(f"{len(_fail)} FAILED: {_fail}"); sys.exit(1)
print("ALL PASS")
