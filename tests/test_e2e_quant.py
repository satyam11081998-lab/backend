"""End-to-end: does the Chennai turn actually reach the model as arithmetic?"""
from services.interviewer_decision import build_interviewer_control_packet
from services.interviewer_mode import build_mode_block
from services import structure_tree as S
import json

def test_packet_carries_the_arithmetic_for_the_real_turn():
    state = S.fold_turn({}, {"tree": ">population#1.2 crore"})
    msg = "upper middle class would be 1/3 that is 40 lacks who ows a car"
    for policy in ("coached", "exam"):
        pkt = build_interviewer_control_packet(
            {"teaching_policy": policy, "case_type": "guesstimate"},
            "TARGETED_QUESTION", candidate_text=msg, prior_state=state)
        cc = pkt["interviewer_control"]["case_context"]
        assert cc.get("from_their_numbers"), f"{policy}: no arithmetic in packet"
        assert any("33" in r for r in cc["from_their_numbers"]), cc["from_their_numbers"]
        assert pkt["interviewer_control"]["response_generation"]["track_structure"] is True
        block = build_mode_block("probe", "x", True, control_packet=pkt)
        assert "WHAT THEIR OWN NUMBERS IMPLY" in block
        assert "DRAWING THEIR STRUCTURE" in block
        print(f"  {policy}: {cc['from_their_numbers']}")
