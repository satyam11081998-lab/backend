"""Adversarial checks for independent guesstimate arithmetic verification."""
from services.guesstimate_backstop import apply_backstop, run_backstop


def test_full_recomputable_chain_can_score_full_arithmetic():
    chain = {"steps": [
        {"id": "pop", "label": "population", "op": "literal", "inputs": [100], "claimedValue": 100},
        {"id": "buyers", "label": "buyers", "op": "percent_of", "inputs": ["10%", "#pop"], "claimedValue": 10},
        {"id": "spend", "label": "annual spend", "op": "multiply", "inputs": ["#buyers", 5], "claimedValue": 50},
    ], "finalRef": "spend", "finalValue": 50}
    result = run_backstop(chain)
    assert result["arithmeticScore"] == 100
    assert result["verificationCoverage"] == 1.0


def test_partial_chain_does_not_claim_full_arithmetic_verification():
    chain = {"steps": [
        {"id": "pop", "label": "population", "op": "literal", "inputs": [100], "claimedValue": 100},
        {"id": "buyers", "label": "buyers", "op": "percent_of", "inputs": ["10%", "#pop"], "claimedValue": 10},
        {"id": "unknown", "label": "unreconstructed step", "op": "multiply", "inputs": ["#missing", 5], "claimedValue": 0},
    ], "finalRef": "buyers", "finalValue": 10}
    result = run_backstop(chain)
    assert result["verifiableCount"] == 1
    assert result["derivedCount"] == 2
    assert result["verificationCoverage"] == 0.5
    assert result["arithmeticScore"] is None
    applied = apply_backstop({"scoping": 70, "structure": 70, "segmentation": 70, "arithmetic": 95, "sanity": 70}, chain)
    assert applied["dimensions"]["arithmetic"] == 95
    assert "Partial arithmetic verification" in result["summary"]


def test_recomputed_mismatch_is_not_hidden_by_other_unverified_steps():
    chain = {"steps": [
        {"id": "pop", "label": "population", "op": "literal", "inputs": [100], "claimedValue": 100},
        {"id": "buyers", "label": "buyers", "op": "percent_of", "inputs": ["10%", "#pop"], "claimedValue": 40},
        {"id": "unknown", "label": "unreconstructed step", "op": "multiply", "inputs": ["#missing", 5], "claimedValue": 0},
    ], "finalRef": "buyers", "finalValue": 40}
    result = run_backstop(chain)
    assert result["arithmeticScore"] == 60
    assert any(f["kind"] == "base_inconsistency" for f in result["findings"])
