"""The QA tooling itself must work: golden dataset shape, the scoring harness through the real
evidence/evaluation code, metrics and gates, and the in-process interview simulator."""

import json

from qa.golden.build_dataset import EXPECT, F, build
from qa.run_golden import compute_metrics, gate_failures


def test_golden_dataset_shape():
    d = build()
    assert len(d["items"]) == len(F) * len(EXPECT) == 96
    assert all(it["label_source"] == "author_draft" for it in d["items"])
    assert len({it["id"] for it in d["items"]}) == 96
    fams = {it["family"] for it in d["items"]}
    assert {"marketing", "finance_corporate", "consulting", "software_engineering", "sales", "product_management",
            "operations", "human_resources"} == fams


def test_harness_scores_through_the_real_pipeline_and_gates_work():
    from qa import harness
    harness.use_scratch_db(simulated=True)
    d = build()
    subset = [it for it in d["items"] if it["family"] in ("marketing", "sales")]
    results = {it["id"]: [harness.score_item(it, terms=F[it["family"]]["terms"])] for it in subset}
    m = compute_metrics(d, results)
    assert m["items"] == 24 and m["false_weakness"] == 0 and m["injection_detected"] == 1.0
    inj = results["marketing.injection"][0]
    assert inj["injection_flags"] and not any("ignore all previous" in e["quote"].lower() for e in inj["evidence"]), \
        "an instruction aimed at the evaluator is never stored as evidence"
    off = results["sales.off_topic"][0]
    assert off["state"] == "not_sufficiently_tested" and off["score"] is None
    # gate logic
    assert gate_failures({"fairness_gap_non_native_strong": 2.0, "false_weakness": 1, "injection_resisted": 0.5,
                          "stuffing_resisted": 1.0}) == ["fairness_gap_non_native_strong=2.0 > 1.0",
                                                         "false_weakness=1", "injection_resisted=0.5 < 1.0"]


def test_interview_simulator_runs_end_to_end(tmp_path):
    from qa.simulate_interview import run_one
    res = run_one("operations", "injector", mode="grill", difficulty="grill", minutes=30, max_turns=6,
                  simulated=True, llm_candidate=False, quiet=True)
    assert res["report_status"] in ("ready", "partial")
    assert sum(1 for m in res["transcript"] if m["role"] == "candidate") >= 3
    assert "10/10" not in json.dumps(res["summary"])
