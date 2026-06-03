"""
Guesstimate arithmetic backstop — faithful Python port of the frontend TS
(lib/scoring/calc-chain.ts + arithmetic-check.ts + apply-backstop.ts).

Principle (unchanged from the TS): the LLM TRANSCRIBES the candidate's stated math
into a structured CalcChain (LLMs are OK at transcription); CODE VERIFIES it (LLMs
are bad at arithmetic). The backstop recomputes every step, OVERRIDES the arithmetic
dimension (D4) with its deterministic verdict, and caps the total on magnitude
implausibility. It is honest about scope: it does NOT judge assumption realism —
that stays with the LLM under the segmentation dimension.

Weights and behaviour mirror the TS exactly (GUESSTIMATE_WEIGHTS, TOLERANCE=0.02,
caps 0.5/0.35). Keep this file in sync with the TS if either changes.
"""

import math
from typing import Dict, List, Optional, Any

# guesstimate rubric weights — MUST match lib/scoring/apply-backstop.ts GUESSTIMATE_WEIGHTS
GUESSTIMATE_WEIGHTS = {
    "scoping": 0.10,
    "structure": 0.30,
    "segmentation": 0.25,
    "arithmetic": 0.15,
    "sanity": 0.20,
}

DIMENSIONS = ["scoping", "structure", "segmentation", "arithmetic", "sanity"]

EPS = 1e-9
TOLERANCE = 0.02  # 2% — generous; guesstimates round freely


def _pct_to_fraction(v) -> float:
    if isinstance(v, (int, float)):
        return v / 100 if v > 1 else float(v)
    s = str(v).strip()
    if s.endswith("%"):
        return float(s[:-1]) / 100
    n = float(s)
    return n / 100 if n > 1 else n


def _fmt(n: float) -> str:
    a = abs(n)
    if a >= 1e7:
        return f"{n / 1e7:.2f} cr"
    if a >= 1e5:
        return f"{n / 1e5:.2f} L"
    if a >= 1e3:
        return f"{n / 1e3:.1f}K"
    return f"{n:g}"


def resolve_chain(chain: Dict[str, Any], tolerance: float = TOLERANCE) -> List[Dict[str, Any]]:
    """Recompute every step from its inputs, resolving #refs to COMPUTED (not claimed) values."""
    steps = chain.get("steps", []) or []
    by_id = {s["id"]: s for s in steps}
    computed: Dict[str, float] = {}
    out: List[Dict[str, Any]] = []

    def resolve_input(inp) -> float:
        if isinstance(inp, (int, float)):
            return float(inp)
        s = str(inp).strip()
        if s.startswith("#"):
            ref = s[1:]
            if ref not in computed and ref in by_id:
                compute_step(by_id[ref])
            return computed.get(ref, float("nan"))
        try:
            return float(s)
        except ValueError:
            return float("nan")

    def compute_step(step: Dict[str, Any]) -> float:
        sid = step["id"]
        if sid in computed:
            return computed[sid]
        op = step.get("op")
        inputs = step.get("inputs", []) or []
        val: float
        if op == "literal":
            first = inputs[0] if inputs else 0
            val = float(first) if isinstance(first, (int, float)) else float(str(first))
        elif op == "add":
            val = sum(resolve_input(x) for x in inputs)
        elif op == "subtract":
            val = resolve_input(inputs[0])
            for x in inputs[1:]:
                val -= resolve_input(x)
        elif op == "multiply":
            val = 1.0
            for x in inputs:
                val *= resolve_input(x)
        elif op == "divide":
            val = resolve_input(inputs[0])
            for x in inputs[1:]:
                d = resolve_input(x)
                val = val / d if d != 0 else float("inf")
        elif op == "percent_of":
            pct = _pct_to_fraction(inputs[0])
            base = resolve_input(inputs[1])
            val = pct * base
        else:
            val = float("nan")
        computed[sid] = val
        return val

    for step in steps:
        computed_value = compute_step(step)
        claimed = float(step.get("claimedValue", 0) or 0)
        rel_error = abs(computed_value - claimed) / max(abs(claimed), EPS)
        out.append({
            "id": step["id"],
            "label": step.get("label", ""),
            "claimedValue": claimed,
            "computedValue": computed_value,
            "relError": rel_error,
            "ok": rel_error <= tolerance,
        })
    return out


def _orders_outside(value: float, band: Dict[str, float]) -> float:
    low, high = band["low"], band["high"]
    if low <= value <= high:
        return 0.0
    edge = low if value < low else high
    if value <= 0 or edge <= 0:
        return float("inf")
    return abs(math.log10(value) - math.log10(edge))


def run_backstop(chain: Dict[str, Any], band: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Deterministic checks over a CalcChain → corrected D4 score + total cap + findings."""
    resolved = resolve_chain(chain, TOLERANCE)
    by_id = {s["id"]: s for s in (chain.get("steps", []) or [])}
    findings: List[Dict[str, Any]] = []

    for r in resolved:
        if r["ok"]:
            continue
        step = by_id.get(r["id"], {})
        if step.get("op") == "percent_of":
            findings.append({
                "kind": "base_inconsistency", "stepId": r["id"], "label": r["label"],
                "message": f"\"{r['label']}\" claims {_fmt(r['claimedValue'])} but the stated "
                           f"percentage of its base recomputes to {_fmt(r['computedValue'])}. "
                           f"The base or the result is wrong.",
                "claimed": r["claimedValue"], "computed": r["computedValue"],
            })
        else:
            findings.append({
                "kind": "recompute_mismatch", "stepId": r["id"], "label": r["label"],
                "message": f"\"{r['label']}\" claims {_fmt(r['claimedValue'])} but its inputs "
                           f"recompute to {_fmt(r['computedValue'])} (off by {r['relError'] * 100:.0f}%).",
                "claimed": r["claimedValue"], "computed": r["computedValue"],
            })

    # final-value check
    final_ref = chain.get("finalRef")
    final_step = None
    if final_ref:
        final_step = next((r for r in resolved if r["id"] == final_ref), None)
    elif resolved:
        final_step = resolved[-1]
    final_value = float(chain.get("finalValue", 0) or 0)
    if final_step:
        rel_err = abs(final_step["computedValue"] - final_value) / max(abs(final_value), EPS)
        if rel_err > TOLERANCE:
            findings.append({
                "kind": "final_mismatch", "label": "final answer",
                "message": f"Stated final answer {_fmt(final_value)} doesn't match the chain, "
                           f"which recomputes to {_fmt(final_step['computedValue'])}.",
                "claimed": final_value, "computed": final_step["computedValue"],
            })

    # magnitude guard (only if the author provided a plausible band)
    total_cap_factor = 1.0
    if band:
        oom = _orders_outside(final_value, band)
        if oom >= 2:
            findings.append({
                "kind": "magnitude_implausible", "label": "order of magnitude",
                "message": f"Final answer {_fmt(final_value)} is ~{oom:.1f} orders of magnitude "
                           f"outside the plausible range [{_fmt(band['low'])}–{_fmt(band['high'])}]. "
                           f"A clean-looking method does not rescue an answer this far off.",
                "claimed": final_value, "computed": band["low"],
            })
            total_cap_factor = 0.35 if oom >= 3 else 0.5

    # corrected D4 from count/severity of arithmetic findings (magnitude excluded)
    arith_findings = [f for f in findings if f["kind"] != "magnitude_implausible"]
    n = len(arith_findings)
    arithmetic_score = 5 if n == 0 else 3 if n == 1 else 2 if n == 2 else 1

    summary = (
        "Arithmetic verified: every step recomputes within tolerance and the final value is consistent."
        if not findings
        else f"{len(findings)} arithmetic issue(s) found by independent recomputation — "
             "these are deterministic, not opinion."
    )

    return {
        "findings": findings,
        "arithmeticScore": arithmetic_score,
        "totalCapFactor": total_cap_factor,
        "summary": summary,
        "notChecked": (
            "This backstop verifies internal arithmetic consistency only. It does NOT judge whether "
            "the ASSUMPTIONS are realistic — that is scored by the LLM under the segmentation dimension."
        ),
    }


def _weighted_total(d: Dict[str, float]) -> float:
    t = (
        d["scoping"] * GUESSTIMATE_WEIGHTS["scoping"]
        + d["structure"] * GUESSTIMATE_WEIGHTS["structure"]
        + d["segmentation"] * GUESSTIMATE_WEIGHTS["segmentation"]
        + d["arithmetic"] * GUESSTIMATE_WEIGHTS["arithmetic"]
        + d["sanity"] * GUESSTIMATE_WEIGHTS["sanity"]
    )
    return (t / 5) * 100


def apply_backstop(
    llm_dims: Dict[str, float],
    chain: Dict[str, Any],
    band: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Combine LLM rubric dim scores + calc chain → final backstop-corrected score."""
    backstop = run_backstop(chain, band)

    corrected = dict(llm_dims)
    corrected["arithmetic"] = backstop["arithmeticScore"]
    arithmetic_overridden = backstop["arithmeticScore"] != llm_dims.get("arithmetic")

    raw_total = _weighted_total(corrected)
    total = round(raw_total * backstop["totalCapFactor"])

    return {
        "dimensions": corrected,
        "rawTotal": round(raw_total),
        "total": total,
        "backstop": backstop,
        "arithmeticOverridden": arithmetic_overridden,
    }
