"""
Guesstimate arithmetic backstop — Python port of the frontend TS
(lib/scoring/calc-chain.ts + arithmetic-check.ts + apply-backstop.ts), hardened.

Principle (unchanged): the LLM TRANSCRIBES the candidate's stated math into a
structured CalcChain; CODE VERIFIES it. The backstop recomputes the steps it can
fully reconstruct, OVERRIDES the arithmetic dimension with its deterministic
verdict, and caps the total on magnitude implausibility.

2026-06-02 hardening (fixes "everything recomputes to 0 / false arithmetic 1/5"):
  - BASE/`literal` steps are assumptions, not recomputations: their computed value
    is their stated value (input if given, else claimedValue) — they NEVER false-flag.
  - A step is only FLAGGED when it is fully recomputable (a derived op whose inputs
    all resolve to finite numbers) AND the recompute genuinely disagrees. Steps we
    can't reconstruct are marked unverified and skipped — never invented as errors.
  - `percent_of` handles "12%", 0.12, 12, or a "#ref" as the percentage without crashing.
  - If NOTHING in the chain is verifiable, the arithmetic dimension falls back to the
    LLM's own score (we add no information, so we don't override).

Weights/tolerance/caps mirror the TS. Keep in sync if either changes.
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
DERIVED_OPS = {"add", "subtract", "multiply", "divide", "percent_of"}


def _is_finite(x) -> bool:
    return isinstance(x, (int, float)) and not (math.isnan(x) or math.isinf(x))


def _to_number(v) -> float:
    """Parse a number from int/float or a clean numeric string; NaN if not parseable."""
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _fmt(n: float) -> str:
    if not _is_finite(n):
        return "—"
    a = abs(n)
    if a >= 1e7:
        return f"{n / 1e7:.2f} cr"
    if a >= 1e5:
        return f"{n / 1e5:.2f} L"
    if a >= 1e3:
        return f"{n / 1e3:.1f}K"
    return f"{n:g}"


def resolve_chain(chain: Dict[str, Any], tolerance: float = TOLERANCE) -> List[Dict[str, Any]]:
    """
    Recompute every step. Each result carries:
      computedValue, claimedValue, relError, ok, verifiable, op
    `verifiable` = we could fully reconstruct this step's math (derived op + all inputs
    resolved to finite numbers). Only verifiable steps can ever be flagged.
    """
    steps = chain.get("steps", []) or []
    by_id = {s.get("id"): s for s in steps if s.get("id") is not None}
    computed: Dict[str, float] = {}
    _verifiable: Dict[str, bool] = {}

    def resolve_ref(inp) -> float:
        if isinstance(inp, (int, float)):
            return float(inp)
        s = str(inp).strip()
        if s.startswith("#"):
            ref = s[1:]
            if ref not in computed and ref in by_id:
                compute_step(by_id[ref])
            return computed.get(ref, float("nan"))
        return _to_number(s)

    def resolve_pct(inp) -> float:
        if isinstance(inp, str) and inp.strip().endswith("%"):
            return _to_number(inp.strip()[:-1]) / 100.0
        v = resolve_ref(inp)
        if not _is_finite(v):
            return float("nan")
        return v / 100.0 if v > 1 else v

    def compute_step(step: Dict[str, Any]) -> float:
        sid = step.get("id")
        if sid in computed:
            return computed[sid]
        op = step.get("op")
        inputs = step.get("inputs", []) or []
        claimed = _to_number(step.get("claimedValue", 0))
        verifiable = False
        val: float

        if op == "literal" or op not in DERIVED_OPS:
            # Base assumption (or an op we don't recompute): the stated value IS the value.
            v = resolve_ref(inputs[0]) if inputs else float("nan")
            val = v if _is_finite(v) else (claimed if _is_finite(claimed) else 0.0)
            verifiable = False
        elif op == "percent_of":
            pct = resolve_pct(inputs[0]) if len(inputs) >= 1 else float("nan")
            base = resolve_ref(inputs[1]) if len(inputs) >= 2 else float("nan")
            if _is_finite(pct) and _is_finite(base):
                val, verifiable = pct * base, True
            else:
                val, verifiable = (claimed if _is_finite(claimed) else float("nan")), False
        else:
            resolved = [resolve_ref(x) for x in inputs]
            if resolved and all(_is_finite(r) for r in resolved):
                if op == "add":
                    val = sum(resolved)
                elif op == "subtract":
                    val = resolved[0]
                    for r in resolved[1:]:
                        val -= r
                elif op == "multiply":
                    val = 1.0
                    for r in resolved:
                        val *= r
                else:  # divide
                    val = resolved[0]
                    for r in resolved[1:]:
                        val = val / r if r != 0 else float("inf")
                verifiable = _is_finite(val)
            else:
                val, verifiable = (claimed if _is_finite(claimed) else float("nan")), False

        computed[sid] = val
        _verifiable[sid] = verifiable
        return val

    out: List[Dict[str, Any]] = []
    for step in steps:
        cv = compute_step(step)
        claimed = _to_number(step.get("claimedValue", 0))
        verifiable = _verifiable.get(step.get("id"), False)
        if verifiable and _is_finite(cv) and _is_finite(claimed):
            rel = abs(cv - claimed) / max(abs(claimed), EPS)
            ok = rel <= tolerance
        else:
            rel, ok = 0.0, True  # not verifiable → never a finding
        out.append({
            "id": step.get("id"),
            "label": step.get("label", ""),
            "op": step.get("op"),
            "claimedValue": claimed,
            "computedValue": cv,
            "relError": rel,
            "ok": ok,
            "verifiable": verifiable,
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
    resolved = resolve_chain(chain, TOLERANCE)
    findings: List[Dict[str, Any]] = []
    verifiable_steps = [r for r in resolved if r["verifiable"]]

    for r in resolved:
        if r["ok"] or not r["verifiable"]:
            continue
        if r["op"] == "percent_of":
            findings.append({
                "kind": "base_inconsistency", "stepId": r["id"], "label": r["label"],
                "message": f"\"{r['label']}\" claims {_fmt(r['claimedValue'])} but the stated "
                           f"percentage of its base recomputes to {_fmt(r['computedValue'])}.",
                "claimed": r["claimedValue"], "computed": r["computedValue"],
            })
        else:
            findings.append({
                "kind": "recompute_mismatch", "stepId": r["id"], "label": r["label"],
                "message": f"\"{r['label']}\" claims {_fmt(r['claimedValue'])} but its inputs "
                           f"recompute to {_fmt(r['computedValue'])} (off by {r['relError'] * 100:.0f}%).",
                "claimed": r["claimedValue"], "computed": r["computedValue"],
            })

    # final-value check — only when the final step is one we actually recomputed
    final_ref = chain.get("finalRef")
    final_step = None
    if final_ref:
        final_step = next((r for r in resolved if r["id"] == final_ref), None)
    elif verifiable_steps:
        final_step = verifiable_steps[-1]
    final_value = _to_number(chain.get("finalValue", 0))
    if final_step and final_step["verifiable"] and _is_finite(final_value) and _is_finite(final_step["computedValue"]):
        rel_err = abs(final_step["computedValue"] - final_value) / max(abs(final_value), EPS)
        if rel_err > TOLERANCE:
            findings.append({
                "kind": "final_mismatch", "label": "final answer",
                "message": f"Stated final answer {_fmt(final_value)} doesn't match the chain, "
                           f"which recomputes to {_fmt(final_step['computedValue'])}.",
                "claimed": final_value, "computed": final_step["computedValue"],
            })

    total_cap_factor = 1.0
    if band and _is_finite(final_value):
        oom = _orders_outside(final_value, band)
        if oom >= 2:
            findings.append({
                "kind": "magnitude_implausible", "label": "order of magnitude",
                "message": f"Final answer {_fmt(final_value)} is ~{oom:.1f} orders of magnitude "
                           f"outside the plausible range [{_fmt(band['low'])}–{_fmt(band['high'])}].",
                "claimed": final_value, "computed": band["low"],
            })
            total_cap_factor = 0.35 if oom >= 3 else 0.5

    could_verify = len(verifiable_steps) > 0 or (final_step is not None and final_step["verifiable"])
    arith_findings = [f for f in findings if f["kind"] != "magnitude_implausible"]
    n = len(arith_findings)

    if not could_verify:
        arithmetic_score = None  # defer to LLM's read
        summary = ("Arithmetic could not be independently recomputed from the steps provided, "
                   "so the interviewer's assessment stands for this dimension.")
    elif n == 0:
        arithmetic_score = 5
        summary = "Arithmetic verified: every recomputable step checks out within tolerance."
    else:
        arithmetic_score = 3 if n == 1 else 2 if n == 2 else 1
        summary = (f"{n} arithmetic issue(s) found by independent recomputation of the steps "
                   "that could be verified — these are deterministic, not opinion.")

    return {
        "findings": findings,
        "arithmeticScore": arithmetic_score,
        "verifiableCount": len(verifiable_steps),
        "totalCapFactor": total_cap_factor,
        "summary": summary,
        "notChecked": (
            "This backstop verifies internal arithmetic consistency of the steps it can reconstruct. "
            "It does NOT judge whether the ASSUMPTIONS are realistic — that is scored under segmentation."
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
    backstop = run_backstop(chain, band)

    corrected = dict(llm_dims)
    bs_score = backstop["arithmeticScore"]
    if bs_score is None:
        try:
            corrected["arithmetic"] = max(1, min(5, int(round(float(llm_dims.get("arithmetic", 3))))))
        except (TypeError, ValueError):
            corrected["arithmetic"] = 3
        arithmetic_overridden = False
    else:
        corrected["arithmetic"] = bs_score
        arithmetic_overridden = bs_score != llm_dims.get("arithmetic")

    raw_total = _weighted_total(corrected)
    total = round(raw_total * backstop["totalCapFactor"])

    return {
        "dimensions": corrected,
        "rawTotal": round(raw_total),
        "total": total,
        "backstop": backstop,
        "arithmeticOverridden": arithmetic_overridden,
    }
