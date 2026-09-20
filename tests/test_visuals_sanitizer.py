"""Unit tests for sanitize_visuals — the trust boundary on scorer `visuals`.

Runs on the standard library alone (no OpenAI, no network):
    python -m tests.test_visuals_sanitizer

WHY THIS FILE EXISTS. `visuals` is model output that gets persisted to
feedback_json and then RENDERED to users. Everything about it is untrusted:
the shape, the types, the string lengths, the array lengths, the numeric
ranges. The frontend re-validates independently (lib/results/visuals.ts), but
neither side may assume the other did it, so the server-side rules get their
own tests. The bar is: malformed input is DROPPED, never repaired into
something plausible, and never raised on — a bad figure must not be able to
fail a real score.

sanitize_visuals is extracted from source rather than imported, so this file
runs without the OpenAI SDK installed (services.ai_scorer imports it at module
scope). If the extraction markers below ever move, this test fails loudly
rather than silently testing nothing.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SRC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "services", "ai_scorer.py")
_src = open(_SRC_PATH, encoding="utf-8").read()
assert "VISUAL_KINDS = {" in _src and "def _str_list(" in _src, (
    "test_visuals_sanitizer: extraction markers not found in services/ai_scorer.py — "
    "the sanitiser block moved; update this test rather than deleting it."
)
_block = _src[_src.index("VISUAL_KINDS = {"):_src.index("def _str_list(")]
_ns = {}
exec("from typing import Any, Dict, Optional\n" + _block, _ns)  # noqa: S102
sanitize_visuals = _ns["sanitize_visuals"]

_fail = []


def check(name, cond):
    if cond:
        print(f"  [PASS] {name}")
    else:
        print(f"  [FAIL] {name}")
        _fail.append(name)


def n(raw):
    return len(sanitize_visuals(raw))


BAR = {"kind": "bar", "title": "T", "points": [{"label": "a", "value": 1}, {"label": "b", "value": 2}]}


def main():
    print("sanitize_visuals — shape rejection")
    check("non-list input is dropped", n({"kind": "bar"}) == 0)
    check("empty list stays empty", n([]) == 0)
    check("unknown kind is dropped", n([dict(BAR, kind="pie")]) == 0)
    check("missing title is dropped", n([{k: v for k, v in BAR.items() if k != "title"}]) == 0)
    check("non-dict members are skipped", n([None, "str", 42, BAR]) == 1)
    check("a valid bar survives", n([BAR]) == 1)

    print("\nsanitize_visuals — data rules")
    check("a one-point series is dropped", n([dict(BAR, points=[{"label": "a", "value": 1}])]) == 0)
    check("numeric strings are coerced", n([dict(BAR, points=[{"label": "a", "value": "12"},
                                                              {"label": "b", "value": "3.5"}])]) == 1)
    check("NaN and inf are dropped", n([dict(BAR, points=[{"label": "a", "value": float("inf")},
                                                          {"label": "b", "value": float("nan")},
                                                          {"label": "c", "value": 1}])]) == 0)
    check("at most 6 figures", n([dict(BAR, title=f"T{i}") for i in range(10)]) == 6)

    print("\nsanitize_visuals — quadrant")
    q_in = [{"kind": "quadrant", "title": "T", "xLabel": "x", "yLabel": "y",
             "points": [{"label": "a", "x": 5, "y": -3}, {"label": "b", "x": 0.5, "y": 0.5}]}]
    q = sanitize_visuals(q_in)
    check("quadrant survives", len(q) == 1)
    check("x clamped to 1.0", q and q[0]["points"][0]["x"] == 1.0)
    check("y clamped to 0.0", q and q[0]["points"][0]["y"] == 0.0)
    check("snake_case axis labels accepted", n([{"kind": "quadrant", "title": "T",
                                                 "x_label": "x", "y_label": "y",
                                                 "points": [{"label": "a", "x": 0.1, "y": 0.2},
                                                            {"label": "b", "x": 0.5, "y": 0.5}]}]) == 1)
    check("quadrant with one point is dropped", n([{"kind": "quadrant", "title": "T", "xLabel": "x",
                                                    "yLabel": "y",
                                                    "points": [{"label": "a", "x": 0.1, "y": 0.2}]}]) == 0)

    print("\nsanitize_visuals — tree")
    check("childless root is dropped", n([{"kind": "tree", "title": "T", "root": {"label": "solo"}}]) == 0)
    check("root with children survives",
          n([{"kind": "tree", "title": "T", "root": {"label": "r", "children": [{"label": "c"}]}}]) == 1)
    deep = {"label": "a", "children": [{"label": "b", "children": [{"label": "c", "children": [
        {"label": "d", "children": [{"label": "e"}]}]}]}]}
    check("over-deep tree is truncated, not rejected",
          n([{"kind": "tree", "title": "T", "root": deep}]) == 1)

    print("\nsanitize_visuals — waterfall")
    check("signed steps survive", n([{"kind": "waterfall", "title": "T",
                                      "steps": [{"label": "R", "value": 100, "total": True},
                                                {"label": "C", "value": -40}]}]) == 1)

    print("\nsanitize_visuals — numeric limits")
    check("booleans are not data points",
          n([dict(BAR, points=[{"label": "a", "value": True}, {"label": "b", "value": 2}])]) == 0)
    check("absurd magnitudes are dropped",
          n([dict(BAR, points=[{"label": "a", "value": 1e308}, {"label": "b", "value": 2}])]) == 0)
    check("a realistic rupee market size survives",
          n([dict(BAR, points=[{"label": "a", "value": 4.2e13}, {"label": "b", "value": 2}])]) == 1)
    check("a waterfall with an absurd step is dropped",
          n([{"kind": "waterfall", "title": "T",
              "steps": [{"label": "R", "value": 1e20, "total": True}, {"label": "C", "value": -40}]}]) == 0)

    print("\nsanitize_visuals — bounds")
    big = sanitize_visuals([{"kind": "bar", "title": "T" * 500, "caption": "C" * 900,
                             "unit": "U" * 90,
                             "points": [{"label": "L" * 200, "value": 1},
                                        {"label": "m", "value": 2}]}])
    check("title capped at 90", big and len(big[0]["title"]) <= 90)
    check("caption capped at 200", big and len(big[0]["caption"]) <= 200)
    check("unit capped at 20", big and len(big[0]["unit"]) <= 20)
    check("point label capped at 40", big and len(big[0]["points"][0]["label"]) <= 40)

    print("\nsanitize_visuals — never raises")
    for hostile in (None, 0, "", {"a": 1}, [{"kind": "tree", "root": object()}],
                    [{"kind": "bar", "title": "T", "points": "nope"}]):
        try:
            sanitize_visuals(hostile)
        except Exception as e:  # noqa: BLE001
            check(f"raised on {hostile!r}: {e}", False)
    check("hostile inputs handled without raising", True)

    print()
    if _fail:
        print(f"FAILED ({len(_fail)}): " + ", ".join(_fail))
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
