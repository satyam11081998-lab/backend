"""The worked approach for a case: load it, generate it, and moderate what people add.

WHAT THIS IS FOR

Every case gets a page like a casebook's: the problem, the facts that matter, the tree a
strong candidate builds, what to notice, where it lands. The candidate practises first and
opens it afterwards -- or opens it deliberately mid-attempt and has that recorded.

THREE RULES THIS MODULE EXISTS TO ENFORCE

1. It is generated from the case's OWN solution. Not from any third-party casebook. The
   decks this layout is modelled on are somebody else's copyrighted teaching material;
   the information architecture is fair to borrow, the content is not.

2. Generated is not publishable. A framework reads with the authority of a model answer,
   so a wrong one teaches the error -- strictly worse than showing nothing. Generation
   writes status='draft'. Only 'approved' is ever served.

3. A contributed note is reviewed before anyone else sees it. It appears beside a worked
   answer and inherits that authority. The adversarial check here is ADVISORY: it marks
   the obvious rejects so a human reviews a short queue instead of a long one. A model
   approving text that will be shown as teaching material next to that same model's
   framework is a system grading its own homework.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

FRAMEWORK_TTL_S = float(os.getenv("CASE_FRAMEWORK_TTL_S", "600") or 600)
FRAMEWORK_MODEL = (os.getenv("CASE_FRAMEWORK_MODEL") or "gpt-4o").strip()
MAX_NOTES = 12

_cache: Dict[str, Any] = {}
_cache_at: Dict[str, float] = {}


def _sb():
    try:
        from services.supabase_client import get_supabase_client
        return get_supabase_client()
    except Exception:  # noqa: BLE001
        return None


def enabled() -> bool:
    return (os.getenv("CASE_FRAMEWORKS", "on") or "on").strip().lower() not in ("0", "off", "false")


# --------------------------------------------------------------------------- read
def load(case_id: str) -> Optional[Dict[str, Any]]:
    """The approved framework for a case, or None. Never raises."""
    if not enabled() or not case_id:
        return None
    now = time.time()
    if case_id in _cache and now - _cache_at.get(case_id, 0) < FRAMEWORK_TTL_S:
        return _cache[case_id]
    sb = _sb()
    if sb is None:
        return None
    row = None
    try:
        res = (sb.table("case_frameworks")
               .select("tree, facts, assumptions, interviewee_notes, observations, "
                       "recommendations, source, model, updated_at")
               .eq("case_id", case_id).eq("status", "approved").limit(1).execute())
        row = (res.data or [None])[0]
    except Exception as e:  # noqa: BLE001 - table may not be migrated yet
        print(f"[framework] read skipped: {type(e).__name__}: {str(e)[:90]}")
    _cache[case_id] = row
    _cache_at[case_id] = now
    return row


def notes(case_id: str, limit: int = MAX_NOTES) -> List[Dict[str, Any]]:
    """Approved candidate notes, most helpful first. Never raises."""
    sb = _sb()
    if sb is None or not case_id:
        return []
    try:
        res = (sb.table("case_framework_notes")
               .select("id, body, kind, helpful_count, created_at")
               .eq("case_id", case_id).eq("status", "approved")
               .order("helpful_count", desc=True).limit(limit).execute())
        return list(res.data or [])
    except Exception as e:  # noqa: BLE001
        print(f"[framework] notes skipped: {type(e).__name__}: {str(e)[:90]}")
        return []


def invalidate(case_id: Optional[str] = None) -> None:
    if case_id:
        _cache.pop(case_id, None)
        _cache_at.pop(case_id, None)
    else:
        _cache.clear()
        _cache_at.clear()


# --------------------------------------------------------------------------- generate
_GEN_COMMON = """You turn a problem's own worked solution into the approach a strong
candidate would build. You are given ONLY that problem's statement and solution. Use
nothing else: no outside casebook, no remembered version of this question.

Return STRICT JSON, no prose, no code fence:
{
 "tree": {"label": "...", "op": "*|/|+|-|null", "value": "figure or null",
          "share": "percent of its parent, or null",
          "driver": true|false, "children": [ ...same shape... ]},
 "calculation": [ {"label": "what this step produces",
                   "terms": [{"label": "...", "value": "600 mins"}, ...],
                   "op": "*|/|+|-", "result": "150 votes"} ],
 "facts": [...], "assumptions": [...], "interviewee_notes": [...],
 "observations": [...], "recommendations": [...]
}

SHARED RULES
- The tree is the DECOMPOSITION, not a list of topics. A parent must equal its children
  combined by its op.
- Node labels are three or four words. Figures go in `value` EXACTLY as the solution
  writes them. Never invent a figure the solution does not contain -- a number you made up
  becomes a wrong model answer on a product people pay for.
- 3 to 6 entries per list, each one line. Omit a list rather than padding it.
- If the solution is too thin to support a tree, return {"insufficient": true} alone."""

_GEN_CASE = _GEN_COMMON + """

THIS IS A CASE.
- `driver`: true on the branch that actually explained THIS case, false elsewhere; usually
  one path from the root. If the solution does not say, mark nothing as driver.
- `calculation` is usually empty for a case. Include a step only where the solution
  actually computes something (a profit bridge, a break-even).
- `recommendations` is where the case lands, and it is not optional."""

_GEN_GUESSTIMATE = _GEN_COMMON + """

THIS IS A GUESSTIMATE. It decomposes AND THEN COMPUTES, and the computation is the half
candidates get wrong, so both halves must be present.
- The tree is the SEGMENTATION: population split by age, geography, segment. Where the
  solution gives a share and an absolute for a node ("Rural 65%, ~900 Mn"), put the
  percent in `share` and the absolute in `value`. Siblings' shares should sum to about
  100% of their parent.
- `calculation` is REQUIRED and is the ordered chain that produces the answer, each step
  feeding the next:
    {"label": "# votes per EVM per phase",
     "terms": [{"label":"Booth hours","value":"600 mins"},
               {"label":"Time per vote","value":"4 mins"}],
     "op": "/", "result": "150 votes"}
  Copy the figures the solution uses. The terms combined by `op` MUST actually produce
  `result` -- this is recomputed and a chain that does not close is thrown away.
- `driver`: true along the spine that carries the answer.
- `observations` should include the sanity check: what the answer implies per person, per
  day, or against a known total."""


CALC_TOLERANCE = 0.15   # casebook chains round hard: 800 Mn / 750 is written 1.07 Mn


def _step_closes(step: Dict[str, Any]) -> bool:
    """Do this step's own terms, combined by its own operator, produce its own result?

    This is the check that makes a generated guesstimate worth more than a reviewer's
    patience: a chain whose arithmetic does not close is wrong with CERTAINTY, and no
    amount of reading catches it as reliably as recomputing it.
    """
    try:
        from services.quant_graph import _value_of, _combine
    except Exception:  # noqa: BLE001
        return True                      # cannot check: do not reject on that basis
    want = _value_of(step.get("result") or "")
    qs = [_value_of(t.get("value") or "") for t in (step.get("terms") or [])]
    if not want or want.pct or len(qs) < 2 or not all(qs):
        return True                      # nothing checkable here
    op = step.get("op")

    # "1.07 Mn + 10% buffer" means increase BY ten percent, not add the number 0.1. A
    # percentage in an additive step is always relative; treating it as a magnitude is
    # how a correct chain gets thrown away.
    if op in ("+", "-") and any(q.pct for q in qs):
        base = next((q.value for q in qs if not q.pct), None)
        if base is None:
            return True
        got = base
        for q in qs:
            if q.pct:
                got = got * (1 + q.value / 100.0) if op == "+" else got * (1 - q.value / 100.0)
    else:
        got = _combine(op, [q.value / 100.0 if q.pct else q.value for q in qs])
    if not got or got <= 0 or want.value <= 0:
        return True
    return abs(got - want.value) <= CALC_TOLERANCE * max(got, want.value)


def _clean_calculation(v: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(v, list):
        return out
    for raw in v[:10]:
        if not isinstance(raw, dict):
            continue
        label = str(raw.get("label") or "").strip()[:48]
        result = str(raw.get("result") or "").strip()[:28]
        op = raw.get("op") if raw.get("op") in ("*", "/", "+", "-") else None
        terms = []
        for t in (raw.get("terms") or [])[:5]:
            if isinstance(t, dict) and str(t.get("value") or "").strip():
                terms.append({"label": str(t.get("label") or "").strip()[:40],
                              "value": str(t.get("value")).strip()[:28]})
        if label and result and op and len(terms) >= 2:
            out.append({"label": label, "terms": terms, "op": op, "result": result})
    return out


def generate(case_id: str, title: str, content: str, solution: str,
             case_type: str = "", *, llm=None, model: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Draft a framework from the case's own solution. Returns the row written, or None.

    Admin/CLI only. Writes status='draft' -- nothing here reaches a candidate until a
    human approves it.
    """
    if not (solution or "").strip():
        return None
    use_model = (model or FRAMEWORK_MODEL).strip()
    try:
        from services.ai_providers import openai_client
        cli = llm or openai_client()
        if cli is None:
            return None
        guess = (case_type or "").strip().lower() == "guesstimate"
        user = (f"PROBLEM: {title}\n\n{(content or '')[:2500]}\n\n"
                f"WORKED SOLUTION:\n{(solution or '')[:6000]}")
        resp = cli.chat.completions.create(
            model=use_model, temperature=0, max_tokens=1900,
            messages=[{"role": "system",
                       "content": _GEN_GUESSTIMATE if guess else _GEN_CASE},
                      {"role": "user", "content": user}],
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001
        print(f"[framework] generate failed for {case_id}: {type(e).__name__}")
        return None

    data = _parse_json(raw)
    if not data or data.get("insufficient"):
        print(f"[framework] solution too thin for {case_id}")
        return None
    tree = _clean_tree(data.get("tree"))
    if not tree or not tree.get("children"):
        print(f"[framework] no usable tree for {case_id}")
        return None

    calc = _clean_calculation(data.get("calculation"))
    broken = [c["label"] for c in calc if not _step_closes(c)]
    if broken:
        # Refuse the WHOLE draft, not just the bad step. A chain is one argument: if the
        # model's arithmetic failed anywhere in it, the figures around it are not to be
        # trusted either, and this never reaches a human queue.
        print(f"[framework] arithmetic does not close for {case_id}: {broken}")
        return None
    if (case_type or "").strip().lower() == "guesstimate" and not calc:
        # A guesstimate decomposes AND computes. A tree with no chain under it is half a
        # worked answer, and half is what a candidate already had.
        print(f"[framework] guesstimate {case_id} produced no calculation chain")
        return None

    row = {
        "case_id": case_id,
        "tree": tree,
        "calculation": calc,
        "facts": _clean_list(data.get("facts")),
        "assumptions": _clean_list(data.get("assumptions")),
        "interviewee_notes": _clean_list(data.get("interviewee_notes")),
        "observations": _clean_list(data.get("observations")),
        "recommendations": _clean_list(data.get("recommendations")),
        "source": "generated",
        "status": "draft",
        "model": use_model,
    }
    sb = _sb()
    if sb is None:
        return row
    try:
        sb.table("case_frameworks").insert(row).execute()
    except Exception as e:  # noqa: BLE001
        print(f"[framework] insert failed for {case_id}: {type(e).__name__}")
    return row


def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
    t = (raw or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        out = json.loads(t)
        return out if isinstance(out, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _clean_list(v: Any, cap: int = 6, chars: int = 180) -> List[str]:
    if not isinstance(v, list):
        return []
    out = []
    for x in v:
        s = str(x).strip()
        if s:
            out.append(s[:chars])
        if len(out) >= cap:
            break
    return out


def _clean_tree(node: Any, depth: int = 0) -> Optional[Dict[str, Any]]:
    """Coerce whatever came back into the node shape the renderer already draws.

    Anything unexpected is dropped rather than passed through: this JSON is written into a
    table that feeds a teaching page, so the time to refuse a malformed branch is before
    it is stored, not when it fails to render.
    """
    if not isinstance(node, dict) or depth > 5:
        return None
    label = str(node.get("label") or "").strip()[:42]
    if not label:
        return None
    op = node.get("op")
    op = op if op in ("*", "/", "+", "-") else None
    value = node.get("value")
    value = str(value).strip()[:24] if value not in (None, "", "null") else None
    kids = [k for k in (_clean_tree(c, depth + 1) for c in (node.get("children") or [])[:8]) if k]
    share = node.get("share")
    share = str(share).strip()[:12] if share not in (None, "", "null") else None
    out: Dict[str, Any] = {
        "id": _slug(label) or f"n{depth}",
        "label": label,
        "value": value,
        "share": share,
        "op": op if kids else None,
        "kind": "driver" if node.get("driver") else "stated",
        "role": "derived" if kids else "input",
    }
    if kids:
        out["children"] = kids
    return out


def _slug(label: str) -> str:
    from services.structure_tree import _slug as s
    return s(label)


# --------------------------------------------------------------------------- moderation
_REVIEW_SYSTEM = """You screen a note a candidate wrote on a case's worked approach. It
would appear beside the model answer, so it carries that authority.

Return STRICT JSON: {"verdict": "accept"|"reject"|"unsure", "reason": "<one short line>"}

reject when the note is: factually wrong about this case, a restatement of what the
framework already says, abuse or spam, an advertisement or link, personal information, or
so vague it teaches nothing ("good case", "nice one").
unsure when it is arguable, opinionated-but-plausible, or you cannot tell from the case.
accept only when it would genuinely help the next candidate and is consistent with the
solution given.

You are a FILTER, not the decision. A human approves what you accept."""


def review_note(body: str, case_title: str, solution: str, *, llm=None) -> Dict[str, str]:
    """Advisory screen on a contributed note. Never raises; defaults to 'unsure'.

    Deliberately cannot approve anything on its own -- the caller stores this verdict
    beside the note and a human still moves it to 'approved'. A model clearing text to be
    displayed next to that same model's framework is grading its own homework.
    """
    out = {"verdict": "unsure", "reason": ""}
    if not (body or "").strip():
        return {"verdict": "reject", "reason": "empty"}
    try:
        from services.ai_providers import openai_client
        cli = llm or openai_client()
        if cli is None:
            return out
        resp = cli.chat.completions.create(
            model=FRAMEWORK_MODEL, temperature=0, max_tokens=120,
            messages=[{"role": "system", "content": _REVIEW_SYSTEM},
                      {"role": "user", "content": (f"CASE: {case_title}\n\n"
                                                   f"SOLUTION:\n{(solution or '')[:3000]}\n\n"
                                                   f"NOTE:\n{body[:600]}")}],
        )
        data = _parse_json((resp.choices[0].message.content or "")) or {}
        v = str(data.get("verdict") or "").lower()
        if v in ("accept", "reject", "unsure"):
            out["verdict"] = v
        out["reason"] = str(data.get("reason") or "")[:200]
    except Exception as e:  # noqa: BLE001
        print(f"[framework] note review failed: {type(e).__name__}")
    return out
