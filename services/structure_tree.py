"""The live structure tree — the candidate's own decomposition, drawn as they talk.

WHY IT IS BUILT THIS WAY

The obvious implementation is a second model call per turn: "read this conversation and
return the tree as JSON". That is one extra round trip on every single turn of a live
conversation, which is the one place in this product where latency is felt. It is also the
least reliable option: a second model re-reading the transcript invents branches the
candidate never said, and it drifts turn to turn because nothing anchors it to what it
emitted last time.

So the tree is not extracted. It is REPORTED, by the model that is already running, on the
control tag it already emits:

    <<mode=probe; intervention=targeted_question; tree=population>car owners#40 lakh>>

Zero extra calls, roughly fifteen output tokens, and StreamShapeGuard already strips the
tag before a single character reaches the candidate. The model reports only what THIS turn
added; this module accumulates. That is what keeps it stable: the model never restates the
whole tree, so it cannot contradict itself, and a turn that adds nothing emits nothing.

GRAMMAR (see interviewer_mode.TREE_RULES for what the model is told)

    tree=<parent>><child>#<value>|<parent>><child>

  - edges separated by  |     ... because parse_control_tag already splits the tag on [;,]
  - parent and child by >     ... a top-level node is written with an empty parent: >revenue
  - an optional figure by #   ... "#40 lakh", shown under the label, never parsed as a number

Everything here is a pure function over (state, tag). No database, no network, no model.
That is deliberate: it makes the whole feature testable without a live conversation, and it
means a malformed tag can only ever produce a smaller tree, never an exception on a turn.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Caps. A model that misbehaves must degrade the picture, never the page: an unbounded
# tree is an unbounded SVG, and this renders inside a live conversation.
MAX_NODES = 24
MAX_DEPTH = 5
MAX_LABEL = 42
MAX_VALUE = 18
MAX_EDGES_PER_TURN = 6

_WS = re.compile(r"\s+")
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def _norm(label: str) -> str:
    """Display form: collapsed whitespace, capped, first letter raised.

    parse_control_tag lowercases every tag value, so labels arrive flat. Sentence case is
    how a consulting tree is written anyway, and raising only the first letter keeps the
    candidate's own words intact — "car owners" stays "Car owners", never "Car Owners",
    which would read as house style imposed on what they said.
    """
    t = _WS.sub(" ", (label or "").strip())[:MAX_LABEL].strip(" -–—:>#|")
    return (t[0].upper() + t[1:]) if t else ""


def _slug(label: str) -> str:
    """Identity is the normalised label, so the same branch named twice is one node and
    dedup costs nothing. Short hash-free slug keeps the id readable in logs."""
    return _SLUG_STRIP.sub("-", (label or "").strip().lower()).strip("-")[:48]


def parse_edges(raw: str) -> List[Dict[str, str]]:
    """'a>b#2 cr|>revenue' -> [{parent:'a', label:'b', value:'2 cr'}, {parent:'', ...}].

    Tolerant by design. Anything unparseable is dropped silently rather than raised: this
    runs on the reply path of a live turn, and a bad tag must cost a branch, not the turn.
    """
    out: List[Dict[str, str]] = []
    for part in (raw or "").split("|"):
        part = part.strip()
        if not part:
            continue
        value = ""
        if "#" in part:
            part, value = part.split("#", 1)
            value = _WS.sub(" ", value.strip())[:MAX_VALUE]
        parent, _, child = part.rpartition(">")
        label = _norm(child)
        if not label:
            continue
        out.append({"parent": _norm(parent), "label": label, "value": value})
        if len(out) >= MAX_EDGES_PER_TURN:
            break
    return out


def fold_turn(state: Optional[Dict[str, Any]], tag: Dict[str, str]) -> Dict[str, Any]:
    """Accumulate this turn's reported edges into session_state['structure'].

    Stored flat — [{id, label, value, parent, kind}] — not nested. A flat list is what makes
    an append cheap and idempotent; nesting is a view, computed at read time by to_tree().
    """
    st = dict(state or {})
    edges = parse_edges(tag.get("tree", ""))
    if not edges:
        return st

    nodes: List[Dict[str, Any]] = list(st.get("structure") or [])
    by_id = {n["id"]: n for n in nodes}
    active = st.get("structure_active")

    for e in edges:
        nid = _slug(e["label"])
        if not nid:
            continue
        pid = _slug(e["parent"]) if e["parent"] else ""
        # A parent nobody has introduced yet becomes a top-level node rather than being
        # dropped: candidates name a sub-branch before its header all the time, and losing
        # the branch is worse than hanging it off the root.
        if pid and pid not in by_id and pid != nid:
            if len(by_id) < MAX_NODES:
                by_id[pid] = {"id": pid, "label": e["parent"], "value": "",
                              "parent": "", "kind": "stated"}
                nodes.append(by_id[pid])
        if pid == nid:
            pid = ""
        # Cycles: a child may not be its own ancestor. Without this one check a model that
        # reports a>b then b>a produces infinite recursion in to_tree().
        if pid and _is_descendant(by_id, pid, nid):
            pid = ""

        if nid in by_id:
            n = by_id[nid]
            if e["value"]:
                n["value"] = e["value"]          # a later figure refines an earlier one
            if pid and not n.get("parent"):
                n["parent"] = pid                # a node first seen loose finds its branch
            n["kind"] = "stated"
        else:
            if len(by_id) >= MAX_NODES:
                continue
            n = {"id": nid, "label": e["label"], "value": e["value"],
                 "parent": pid, "kind": "stated"}
            by_id[nid] = n
            nodes.append(n)
        active = nid

    st["structure"] = nodes
    st["structure_active"] = active
    return st


def _is_descendant(by_id: Dict[str, Dict[str, Any]], node_id: str, maybe_ancestor: str) -> bool:
    seen = set()
    cur = node_id
    while cur and cur not in seen:
        seen.add(cur)
        if cur == maybe_ancestor:
            return True
        cur = (by_id.get(cur) or {}).get("parent") or ""
    return False


def to_tree(state: Optional[Dict[str, Any]], root_label: str = "") -> Optional[Dict[str, Any]]:
    """Flat nodes -> the nested StructureNode the component renders, or None when the
    candidate has not put anything on the table yet (the component draws its own empty
    state, so None is a valid, expected answer — not an error)."""
    nodes = list((state or {}).get("structure") or [])
    if not nodes:
        return None

    by_id = {n["id"]: {"id": n["id"], "label": n["label"],
                       "value": n.get("value") or None,
                       "kind": n.get("kind") or "stated",
                       "children": []} for n in nodes}

    tops: List[Dict[str, Any]] = []
    for n in nodes:
        pid = n.get("parent") or ""
        node = by_id[n["id"]]
        if pid and pid in by_id and pid != n["id"]:
            by_id[pid]["children"].append(node)
        else:
            tops.append(node)

    _prune(tops, 1)
    for node in by_id.values():
        if not node["children"]:
            node.pop("children")

    # One synthetic root so the diagram is always a single tree. Its label is the case's
    # own question, truncated — the thing every branch is ultimately answering.
    if len(tops) == 1 and not root_label:
        return tops[0]
    return {"id": "__root", "label": _norm(root_label) or "The question",
            "kind": "stated", "children": tops}


def _prune(nodes: List[Dict[str, Any]], depth: int) -> None:
    """Depth cap, applied to the view rather than to storage: a branch that is too deep to
    draw is still worth keeping in state, because the interviewer reads state."""
    for n in nodes:
        if depth >= MAX_DEPTH:
            n["children"] = []
        else:
            _prune(n["children"], depth + 1)


def active_id(state: Optional[Dict[str, Any]]) -> Optional[str]:
    return (state or {}).get("structure_active") or None


# =============================================================================
# INFERRING THE EDGES WHEN THE MODEL DOES NOT REPORT THEM
# =============================================================================
#
# The tag-reporting path above is free -- the interviewer is already running and the key
# rides along. It is also a COMPLIANCE DEPENDENCY, and measuring it on the live site showed
# gpt-4o-mini simply not emitting `tree=`, twice, through two rounds of prompt work. A
# feature that silently produces nothing when a small model ignores one instruction is the
# wrong foundation, however cheap it is.
#
# So the tag is now the fast path, not the only path. When a turn arrives without it, a
# dedicated call infers the same edges from the last few turns.
#
# WHY A SECOND CALL IS ACCEPTABLE HERE AND WAS NOT BEFORE
#
# I rejected a second model call earlier on latency grounds. That reasoning was right about
# the CRITICAL PATH and wrong as a blanket rule. This runs inside _fold_session_state, which
# _after_turn already executes once the reply has finished streaming -- the candidate is
# reading the interviewer's line while it happens. It cannot delay a turn, because the turn
# is over. The diagram therefore lags by one turn, which for a picture that builds up as you
# talk is not a defect.
#
# It returns the SAME grammar the tag uses, so parse_edges and its tests are unchanged: one
# parser, one format, two sources.

_INFER_SYSTEM = (
    "You extract the structure of a candidate's estimate. Output ONE line and nothing else.\n"
    "Format: parent>child#figure, several joined by |, a top-level node written with nothing "
    "before the > . Omit #figure when the candidate gave no number for that node.\n"
    "Use the candidate's own words, three or four words per node. Report ONLY branches the "
    "LAST candidate message added. If it added none, output exactly: none\n"
    "Examples:\n"
    "candidate: \"start with chennai population, 1.2 crore\" -> >population#1.2 crore\n"
    "candidate: \"a third of them own cars, so 40 lakh\" (known: population) -> "
    "population>car owners#40 lakh\n"
    "candidate: \"what do you mean by pre-owned?\" -> none"
)


def infer_edges(transcript, new_user_message: str,
                state: Optional[Dict[str, Any]] = None, *, llm=None) -> str:
    """Infer this turn's edges. Returns a `tree=` value, or "" when there is nothing.

    Never raises and never retries: this is a side-channel on a background path, so a
    failure means the diagram waits for the next turn, nothing more.
    """
    msg = (new_user_message or "").strip()
    if not msg:
        return ""
    try:
        from services.ai_providers import openai_client
        import os
        cli = llm or openai_client()
        if cli is None:
            return ""
        known = [n["label"] for n in ((state or {}).get("structure") or [])][:12]
        recent = [t.get("content") or "" for t in (transcript or []) if t.get("role") == "user"][-2:]
        user = ""
        if known:
            user += "known nodes: " + ", ".join(known) + "\n"
        if recent:
            user += "earlier: " + " | ".join(r[:160] for r in recent) + "\n"
        user += "candidate: " + msg[:400]
        resp = cli.chat.completions.create(
            model=(os.getenv("STRUCTURE_MODEL") or "gpt-4o-mini").strip(),
            messages=[{"role": "system", "content": _INFER_SYSTEM},
                      {"role": "user", "content": user}],
            temperature=0,
            max_tokens=60,
        )
        out = (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001
        print(f"[structure] infer failed: {type(e).__name__}")
        return ""
    # The model may wrap it, prefix "tree=", or answer "none".
    out = out.strip().strip("`").strip()
    if out.lower().startswith("tree="):
        out = out[5:]
    if not out or out.lower() in ("none", "none.", "-"):
        return ""
    out = out.splitlines()[0].strip()
    # Anything that survives still goes through parse_edges, which caps and validates it.
    return out if parse_edges(out) else ""
