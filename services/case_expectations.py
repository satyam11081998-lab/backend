"""
What a case expects, and what candidates reliably get wrong on it — the per-case memory
that sharpens the CONVERSATION, and that grows with every attempt.

WHY THIS EXISTS
---------------
services/session_signals.py classifies the CANDIDATE: intent, frustration, stuckness. That
set is infinite. Every unseen phrasing is a wrong read waiting to happen, and every fix is
a patch on a tail with no end — measured 2026-11-02, the regex layer mis-read four of
seven realistically-phrased turns, and the model only gets consulted on about 10% of them.

Matching against the CASE inverts the problem. One case has perhaps eight things a good
answer contains and a dozen things people reliably get wrong, and THAT set is finite and
knowable. A three-word fragment, a typo, hinglish or something nobody predicted all
resolve the same way: which expectation does this touch, which misconception does it look
like, what is still missing. (Expectation–misconception tailored dialogue — Graesser et
al.'s AutoTutor work, where tutors anticipate both before the conversation starts.)

AND IT IMPROVES WITH USE
------------------------
Every attempt records, on its own session_state, which expectations it touched and which
misconceptions it fell into. At submit those are folded into counters on the shared table.
The next candidate on the same case meets an interviewer that knows where people fall
over, and can spend its one cross-question on the branch that actually costs people marks.
No weights change. The evidence simply accumulates.

A misconception OBSERVED rather than authored lands as status='candidate' and does nothing
until a human promotes it. A loop that promotes its own guesses drifts, and drift here is
invisible — it would quietly teach the interviewer to chase an error that does not exist.

COST
----
Zero extra model calls and zero extra round trips on a turn:
  * the list is read once per case and cached in-process (CASE_EXPECTATIONS_TTL_S);
  * the match is reported by the interviewer in the control tag it ALREADY emits
    (`<<... touched=e1,e3 missed=m2 ...>>`), which StreamShapeGuard already strips before
    a single token reaches the candidate;
  * counters are written once, at submit, not per turn.

EVERY function is defensive. A missing table, a DB outage or a malformed tag degrades to
"no expectations", which is exactly how the interviewer behaved before this existed. This
side-channel must never be able to break a turn or a submit.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- tunables
CASE_EXPECTATIONS_TTL_S = float(os.getenv("CASE_EXPECTATIONS_TTL_S", "300") or 300)
# How many to put in front of the model. The packet is read on every turn, so this is a
# latency and attention budget, not a storage one: a long list buries the useful entries.
MAX_IN_PACKET = int(os.getenv("CASE_EXPECTATIONS_IN_PACKET", "8") or 8)
BOOTSTRAP_MODEL = os.getenv("CASE_EXPECTATIONS_MODEL", "gpt-4o-mini") or "gpt-4o-mini"

_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_cache_lock = threading.Lock()


def _sb():
    try:
        from services.supabase_client import get_supabase_client
        return get_supabase_client()
    except Exception:  # noqa: BLE001
        return None


def enabled() -> bool:
    """Off by default until the migration has run and a human has looked at the seed."""
    return (os.getenv("CASE_EXPECTATIONS", "on") or "on").strip().lower() not in ("off", "0", "false", "no")


# --------------------------------------------------------------------------- read
def load(case_id: Optional[str], case_type: Optional[str]) -> List[Dict[str, Any]]:
    """Active rows for this case, plus the generic rows for its type.

    Cached per (case_id, case_type) for CASE_EXPECTATIONS_TTL_S. Returns [] on any
    failure — the caller then behaves exactly as it did before this module existed.
    """
    if not enabled():
        return []
    key = f"{case_id or '-'}::{(case_type or '-').lower()}"
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < CASE_EXPECTATIONS_TTL_S:
            return hit[1]

    rows: List[Dict[str, Any]] = []
    sb = _sb()
    if sb is not None:
        cols = "id, kind, label, cue, probe, seen_count, hit_count, helped_count, case_id"
        for q in (
            (lambda: sb.table("case_expectations").select(cols)
             .eq("case_id", case_id).eq("status", "active").limit(60).execute()) if case_id else None,
            (lambda: sb.table("case_expectations").select(cols)
             .is_("case_id", "null").eq("case_type", (case_type or "").lower())
             .eq("status", "active").limit(60).execute()) if case_type else None,
        ):
            if q is None:
                continue
            try:
                rows += list((q().data) or [])
            except Exception as e:  # noqa: BLE001 - table may not be migrated yet
                print(f"[expectations] read skipped: {type(e).__name__}: {str(e)[:90]}")

    # Case-specific first, then the ones people actually fall into most often. A
    # misconception nobody has hit is worth less of the packet than one that is costing
    # every second candidate the case.
    rows.sort(key=lambda r: (0 if r.get("case_id") else 1, -int(r.get("hit_count") or 0)))
    with _cache_lock:
        _cache[key] = (now, rows)
    return rows


def invalidate(case_id: Optional[str] = None) -> None:
    with _cache_lock:
        if case_id is None:
            _cache.clear()
        else:
            for k in [k for k in _cache if k.startswith(f"{case_id}::")]:
                _cache.pop(k, None)


# --------------------------------------------------------------------------- packet
def packet_block(case_id: Optional[str], case_type: Optional[str],
                 state: Optional[dict] = None) -> Dict[str, Any]:
    """The slice of the case's answer space the interviewer should have in mind.

    `open` is what a good answer still has not touched — that is where a cross-question
    buys the most. `watch_for` is what people fall into on this case, most frequent first.
    Both carry short ids so the model can report what it saw in the control tag.
    """
    rows = load(case_id, case_type)
    if not rows:
        return {}
    seen = set((state or {}).get("expectations_touched") or [])
    out: Dict[str, Any] = {}
    opens = [{"id": _short_id(r), "is": r["label"]}
             for r in rows if r["kind"] == "expectation" and _short_id(r) not in seen]
    watch = [{"id": _short_id(r), "looks_like": r["label"],
              **({"ask": r["probe"]} if (r.get("probe") or "").strip() else {})}
             for r in rows if r["kind"] == "misconception"]
    if opens:
        out["not_covered_yet"] = opens[:MAX_IN_PACKET]
    if watch:
        out["common_errors_on_this_case"] = watch[:MAX_IN_PACKET]
    return out


def _short_id(row: Dict[str, Any]) -> str:
    """A tag-safe handle. The control tag is parsed on [;,] and lowercased, so the id has
    to survive that: 8 hex characters of the row id, prefixed by kind."""
    rid = str(row.get("id") or "")
    return ("e" if row.get("kind") == "expectation" else "m") + rid.replace("-", "")[:8]


# --------------------------------------------------------------------------- fold
_ID_RE = re.compile(r"[em][0-9a-f]{8}")


def fold_turn(state: Optional[dict], tag: Optional[Dict[str, str]],
              helped: bool = False) -> Dict[str, Any]:
    """Record what this turn touched onto the attempt's own session_state.

    No database write: a turn is the latency-critical path, and the counters only have to
    be right at submit. Unknown or malformed ids are dropped rather than stored, so a
    model that invents a handle cannot pollute the bank.
    """
    s = dict(state or {})
    if not tag:
        return s
    touched = set(s.get("expectations_touched") or [])
    missed = set(s.get("misconceptions_hit") or [])
    helped_ids = set(s.get("expectations_helped") or [])
    for field, bucket in (("touched", touched), ("missed", missed), ("hit", missed)):
        raw = (tag.get(field) or "").strip().lower()
        if not raw:
            continue
        found = _ID_RE.findall(raw)
        bucket.update(found)
        if helped and field in ("missed", "hit"):
            helped_ids.update(found)
    s["expectations_touched"] = sorted(touched)
    s["misconceptions_hit"] = sorted(missed)
    if helped_ids:
        s["expectations_helped"] = sorted(helped_ids)
    return s


def coverage(state: Optional[dict], case_id: Optional[str],
             case_type: Optional[str]) -> Dict[str, Any]:
    """What the attempt covered and what it never reached — for the debrief.

    Deliberately NO percentage and NO percentile: on a Whiteboard run the candidate is
    never shown a number (owner, 2026-10-29), and a count of branches is a score wearing a
    different hat.
    """
    rows = load(case_id, case_type)
    if not rows:
        return {}
    seen = set((state or {}).get("expectations_touched") or [])
    hit = set((state or {}).get("misconceptions_hit") or [])
    return {
        "covered": [r["label"] for r in rows if r["kind"] == "expectation" and _short_id(r) in seen],
        "never_reached": [r["label"] for r in rows if r["kind"] == "expectation" and _short_id(r) not in seen],
        "fell_into": [r["label"] for r in rows if r["kind"] == "misconception" and _short_id(r) in hit],
    }


# --------------------------------------------------------------------------- the loop
def commit_attempt(case_id: Optional[str], case_type: Optional[str],
                   state: Optional[dict]) -> None:
    """Fold one finished attempt into the shared counters. Called at submit, once.

    Best-effort to the point of silence: this runs inside a scored submit, and a counter
    is never worth failing a candidate's result for.
    """
    if not enabled():
        return
    rows = load(case_id, case_type)
    if not rows:
        return
    seen = set((state or {}).get("expectations_touched") or [])
    hit = set((state or {}).get("misconceptions_hit") or [])
    helped = set((state or {}).get("expectations_helped") or [])
    if not (seen or hit):
        return
    sb = _sb()
    if sb is None:
        return
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for r in rows:
        sid = _short_id(r)
        touched = sid in seen or sid in hit
        if not touched:
            continue
        try:
            sb.table("case_expectations").update({
                "seen_count": int(r.get("seen_count") or 0) + 1,
                "hit_count": int(r.get("hit_count") or 0) + (1 if sid in hit or (r["kind"] == "expectation" and sid in seen) else 0),
                "helped_count": int(r.get("helped_count") or 0) + (1 if sid in helped else 0),
                "last_seen_at": now,
                "updated_at": now,
            }).eq("id", r["id"]).execute()
        except Exception as e:  # noqa: BLE001
            print(f"[expectations] counter skipped id={r.get('id')}: {type(e).__name__}")
            return
    invalidate(case_id)


def propose(case_id: Optional[str], case_type: Optional[str], kind: str,
            label: str, cue: str = "", probe: str = "") -> bool:
    """File something observed in a real session for a human to look at.

    Lands as status='candidate' and influences nothing until promoted. This is the only
    way the bank grows on its own, and the gate is the whole point: an automatic loop that
    promotes its own guesses teaches the interviewer to chase errors nobody makes.
    """
    if not enabled() or kind not in ("expectation", "misconception"):
        return False
    label = re.sub(r"\s+", " ", (label or "")).strip()[:160]
    if len(label) < 8:
        return False
    sb = _sb()
    if sb is None:
        return False
    try:
        sb.table("case_expectations").insert({
            "case_id": case_id, "case_type": (case_type or "").lower() or None,
            "kind": kind, "label": label, "cue": (cue or "")[:300],
            "probe": (probe or "")[:300], "source": "observed", "status": "candidate",
        }).execute()
        return True
    except Exception as e:  # noqa: BLE001 - a duplicate label hits the unique index; fine
        print(f"[expectations] propose skipped: {type(e).__name__}: {str(e)[:90]}")
        return False


# --------------------------------------------------------------------------- bootstrap
_BOOTSTRAP_SYSTEM = (
    "You prepare an interviewer's private notes for ONE case. From the case and its model "
    "solution, list what a good answer must contain, and the mistakes candidates actually "
    "make on a case like this.\n"
    "Return JSON only: {\"expectations\":[{\"label\":\"...\"}], \"misconceptions\":[{\"label\":\"...\","
    "\"cue\":\"...\",\"probe\":\"...\"}]}\n"
    "label: one short concrete clause in the candidate's own terms, not a lecture "
    "(\"splits revenue into price x volume\", \"crore and million used interchangeably\").\n"
    "cue: what it looks like in a candidate's message.\n"
    "probe: ONE question that exposes the error WITHOUT naming the answer or the fix. "
    "\"Crore or million?\" is a probe; \"Have you considered the cost side?\" is a hint "
    "wearing a question mark and is wrong.\n"
    "At most 8 of each. Indian context: figures in lakh and crore."
)


def bootstrap(case_id: str, case_content: str, case_type: str,
              solution: str = "", llm=None) -> int:
    """One model call, offline, to seed a case from its own solution. Returns rows written.

    Deliberately NOT called from a turn or a submit: this is an admin/CLI action, because
    it costs a call and its output deserves eyes before it shapes live conversations.
    Rows land as source='solution', status='candidate' for exactly that reason.
    """
    if not enabled() or not (case_content or "").strip():
        return 0
    try:
        from services.ai_providers import openai_client
        cli = llm or openai_client()
        if cli is None:
            return 0
        user = (f"CASE TYPE: {case_type}\n\nCASE:\n{(case_content or '')[:3000]}\n\n"
                f"MODEL SOLUTION:\n{(solution or '(none stored)')[:3000]}")
        resp = cli.chat.completions.create(
            model=BOOTSTRAP_MODEL, temperature=0.0, max_tokens=900,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": _BOOTSTRAP_SYSTEM},
                      {"role": "user", "content": user}])
        data = json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:  # noqa: BLE001
        print(f"[expectations] bootstrap failed case={case_id}: {type(e).__name__}: {str(e)[:90]}")
        return 0

    sb = _sb()
    if sb is None:
        return 0
    written = 0
    for kind, key in (("expectation", "expectations"), ("misconception", "misconceptions")):
        for item in (data.get(key) or [])[:8]:
            label = re.sub(r"\s+", " ", str((item or {}).get("label") or "")).strip()[:160]
            if len(label) < 8:
                continue
            try:
                sb.table("case_expectations").insert({
                    "case_id": case_id, "case_type": (case_type or "").lower() or None,
                    "kind": kind, "label": label,
                    "cue": str((item or {}).get("cue") or "")[:300],
                    "probe": str((item or {}).get("probe") or "")[:300],
                    "source": "solution", "status": "candidate",
                }).execute()
                written += 1
            except Exception:  # noqa: BLE001 - duplicate label, or not migrated
                continue
    invalidate(case_id)
    return written
