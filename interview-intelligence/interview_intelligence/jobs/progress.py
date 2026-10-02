"""Live progress of slow background work — document analysis, building an interview, writing the
report — so the person waiting sees what is actually happening instead of a bare spinner.

Every step reported here is a real stage of the work, reported by the code doing it; facts are
real findings (counts from the parsed CV, the competency match...). The only estimate is how far
along the CURRENT step is: a model call cannot report its own progress, so it is projected from
how long the same prompt took recently (median of past model runs), and capped below 100 % until
the step really ends.

Kept in memory: progress is ephemeral, and in host mode the worker runs in the same process as the
API. A restarted process simply has no entry, and the UI falls back to a plain "working" state.
Nothing here may ever break the work it describes: every public function swallows its errors.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

log = logging.getLogger("ii.progress")

_lock = threading.Lock()
_runs: Dict[str, dict] = {}
_eta_cache: Dict[str, Tuple[float, float]] = {}  # prompt id -> (seconds, fetched at)

MAX_ENTRIES = 500
KEEP_DONE_S = 900.0
ETA_TTL_S = 600.0
#: A running step never shows more than this share of itself as done.
STEP_CAP = 0.92

#: Typical seconds per call when there is no history yet (real models, not the simulator).
DEFAULT_ETA_S = {
    "cv_parser": 25.0, "jd_parser": 15.0, "role_classifier": 5.0, "competency_mapper": 18.0,
    "rubric_builder": 18.0, "question_generator": 12.0, "blueprint_qa": 8.0, "evidence_extractor": 6.0,
    "competency_evaluator": 7.0, "assessment_qa": 10.0, "feedback_writer": 30.0, "feedback_qa": 8.0,
}

StepSpec = Tuple[str, str, float]  # (id, label, expected seconds)


def _now() -> float:
    return time.monotonic()


def _prune() -> None:
    now = _now()
    stale = [k for k, r in _runs.items() if r.get("ended") and now - r["ended"] > KEEP_DONE_S]
    for k in stale:
        _runs.pop(k, None)
    if len(_runs) > MAX_ENTRIES:
        for k, _ in sorted(_runs.items(), key=lambda kv: kv[1]["started"])[: len(_runs) - MAX_ENTRIES]:
            _runs.pop(k, None)


def _step(spec: StepSpec) -> dict:
    sid, label, eta = spec
    return {"id": sid, "label": label, "eta": max(0.3, float(eta or 0.3)), "state": "todo", "detail": "",
            "facts": [], "started": None, "ended": None, "portion": 0.0}


def start(key: str, steps: Sequence[StepSpec]) -> None:
    """(Re)start tracking `key` with its planned steps."""
    try:
        with _lock:
            _prune()
            _runs[key] = {"steps": [_step(s) for s in steps], "started": _now(), "ended": None, "failed": False}
    except Exception:  # noqa: BLE001
        log.exception("progress.start")


def insert(key: str, steps: Sequence[StepSpec], *, before: Optional[str] = None) -> None:
    """Add steps known only once the work is under way (e.g. one per interview section)."""
    try:
        with _lock:
            r = _runs.get(key)
            if not r:
                return
            new = [_step(s) for s in steps]
            idx = next((i for i, s in enumerate(r["steps"]) if s["id"] == before), len(r["steps"]))
            r["steps"][idx:idx] = new
    except Exception:  # noqa: BLE001
        log.exception("progress.insert")


def set_eta(key: str, step_id: str, eta_s: float) -> None:
    try:
        with _lock:
            s = _find(key, step_id)
            if s:
                s["eta"] = max(0.3, float(eta_s))
    except Exception:  # noqa: BLE001
        log.exception("progress.set_eta")


def begin(key: str, step_id: str, *, detail: str = "", eta_s: Optional[float] = None) -> None:
    """`step_id` is now running; every step before it is done."""
    try:
        with _lock:
            r = _runs.get(key)
            if not r:
                return
            now = _now()
            ids = [s["id"] for s in r["steps"]]
            if step_id not in ids:
                r["steps"].append(_step((step_id, step_id, eta_s or 1.0)))
                ids.append(step_id)
            target = ids.index(step_id)
            for i, s in enumerate(r["steps"]):
                if i < target and s["state"] != "done":
                    s["state"], s["ended"] = "done", now
                    s["started"] = s["started"] or now
            s = r["steps"][target]
            if s["state"] != "active":
                s["state"], s["started"] = "active", now
            if eta_s is not None:
                s["eta"] = max(0.3, float(eta_s))
            s["detail"] = detail or s["detail"]
    except Exception:  # noqa: BLE001
        log.exception("progress.begin")


def detail(key: str, text: str) -> None:
    """What the running step is doing right now (e.g. "Section 2 of 4: CV deep-dive")."""
    try:
        with _lock:
            s = _active(key)
            if s:
                s["detail"] = text[:160]
    except Exception:  # noqa: BLE001
        log.exception("progress.detail")


def portion(key: str, done: int, total: int) -> None:
    """A step made of countable parts (6 of 9 skills scored): use the real count, not only time."""
    try:
        with _lock:
            s = _active(key)
            if s and total > 0:
                s["portion"] = max(s["portion"], min(1.0, done / total))
    except Exception:  # noqa: BLE001
        log.exception("progress.portion")


def fact(key: str, text: str, *, step_id: Optional[str] = None) -> None:
    """A real finding, shown under its step (e.g. "9 skills this role needs, 6 clearly backed by your CV")."""
    try:
        with _lock:
            s = _find(key, step_id) if step_id else _active(key)
            if s and text and text not in s["facts"]:
                s["facts"].append(text[:160])
    except Exception:  # noqa: BLE001
        log.exception("progress.fact")


def finish(key: str) -> None:
    try:
        with _lock:
            r = _runs.get(key)
            if not r:
                return
            now = _now()
            for s in r["steps"]:
                if s["state"] != "done":
                    s["state"], s["ended"] = "done", now
                    s["started"] = s["started"] or now
            r["ended"] = now
    except Exception:  # noqa: BLE001
        log.exception("progress.finish")


def fail(key: str) -> None:
    try:
        with _lock:
            r = _runs.get(key)
            if r:
                r["failed"], r["ended"] = True, _now()
    except Exception:  # noqa: BLE001
        log.exception("progress.fail")


def forget(key: str) -> None:
    with _lock:
        _runs.pop(key, None)


def reset_for_tests() -> None:
    with _lock:
        _runs.clear()
        _eta_cache.clear()


def _find(key: str, step_id: Optional[str]) -> Optional[dict]:
    r = _runs.get(key)
    if not r:
        return None
    return next((s for s in r["steps"] if s["id"] == step_id), None)


def _active(key: str) -> Optional[dict]:
    r = _runs.get(key)
    if not r:
        return None
    return next((s for s in r["steps"] if s["state"] == "active"), None)


def view(key: str) -> Optional[dict]:
    """Public, JSON-safe progress for `key` (None if nothing is tracked in this process)."""
    try:
        with _lock:
            r = _runs.get(key)
            if not r:
                return None
            now = _now()
            total = sum(s["eta"] for s in r["steps"]) or 1.0
            acc = 0.0
            active = None
            for s in r["steps"]:
                if s["state"] == "done":
                    acc += s["eta"]
                elif s["state"] == "active":
                    active = s
                    timed = (now - (s["started"] or now)) / s["eta"]
                    acc += s["eta"] * min(STEP_CAP, max(s["portion"], timed))
            done = bool(r["ended"]) and not r["failed"]
            pct = 100 if done else min(99, int(round(100 * acc / total)))
            return {
                "steps": [{"id": s["id"], "label": s["label"], "state": s["state"], "detail": s["detail"],
                           "facts": list(s["facts"])} for s in r["steps"]],
                "pct": pct,
                "done": done,
                "failed": bool(r["failed"]),
                "elapsed_s": round(now - r["started"], 1),
                # lets the page move the bar smoothly between polls without guessing
                "step_elapsed_s": round(now - active["started"], 1) if active and active["started"] else 0.0,
                "step_eta_s": round(active["eta"], 1) if active else 0.0,
                "step_weight": round(active["eta"] / total, 4) if active else 0.0,
            }
    except Exception:  # noqa: BLE001
        log.exception("progress.view")
        return None


def eta_for(db, prompt_id: str, *, calls: int = 1) -> float:
    """Expected seconds for `calls` runs of `prompt_id`: the median of its recent real runs
    (cached), else a typical default."""
    calls = max(1, int(calls))
    per = DEFAULT_ETA_S.get(prompt_id, 8.0)
    try:
        hit = _eta_cache.get(prompt_id)
        if hit and time.monotonic() - hit[1] < ETA_TTL_S:
            return hit[0] * calls
        if db is not None:
            from sqlalchemy import select
            from ..db.models import ModelRun
            with db.begin_nested():  # a failed read must never poison the job's transaction
                rows = db.execute(
                    select(ModelRun.latency_ms).where(ModelRun.prompt_id == prompt_id,
                                                      ModelRun.status.in_(("ok", "repaired")),
                                                      ModelRun.provider != "simulated")
                    .order_by(ModelRun.created_at.desc()).limit(25)).scalars().all()
            vals = sorted(int(v) for v in rows if v and v > 0)
            if len(vals) >= 3:
                per = vals[len(vals) // 2] / 1000.0
        _eta_cache[prompt_id] = (per, time.monotonic())
    except Exception:  # noqa: BLE001
        log.debug("progress.eta_for fell back to the default for %s", prompt_id, exc_info=True)
    return per * calls


SENIORITY_LABEL = {"intern": "Internship", "entry": "Entry level", "mid": "Mid-level", "senior": "Senior",
                   "lead": "Lead", "executive": "Executive"}


def seniority_label(level: str) -> str:
    return SENIORITY_LABEL.get((level or "").strip().lower(), "")


def clean_chip(term: str) -> str:
    t = " ".join(str(term or "").split())
    return t if 1 < len(t) <= 40 else ""


def plural(n: int, one: str, many: Optional[str] = None) -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


def join(parts: Iterable[str]) -> str:
    return ", ".join(p for p in parts if p)


__all__: List[str] = ["start", "insert", "begin", "detail", "fact", "portion", "finish", "fail", "view", "eta_for",
                      "set_eta", "plural", "join", "forget", "reset_for_tests", "seniority_label", "clean_chip"]
