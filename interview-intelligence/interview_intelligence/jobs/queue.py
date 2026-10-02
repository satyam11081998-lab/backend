"""Durable job queue on the II database (no external broker).

* enqueue() is called inside the caller's transaction, so a job exists iff the work
  that created it committed.
* claim() uses FOR UPDATE SKIP LOCKED on Postgres so N workers never take the same job.
* Failures retry with exponential backoff; after max_attempts the job is `dead` and the
  handler's on_dead hook (if any) marks the owning entity failed.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
import traceback
import uuid
from datetime import timedelta
from typing import Callable, Dict, List, Optional

from sqlalchemy import and_, or_, select, update
from sqlalchemy.orm import Session

from ..db.models import Job, utcnow
from ..db.session import db_session, is_postgres

log = logging.getLogger("ii.jobs")

Handler = Callable[[Session, dict], None]
_handlers: Dict[str, Handler] = {}
_dead_hooks: Dict[str, Callable[[Session, dict, str], None]] = {}
BACKOFF_S = [30, 120, 600, 1800, 7200, 43200]
STALE_LOCK = timedelta(minutes=10)
WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"
# Set by enqueue() so an idle worker in this process starts polling fast at once (the job
# becomes visible when the enqueuing transaction commits, a moment later).
_wake = threading.Event()


def register(kind: str, *, on_dead: Optional[Callable[[Session, dict, str], None]] = None):
    def deco(fn: Handler) -> Handler:
        _handlers[kind] = fn
        if on_dead:
            _dead_hooks[kind] = on_dead
        return fn
    return deco


def enqueue(db: Session, kind: str, payload: dict, *, dedupe_key: Optional[str] = None,
            max_attempts: int = 3, delay_s: float = 0) -> Optional[Job]:
    """Add a job in the caller's transaction. A duplicate dedupe_key of a job that is still
    queued/running is a no-op (returns None)."""
    if dedupe_key:
        existing = db.execute(select(Job).where(Job.dedupe_key == dedupe_key)).scalar_one_or_none()
        if existing is not None:
            if existing.status in ("queued", "running"):
                return None
            # finished job with the same key: recycle the key so it can run again
            existing.dedupe_key = None
            db.flush()
    job = Job(kind=kind, payload=payload, dedupe_key=dedupe_key, max_attempts=max_attempts,
              run_after=utcnow() + timedelta(seconds=delay_s))
    # Callers hold the owning row's lock (session/document), so a concurrent duplicate
    # is not expected; the UNIQUE(dedupe_key) constraint is the backstop and would fail
    # the caller's transaction loudly rather than double-run work.
    db.add(job)
    db.flush()
    _wake.set()
    return job


def _claim(db: Session, kinds: Optional[List[str]] = None, include_delayed: bool = False) -> Optional[Job]:
    now = utcnow()
    due = now + timedelta(days=3650) if include_delayed else now
    conds = [
        or_(
            and_(Job.status == "queued", Job.run_after <= due),
            and_(Job.status == "running", Job.locked_at < now - STALE_LOCK),
        )
    ]
    if kinds:
        conds.append(Job.kind.in_(kinds))
    q = select(Job).where(*conds).order_by(Job.run_after).limit(1)
    if is_postgres():
        q = q.with_for_update(skip_locked=True)
    job = db.execute(q).scalar_one_or_none()
    if job is None:
        return None
    res = db.execute(
        update(Job)
        .where(Job.id == job.id, Job.status == job.status)
        .values(status="running", locked_at=now, locked_by=WORKER_ID, attempts=Job.attempts + 1,
                updated_at=now)
    )
    if res.rowcount != 1:
        return None
    db.flush()
    db.refresh(job)
    return job


def run_one(kinds: Optional[List[str]] = None, include_delayed: bool = False) -> bool:
    """Claim and run a single job. Returns False when the queue had nothing to do."""
    with db_session() as db:
        job = _claim(db, kinds, include_delayed)
        if job is None:
            return False
        job_id, kind, payload, attempts, max_attempts = job.id, job.kind, dict(job.payload or {}), job.attempts, job.max_attempts

    handler = _handlers.get(kind)
    err: Optional[str] = None
    if handler is None:
        err = f"no handler registered for {kind}"
    else:
        try:
            # Model runs are collected and written AFTER the handler's transaction ends (commit or
            # rollback). Writing them mid-job from a second connection would wait on rows the job
            # itself holds (the session's cost_usd) — a deadlock the database cannot see — and on
            # SQLite it blocks on the single-writer lock.
            from ..ai.runner import collect_runs
            with collect_runs():
                with db_session() as db:
                    handler(db, payload)
        except Exception as e:  # noqa: BLE001 - every failure is recorded and retried
            err = f"{type(e).__name__}: {e}"
            log.warning("job %s (%s) failed: %s\n%s", job_id, kind, err, traceback.format_exc(limit=4))

    with db_session() as db:
        j = db.get(Job, job_id)
        if j is None:
            return True
        now = utcnow()
        if err is None:
            j.status, j.finished_at, j.last_error = "succeeded", now, ""
        elif attempts >= max_attempts:
            j.status, j.finished_at, j.last_error = "dead", now, err[:4000]
            hook = _dead_hooks.get(kind)
            if hook:
                try:
                    hook(db, payload, err)
                except Exception:  # noqa: BLE001
                    log.exception("on_dead hook failed for %s", kind)
        else:
            j.status = "queued"
            j.last_error = err[:4000]
            j.run_after = now + timedelta(seconds=BACKOFF_S[min(attempts - 1, len(BACKOFF_S) - 1)])
        j.locked_at, j.locked_by, j.updated_at = None, "", now
    return True


def run_pending(max_jobs: int = 200, kinds: Optional[List[str]] = None, include_delayed: bool = False) -> int:
    """Drain jobs synchronously (tests, CLI). include_delayed=True also runs jobs scheduled in
    the future — initial delays AND retry backoffs — so a failing job is retried to `dead`
    immediately. Never use include_delayed in the production worker."""
    n = 0
    while n < max_jobs and run_one(kinds, include_delayed):
        n += 1
    return n


class Worker:
    """Polls fast (poll_s) while there is work or something was just enqueued in this process,
    and slowly (idle_poll_s) otherwise — so an idle II costs one query every ~20 s, not every
    1.5 s, on a shared box. Jobs enqueued elsewhere are still picked up within idle_poll_s."""

    def __init__(self, threads: int = 2, poll_s: float = 1.5, idle_poll_s: float = 20.0,
                 active_window_s: float = 120.0):
        self.threads = threads
        self.poll_s = poll_s
        self.idle_poll_s = idle_poll_s
        self.active_window_s = active_window_s
        self._stop = threading.Event()
        self._ts: List[threading.Thread] = []

    def start(self) -> None:
        for i in range(self.threads):
            t = threading.Thread(target=self._loop, name=f"ii-worker-{i}", daemon=True)
            t.start()
            self._ts.append(t)

    def stop(self) -> None:
        self._stop.set()
        _wake.set()  # break an idle wait

    def _loop(self) -> None:
        fast_until = time.monotonic() + self.active_window_s
        while not self._stop.is_set():
            try:
                did = run_one()
            except Exception:  # noqa: BLE001
                log.exception("worker loop error")
                did = False
            if did:
                fast_until = time.monotonic() + self.active_window_s
                continue
            delay = self.poll_s if time.monotonic() < fast_until else self.idle_poll_s
            if _wake.wait(delay):
                _wake.clear()
                fast_until = time.monotonic() + self.active_window_s


def new_correlation_id() -> str:
    return uuid.uuid4().hex[:12]


def wait_idle(timeout_s: float = 10.0) -> None:  # pragma: no cover - dev helper
    end = time.time() + timeout_s
    while time.time() < end:
        with db_session() as db:
            busy = db.execute(select(Job).where(Job.status.in_(["queued", "running"]))).first()
        if not busy:
            return
        time.sleep(0.2)
