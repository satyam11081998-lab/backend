"""Periodic housekeeping: session timeouts for all users (lazy sweeps also run per request)."""

from __future__ import annotations

import logging
import threading

from sqlalchemy import select

from ..db.models import InterviewSession
from ..db.session import db_session
from ..interview_engine.lifecycle import SLOT_STATES, sweep_user
from ..interview_engine.sessions import _enqueue_assessments

log = logging.getLogger("ii.maintenance")


def sweep_all() -> int:
    n = 0
    with db_session() as db:
        users = db.execute(select(InterviewSession.user_id).where(InterviewSession.status.in_(list(SLOT_STATES)))
                           .distinct()).scalars().all()
    for uid in users:
        try:
            with db_session() as db:
                _enqueue_assessments(db, sweep_user(db, uid))
                n += 1
        except Exception:  # noqa: BLE001
            log.exception("sweep failed for a user")
    return n


class Sweeper:
    def __init__(self, every_s: float = 600.0):
        self.every_s = every_s
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._loop, name="ii-sweeper", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self.every_s):
            try:
                sweep_all()
            except Exception:  # noqa: BLE001
                log.exception("sweeper error")
