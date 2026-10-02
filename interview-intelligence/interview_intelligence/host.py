"""Host mode — run Interview Intelligence inside another FastAPI process.

II normally runs as its own service and receives identity as a signed assertion (C10).
Host mode is the low-cost alternative chosen for the first deployment: the MECE backend
mounts II under a path prefix in its own process, and gives II **only** a function that
turns the request's Authorization header into an identity (who, tier, admin, guest).

What stays independent even in host mode: II's code (this package imports nothing from the
host), its database schema and connection, prompts, model routing, scoring, admin, audit and
spend accounting. What is shared: the process (memory/CPU) and, by default, the host's AI
provider keys (read from the host's env names when II_* keys are not set).

This module must stay import-light: the host imports it at startup, and nothing heavy
(SQLAlchemy, document parsers, the routers) loads until the first request to II.

    from interview_intelligence.host import mount
    mount(app, "/ii", resolve_identity)   # resolve_identity(authorization) -> HostIdentity
    mount(app, "/ii", resolve_identity, news=recent_headlines)   # optional: real news for awareness questions
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from starlette.concurrency import run_in_threadpool

log = logging.getLogger("ii.host")


@dataclass(frozen=True)
class HostIdentity:
    user_id: str
    email: str
    tier: str  # effective tier: free | lite | pro
    is_admin: bool = False
    is_guest: bool = False


Resolver = Callable[[Optional[str]], HostIdentity]
# Optional: recent business headlines the host already collects (title, summary, source,
# published_at, category, keywords). II only reads them to ask a business-awareness question
# built on real news; without a provider the candidate picks the story instead.
NewsProvider = Callable[[], List[dict]]

_resolver: Optional[Resolver] = None
_news: Optional[NewsProvider] = None


def set_identity_resolver(fn: Optional[Resolver]) -> None:
    global _resolver
    _resolver = fn


def identity_resolver() -> Optional[Resolver]:
    return _resolver


def set_news_provider(fn: Optional[NewsProvider]) -> None:
    global _news
    _news = fn


def recent_news() -> List[dict]:
    """The host's recent headlines, or [] (no provider, or it failed — never an error)."""
    if _news is None:
        return []
    try:
        rows = _news() or []
        return [r for r in rows if isinstance(r, dict) and r.get("title")]
    except Exception:  # noqa: BLE001 - news is a nice-to-have; an interview never fails for it
        log.warning("news provider failed", exc_info=True)
        return []


def configured() -> bool:
    """II needs its own database URL and encryption key; without them it stays dormant."""
    return bool(os.environ.get("II_DATABASE_URL", "").strip())


async def _send_json(send, status: int, body: dict) -> None:
    payload = json.dumps(body).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"cache-control", b"no-store")]})
    await send({"type": "http.response.body", "body": payload})


class LazyApp:
    """ASGI app that builds the real II app on the first request (memory is only spent once
    someone actually uses Interview Intelligence) and starts II's background workers then."""

    RETRY_AFTER_S = 30.0  # after a failed start, do not retry on every request

    def __init__(self) -> None:
        self._app: Any = None
        self._lock = threading.Lock()
        self._workers: list = []
        self._failed_at = 0.0

    def _load(self):
        if self._app is None:
            with self._lock:
                if self._app is None:
                    from .main import create_app, start_background
                    app = create_app(host_mode=True)
                    self._workers = start_background()
                    self._app = app
                    log.info("Interview Intelligence loaded in host mode")
        return self._app

    def stop(self) -> None:
        for w in self._workers:
            try:
                w.stop()
            except Exception:  # noqa: BLE001
                pass

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":  # the host owns the lifespan
            return
        if not configured():
            if scope["type"] == "http":
                await _send_json(send, 503, {"error": {
                    "code": "not_configured",
                    "message": "Interview Intelligence is not available yet."}})
            return
        app = self._app
        if app is None:
            recently_failed = self._failed_at and time.monotonic() - self._failed_at < self.RETRY_AFTER_S
            if not recently_failed:
                try:
                    # First use imports SQLAlchemy, parsers and the routers and touches the
                    # database: do it off the event loop so the host keeps serving meanwhile.
                    app = await run_in_threadpool(self._load)
                except Exception:  # noqa: BLE001 - never take the host down with us
                    self._failed_at = time.monotonic()
                    log.exception("Interview Intelligence failed to start")
                    app = None
            if app is None:
                if scope["type"] == "http":
                    await _send_json(send, 503, {"error": {
                        "code": "unavailable",
                        "message": "Interview Intelligence is temporarily unavailable."}})
                return
        await app(scope, receive, send)


def mount(host_app, path: str, resolver: Resolver, *, news: Optional[NewsProvider] = None) -> LazyApp:
    """Mount II under `path` of a FastAPI/Starlette host and stop II's workers on host shutdown."""
    os.environ.setdefault("II_WORKER_THREADS", "1")  # share the host's box politely
    os.environ.setdefault("II_DB_POOL_SIZE", "3")
    set_identity_resolver(resolver)
    set_news_provider(news)
    lazy = LazyApp()
    host_app.mount(path, lazy)

    original = host_app.router.lifespan_context

    @asynccontextmanager
    async def chained(app):
        try:
            async with original(app) as state:
                yield state
        finally:
            lazy.stop()

    host_app.router.lifespan_context = chained
    return lazy
