"""MECE Interview Intelligence — FastAPI application (independent service).

    uvicorn interview_intelligence.main:app --host 0.0.0.0 --port $PORT --workers 1
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from .config import get_settings
from .errors import IIError
from .versions import versions


def _register_job_handlers() -> None:
    # Importing registers @register(...) handlers with the queue.
    from .documents import analysis  # noqa: F401
    from .drive_integration import sync  # noqa: F401
    from .evidence_engine import extractor  # noqa: F401
    from .interview_engine import blueprint  # noqa: F401
    from .report_engine import assessment  # noqa: F401


def start_background() -> list:
    """Startup checks, job handlers, test-grant bootstrap and background workers.
    Used by the standalone lifespan and by host mode (host.py) on first use."""
    s = get_settings()
    _register_job_handlers()
    from .access.policy import bootstrap_test_grants
    from .db.session import db_session
    from .security.crypto import _get as _crypto_ready
    if s.is_production and s.is_sqlite:
        # Fail fast: a forgotten II_DATABASE_URL would otherwise run on an in-memory database.
        raise RuntimeError("II_DATABASE_URL must point to Postgres in production (II_ENV is not dev/test/qa).")
    if s.is_production and s.drive_configured and not s.drive_folder_salt:
        logging.getLogger("ii").error("II_DRIVE_FOLDER_SALT is not set: Drive folder names use the public default salt.")
    _crypto_ready()  # fail fast in prod without an encryption key
    try:
        with db_session() as db:
            bootstrap_test_grants(db)
    except Exception:  # noqa: BLE001 - the service must still boot to report health
        logging.getLogger("ii").exception("bootstrap failed (database reachable? migration run?)")
    started: list = []
    if s.worker_threads > 0:
        from .jobs.maintenance import Sweeper
        from .jobs.queue import Worker
        worker = Worker(threads=s.worker_threads)
        worker.start()
        sweeper = Sweeper()
        sweeper.start()
        started = [worker, sweeper]
    return started


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=get_settings().log_level)
    started = start_background()
    yield
    for w in started:
        w.stop()


def create_app(*, host_mode: bool = False) -> FastAPI:
    """host_mode=True: mounted inside another app (see host.py) — the host owns the lifespan
    and CORS (a second CORS layer would duplicate headers), II's workers are started by host.py."""
    s = get_settings()
    app = FastAPI(title="MECE Interview Intelligence", version=versions()["service"],
                  lifespan=None if host_mode else lifespan,
                  docs_url=None if s.is_production else "/docs", redoc_url=None, openapi_url=None if s.is_production else "/openapi.json")
    if not host_mode:
        app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_origin_regex=s.cors_origin_regex,
                           allow_credentials=False, allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
                           allow_headers=["Authorization", "Content-Type"], max_age=600)

    @app.exception_handler(IIError)
    async def _ii(_: Request, exc: IIError):
        return JSONResponse(status_code=exc.status, content=exc.to_dict())

    @app.exception_handler(RequestValidationError)
    async def _val(_: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(x) for x in first.get("loc", [])[1:])
        return JSONResponse(status_code=422, content={"error": {"code": "invalid_request",
                                                                "message": f"Invalid request: {loc} {first.get('msg', '')}".strip()}})

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        logging.getLogger("ii").exception("unhandled error")
        return JSONResponse(status_code=500, content={"error": {"code": "internal",
                                                                "message": "Something went wrong. Please try again."}})

    @app.get("/healthz")
    def healthz():
        ok = True
        try:
            from .db.session import get_engine
            with get_engine().connect() as c:
                c.execute(text("SELECT 1"))
        except Exception:  # noqa: BLE001
            ok = False
        return {"ok": ok, "service": "interview-intelligence", "version": versions()["service"],
                "assertion_keys": len(get_settings().assertion_public_keys),
                "drive_configured": get_settings().drive_configured}

    from .api.admin_routes import router as admin_router
    from .api.routes import router as api_router
    from .voice.routes import router as voice_router
    app.include_router(api_router)
    app.include_router(admin_router)
    app.include_router(voice_router)
    return app


_standalone = None


def __getattr__(name: str):
    """`interview_intelligence.main:app` for uvicorn, built on first access (so importing this
    module in host mode does not build a second app)."""
    global _standalone
    if name == "app":
        if _standalone is None:
            _standalone = create_app()
        return _standalone
    raise AttributeError(name)
