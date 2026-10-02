"""Engine + session management.

Postgres (production): direct connection (psycopg3) to the `interview_intel` schema.
SQLite (unit tests only): the schema name is translated away so the same models run.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ..config import get_settings
from .models import SCHEMA, Base

_engine: Optional[Engine] = None
_factory: Optional[sessionmaker] = None
_lock = threading.Lock()


def _build_engine(url: str, schema: str) -> Engine:
    if url.startswith("sqlite"):
        eng = create_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool if ":memory:" in url else None,
            execution_options={"schema_translate_map": {SCHEMA: None}},
            future=True,
        )

        @event.listens_for(eng, "connect")
        def _fk_on(dbapi_conn, _):  # pragma: no cover - trivial
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

        return eng
    translate = {SCHEMA: schema} if schema != SCHEMA else {}
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=max(1, get_settings().db_pool_size),
        max_overflow=max(1, get_settings().db_pool_size),
        pool_recycle=1800,
        execution_options={"schema_translate_map": translate} if translate else {},
        future=True,
    )


def get_engine() -> Engine:
    global _engine, _factory
    if _engine is None:
        with _lock:
            if _engine is None:
                s = get_settings()
                _engine = _build_engine(s.database_url, s.db_schema)
                _factory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


def reset_engine_for_tests() -> None:
    global _engine, _factory
    with _lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None
        _factory = None


def session_factory() -> sessionmaker:
    get_engine()
    assert _factory is not None
    return _factory


@contextmanager
def db_session() -> Iterator[Session]:
    """Unit of work: commit on success, rollback on error."""
    s = session_factory()()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def create_all_for_dev() -> None:
    """Tests / local dev only. Production uses migrations/0001_interview_intel.sql."""
    eng = get_engine()
    if eng.dialect.name == "postgresql":
        with eng.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{get_settings().db_schema}"'))
    Base.metadata.create_all(eng)


def is_postgres() -> bool:
    return get_engine().dialect.name == "postgresql"
