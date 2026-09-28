"""
Supabase client - centralized database connection for the backend.

Uses the service_role key to bypass Row Level Security (RLS),
since the backend operates with admin privileges to write submissions.

SPEED (2026-09-28): ONE client per process, reused by every request.
The backend runs in Oregon and the database in Tokyo (~100 ms per round trip).
Creating a new client per call meant a new TCP+TLS handshake (~2 extra round
trips) on almost every query, plus the CPU cost of building a client, which is
large on Render's 0.1-CPU instance. One interview turn made 4-5 new clients.

The shared client keeps a pool of warm HTTP/1.1 connections (keep-alive
SUPABASE_KEEPALIVE_SECONDS, default 60, instead of httpx's 5 s) that PostgREST,
auth and storage all use. HTTP/1.1, NOT HTTP/2: requests arrive from many
threads, and httpx's sync HTTP/2 cannot share one connection between threads
(production 2026-09-28: "ReadError: [Errno 11] Resource temporarily
unavailable" on a session_state write). HTTP/1.1 gives each in-flight request
its own pooled connection, so nothing is shared mid-request. Safe to share: the client only ever carries the
service-role key -- nothing in this backend signs a user in on it -- and
supabase-py builds fresh headers per request. Verified to send byte-identical
URLs and auth headers to a default client.

Rollback without a code change: SUPABASE_CLIENT_MODE=per_call restores the old
new-client-per-call behaviour.
"""

import os
import threading

import httpx
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError(
        "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env file"
    )

_PER_CALL = os.getenv("SUPABASE_CLIENT_MODE", "").strip().lower() == "per_call"


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


_KEEPALIVE_S = _float_env("SUPABASE_KEEPALIVE_SECONDS", 60.0)

_client = None
_client_lock = threading.Lock()


class _RetryStaleConnection(httpx.BaseTransport):
    """Retry ONCE when a pooled connection turns out to be dead.

    A long-lived pool can hand out a connection the server has already closed.
    Retrying is always safe when the request never left (connect errors), and
    for reads (GET/HEAD). A write that may have reached the server is never
    retried, so an insert can not be applied twice.
    """

    def __init__(self, inner: httpx.BaseTransport):
        self._inner = inner

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        try:
            return self._inner.handle_request(request)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return self._inner.handle_request(request)
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError):
            if request.method in ("GET", "HEAD", "OPTIONS"):
                return self._inner.handle_request(request)
            raise

    def close(self) -> None:
        self._inner.close()


def _build_shared_client() -> Client:
    try:
        from supabase.lib.client_options import SyncClientOptions

        http = httpx.Client(
            http2=False,
            follow_redirects=True,
            # >= every supabase-py default (PostgREST 120 s, storage 20 s).
            timeout=httpx.Timeout(120.0, connect=10.0),
            transport=_RetryStaleConnection(httpx.HTTPTransport(
                http2=False,
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                    keepalive_expiry=_KEEPALIVE_S,
                ),
            )),
        )
        return create_client(
            SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY,
            options=SyncClientOptions(httpx_client=http),
        )
    except Exception as e:  # noqa: BLE001 -- never let a tuning step break the DB
        print(f"[supabase] shared-connection setup failed ({type(e).__name__}: {e}); using default client")
        return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def get_supabase_client() -> Client:
    """
    Returns a Supabase client configured with the service_role key.
    This client bypasses RLS - only use it from trusted backend code.
    """
    if _PER_CALL:
        return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    global _client
    c = _client
    if c is None:
        with _client_lock:
            if _client is None:
                _client = _build_shared_client()
            c = _client
    return c
