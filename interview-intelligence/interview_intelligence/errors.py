"""Typed API errors -> `{"error": {"code", "message", ...}}` responses."""

from __future__ import annotations

from typing import Any, Dict, Optional


class IIError(Exception):
    status = 400
    code = "bad_request"

    def __init__(self, message: str, *, code: Optional[str] = None, status: Optional[int] = None,
                 extra: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        if status:
            self.status = status
        self.extra = extra or {}

    def to_dict(self) -> Dict[str, Any]:
        body = {"code": self.code, "message": self.message}
        body.update(self.extra)
        return {"error": body}


class Unauthorized(IIError):
    status = 401
    code = "unauthorized"


class Forbidden(IIError):
    status = 403
    code = "forbidden"


class NotFound(IIError):
    status = 404
    code = "not_found"


class Conflict(IIError):
    status = 409
    code = "conflict"


class TooMany(IIError):
    status = 429
    code = "rate_limited"


class Unprocessable(IIError):
    status = 422
    code = "unprocessable"


class Unavailable(IIError):
    status = 503
    code = "unavailable"
