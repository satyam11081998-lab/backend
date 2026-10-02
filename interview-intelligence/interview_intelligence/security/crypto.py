"""Field-level encryption for the most sensitive payloads (raw CV/JD bytes and text).

Fernet (AES-128-CBC + HMAC-SHA256). The key is `II_ENCRYPTION_KEY`. In production the
service refuses to start without it; in dev/test an ephemeral key is generated (data
encrypted with it is unreadable after a restart, which is the safe failure).
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

from ..config import get_settings

log = logging.getLogger("ii.crypto")
_lock = threading.Lock()
_fernet: Optional[Fernet] = None
_ephemeral = False


class EncryptionUnavailable(RuntimeError):
    pass


def _get() -> Fernet:
    global _fernet, _ephemeral
    if _fernet is None:
        with _lock:
            if _fernet is None:
                s = get_settings()
                key = s.encryption_key
                if not key:
                    if s.is_production:
                        raise EncryptionUnavailable("II_ENCRYPTION_KEY is required in production")
                    key = Fernet.generate_key().decode()
                    _ephemeral = True
                    log.warning("II_ENCRYPTION_KEY not set — using an EPHEMERAL key (dev/test only)")
                _fernet = Fernet(key.encode() if isinstance(key, str) else key)
    return _fernet


def reset_for_tests() -> None:
    global _fernet, _ephemeral
    with _lock:
        _fernet = None
        _ephemeral = False


def encrypt_bytes(data: bytes) -> bytes:
    return _get().encrypt(data)


def decrypt_bytes(token: bytes) -> bytes:
    try:
        return _get().decrypt(token)
    except InvalidToken as e:  # wrong key / tampered
        raise EncryptionUnavailable("Stored content could not be decrypted") from e


def encrypt_text(text: str) -> bytes:
    return encrypt_bytes(text.encode("utf-8"))


def decrypt_text(token: Optional[bytes]) -> str:
    if not token:
        return ""
    return decrypt_bytes(bytes(token)).decode("utf-8")
