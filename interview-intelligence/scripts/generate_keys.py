"""Generate the secrets Interview Intelligence needs. Prints them; writes nothing.

    python -m scripts.generate_keys

Paste each value into the right place and nowhere else:
  * II_ASSERTION_PRIVATE_KEY  -> Vercel (MECE frontend) server env ONLY. Never NEXT_PUBLIC_.
  * II_ASSERTION_PUBLIC_KEY   -> II service env (Render). Public by nature; II cannot mint tokens.
  * II_ENCRYPTION_KEY         -> II service env. Losing it makes stored documents unreadable:
                                 keep a copy in your password manager.
  * II_DRIVE_FOLDER_SALT      -> II service env (if Drive is enabled).
Do not commit any of them, and do not put them in .brain/ or handoff files.
"""

from __future__ import annotations

import secrets

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def main() -> None:
    key = Ed25519PrivateKey.generate()
    priv = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption()).decode()
    pub = key.public_key().public_bytes(serialization.Encoding.PEM,
                                        serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    one_line = lambda pem: pem.strip().replace("\n", "\\n")  # noqa: E731 - env UIs prefer one line
    print("# ---- MECE frontend (Vercel, server-side env) -----------------------------")
    print(f'II_ASSERTION_PRIVATE_KEY="{one_line(priv)}"')
    print("II_ASSERTION_KID=k1")
    print()
    print("# ---- Interview Intelligence service (Render env) ------------------------")
    print(f'II_ASSERTION_PUBLIC_KEY="{one_line(pub)}"')
    print("II_ASSERTION_KID=k1")
    print(f"II_ENCRYPTION_KEY={Fernet.generate_key().decode()}")
    print(f"II_DRIVE_FOLDER_SALT={secrets.token_urlsafe(24)}")


if __name__ == "__main__":
    main()
