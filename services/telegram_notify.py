"""
Telegram admin notifications — best-effort, never blocks or fails a request.

Setup (both must be set, otherwise every call is a silent no-op):
  TELEGRAM_BOT_TOKEN     — from @BotFather
  TELEGRAM_ADMIN_CHAT_ID — your chat id (message the bot once, then
                           GET https://api.telegram.org/bot<token>/getUpdates
                           and read message.chat.id)

Used by the Deck Vault: pings the admin the moment a deck submission lands.
Sent from a daemon thread with a short timeout so a Telegram outage can never
slow down or break the user-facing upload.
"""

import os
import threading

import httpx


def _send_sync(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    # Accept both names — an env-var name mismatch must not silently kill alerts.
    chat_id = (
        os.environ.get("TELEGRAM_ADMIN_CHAT_ID", "").strip()
        or os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    )
    if not token or not chat_id:
        return
    try:
        httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text[:4000],  # Telegram hard limit is 4096
                "disable_web_page_preview": True,
            },
            timeout=8.0,
        )
    except Exception:
        # Notification is a nice-to-have; the submission itself already succeeded.
        pass


def send_admin_alert(text: str) -> None:
    """Fire-and-forget Telegram message to the admin chat."""
    threading.Thread(target=_send_sync, args=(text,), daemon=True).start()


def notify_deck_submission(
    *,
    submission_id: str,
    user_name: str,
    user_email: str,
    competition_name: str,
    organizer: str,
    competition_type: str,
    position: str,
    year: int,
    default_pct: int,
) -> None:
    """Alert the admin that a new deck landed in the vault and needs review."""
    kind = "Corporate" if competition_type == "corporate" else "B-school"
    pos = position.replace("_", " ").title()
    send_admin_alert(
        "📥 New Deck Vault submission\n"
        f"From: {user_name or 'Unknown'} ({user_email or 'no email'})\n"
        f"Competition: {competition_name} ({year})\n"
        f"Organizer: {organizer or '—'}\n"
        f"Type: {kind} · Position: {pos}\n"
        f"Default discount: {default_pct}% (editable on approval)\n"
        f"Review: https://mece.in/admin/deck-vault\n"
        f"ID: {submission_id}"
    )
