"""
Telegram review for the daily blog (services/growth/daily_blog.py).

Every new daily draft is sent, in full, to the admin's Telegram chat. Replying there decides it:

  publish   (or /publish, approve, post it)   -> published on mece.in/insights at once
  another   (or /another, new, rewrite)       -> this draft is dropped and a new one is written
                                                  on a different topic (3-5 minutes), then sent
  reject    (or /reject, drop, skip)          -> dropped
  status / help

A reply to one of the draft's messages acts on that draft; a plain message acts on the newest
draft waiting for review. Only the admin chat is obeyed (TELEGRAM_ADMIN_CHAT_ID).

Setup is automatic: the same TELEGRAM_BOT_TOKEN / TELEGRAM_ADMIN_CHAT_ID the Deck Vault alerts
use; the webhook (POST /seo/telegram/webhook) registers itself the first time a draft is sent,
at RENDER_EXTERNAL_URL (Render sets it) or DAILY_BLOG_PUBLIC_API_URL, with a secret token derived
from the bot token so nobody else can call it.
"""

from __future__ import annotations

import hashlib
import hmac
import html
import os
import re
import threading
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

SITE = "https://mece.in"
WEBHOOK_PATH = "/seo/telegram/webhook"
MAX_CHARS = 3800  # Telegram's limit is 4096 per message

_seen_updates: deque = deque(maxlen=300)
_seen_lock = threading.Lock()
_webhook_ok = {"url": ""}


# ---------------------------------------------------------------------------- config
def _token() -> str:
    return (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()


def _chat_id() -> str:
    return (os.getenv("TELEGRAM_ADMIN_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID") or "").strip()


def configured() -> bool:
    return bool(_token() and _chat_id())


def webhook_secret() -> str:
    """Telegram sends this back in X-Telegram-Bot-Api-Secret-Token; derived, so no new env var."""
    return hmac.new(_token().encode(), b"mece-daily-blog-review", hashlib.sha256).hexdigest()[:48]


def base_url() -> str:
    for k in ("DAILY_BLOG_PUBLIC_API_URL", "RENDER_EXTERNAL_URL", "API_BASE_URL", "BACKEND_URL"):
        v = (os.getenv(k) or "").strip().rstrip("/")
        if v.startswith("https://"):
            return v
    return ""


def webhook_url() -> str:
    b = base_url()
    return f"{b}{WEBHOOK_PATH}" if b else ""


def _call(method: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 10.0) -> Dict[str, Any]:
    import httpx
    r = httpx.post(f"https://api.telegram.org/bot{_token()}/{method}", json=payload or {}, timeout=timeout)
    try:
        data = r.json()
    except Exception:
        data = {"ok": False, "description": f"HTTP {r.status_code}"}
    return data


def ensure_webhook(force: bool = False) -> Dict[str, Any]:
    """Point the bot's webhook at this backend (idempotent)."""
    if not configured():
        return {"ok": False, "reason": "TELEGRAM_BOT_TOKEN / TELEGRAM_ADMIN_CHAT_ID not set"}
    url = webhook_url()
    if not url:
        return {"ok": False, "reason": "no public backend URL (RENDER_EXTERNAL_URL or DAILY_BLOG_PUBLIC_API_URL)"}
    if not force and _webhook_ok["url"] == url:
        return {"ok": True, "url": url, "cached": True}
    try:
        info = _call("getWebhookInfo").get("result") or {}
        if force or info.get("url") != url:
            res = _call("setWebhook", {"url": url, "secret_token": webhook_secret(),
                                       "allowed_updates": ["message"], "drop_pending_updates": False})
            if not res.get("ok"):
                return {"ok": False, "reason": res.get("description") or "setWebhook failed"}
        _webhook_ok["url"] = url
        return {"ok": True, "url": url}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"[:200]}


def state() -> Dict[str, Any]:
    out: Dict[str, Any] = {"configured": configured(), "webhook_url": webhook_url()}
    if not configured():
        return out
    try:
        info = _call("getWebhookInfo", timeout=6).get("result") or {}
        out.update({"webhook_set": info.get("url") == webhook_url() and bool(webhook_url()),
                    "pending_updates": info.get("pending_update_count"),
                    "last_error": info.get("last_error_message")})
    except Exception as e:  # noqa: BLE001
        out["error"] = type(e).__name__
    return out


def send_text(text: str, *, reply_to: Optional[int] = None, html_mode: bool = False) -> Optional[int]:
    if not configured():
        return None
    payload: Dict[str, Any] = {"chat_id": _chat_id(), "text": text[:4000], "disable_web_page_preview": True}
    if html_mode:
        payload["parse_mode"] = "HTML"
    if reply_to:
        payload["reply_to_message_id"] = reply_to
        payload["allow_sending_without_reply"] = True
    try:
        res = _call("sendMessage", payload)
        if not res.get("ok") and html_mode:  # a formatting slip must not lose the message
            payload.pop("parse_mode", None)
            payload["text"] = re.sub(r"</?[a-z]+>", "", text)[:4000]
            res = _call("sendMessage", payload)
        return ((res.get("result") or {}).get("message_id")) if res.get("ok") else None
    except Exception:
        return None


# ---------------------------------------------------------------------------- the review message
def _e(s: Any) -> str:
    return html.escape(str(s or ""), quote=False)


def render_messages(page: Dict[str, Any]) -> List[str]:
    """The draft as Telegram messages (HTML), split under Telegram's size limit."""
    c = page.get("content") or {}
    meta = page.get("agent_meta") or {}
    blocks: List[str] = []
    serious = meta.get("serious_problems") or []
    problems = meta.get("problems") or []
    checks = "all passed ✅" if not problems else (
        "⚠️ " + "; ".join(serious) if serious else f"{len(problems)} style note(s): " + "; ".join(problems[:3]))
    head = [f"📝 <b>MECE Insights — draft for review</b>", "", f"<b>{_e(page.get('title'))}</b>"]
    if page.get("dek"):
        head.append(f"<i>{_e(page.get('dek'))}</i>")
    head += ["", f"Score {page.get('quality_score') if page.get('quality_score') is not None else '—'}/100 · "
                 f"{c.get('words') or '?'} words · {len(c.get('sources') or [])} sources · "
                 f"{meta.get('linked_facts', meta.get('facts', '?'))} linked facts",
             f"Checks: {_e(checks)}"]
    if meta.get("reasons"):
        head.append("Why this topic: " + _e("; ".join(str(r) for r in meta.get("reasons") or [])[:300]))
    if page.get("quality_notes") and not problems:
        head.append("Editor's note: " + _e(str(page.get("quality_notes"))[:300]))
    head += ["", "Reply <b>publish</b> to post it · <b>another</b> for a new topic · <b>reject</b> to drop it."]
    blocks.append("\n".join(head))

    body: List[str] = []
    if c.get("key_points"):
        body.append("<b>Key points</b>\n" + "\n".join(f"• {_e(k)}" for k in c["key_points"]))
    if c.get("lede"):
        body.append(_e(c["lede"]))
    for s in c.get("sections") or []:
        part = [f"<b>{_e(s.get('heading'))}</b>"] + [_e(p) for p in s.get("paragraphs") or []]
        if s.get("bullets"):
            part.append("\n".join(f"• {_e(b)}" for b in s["bullets"]))
        body.append("\n\n".join(part))
    if c.get("numbers"):
        body.append("<b>By the numbers</b>\n" + "\n".join(
            f"• {_e(n.get('figure'))}: {_e(n.get('what'))}" for n in c["numbers"]))
    fw = c.get("framework") or {}
    if fw.get("steps"):
        body.append(f"<b>{_e(fw.get('heading') or 'How to break it down')}</b>"
                    + (f" ({_e(fw.get('name'))})" if fw.get("name") else "") + "\n"
                    + "\n".join(f"{i + 1}. {_e(x)}" for i, x in enumerate(fw["steps"])))
    if c.get("what_to_watch"):
        body.append("<b>What to watch</b>\n" + "\n".join(f"• {_e(x)}" for x in c["what_to_watch"]))
    asp = c.get("aspirants") or {}
    if asp:
        lines = ["<b>For MBA aspirants</b>"]
        if asp.get("gd_topic"):
            lines.append(f"GD topic: {_e(asp['gd_topic'])}")
        if asp.get("for"):
            lines.append("For: " + _e("; ".join(asp["for"])))
        if asp.get("against"):
            lines.append("Against: " + _e("; ".join(asp["against"])))
        if asp.get("pi_questions"):
            lines.append("PI questions:\n" + "\n".join(f"• {_e(q)}" for q in asp["pi_questions"]))
        if asp.get("wat_prompt"):
            lines.append(f"WAT: {_e(asp['wat_prompt'])}")
        if asp.get("case_question"):
            lines.append(f"As a case: {_e(asp['case_question'])}")
        body.append("\n".join(lines))
    if c.get("faq"):
        body.append("<b>FAQ</b>\n" + "\n\n".join(f"<b>{_e(f.get('q'))}</b>\n{_e(f.get('a'))}" for f in c["faq"]))
    rel = c.get("related") or {}
    if rel:
        body.append("Related practice: " + _e("; ".join(f"{v.get('title')}" for v in rel.values() if v)))
    if c.get("sources"):
        body.append("<b>Sources</b>\n" + "\n".join(
            f"{s.get('n')}. {_e(s.get('label'))}" + (f" — {_e(s.get('url'))}" if s.get("url") else " (named, no link)")
            for s in c["sources"]))

    msgs, cur = [], ""
    for b in body:
        while len(b) > MAX_CHARS:  # one huge block: cut at a paragraph or line break
            cut = b.rfind("\n", 0, MAX_CHARS)
            cut = cut if cut > 1000 else MAX_CHARS
            if cur:
                msgs.append(cur)
                cur = ""
            msgs.append(b[:cut])
            b = b[cut:].lstrip()
        if len(cur) + len(b) + 2 > MAX_CHARS:
            msgs.append(cur)
            cur = b
        else:
            cur = f"{cur}\n\n{b}" if cur else b
    if cur:
        msgs.append(cur)
    return blocks + msgs + ["👆 Reply <b>publish</b> to post this on mece.in · <b>another</b> for a new topic · "
                            "<b>reject</b> to drop it."]


def send_for_review(supabase, page: Dict[str, Any]) -> bool:
    """Send a draft (or a just-published post, as a notice) to the admin chat; remember the message ids."""
    if not configured():
        return False
    ensure_webhook()
    if page.get("status") == "published":
        mid = send_text(f"✅ Published automatically: <b>{_e(page.get('title'))}</b>\n{SITE}/insights/{page.get('slug')}",
                        html_mode=True)
        return bool(mid)
    ids: List[int] = []
    hero = (page.get("content") or {}).get("hero") or {}
    if hero.get("url"):
        try:
            res = _call("sendPhoto", {"chat_id": _chat_id(), "photo": hero.get("og_url") or hero["url"],
                                      "caption": f"{page.get('title') or ''}"[:1000]})
            if res.get("ok"):
                ids.append((res.get("result") or {}).get("message_id"))
        except Exception:
            pass
    ids += [m for m in (send_text(t, html_mode=True) for t in render_messages(page)) if m]
    ids = [i for i in ids if i]
    if not ids:
        return False
    try:
        cur = supabase.table("seo_pages").select("agent_meta").eq("id", page["id"]).limit(1).execute()
        meta = dict(((cur.data or [{}])[0]).get("agent_meta") or {})
        meta["telegram"] = {"message_ids": ids, "sent_at": datetime.now(timezone.utc).isoformat()}
        meta["review"] = "pending"
        supabase.table("seo_pages").update({"agent_meta": meta}).eq("id", page["id"]).execute()
    except Exception:
        pass
    return True


# ---------------------------------------------------------------------------- replies
_PUBLISH = {"publish", "publish it", "post", "post it", "approve", "approved", "go live", "publish now", "yes publish"}
_ANOTHER = {"another", "another one", "new", "new one", "new topic", "next", "rewrite", "try again", "different topic"}
_REJECT = {"reject", "drop", "skip", "discard", "delete", "no"}
_STATUS = {"status", "what's pending", "pending"}
_HELP = {"help", "start", "commands"}

HELP_TEXT = ("Reply to a draft with:\n• publish — post it on mece.in now\n• another — drop it and write a new one "
             "on a different topic\n• reject — drop it\n• status — what is waiting")


def parse_command(text: str) -> str:
    t = re.sub(r"@\w+", "", (text or "").strip().lower())          # /publish@MyBot
    t = re.sub(r"[^a-z' ]+", " ", t).strip()                        # emojis, punctuation, the leading '/'
    t = re.sub(r"\s+", " ", t)
    for name, words in (("publish", _PUBLISH), ("another", _ANOTHER), ("reject", _REJECT), ("status", _STATUS),
                        ("help", _HELP)):
        if t in words:
            return name
    first = t.split(" ")[0] if t else ""
    if first in {"publish", "approve"}:
        return "publish"
    if first in {"another", "rewrite"}:
        return "another"
    if first in {"reject", "drop", "discard"}:
        return "reject"
    return ""


def pending_drafts(supabase, days: int = 7) -> List[Dict[str, Any]]:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    try:
        r = (supabase.table("seo_pages").select("id, slug, title, status, topic, agent_meta, source_headline_id, created_at")
             .eq("kind", "daily").eq("status", "draft").gte("created_at", since)
             .order("created_at", desc=True).limit(20).execute())
        return [p for p in (r.data or []) if (p.get("agent_meta") or {}).get("review", "pending") == "pending"]
    except Exception:
        return []


def find_target(supabase, reply_to_id: Optional[int]) -> Optional[Dict[str, Any]]:
    drafts = pending_drafts(supabase)
    if reply_to_id:
        for p in drafts:
            if reply_to_id in (((p.get("agent_meta") or {}).get("telegram") or {}).get("message_ids") or []):
                return p
    return drafts[0] if drafts else None


def handle_update(supabase, update: Dict[str, Any], *, run_another: Callable[[List[str]], Any]) -> Dict[str, Any]:
    """Act on one Telegram update. `run_another(exclude_titles)` writes a new draft (in the background)."""
    uid = update.get("update_id")
    with _seen_lock:
        if uid is not None and uid in _seen_updates:
            return {"ok": True, "duplicate": True}
        if uid is not None:
            _seen_updates.append(uid)
    msg = update.get("message") or {}
    if not msg or str((msg.get("chat") or {}).get("id")) != _chat_id():
        return {"ok": True, "ignored": "not the admin chat"}
    text = msg.get("text") or ""
    cmd = parse_command(text)
    reply_to = (msg.get("reply_to_message") or {}).get("message_id")
    me = msg.get("message_id")
    from services.growth import daily_blog

    if cmd == "publish":
        page = find_target(supabase, reply_to)
        if not page:
            send_text("Nothing is waiting for review.", reply_to=me)
            return {"ok": True, "action": "none"}
        done = daily_blog.publish_page(supabase, page["id"], via="telegram")
        if done:
            send_text(f"✅ Published: {page.get('title')}\n{SITE}/insights/{page.get('slug')}\n"
                      "(The Insights list refreshes within a few minutes.)", reply_to=me)
            return {"ok": True, "action": "published", "id": page["id"]}
        send_text("Couldn't publish it (database error). Try again, or publish from Admin → Growth.", reply_to=me)
        return {"ok": False, "action": "publish_failed"}
    if cmd == "reject":
        page = find_target(supabase, reply_to)
        if not page:
            send_text("Nothing is waiting for review.", reply_to=me)
            return {"ok": True, "action": "none"}
        daily_blog.set_review_state(supabase, page["id"], status="rejected", review="rejected", via="telegram")
        send_text("Dropped. Reply another for a new one.", reply_to=me)
        return {"ok": True, "action": "rejected", "id": page["id"]}
    if cmd == "another":
        page = find_target(supabase, reply_to)
        exclude: List[str] = []
        if page:
            daily_blog.set_review_state(supabase, page["id"], status="rejected", review="replaced", via="telegram")
            exclude = [page.get("topic") or "", page.get("title") or ""]
        send_text("Writing a new one on a different topic. It takes 3-5 minutes; it will arrive here.", reply_to=me)
        threading.Thread(target=run_another, args=(exclude,), daemon=True).start()
        return {"ok": True, "action": "another"}
    if cmd == "status":
        drafts = pending_drafts(supabase)
        send_text(("Waiting for review:\n" + "\n".join(f"• {p.get('title')}" for p in drafts[:5])) if drafts
                  else "Nothing is waiting for review.", reply_to=me)
        return {"ok": True, "action": "status"}
    if cmd == "help" or len(text.split()) <= 4:
        send_text(HELP_TEXT, reply_to=me)
        return {"ok": True, "action": "help"}
    return {"ok": True, "action": "ignored"}
