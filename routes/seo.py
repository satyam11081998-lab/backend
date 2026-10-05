"""
Growth Agent routes — programmatic SEO generation (admin only).

  GET  /seo/candidates -> fresh, GD-worthy headlines not yet turned into a page.
  POST /seo/generate   -> generate ONE grounded SEO draft from a headline (or an
                          explicit topic), self-critique it, and store it as a
                          DRAFT in seo_pages. Never publishes — an admin approves
                          in /admin/growth, which flips status to 'published'.
  GET  /seo/daily/status -> the daily blog: switches, today's post, the last
                          fortnight, and the topics it would pick right now.
  POST /seo/daily/run    -> write today's daily post now (services/growth/daily_blog.py).
                          Works even while DAILY_BLOG_ENABLED is off, so the owner can
                          try it; saves a DRAFT (sent to Telegram for review) unless
                          {"publish": true} and it passes every check. {"dry_run": true}
                          = preview, nothing saved.
  POST /seo/daily/send/{id} -> send a daily draft to Telegram for review again (admin).
  POST /seo/telegram/setup  -> (re)register the Telegram webhook (admin).
  POST /seo/telegram/webhook -> Telegram calls this when the admin replies to a draft
                          (publish / another / reject). Not a user route: it checks the
                          secret token Telegram sends and only obeys the admin chat.

Admin-gated with the same contract as routes/agentic.py & routes/coach.py.
The daily-budget kill switch is enforced before any model call.
"""

from typing import Optional

import hmac

from fastapi import APIRouter, Body, Header, HTTPException
from pydantic import BaseModel

from services.auth import get_verified_user, is_guest_user
from services.supabase_client import get_supabase_client
from services.rate_limit import check_rate_limit
from services.ai_usage import assert_daily_budget

router = APIRouter(prefix="/seo", tags=["seo", "growth", "admin"])


def _require_admin(authorization: Optional[str]) -> str:
    supabase = get_supabase_client()
    uid, user_obj = get_verified_user(supabase, authorization)
    if is_guest_user(user_obj):
        raise HTTPException(status_code=403, detail="Admins only")
    try:
        res = supabase.table("users").select("is_admin").eq("id", uid).single().execute()
        is_admin = bool((res.data or {}).get("is_admin"))
    except Exception:
        raise HTTPException(status_code=403, detail="Admins only")
    if not is_admin:
        raise HTTPException(status_code=403, detail="Admins only")
    return uid


class GenerateRequest(BaseModel):
    headline_id: Optional[str] = None
    topic: Optional[str] = None


@router.get("/candidates")
async def seo_candidates(authorization: Optional[str] = Header(default=None)):
    _require_admin(authorization)
    from services.growth import list_candidate_headlines
    return {"candidates": list_candidate_headlines(get_supabase_client(), limit=12)}


@router.post("/generate")
async def seo_generate(body: GenerateRequest, authorization: Optional[str] = Header(default=None)):
    uid = _require_admin(authorization)
    check_rate_limit(f"seo:gen:{uid}", max_calls=20, window_seconds=60)
    assert_daily_budget()  # 503 if the day's AI spend is over budget

    from services.growth import generate_seo_page
    try:
        draft = generate_seo_page(
            get_supabase_client(), uid,
            headline_id=(body.headline_id or None),
            topic=(body.topic or None),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Generation failed: {type(e).__name__}")
    return draft


class DailyRunRequest(BaseModel):
    force: bool = False      # write another even if today's exists
    dry_run: bool = False    # preview only, nothing saved
    publish: bool = False    # publish if it passes every check (else a draft)


@router.get("/daily/status")
def seo_daily_status(authorization: Optional[str] = Header(default=None)):
    _require_admin(authorization)
    from services.growth import daily_blog
    return daily_blog.status(get_supabase_client())


@router.post("/daily/run")
def seo_daily_run(body: DailyRunRequest, authorization: Optional[str] = Header(default=None)):
    """Plain `def`: a run takes a minute or two of blocking calls (threadpool, not the event loop)."""
    uid = _require_admin(authorization)
    check_rate_limit(f"seo:daily:{uid}", max_calls=6, window_seconds=600)
    assert_daily_budget()
    from services.growth import daily_blog
    return daily_blog.run_daily(get_supabase_client(), user_id=uid, force=body.force, dry_run=body.dry_run,
                                publish=body.publish)


@router.post("/daily/send/{page_id}")
def seo_daily_send(page_id: str, authorization: Optional[str] = Header(default=None)):
    _require_admin(authorization)
    from services.growth import telegram_review
    sb = get_supabase_client()
    try:
        rows = sb.table("seo_pages").select("*").eq("id", page_id).limit(1).execute().data or []
    except Exception:
        rows = []
    if not rows:
        raise HTTPException(status_code=404, detail="Not found")
    if not telegram_review.configured():
        raise HTTPException(status_code=422, detail="Telegram is not set up (TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_CHAT_ID)")
    return {"sent": telegram_review.send_for_review(sb, rows[0])}


@router.post("/telegram/setup")
def seo_telegram_setup(authorization: Optional[str] = Header(default=None)):
    _require_admin(authorization)
    from services.growth import telegram_review
    return telegram_review.ensure_webhook(force=True)


def _write_another(exclude_titles):
    """Background: a new daily draft after the reviewer asked for 'another' on Telegram."""
    from services.growth import daily_blog, telegram_review
    res = daily_blog.run_daily(get_supabase_client(), force=True, publish=False, exclude_titles=exclude_titles)
    if res.get("status") not in ("draft", "published"):
        telegram_review.send_text("Couldn't write a new one: " + str(res.get("reason") or "unknown error")
                                  + ". Reply another to try again.")


@router.post("/telegram/webhook")
def seo_telegram_webhook(update: dict = Body(default_factory=dict),
                         x_telegram_bot_api_secret_token: Optional[str] = Header(default=None)):
    from services.growth import telegram_review
    if not telegram_review.configured():
        raise HTTPException(status_code=404, detail="Not found")
    if not hmac.compare_digest(x_telegram_bot_api_secret_token or "", telegram_review.webhook_secret()):
        raise HTTPException(status_code=403, detail="Forbidden")
    try:
        return telegram_review.handle_update(get_supabase_client(), update, run_another=_write_another)
    except Exception as e:  # noqa: BLE001 - always 200 to Telegram, or it re-sends the same update
        print(f"[telegram] update failed: {type(e).__name__}: {e}")
        return {"ok": False}
