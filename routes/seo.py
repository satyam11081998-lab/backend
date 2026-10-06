"""
Growth Agent routes — programmatic SEO generation (admin only).

  GET  /seo/candidates -> fresh, GD-worthy headlines not yet turned into a page.
  POST /seo/generate   -> one full essay on a headline (or an explicit topic) through the
                          daily pipeline (research, writing, line edit, checks, pictures),
                          saved as a DRAFT and sent to Telegram. Never publishes on its own.
  GET  /seo/daily/status -> the daily blog: switches, today's post, the last
                          fortnight, and the topics it would pick right now.
  POST /seo/daily/run    -> write today's daily post now (services/growth/daily_blog.py).
                          Works even while DAILY_BLOG_ENABLED is off, so the owner can
                          try it; saves a DRAFT (sent to Telegram for review) unless
                          {"publish": true} and it passes every check. {"dry_run": true}
                          = preview, nothing saved.
  POST /seo/daily/send/{id} -> send a daily draft to Telegram for review again (admin).
  POST /seo/daily/images/{id} -> pictures for an existing article: real open-licensed photos, else Gemini (admin).
  POST /seo/daily/images/test -> what works for pictures (photo search, Gemini image models, storage) (admin).
  POST /seo/daily/images/backfill -> pictures for up to three published posts that have none (admin).
  POST /seo/daily/rewrite/{id} -> re-research and rewrite a post as a full essay (admin); a live post keeps its
                          version until the rewrite is approved.
  POST /seo/daily/apply/{id} -> publish a draft, or put a pending rewrite live (admin).
  POST /seo/daily/drop-rewrite/{id} -> discard a pending rewrite (admin).
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
def seo_generate(body: GenerateRequest, authorization: Optional[str] = Header(default=None)):
    """A full essay on a headline or topic through the daily pipeline (research, writing, line edit, checks,
    pictures), saved as a draft and sent to Telegram. Plain `def`: it takes a few minutes of blocking calls.
    Falls back to the old one-shot writer only when no web research is configured."""
    uid = _require_admin(authorization)
    check_rate_limit(f"seo:gen:{uid}", max_calls=6, window_seconds=600)
    assert_daily_budget()  # 503 if the day's AI spend is over budget

    from services.growth import daily_blog
    if daily_blog.config().get("research_available"):
        res = daily_blog.write_on(get_supabase_client(), headline_id=(body.headline_id or None),
                                  topic=(body.topic or None), user_id=uid)
        if res.get("page") and res.get("status") in ("draft", "published"):
            return res["page"]
        raise HTTPException(status_code=422, detail=str(res.get("reason") or "No essay this time")[:400])

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


@router.post("/daily/images/test")
def seo_daily_images_test(authorization: Optional[str] = Header(default=None)):
    """What works for pictures right now: photo search, Gemini image models, one test generation, storage."""
    uid = _require_admin(authorization)
    check_rate_limit(f"seo:imgtest:{uid}", max_calls=4, window_seconds=600)
    from services.growth import images
    return images.diagnose(get_supabase_client())


@router.post("/daily/images/backfill")
def seo_daily_images_backfill(authorization: Optional[str] = Header(default=None)):
    """Pictures for up to three published posts that have none (call again for the next three)."""
    uid = _require_admin(authorization)
    check_rate_limit(f"seo:imgfill:{uid}", max_calls=10, window_seconds=600)
    assert_daily_budget()
    from services.growth import daily_blog
    return daily_blog.backfill_images(get_supabase_client(), limit=3)


@router.post("/daily/rewrite/{page_id}")
def seo_daily_rewrite(page_id: str, authorization: Optional[str] = Header(default=None)):
    """Re-research and rewrite a post as a full essay with pictures. A live post keeps its current version until
    the rewrite is approved (Telegram 'publish', or POST /seo/daily/apply/{id})."""
    uid = _require_admin(authorization)
    check_rate_limit(f"seo:rewrite:{uid}", max_calls=6, window_seconds=600)
    assert_daily_budget()
    from services.growth import daily_blog
    return daily_blog.rewrite_page(get_supabase_client(), page_id, user_id=uid)


@router.post("/daily/apply/{page_id}")
def seo_daily_apply(page_id: str, authorization: Optional[str] = Header(default=None)):
    """Publish a draft, or put a pending rewrite of a live post live (same link, same date)."""
    _require_admin(authorization)
    from services.growth import daily_blog
    page = daily_blog.publish_page(get_supabase_client(), page_id, via="admin")
    if not page:
        raise HTTPException(status_code=404, detail="Not found or not saved")
    return page


@router.post("/daily/drop-rewrite/{page_id}")
def seo_daily_drop_rewrite(page_id: str, authorization: Optional[str] = Header(default=None)):
    _require_admin(authorization)
    from services.growth import daily_blog
    return {"ok": daily_blog.drop_rewrite(get_supabase_client(), page_id, via="admin")}


@router.post("/daily/images/{page_id}")
def seo_daily_images(page_id: str, authorization: Optional[str] = Header(default=None)):
    """Commission Gemini images for an existing article (older posts, or a retry)."""
    uid = _require_admin(authorization)
    check_rate_limit(f"seo:images:{uid}", max_calls=6, window_seconds=600)
    assert_daily_budget()
    from services.growth import daily_blog
    return daily_blog.add_images(get_supabase_client(), page_id)


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
