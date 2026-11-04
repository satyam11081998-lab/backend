"""MECE Backend - FastAPI server."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from routes import submit, news, cron, daily, transcribe, attempts, vision, usage, deck_vault, speak, realtime
import os

load_dotenv(override=True)

# MECE Interviewer V11 is the only interviewer, on every path. The frozen
# engine's non-adaptive branch used the static prompt that has been removed
# (prompts/interview_prompts.py), so the adaptive switch is pinned on here
# instead of being left as a fallback that would now fail at the first turn.
if os.getenv("ADAPTIVE_INTERVIEWER", "").strip().lower() not in {"1", "true", "yes", "on"}:
    print("[interviewer] ADAPTIVE_INTERVIEWER pinned to 'true': V11 is the only interviewer")
os.environ["ADAPTIVE_INTERVIEWER"] = "true"

app = FastAPI(title="MECE Backend", version="0.2.0")

# WHICH ORIGINS MAY CALL THIS API, AS CONFIGURATION RATHER THAN CODE
#
# This list was hardcoded and it has now been wrong twice in one migration: the preview
# regex was pinned to a Vercel team slug that no longer exists, and the site origins were
# pinned to a domain that is about to change. Both failures look identical from the
# server -- nothing in any log, every request failing only in the browser -- so the cost
# of getting it wrong is an outage you find out about from users.
#
# So: set CORS_ALLOW_ORIGINS on the host and a domain move is a settings change, with the
# old and new origins allowed side by side for as long as the cutover takes. Unset, it
# behaves exactly as before.
#
# allow_credentials is True, which means every allowed origin can make requests carrying
# a signed-in user's session. That is why this is an explicit list and NOT a wildcard or
# a broad "any .vercel.app" pattern: the set of origins trusted to act as your users has
# to be a set you can name.
_SITE_ORIGINS_DEFAULT = "https://mece.in,https://www.mece.in"
_SITE_ORIGINS = [o.strip().rstrip("/") for o in
                 (os.getenv("CORS_ALLOW_ORIGINS") or _SITE_ORIGINS_DEFAULT).split(",")
                 if o.strip()]

# Local development, always. Kept OUT of the env var so that setting CORS_ALLOW_ORIGINS
# for a production cutover cannot accidentally break every developer's machine.
_DEV_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]

# Vercel preview deployments get a fresh hashed hostname per build, so they cannot be
# listed. Supply a regex for the CURRENT team only.
#
# This deliberately defaults to EMPTY, where it used to default to a pattern matching the
# old team's slug. A regex that matches a team you no longer control is worse than no
# regex at all: slugs can be re-registered, and this one grants credentialed access.
_CORS_PREVIEW_REGEX = (os.getenv("CORS_ALLOW_ORIGIN_REGEX") or "").strip() or None

app.add_middleware(
    CORSMiddleware,
    allow_origins=[*_DEV_ORIGINS, *_SITE_ORIGINS],
    allow_origin_regex=_CORS_PREVIEW_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "service": "MECE Backend"}


@app.api_route("/health", methods=["GET", "HEAD"])
def health_check():
    return {
        "status": "ok",
        "openai_key_loaded": bool(os.getenv("OPENAI_API_KEY")),
        "supabase_url_loaded": bool(os.getenv("SUPABASE_URL")),
        "supabase_key_loaded": bool(os.getenv("SUPABASE_SERVICE_ROLE_KEY")),
        "gnews_key_loaded": bool(os.getenv("GNEWS_API_KEY")),
        "newsapi_key_loaded": bool(os.getenv("NEWSAPI_KEY")),
        "cron_secret_loaded": bool(os.getenv("CRON_SECRET")),
        "gemini_key_loaded": bool(os.getenv("GEMINI_API_KEY")),
    }


# Register all routes
app.include_router(submit.router)
app.include_router(news.router)
app.include_router(cron.router)
app.include_router(daily.router)
app.include_router(transcribe.router, prefix="/transcribe")
app.include_router(speak.router, prefix="/speak")
app.include_router(realtime.router, prefix="/realtime")
app.include_router(attempts.router)
app.include_router(vision.router, prefix="/extract-text")
app.include_router(usage.router)
app.include_router(deck_vault.router)
from routes import resume as _resume
app.include_router(_resume.router)
from routes import certificates as _certificates
app.include_router(_certificates.router)
from routes import decks as _decks
app.include_router(_decks.router)
from routes import deck_ingestion as _deck_ingestion
app.include_router(_deck_ingestion.router)
from routes import ai_providers as _ai_providers
app.include_router(_ai_providers.router)
from routes import public_config as _public_config
app.include_router(_public_config.router)
from routes import realtime_gemini as _realtime_gemini
app.include_router(_realtime_gemini.router, prefix="/realtime-gemini")
from routes import agentic as _agentic
app.include_router(_agentic.router)
from routes import coach as _coach
app.include_router(_coach.router)
# Prep Copilot v2 (role/company-aware, isolated). Self-gates on COPILOT_V2_ENABLED;
# heavy pieces import lazily inside handlers so this can never affect app startup.
from routes import copilot as _copilot
app.include_router(_copilot.router)
from routes import seo as _seo
app.include_router(_seo.router)
from routes import broadcast as _broadcast
app.include_router(_broadcast.router)
from routes import voice_coach as _voice_coach
app.include_router(_voice_coach.router)
# Admin interviewer arena (/admin/arena): blind A/B of two frozen prompt variants
# on a real case. Admin-gated, budget-gated, and isolated from live user data --
# it creates no attempt, spends no clarification quota and persists no learner
# profile. See services/arena.py for the isolation contract.
from routes import arena as _arena
app.include_router(_arena.router)

# Interview Intelligence: its own package in ./interview-intelligence, mounted at /ii.
# Dormant until II_DATABASE_URL is set; nothing of it loads until the first /ii call.
from routes import interview_intelligence as _interview_intelligence
_interview_intelligence.mount_interview_intelligence(app)
