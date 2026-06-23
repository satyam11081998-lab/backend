"""MECE Backend - FastAPI server."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from routes import submit, news, cron, daily, transcribe, attempts, vision
import os

load_dotenv()

app = FastAPI(title="MECE Backend", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://mece.in",
        "https://www.mece.in",
    ],
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
    }


# Register all routes
app.include_router(submit.router)
app.include_router(news.router)
app.include_router(cron.router)
app.include_router(daily.router)
app.include_router(transcribe.router, prefix="/transcribe")
app.include_router(attempts.router)
app.include_router(vision.router, prefix="/extract-text")
