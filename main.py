"""Consilio Backend - FastAPI server."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="Consilio Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "service": "Consilio Backend"}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "openai_key_loaded": bool(os.getenv("OPENAI_API_KEY")),
        "supabase_url_loaded": bool(os.getenv("SUPABASE_URL")),
        "supabase_key_loaded": bool(os.getenv("SUPABASE_SERVICE_ROLE_KEY")),
    }