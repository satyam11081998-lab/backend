"""
Supabase client - centralized database connection for the backend.

Uses the service_role key to bypass Row Level Security (RLS),
since the backend operates with admin privileges to write submissions.
"""

from supabase import create_client, Client
from dotenv import load_dotenv
import os

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError(
        "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env file"
    )


def get_supabase_client() -> Client:
    """
    Returns a Supabase client configured with the service_role key.
    This client bypasses RLS - only use it from trusted backend code.
    """
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)