"""
Public runtime config — small, unauthenticated feature flags the frontend needs
BEFORE it knows who the user is (e.g. which voice-interview mode to mount).

Only NON-SENSITIVE switches belong here. Right now it exposes `voice_mode`
(realtime vs pipeline), which the admin flips live via the AI-providers panel;
the interview page fetches this on mount so the toggle takes effect with no
redeploy and no rebuild.
"""

from fastapi import APIRouter

from services.ai_providers import current_provider

router = APIRouter(prefix="/public-config", tags=["config"])


@router.get("")
async def public_config():
    return {"voice_mode": current_provider("voice_mode")}
