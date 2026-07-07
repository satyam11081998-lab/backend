import os
import time
from typing import Optional
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

from services.supabase_client import get_supabase_client
from services.auth import get_verified_user_id
from services.rate_limit import check_rate_limit
from services.ai_usage import (
    assert_ocr_quota,
    assert_daily_budget,
    get_ai_input_quota,
    log_ai_usage,
)

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=60.0, max_retries=1)

router = APIRouter()

# ~6M chars of base64 ≈ 4.5 MB decoded. The client already downscales to 1200px,
# so anything larger is anomalous — reject before paying for a vision call.
MAX_B64_LEN = 6_000_000


class ExtractTextRequest(BaseModel):
    base64_image: str


@router.post("")
async def extract_text(
    request: ExtractTextRequest,
    authorization: Optional[str] = Header(default=None),
):
    """
    OCR a handwritten/printed image via gpt-4o-mini vision (OCR is a mini-strength
    task, ~94% cheaper than gpt-4o).

    Guarded: requires a valid Supabase JWT, rate-limited, bounded by the caller's
    per-tier daily image quota, payload-size-capped, and logged to ai_usage_log.
    """
    supabase = get_supabase_client()
    uid = get_verified_user_id(supabase, authorization)          # 401 if missing/invalid
    check_rate_limit(f"ocr:{uid}", max_calls=12, window_seconds=60)
    assert_daily_budget()                                        # 503 if global cap hit
    assert_ocr_quota(supabase, uid)                             # 429 if user out of images

    try:
        if not request.base64_image:
            raise HTTPException(status_code=400, detail="Empty image data")
        if len(request.base64_image) > MAX_B64_LEN:
            raise HTTPException(status_code=413, detail="Image too large — please use a smaller photo.")

        base64_data = request.base64_image
        if not base64_data.startswith("data:image"):
            base64_data = f"data:image/jpeg;base64,{base64_data}"

        t0 = time.time()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert OCR system. Extract all handwritten or printed text from this image exactly as written. Preserve structural elements like bullet points, numbering, and indentation. Do not add any conversational filler or introductions; return ONLY the extracted text.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": base64_data}}
                    ],
                },
            ],
            max_tokens=1500,  # a dense handwritten page can be long; avoid truncating the OCR text
        )
        latency_ms = int((time.time() - t0) * 1000)

        extracted_text = response.choices[0].message.content
        log_ai_usage(
            user_id=uid, endpoint="/extract-text", model="gpt-4o-mini",
            response=response, latency_ms=latency_ms, success=True,
        )
        return {"text": extracted_text, "quota": get_ai_input_quota(supabase, uid)}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")
