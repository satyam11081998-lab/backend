"""
Deck page renderer using pypdfium2 (BSD/Apache-2.0).

Rasterises every page of a competition deck PDF to WebP at ~1600px width with
lossy compression (quality ~80). Watermarks the free preview pages with the
public deck URL (mece.in/decks/<slug>). Stores all rendered images in the
private Supabase bucket `deck-pages/<deck_id>/<n>.webp`.

Also generates a standard JPEG for slide 1 (`deck-pages/<deck_id>/og.jpg`) to
ensure link previews on social/chat platforms and AI search scrapers render
reliably without WebP compatibility issues.
"""

import io
import time
from typing import Optional

from PIL import Image, ImageDraw, ImageFont
import pypdfium2 as pdfium

from services.supabase_client import get_supabase_client

BUCKET_NAME = "deck-pages"
TARGET_WIDTH = 1600
WEBP_QUALITY = 80
JPEG_QUALITY = 85


def _ensure_bucket_exists(supabase):
    """Ensure the private deck-pages bucket exists in Supabase Storage."""
    try:
        buckets = supabase.storage.list_buckets()
        existing = [b.name for b in buckets] if hasattr(buckets[0], 'name') else [b.get('name') for b in buckets]
        if BUCKET_NAME not in existing:
            supabase.storage.create_bucket(BUCKET_NAME, options={"public": False})
    except Exception:
        pass


def _apply_watermark(image: Image.Image, watermark_text: str) -> Image.Image:
    """Draw a clean, professional footer watermark on free preview pages."""
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    width, height = base.size
    font_size = max(16, int(width * 0.016))

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    label = f"MECE Deck Vault  •  {watermark_text}"
    
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    pad_x = int(width * 0.02)
    pad_y = int(height * 0.02)
    bar_padding = 8

    x1 = width - pad_x - text_w - (bar_padding * 2)
    y1 = height - pad_y - text_h - (bar_padding * 2)
    x2 = width - pad_x
    y2 = height - pad_y

    draw.rounded_rectangle(
        [x1, y1, x2, y2],
        radius=6,
        fill=(15, 23, 42, 180),  # slate-900 at ~70% opacity
    )
    draw.text(
        (x1 + bar_padding, y1 + bar_padding - bbox[1]),
        label,
        font=font,
        fill=(255, 255, 255, 230),  # bright white at ~90% opacity
    )

    watermarked = Image.alpha_composite(base, overlay)
    return watermarked.convert("RGB")


def render_deck_pages(
    deck_id: str,
    pdf_bytes: bytes,
    slug: str,
    effective_free_pages: int,
) -> dict:
    """Rasterise all pages of a PDF using pypdfium2 and save to Supabase private storage."""
    if not pdf_bytes or len(pdf_bytes) < 10:
        raise ValueError("PDF content is empty or corrupted")

    pdf = pdfium.PdfDocument(pdf_bytes)
    page_count = len(pdf)
    if page_count < 1:
        raise ValueError("PDF contains no pages")

    supabase = get_supabase_client()
    _ensure_bucket_exists(supabase)
    storage = supabase.storage.from_(BUCKET_NAME)

    watermark_url = f"mece.in/decks/{slug}" if slug else "mece.in/decks"

    for i in range(page_count):
        page_num = i + 1
        page = pdf[i]

        w, h = page.get_size()
        scale = TARGET_WIDTH / w if w > 0 else 2.0
        bitmap = page.render(scale=scale)
        img = bitmap.to_pil()

        if page_num <= effective_free_pages:
            img = _apply_watermark(img, watermark_url)

        # 1. Save WebP: <deck_id>/<page_num>.webp
        webp_buffer = io.BytesIO()
        img.save(webp_buffer, format="WEBP", quality=WEBP_QUALITY, method=4)
        webp_bytes = webp_buffer.getvalue()

        dest_path = f"{deck_id}/{page_num}.webp"
        storage.upload(
            dest_path,
            webp_bytes,
            file_options={"content-type": "image/webp", "upsert": "true"},
        )

        # 2. For Slide 1, also render JPEG for social/AI scrapers (OG preview)
        if page_num == 1:
            jpeg_buffer = io.BytesIO()
            img.save(jpeg_buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
            jpeg_bytes = jpeg_buffer.getvalue()

            storage.upload(
                f"{deck_id}/og.jpg",
                jpeg_bytes,
                file_options={"content-type": "image/jpeg", "upsert": "true"},
            )
            storage.upload(
                f"{deck_id}/1.jpg",
                jpeg_bytes,
                file_options={"content-type": "image/jpeg", "upsert": "true"},
            )

    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    supabase.table("deck_skeletons").update({
        "page_count": page_count,
        "pages_rendered_at": now_iso,
    }).eq("id", deck_id).execute()

    return {
        "page_count": page_count,
        "rendered_pages": page_count,
        "rendered_at": now_iso,
    }