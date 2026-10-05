"""
Images for MECE Insights articles — generated with Google's Gemini image models only (owner,
2026-10-06: "if image generation is needed, use Gemini only"), never another provider.

  plan        the writer's art direction: a hero and 1-2 inline images, each a scene description,
              alt text and a caption (services/growth/daily_blog.py asks for it; art_plan() makes
              one for an older article that has none)
  generate    Gemini image model (DAILY_BLOG_IMAGE_MODEL, else the newest gemini-*-image model this
              key lists) with one house style, so every article looks like the same magazine:
              documentary editorial photography, India, natural light, no text, logos or real people
  store       Pillow -> WebP (page) + JPEG (hero only, for link previews), uploaded to the public
              Supabase Storage bucket `insights` (created on first use); the public URLs go into the
              article's content (hero / images), with a plain "Generated with Gemini" credit

Every step degrades: no model, no bucket or a failed call means an article without that image,
never a failed article. The page has a typographic cover for posts without a hero.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

BUCKET = "insights"
HERO_WIDTH = 2000
INLINE_WIDTH = 1600
OG_WIDTH = 1200
CREDIT = "Image generated with Gemini for MECE Insights"

HOUSE_STYLE = (
    "Editorial photograph for a long-form business and economics magazine, in the style of a documentary "
    "photo essay. Set in India. Natural light, real textures, calm and considered composition with negative "
    "space, muted and warm colour palette, shallow depth of field, shot on a 35mm lens. "
    "Strictly no text, no letters, no numbers, no logos, no brand names, no signage with writing, no watermarks, "
    "no recognisable real people or celebrities; any people are anonymous, seen from behind, at a distance or out of focus."
)

_models: Dict[str, Any] = {"at": 0.0, "list": []}
_dead: Dict[str, float] = {}


def available() -> bool:
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        return False
    try:
        import google.genai  # noqa: F401
        return True
    except Exception:
        return False


def _client():
    from google import genai
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def image_models() -> List[str]:
    """Gemini image models to try: DAILY_BLOG_IMAGE_MODEL, then what this key lists (newest first)."""
    out: List[str] = []
    env = (os.getenv("DAILY_BLOG_IMAGE_MODEL") or "").strip()
    if env:
        out.append(env.removeprefix("models/"))
    if time.time() - _models["at"] > 3600:
        names: List[str] = []
        try:
            for m in _client().models.list():
                name = (getattr(m, "name", "") or "").removeprefix("models/")
                actions = getattr(m, "supported_actions", None) or []
                if name.startswith("gemini") and "image" in name and (not actions or "generateContent" in actions):
                    names.append(name)
        except Exception as e:  # noqa: BLE001
            print(f"[insights images] model list failed: {type(e).__name__}: {e}")

        def key(n: str):
            m = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
            return (-(float(m.group(1)) if m else 0.0), "preview" in n, "pro" in n)
        _models.update(at=time.time(), list=sorted(dict.fromkeys(names), key=key))
    for n in _models["list"] + ["gemini-2.5-flash-image"]:
        if n not in out:
            out.append(n)
    return [m for m in out if time.time() - _dead.get(m, 0) > 6 * 3600]


def gemini_image(prompt: str, aspect: str = "16:9") -> Dict[str, Any]:
    """{bytes, mime, model} from the first Gemini image model that answers. Raises if none does."""
    from google.genai import types
    client = _client()
    errors = []
    for model in image_models():
        for with_config in (True, False):
            try:
                cfg = types.GenerateContentConfig(response_modalities=["IMAGE"])
                if with_config:
                    cfg = types.GenerateContentConfig(response_modalities=["IMAGE"],
                                                      image_config=types.ImageConfig(aspect_ratio=aspect))
                resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
                for cand in getattr(resp, "candidates", None) or []:
                    for part in getattr(getattr(cand, "content", None), "parts", None) or []:
                        blob = getattr(part, "inline_data", None)
                        if blob is not None and getattr(blob, "data", None):
                            return {"bytes": blob.data, "mime": getattr(blob, "mime_type", "image/png"), "model": model}
                errors.append(f"{model}: no image in the answer")
                break
            except Exception as e:  # noqa: BLE001
                t = f"{type(e).__name__} {e}".lower()
                if any(k in t for k in ("404", "not_found", "not found", "no longer available")):
                    _dead[model] = time.time()
                    errors.append(f"{model}: not available on this key")
                    break
                errors.append(f"{model}: {type(e).__name__}: {str(e)[:100]}")
                if not with_config:
                    break
    raise RuntimeError("; ".join(errors)[:500] or "no Gemini image model available")


def to_web(data: bytes, width: int) -> Dict[str, Any]:
    """WebP for the page (resized to `width`) and a 1200 px JPEG for link previews, plus the page size."""
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        if im.width > width:
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        w, h = im.size
        webp = io.BytesIO()
        try:
            im.save(webp, format="WEBP", quality=82, method=4)
            webp_bytes, webp_type = webp.getvalue(), "image/webp"
        except Exception:
            webp = io.BytesIO()
            im.save(webp, format="JPEG", quality=85, optimize=True, progressive=True)
            webp_bytes, webp_type = webp.getvalue(), "image/jpeg"
        # link previews (WhatsApp, LinkedIn, X) want a modest JPEG: 1200 px wide keeps it well under 300 KB
        og = im.resize((OG_WIDTH, round(h * OG_WIDTH / w)), Image.LANCZOS) if w > OG_WIDTH else im
        jpg = io.BytesIO()
        og.save(jpg, format="JPEG", quality=82, optimize=True, progressive=True)
        return {"web": webp_bytes, "web_type": webp_type, "jpeg": jpg.getvalue(), "width": w, "height": h}


_bucket_ready = {"ok": False}


def _ensure_bucket(supabase) -> None:
    if _bucket_ready["ok"]:
        return
    try:
        buckets = supabase.storage.list_buckets() or []
        names = [getattr(b, "name", None) or (b.get("name") if isinstance(b, dict) else None) for b in buckets]
        if BUCKET not in names:
            supabase.storage.create_bucket(BUCKET, options={"public": True})
        _bucket_ready["ok"] = True
    except Exception as e:  # noqa: BLE001 - the upload will say if it really is missing
        print(f"[insights images] bucket check: {type(e).__name__}: {e}")


def store(supabase, path: str, data: bytes, content_type: str) -> str:
    _ensure_bucket(supabase)
    st = supabase.storage.from_(BUCKET)
    st.upload(path, data, file_options={"content-type": content_type, "upsert": "true",
                                        "cache-control": "31536000"})
    url = st.get_public_url(path)
    return (url if isinstance(url, str) else (url or {}).get("publicUrl", "")).rstrip("?")


def _one(supabase, slug: str, item: Dict[str, Any], role: str, n: int, *, generate: Callable, upload: Callable) -> Optional[Dict[str, Any]]:
    scene = (item.get("prompt") or "").strip()
    if not scene:
        return None
    aspect = "16:9" if role == "hero" else "3:2"
    out = generate(f"{scene}\n\n{HOUSE_STYLE}", aspect)
    web = to_web(out["bytes"], HERO_WIDTH if role == "hero" else INLINE_WIDTH)
    tag = hashlib.sha1(scene.encode()).hexdigest()[:8]
    ext = "webp" if web["web_type"] == "image/webp" else "jpg"
    url = upload(supabase, f"{slug}/{role}-{n}-{tag}.{ext}", web["web"], web["web_type"])
    res = {"url": url, "width": web["width"], "height": web["height"], "alt": (item.get("alt") or "")[:300],
           "caption": (item.get("caption") or "")[:300], "credit": CREDIT, "model": out.get("model")}
    if role == "hero":
        res["og_url"] = upload(supabase, f"{slug}/og-{tag}.jpg", web["jpeg"], "image/jpeg")
    else:
        res["after_section"] = int(item.get("after_section") or 1)
    return res


def make_images(supabase, slug: str, art: Dict[str, Any], *, generate: Callable = gemini_image,
                upload: Callable = store) -> Dict[str, Any]:
    """{hero?, images: [...], errors: [...]} for an article's art direction. Never raises."""
    if not art or not (generate is not gemini_image or available()):
        return {"images": [], "errors": ["Gemini image generation not available (GEMINI_API_KEY)"]}
    jobs = []
    if art.get("hero"):
        jobs.append(("hero", 0, art["hero"]))
    for i, item in enumerate((art.get("inline") or [])[:2]):
        jobs.append(("inline", i + 1, item))
    out: Dict[str, Any] = {"images": [], "errors": []}

    def run(job):
        role, n, item = job
        try:
            return role, _one(supabase, slug, item, role, n, generate=generate, upload=upload), None
        except Exception as e:  # noqa: BLE001
            return role, None, f"{role}: {type(e).__name__}: {str(e)[:160]}"

    with ThreadPoolExecutor(max_workers=3) as ex:
        for role, res, err in ex.map(run, jobs):
            if err:
                out["errors"].append(err)
            elif res and role == "hero":
                out["hero"] = res
            elif res:
                out["images"].append(res)
    out["images"].sort(key=lambda x: x.get("after_section", 1))
    return out


_ART_SYSTEM = """You are the picture editor of MECE Insights, a long-form business magazine for Indian MBA students. \
For the article below, describe the photographs to commission: one hero image and two inline images. Each is a \
concrete, visual scene set in India that evokes the story without illustrating a specific company: places, objects, \
work, crowds from a distance, hands, streets, warehouses, markets, offices. No text, logos, brands, charts or real \
people. Return ONLY JSON: {"hero": {"prompt": "one or two sentences describing the scene", "alt": "what the image \
shows", "caption": "one short sentence connecting the image to the story"}, "inline": [{"after_section": 1, \
"prompt": "...", "alt": "...", "caption": "..."}, {"after_section": 3, "prompt": "...", "alt": "...", "caption": "..."}]}"""


def art_plan(chat: Callable, article: Dict[str, Any]) -> Dict[str, Any]:
    """Art direction for an article written without one (older posts)."""
    import json
    c = article.get("content") or {}
    brief = {"title": article.get("title"), "dek": article.get("dek"),
             "sections": [s.get("heading") for s in c.get("sections") or []],
             "lede": c.get("lede") or c.get("summary") or c.get("intro")}
    try:
        resp, _, _ = chat("seo_critique", messages=[{"role": "system", "content": _ART_SYSTEM},
                                                     {"role": "user", "content": json.dumps(brief, ensure_ascii=False)}],
                          response_format={"type": "json_object"}, temperature=0.6, max_tokens=600)
        txt = resp.choices[0].message.content or "{}"
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}
