# Render "memory limit exceeded" — diagnosis & fix

**Service:** consilio-backend (FastAPI, `main.py`) on Render
**Symptom:** "memory limit exceeded" / the service restarts; scoring calls fail intermittently
**Diagnosed:** 2026-09-11 (Cowork), from `main.py` + `requirements.txt`. Two settings on the Render dashboard still need your eyes (marked ⬜ below).

---

## What "memory limit exceeded" actually is

Render kills your process when its RSS crosses the instance's RAM cap (the free/Starter tier is **512 MB**). A killed process exits with **status 137** — that's the tell in the logs. It's not a code bug per se; it's that the backend's resident memory is too close to the cap, so any request that allocates a bit more (a PDF, an image, a burst) tips it over and Render OOM-kills it. Mid-scoring, that looks like a failed evaluation to the user — which quietly costs you the activation you're fighting for.

Your backend is unusually heavy at **import time** for two reasons:

1. **`main.py` imports every route at startup** — `submit, news, cron, daily, transcribe, speak, realtime, attempts, vision, usage, deck_vault, resume, certificates, decks, deck_ingestion, ai_providers, public_config, realtime_gemini, agentic, coach`. Each pulls its services, which import openai + supabase + the Google SDKs + Pillow + pypdfium2. So the *entire* dependency graph is resident before the first request, even for routes nobody hits that day.

2. **The dependency list carries heavy, partly-unnecessary libraries:**
   - **`pyiceberg==0.11.1`** — Apache Iceberg (a data-lake table format). It drags in **PyArrow**, which alone is **~120–200 MB resident**. A case-scoring API has no obvious reason to import Iceberg. **This is the single biggest suspect.**
   - **Three Google SDKs** — `google-generativeai`, `google-genai`, `google-cloud-texttospeech` — all gRPC/protobuf-heavy (grpcio is ~50–100 MB).
   - **`pypdfium2` + `Pillow`** — PDF/image rendering, only needed by the vision/deck routes but loaded for everyone.

On 512 MB, baseline RSS can sit at 350–450 MB before serving anything — so you live one PDF or one traffic burst away from an OOM kill.

---

## The fix, in priority order

### ⬜ 1. Confirm you're running ONE worker (do this first — it's free)
On Render → your service → **Settings → Start Command**. It should be:
```
uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
```
If it says `--workers 2` (or more), or uses `gunicorn -w 2 …`, **that's very likely the whole problem**: each worker is a full copy of that ~400 MB import graph, so 2 workers = ~800 MB on a 512 MB box = guaranteed OOM. Set it to a single worker. One worker handles your current traffic (~6 sessions/day) with room to spare.

### ⬜ 2. Bump the instance one tier: 512 MB → 1 GB (cheapest reliable fix)
Render → service → **Settings → Instance Type**. A backend that OOM-kills mid-scoring is directly losing you conversions; the next tier up is a few dollars a month and buys immediate headroom. Do this even after the code fixes below — heavy Python API backends are simply tight on 512 MB.

### 3. Remove `pyiceberg` if nothing imports it (frees ~120–200 MB)
First verify it's actually unused:
```bash
grep -rin "iceberg\|pyarrow" routes/ services/ prompts/ main.py
```
If that returns nothing, `pyiceberg` is a stray dependency (likely pulled in once and never removed). Delete this line from `requirements.txt`:
```
pyiceberg==0.11.1
```
…and, if `pyarrow`/`pyroaring`/`zstandard`/`fsspec`/`mmh3` are also only there because of it (grep them too), drop those. Redeploy and watch the memory graph fall. **Don't remove it blind** — if the grep finds a real import, leave it and lean on fixes 1, 2 and 4 instead.

### 4. Lazy-import the heavy, rarely-used modules (lowers baseline RSS)
Move the expensive imports out of module top-level and into the functions that use them, so they only load when that route is actually hit:
- **`pypdfium2` and `Pillow`** — import inside the vision/deck-ingestion handlers, not at the top of their route/service modules.
- **`google-cloud-texttospeech`** — the code comment says it's already lazily imported in `routes/speak.py`; confirm that's still true.
- **`routes/agentic.py`, `routes/deck_ingestion.py`, `routes/certificates.py`** — these are admin/occasional. Consider importing their routers lazily, or at minimum keep their heavy service imports function-local.

Pattern:
```python
def extract_text_from_pdf(data: bytes):
    import pypdfium2 as pdfium   # imported on first use, not at startup
    ...
```

### 5. If you keep gunicorn, recycle workers to bleed off slow leaks
If you run gunicorn with the uvicorn worker, add:
```
gunicorn main:app -k uvicorn.workers.UvicornWorker -w 1 --max-requests 300 --max-requests-jitter 50 --timeout 120
```
`--max-requests` restarts the worker every ~300 requests, releasing any gradually-leaked memory before it reaches the cap. (Plain `uvicorn` has no equivalent; that's a reason to prefer the gunicorn+uvicorn combo here.)

---

## How to confirm the cause (2 minutes on the Render dashboard)
- **Metrics tab** → the Memory graph. If it climbs to the cap and sawtooths (drops on restart), that's the OOM pattern. If it's flat but near the ceiling, it's baseline-too-high (fixes 3–4). If it spikes on specific requests, it's per-request allocation (PDF/image — fix 4).
- **Logs tab** → search `Out of memory`, `Ran out of memory`, `SIGKILL`, or `exited with status 137`. Status **137** confirms an OOM kill rather than a crash.

---

## Expected result
Fix 1 (one worker) + fix 2 (1 GB) will almost certainly stop the OOM kills on their own. Fixes 3–4 lower baseline RSS so you could even stay on a smaller instance — but given it's your revenue-scoring path, the few-dollar instance bump is the honest first move. Order of effort-to-payoff: **1 → 2 → 3 → 4**.
