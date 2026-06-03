"""
Render Keep-Alive Pinger (self-contained, lives inside the backend repo).

Keeps the Render free-tier backend warm so the scheduled daily jobs
(/cron/schedule-daily at 00:01 IST, /cron/fetch-news at 06:00 IST) hit a
live server instead of a cold-starting one.

Run by .github/workflows/keep-alive.yml every minute. A deterministic
7-minute window picks one random minute per window to actually ping, so
most invocations exit immediately (cheap) and the max gap between real
pings is < 13 minutes — comfortably under Render's 15-minute sleep.

Zero dependencies (Python standard library only).
Target URL: env RENDER_URL, else the default below.
"""

import urllib.request
import urllib.error
import time
import sys
import json
import random
import os

# Windows-only emoji safety (harmless on Linux CI).
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Override in CI with the RENDER_URL repo variable/secret if the URL ever changes.
RENDER_URL = os.environ.get("RENDER_URL", "https://consilio-backend.onrender.com").rstrip("/")

WINDOW_MINUTES = 7  # max gap = 2*7 - 1 = 13 min

ENDPOINTS = ["/", "/health"]  # real endpoints on this backend


def _ping(url: str, method: str = "GET", body: bytes | None = None, timeout: int = 30) -> bool:
    headers = {
        "User-Agent": "MECE-KeepAlive/1.0 (+github-actions)",
        "Accept": "application/json, text/html, */*",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read(512)  # force the connection to complete
            print(f"  OK   {method} {url} -> {resp.status}")
            return True
    except urllib.error.HTTPError as e:
        # Any HTTP response (even 404/405) means the server is AWAKE.
        print(f"  AWAKE {method} {url} -> HTTP {e.code}")
        return True
    except Exception as e:
        print(f"  FAIL {method} {url} -> {type(e).__name__}: {e}")
        return False


def main() -> None:
    print("=" * 56)
    print("MECE Render Keep-Alive")
    print(f"Target: {RENDER_URL}")
    print(f"Time:   {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("=" * 56)

    # Deterministic window scheduling: one ping per 7-min window.
    current_minute = int(time.time()) // 60
    window_id = current_minute // WINDOW_MINUTES
    minute_in_window = current_minute % WINDOW_MINUTES
    chosen = random.Random(window_id).randint(0, WINDOW_MINUTES - 1)

    print(f"Window #{window_id} | minute {minute_in_window}/{WINDOW_MINUTES - 1} | chosen {chosen}")
    if minute_in_window != chosen:
        print("Not this minute — skipping.")
        sys.exit(0)

    print("Pinging now.")
    ok = False

    # Primary wake-up; retry once after a cold-start pause if it fails.
    if _ping(RENDER_URL + "/"):
        ok = True
    else:
        print("  Possibly cold-starting; waiting 12s then retrying...")
        time.sleep(12)
        ok = _ping(RENDER_URL + "/") or ok

    # Warm the listed endpoints.
    for ep in ENDPOINTS:
        ok = _ping(RENDER_URL + ep) or ok
        time.sleep(0.4)

    print("=" * 56)
    print("Server is ALIVE." if ok else "Could not reach server — check RENDER_URL.")
    print("=" * 56)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
