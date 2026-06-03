"""
Eventide Frontend Server
========================
Serves the dashboard UI and proxies OpenStreetMap tile requests.

This server handles everything that is NOT backend-device-specific:
  - The dashboard HTML (dashboard.html)
  - OSM map tile proxy with caching (/tiles/<z>/<x>/<y>.png)

All camera streams, API calls, gimbal, supervisor, recordings, and
playback endpoints are handled by the backend device (dashboard.py)
and are reached directly from the browser using the IP entered in the
dashboard header — no proxy needed here for any of that.

Usage:
    pip install flask requests
    python frontend_server.py \\
        --html-file ./dashboard.html \\
        --host 0.0.0.0 \\
        --port 8000

Then open http://<this-machine>:8000   (or via nginx at http://<host>/)
"""

import argparse
import os
import threading
from pathlib import Path

import requests as _http
from flask import Flask, abort, make_response, send_file

app = Flask(__name__, static_folder=None)

# ── Config (populated in main) ────────────────────────────────────────────────
cfg: dict = {}

# ── Simple in-memory tile cache ───────────────────────────────────────────────
# Maps (z, x, y) → bytes.  Good enough for a single-user dashboard; replace
# with a proper disk cache (e.g. diskcache or nginx proxy_cache) for heavier use.
_tile_cache: dict[tuple, bytes] = {}
_tile_lock  = threading.Lock()

OSM_BASE    = "https://tile.openstreetmap.org"
OSM_HEADERS = {
    "User-Agent": "Eventide-Frontend-Server/1.0",
    "Referer":    "https://www.openstreetmap.org/",
}

# ── Static HTML ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    html_path = Path(cfg.get("html_file", "dashboard.html")).resolve()
    if not html_path.exists():
        return (
            "dashboard.html not found. "
            "Pass --html-file or place it alongside frontend_server.py."
        ), 404
    return send_file(html_path, mimetype="text/html")


# ── OSM tile proxy ────────────────────────────────────────────────────────────

@app.route("/tiles/<int:z>/<int:x>/<int:y>.png")
def tile_proxy(z, x, y):
    """
    Proxy & cache OpenStreetMap tiles.

    Caching here means:
      - The browser doesn't need to hit the internet for tiles it has seen.
      - Tiles are served from this process's memory for repeated map pans.

    For production you'd want a disk-backed cache (nginx proxy_cache or
    diskcache) so tiles survive a server restart.
    """
    key = (z, x, y)
    with _tile_lock:
        cached = _tile_cache.get(key)

    if cached:
        resp = make_response(cached)
        resp.headers["Content-Type"]  = "image/png"
        resp.headers["Cache-Control"] = "public, max-age=2592000"  # 30 days
        resp.headers["X-Tile-Cache"]  = "HIT"
        return resp

    url = f"{OSM_BASE}/{z}/{x}/{y}.png"
    try:
        upstream = _http.get(url, headers=OSM_HEADERS, timeout=8)
        if upstream.status_code != 200:
            abort(upstream.status_code)
    except _http.exceptions.RequestException:
        abort(502)

    data = upstream.content
    with _tile_lock:
        # Limit cache size: evict oldest entry if over 4 000 tiles (~250 MB).
        if len(_tile_cache) >= 4000:
            oldest = next(iter(_tile_cache))
            del _tile_cache[oldest]
        _tile_cache[key] = data

    resp = make_response(data)
    resp.headers["Content-Type"]  = upstream.headers.get("Content-Type", "image/png")
    resp.headers["Cache-Control"] = "public, max-age=2592000"
    resp.headers["X-Tile-Cache"]  = "MISS"
    return resp


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Eventide Frontend Server")
    parser.add_argument(
        "--html-file", default="dashboard.html",
        help="Path to the dashboard HTML file (default: dashboard.html alongside this script)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    cfg.update(vars(args))

    print(f"[frontend] HTML file:   {args.html_file}")
    print(f"[frontend] Tile proxy:  http://{args.host}:{args.port}/tiles/<z>/<x>/<y>.png  →  {OSM_BASE}")
    print(f"[frontend] Serving at:  http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()