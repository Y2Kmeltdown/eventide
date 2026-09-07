"""
Eventide Backend
================
The backbone process of an eventide payload: serves the dashboard UI and
OSM tile proxy, exposes the control/file APIs, hosts the module manager
(/api/modules/*), and proxies supervisord's XML-RPC endpoint.

This process CAN serve the full frontend itself (HTML at / plus an OSM
tile proxy at /tiles/), which is handy when talking to the device
directly (field laptop, local network).  For data-constrained links the
decoupled frontend server (frontend.py) is still preferred: it keeps
HTML/tile bandwidth off the link — see eventide.nginx.

Module network locations are NOT hardcoded in nginx.  TCP ports for module
sockets are allocated by this server (see --port-pool) — at install time, or
per instance at instance-creation time for instanceable modules — and any
HTTP service a module exposes is reachable through the generic proxy

  /proxy/<module>/<socket>/<upstream path>         →  127.0.0.1:<allocated port>/<upstream path>
  /proxy/<module>/inst/<iid>/<socket>/<upstream…>  →  same, for one instance

resolved from the installed-modules registry at request time.  nginx only
fronts this process (location /) and the base playback server (/playback/).

Usage:
    pip install flask
    python eventide.py \\
        --recordings-dir /tmp/evk4_raw \\
        --viewfinder-bin ./target/release/viewfinder \\
        --replay-bin     ./target/release/replay

Then open http://localhost:5000  (or via nginx at http://<host>/)
"""

import argparse
import glob
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests as _http

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    make_response,
    request,
    send_file,
    send_from_directory,
)

app = Flask(__name__, static_folder=None)

# ── CORS — allow the frontend server (any origin) to call the API ─────────────
# The dashboard HTML can be served two ways: by this process (same-origin —
# CORS irrelevant) or by the decoupled frontend server, in which case the
# browser makes cross-origin requests here.  We allow all origins because the
# frontend host/port is not known at deploy time.  If you want to restrict
# this, set ALLOWED_ORIGIN in the environment or hardcode it below.
_ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = _ALLOWED_ORIGIN
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>",             methods=["OPTIONS"])
def cors_preflight(path=""):
    resp = app.make_default_options_response()
    resp.headers["Access-Control-Allow-Origin"]  = _ALLOWED_ORIGIN
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

# ── Frontend: dashboard HTML + OSM tile proxy ─────────────────────────────────
# eventide.py can serve the whole UI itself for direct/standalone access.
# When the page is loaded from here, the browser's relative API/stream/tile
# URLs resolve to this process (or the nginx vhost in front of it) — no
# backend IP needs to be configured in the UI.  The tile proxy is the same
# one the decoupled frontend.py runs; keep in mind that tiles served from
# here cross the device's link, so constrained-link deployments should keep
# using frontend.py for the page.

@app.route("/")
def index():
    html_path = Path(cfg.get("html_file", "dashboard.html")).resolve()
    if not html_path.exists():
        return (
            "dashboard.html not found. "
            "Pass --html-file or place it alongside eventide.py."
        ), 404
    return send_file(html_path, mimetype="text/html")


# In-memory tile cache: (z, x, y) → bytes.  Fine for a single-user dashboard.
_tile_cache: dict[tuple, bytes] = {}
_tile_lock  = threading.Lock()

_OSM_BASE    = "https://tile.openstreetmap.org"
_OSM_HEADERS = {
    "User-Agent": "Eventide-Dashboard/1.0",
    "Referer":    "https://www.openstreetmap.org/",
}


@app.route("/tiles/<int:z>/<int:x>/<int:y>.png")
def tile_proxy(z, x, y):
    key = (z, x, y)
    with _tile_lock:
        cached = _tile_cache.get(key)
    if cached:
        resp = make_response(cached)
        resp.headers["Content-Type"]  = "image/png"
        resp.headers["Cache-Control"] = "public, max-age=2592000"  # 30 days
        resp.headers["X-Tile-Cache"]  = "HIT"
        return resp

    try:
        upstream = _http.get(f"{_OSM_BASE}/{z}/{x}/{y}.png",
                             headers=_OSM_HEADERS, timeout=8)
        if upstream.status_code != 200:
            abort(upstream.status_code)
    except _http.exceptions.RequestException:
        abort(502)

    data = upstream.content
    with _tile_lock:
        # Evict oldest entry over 4 000 tiles (~250 MB).
        if len(_tile_cache) >= 4000:
            del _tile_cache[next(iter(_tile_cache))]
        _tile_cache[key] = data

    resp = make_response(data)
    resp.headers["Content-Type"]  = upstream.headers.get("Content-Type", "image/png")
    resp.headers["Cache-Control"] = "public, max-age=2592000"
    resp.headers["X-Tile-Cache"]  = "MISS"
    return resp

# ── Global process state ──────────────────────────────────────────────────────

proc_lock = threading.Lock()

# Managed viewfinder processes keyed by mode
viewfinders: dict[str, subprocess.Popen | None] = {"live": None}


# Current configs (persisted so the UI can reflect them)
vf_configs: dict[str, dict] = {
    "live": {"fps": 50, "quality": 80, "width": 1280, "height": 720},
}

# Per-camera stream configs (sent to hardware/mjpeg process)
stream_configs: dict[str, dict] = {
    "evk":   {"quality": 80, "width": 1280, "height": 720},
    "picam": {"quality": 80, "width": 1280, "height": 720},
    "ircam": {"quality": 80, "width": 1280, "height": 720},
}

# Config (populated in main)
cfg: dict = {}

# ── System-wide settings ──────────────────────────────────────────────────────
# Editable from the dashboard SETTINGS tab. Persisted to disk so values survive
# backend restarts. New keys can be added freely; the UI only sends values that
# the user actually changed.

DEFAULT_SETTINGS = {
    "ui_refresh_interval_ms": 5000,
    "default_playback_speed": 1.0,
    "timezone": "UTC",
}

_settings: dict = {}
_settings_lock = threading.Lock()


def settings_path() -> Path:
    return Path(cfg.get("settings_file", "/usr/local/eventide/data/settings.json"))


def load_settings() -> dict:
    path = settings_path()
    if not path.exists():
        return dict(DEFAULT_SETTINGS)
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            merged = dict(DEFAULT_SETTINGS)
            merged.update(data)
            return merged
    except (json.JSONDecodeError, OSError):
        pass
    return dict(DEFAULT_SETTINGS)


def save_settings(data: dict) -> None:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(data, indent=2) + "\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def kill_proc(proc: subprocess.Popen | None) -> None:
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


def start_viewfinder(mode: str, params: dict) -> tuple[subprocess.Popen | None, str | None]:
    """Start or restart a viewfinder process for `mode` ('live' or 'replay')."""
    bind_port    = cfg["live_port"]   if mode == "live" else cfg["replay_port"]
    events_sock  = cfg["live_events_socket"] if mode == "live" else cfg["replay_events_socket"]

    cmd = [
        cfg["viewfinder_bin"],
        "--bind",          f"0.0.0.0:{bind_port}",
        "--events-socket", events_sock,
        "--fps",           str(params.get("fps", 50)),
        "--quality",       str(params.get("quality", 80)),
        "--width",         str(params.get("width", 1280)),
        "--height",        str(params.get("height", 720)),
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return proc, None
    except FileNotFoundError:
        return None, f"viewfinder binary not found: {cfg['viewfinder_bin']}"

# ── Supervisord XML-RPC proxy ─────────────────────────────────────────────────
# Previously the browser hit /supervisor/ via nginx, which set Content-Type:
# text/xml and forwarded to supervisord's RPC2 endpoint.  Now the dashboard is
# cross-origin the browser issues a CORS preflight (OPTIONS) first — and
# supervisord's own HTTP server rejects OPTIONS with a 501, killing the call
# before the real POST ever arrives.
#
# Solution: Flask intercepts /supervisor/ and proxies it itself.  OPTIONS is
# answered immediately with the right CORS headers (handled by the global
# cors_preflight route above), and POST is forwarded server-side to
# supervisord, which never sees the cross-origin problem.

_SUPERVISOR_RPC_URL = os.environ.get(
    "SUPERVISOR_RPC_URL", "http://127.0.0.1:8080/RPC2"
)

@app.route("/supervisor/", methods=["POST", "OPTIONS"])
def supervisor_proxy():
    """Forward XML-RPC calls to supervisord's inet_http_server.

    OPTIONS is handled here explicitly so Flask doesn't 405 it before the
    global cors_preflight catch-all can respond — a 405 on the preflight
    causes the browser to abort the real POST immediately.
    """
    if request.method == "OPTIONS":
        resp = app.make_default_options_response()
        resp.headers["Access-Control-Allow-Origin"]  = _ALLOWED_ORIGIN
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    body = request.get_data()
    try:
        upstream = _http.post(
            _SUPERVISOR_RPC_URL,
            data=body,
            headers={"Content-Type": "text/xml"},
            timeout=10,
        )
    except _http.exceptions.ConnectionError:
        return (
            '<?xml version="1.0"?><methodResponse><fault><value>'
            '<struct><member><name>faultCode</name><value><int>-1</int></value></member>'
            '<member><name>faultString</name><value><string>'
            'supervisord unreachable'
            '</string></value></member></struct>'
            '</value></fault></methodResponse>',
            502,
            {"Content-Type": "text/xml"},
        )
    return upstream.content, upstream.status_code, {"Content-Type": "text/xml"}

# ── Playback server proxy ─────────────────────────────────────────────────────
# On a real payload nginx fronts the base playback server (/playback/ →
# 127.0.0.1:8084 — see eventide.nginx).  When this process is talked to
# directly instead (a remote frontend pointed at this port, bench installs
# without nginx), /playback/ would 404/405 here.  Forward it so the backend
# behaves identically with or without nginx in front.

_PLAYBACK_UPSTREAM_URL = os.environ.get(
    "PLAYBACK_UPSTREAM_URL", "http://127.0.0.1:8084"
)


def _playback_upstream_url() -> str:
    """Resolve the playback server from the eventide-core module when installed.

    If the default module is registered and has a 'playback' TCP socket, use
    its allocated/declared port. Otherwise fall back to the PLAYBACK_UPSTREAM_URL
    environment variable / default so the proxy keeps working during install or
    when the module is not yet registered.
    """
    try:
        entry = load_registry().get("modules", {}).get("eventide-core")
        if entry:
            inst = (entry.get("instances") or {}).get("default")
            if inst:
                for s in (inst.get("sockets") or []):
                    if (
                        isinstance(s, dict)
                        and s.get("name") == "playback"
                        and s.get("type") == "tcp"
                        and isinstance(s.get("port"), int)
                    ):
                        return f"http://127.0.0.1:{s['port']}"
    except Exception:
        pass
    return _PLAYBACK_UPSTREAM_URL


@app.route("/playback/", defaults={"rest": ""},
           methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
@app.route("/playback/<path:rest>",
           methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
def playback_proxy(rest):
    if request.method == "OPTIONS":
        resp = app.make_default_options_response()
        resp.headers["Access-Control-Allow-Origin"]  = _ALLOWED_ORIGIN
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    url = f"{_playback_upstream_url()}/{rest}"
    if request.query_string:
        url += "?" + request.query_string.decode()
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP_HEADERS
    }
    try:
        upstream = _http.request(
            request.method,
            url,
            headers=headers,
            data=request.get_data(),
            stream=True,
            # Connect deadline only — no read deadline, or MJPEG streams die.
            timeout=(5, None),
        )
    except _http.exceptions.ConnectionError:
        return jsonify({"error": "playback server is unreachable"}), 502
    except _http.exceptions.RequestException as exc:
        return jsonify({"error": f"proxy error: {exc}"}), 502

    resp_headers = {
        k: v for k, v in upstream.headers.items()
        if k.lower() not in _HOP_BY_HOP_HEADERS
    }
    return Response(
        upstream.iter_content(chunk_size=64 * 1024),
        status=upstream.status_code,
        headers=resp_headers,
    )

# ── Module socket proxy ───────────────────────────────────────────────────────
# Generic reverse proxy for HTTP services exposed by installed modules.  The
# dashboard resolves module/socket names from /api/modules and calls
#   /proxy/<module>/<socket>/<upstream path>
# which is forwarded to 127.0.0.1:<allocated port>/<upstream path>.  This
# replaces per-module nginx locations: nginx only fronts eventide.py, and a
# module's network location is derived from the registry at request time, so
# modules can come and go without any web-server config changes.
#
# Responses are streamed (requests stream=True + a generator Response) so
# long-lived MJPEG streams work; Flask's dev server is threaded by default,
# so an open stream does not block ordinary API calls.

_HOP_BY_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "host", "content-length",
}


def _instance_tcp_port(module: str, iid: str, socket_name: str) -> int | None:
    """Allocated TCP port of one module instance's socket, or None."""
    entry = load_registry().get("modules", {}).get(module)
    if entry is None:
        return None
    inst = (entry.get("instances") or {}).get(iid)
    if inst is None:
        return None
    for s in (inst or {}).get("sockets", []):
        if isinstance(s, dict) and s.get("name") == socket_name and s.get("type") == "tcp":
            port = s.get("port")
            return port if isinstance(port, int) else None
    return None


def _module_tcp_port(module: str, socket_name: str) -> int | None:
    """Allocated TCP port of an installed module's socket, or None.

    Resolves against the implicit "default" instance, so this only works for
    non-instanceable modules (instanceable modules are proxied per instance —
    see /proxy/<module>/inst/<iid>/<socket>/...).
    """
    return _instance_tcp_port(module, "default", socket_name)


def _proxy_options_response():
    resp = app.make_default_options_response()
    resp.headers["Access-Control-Allow-Origin"]  = _ALLOWED_ORIGIN
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


def _proxy_to_port(port: int, rest: str, desc: str):
    """Forward the current request to 127.0.0.1:<port>/<rest> (streamed)."""
    url = f"http://127.0.0.1:{port}/{rest}"
    if request.query_string:
        url += "?" + request.query_string.decode()
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP_HEADERS
    }
    try:
        upstream = _http.request(
            request.method,
            url,
            headers=headers,
            data=request.get_data(),
            stream=True,
            # Connect deadline only — no read deadline, or MJPEG streams die.
            timeout=(5, None),
        )
    except _http.exceptions.ConnectionError:
        return jsonify({
            "error": f"{desc} is unreachable on port {port}"
        }), 502
    except _http.exceptions.RequestException as exc:
        return jsonify({"error": f"proxy error: {exc}"}), 502

    resp_headers = {
        k: v for k, v in upstream.headers.items()
        if k.lower() not in _HOP_BY_HOP_HEADERS
    }
    return Response(
        upstream.iter_content(chunk_size=64 * 1024),
        status=upstream.status_code,
        headers=resp_headers,
    )


@app.route("/proxy/<module>/<socket_name>/", defaults={"rest": ""},
           methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
@app.route("/proxy/<module>/<socket_name>/<path:rest>",
           methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
def module_socket_proxy(module, socket_name, rest):
    port = _module_tcp_port(module, socket_name)
    if port is None:
        entry = load_registry().get("modules", {}).get(module)
        if entry is not None and entry.get("manifest", {}).get("instance"):
            return jsonify({
                "error": f"module '{module}' is instanceable — proxy per "
                         f"instance at /proxy/{module}/inst/<iid>/{socket_name}/..."
            }), 404
        return jsonify({
            "error": f"no such module tcp socket: {module}/{socket_name}"
        }), 404

    if request.method == "OPTIONS":
        return _proxy_options_response()
    return _proxy_to_port(port, rest, f"{module}/{socket_name}")


@app.route("/proxy/<module>/inst/<iid>/<socket_name>/", defaults={"rest": ""},
           methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
@app.route("/proxy/<module>/inst/<iid>/<socket_name>/<path:rest>",
           methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
def module_instance_socket_proxy(module, iid, socket_name, rest):
    port = _instance_tcp_port(module, iid, socket_name)
    if port is None:
        return jsonify({
            "error": f"no such module instance tcp socket: {module}/{iid}/{socket_name}"
        }), 404

    if request.method == "OPTIONS":
        return _proxy_options_response()
    return _proxy_to_port(port, rest, f"{module}/{iid}/{socket_name}")

# ── Viewfinder API ────────────────────────────────────────────────────────────

@app.route("/api/viewfinder/<mode>/start", methods=["POST"])
def api_vf_start(mode):
    if mode not in ("live", "replay"):
        abort(400)
    data = request.get_json() or {}
    params = {
        "fps":     int(data.get("fps",     vf_configs[mode]["fps"])),
        "quality": int(data.get("quality", vf_configs[mode]["quality"])),
        "width":   int(data.get("width",   vf_configs[mode]["width"])),
        "height":  int(data.get("height",  vf_configs[mode]["height"])),
    }
    with proc_lock:
        kill_proc(viewfinders[mode])
        proc, err = start_viewfinder(mode, params)
        if err:
            return jsonify({"error": err}), 500
        viewfinders[mode] = proc
        vf_configs[mode]  = params
    return jsonify({"ok": True, "params": params})


@app.route("/api/viewfinder/<mode>/stop", methods=["POST"])
def api_vf_stop(mode):
    if mode not in ("live", "replay"):
        abort(400)
    with proc_lock:
        kill_proc(viewfinders[mode])
        viewfinders[mode] = None
    return jsonify({"ok": True})


@app.route("/api/viewfinder/<mode>/status")
def api_vf_status(mode):
    if mode not in ("live", "replay"):
        abort(400)
    with proc_lock:
        proc    = viewfinders[mode]
        running = proc is not None and proc.poll() is None
    return jsonify({"running": running, "params": vf_configs[mode]})

# ── Per-camera stream config ──────────────────────────────────────────────────

@app.route("/api/stream/<cam>/config", methods=["POST"])
def api_stream_config(cam):
    """Apply JPEG quality / resolution settings for a live stream camera."""
    if cam not in stream_configs:
        abort(400)
    data = request.get_json() or {}
    cfg_update = {
        "quality": int(data.get("quality", stream_configs[cam]["quality"])),
        "width":   int(data.get("width",   stream_configs[cam]["width"])),
        "height":  int(data.get("height",  stream_configs[cam]["height"])),
    }
    stream_configs[cam].update(cfg_update)
    # Hook: if you have a socket/IPC interface to the camera processes, send
    # the new settings here.  For now we just persist and acknowledge.
    return jsonify({"ok": True, "cam": cam, "params": stream_configs[cam]})


@app.route("/api/stream/<cam>/config")
def api_stream_config_get(cam):
    if cam not in stream_configs:
        abort(400)
    return jsonify(stream_configs[cam])

# ── System settings ───────────────────────────────────────────────────────────
# The SETTINGS tab reads/writes a small persisted JSON blob. Unknown keys are
# accepted so the UI can evolve without backend changes; type coercion is left
# to the UI for now.

@app.route("/api/settings")
def api_settings_get():
    global _settings
    with _settings_lock:
        _settings = load_settings()
        return jsonify(_settings)


@app.route("/api/settings", methods=["POST"])
def api_settings_post():
    global _settings
    data = request.get_json() or {}
    if not isinstance(data, dict):
        return jsonify({"error": "settings body must be a JSON object"}), 400
    with _settings_lock:
        _settings = load_settings()
        _settings.update(data)
        save_settings(_settings)
        return jsonify(_settings)


# ── Recordings ────────────────────────────────────────────────────────────────
# Recording sources are module-driven: every installed module that declares
# `recordings_subdir` in its manifest gets a recordings directory
# (<recordings_dir>/<recordings_subdir>) and shows up as a source here.  For
# instanceable modules each instance records to its own directory
# (<recordings_subdir>-<iid>) and appears as its own source.  The dashboard's
# PLAYBACK tab builds one inner tab per source — sources no installed module
# declares simply do not exist.

def _recording_sources() -> list[dict]:
    """Recording sources declared by installed modules: [{name, module}]."""
    sources = []
    for mod_name, entry in load_registry().get("modules", {}).items():
        manifest = entry.get("manifest", {})
        instanceable = bool(manifest.get("instance"))
        for inst in (entry.get("instances") or {}).values():
            sub = _instance_recordings_subdir(manifest, inst or {})
            if sub:
                label = f"{mod_name} ({(inst or {}).get('id')})" if instanceable else mod_name
                sources.append({"name": sub, "module": label})
    sources.sort(key=lambda s: s["name"])
    return sources


def _recordings_dir(source: str) -> Path:
    return Path(cfg["recordings_dir"]) / source


def _favorites_path(source: str) -> Path:
    return _recordings_dir(source) / ".favorites.json"


def _load_favorites(source: str) -> set[str]:
    path = _favorites_path(source)
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return {str(x) for x in data if isinstance(x, str)}
    except (json.JSONDecodeError, OSError):
        pass
    return set()


def _save_favorites(source: str, favorites: set[str]) -> None:
    path = _favorites_path(source)
    _atomic_write(path, json.dumps(sorted(favorites), indent=2) + "\n")


@app.route("/api/recordings")
def list_recording_sources():
    """List the recording sources declared by installed modules."""
    return jsonify({"sources": _recording_sources()})


@app.route("/api/recordings/<source>")
def list_recordings_source(source):
    if source not in {s["name"] for s in _recording_sources()}:
        return jsonify({"error": f"unknown recording source: {source}"}), 404
    return _list_recordings_for(source)


RECORDING_EXTENSIONS = ("*.raw", "*.mp4", "*.h264", "*.jsonl", "*.basler")

def _list_recordings_for(source: str):
    recordings_dir = _recordings_dir(source)
    if not recordings_dir.exists():
        return jsonify({"files": []})
    favorites = _load_favorites(source)
    seen = set()
    entries = []
    for pattern in RECORDING_EXTENSIONS:
        for f in recordings_dir.glob(pattern):
            if f.is_file() and f.name not in seen:
                seen.add(f.name)
                entries.append({
                    "name": f.name,
                    "size": f.stat().st_size,
                    "ext": f.suffix.lstrip("."),
                    "favorited": f.name in favorites,
                })
    # Favorites first, then newest-by-name descending.
    entries.sort(key=lambda x: (-int(x["favorited"]), x["name"]), reverse=False)
    return jsonify({"files": entries})


@app.route("/api/recordings/<source>/<filename>/download")
def download_recording(source, filename):
    if source not in {s["name"] for s in _recording_sources()}:
        return jsonify({"error": f"unknown recording source: {source}"}), 404
    recordings_dir = _recordings_dir(source)
    filepath = (recordings_dir / filename).resolve()
    if filepath.parent != recordings_dir.resolve():
        abort(400)
    if not filepath.exists():
        abort(404)
    return send_file(filepath, as_attachment=True, download_name=filename)


@app.route("/api/recordings/<source>/<filename>/favorite", methods=["POST"])
def favorite_recording(source, filename):
    if source not in {s["name"] for s in _recording_sources()}:
        return jsonify({"error": f"unknown recording source: {source}"}), 404
    recordings_dir = _recordings_dir(source)
    filepath = (recordings_dir / filename).resolve()
    if filepath.parent != recordings_dir.resolve():
        return jsonify({"error": "invalid filename"}), 400
    if not filepath.exists():
        return jsonify({"error": "file not found"}), 404
    favorites = _load_favorites(source)
    if filename in favorites:
        favorites.discard(filename)
        favorited = False
    else:
        favorites.add(filename)
        favorited = True
    _save_favorites(source, favorites)
    return jsonify({"favorited": favorited, "favorites": sorted(favorites)})


@app.route("/api/recordings/<source>/<filename>", methods=["DELETE"])
def delete_recording(source, filename):
    if source not in {s["name"] for s in _recording_sources()}:
        return jsonify({"error": f"unknown recording source: {source}"}), 404
    recordings_dir = _recordings_dir(source)
    filepath = (recordings_dir / filename).resolve()
    if filepath.parent != recordings_dir.resolve():
        return jsonify({"error": "invalid filename"}), 400
    if not filepath.exists():
        return jsonify({"error": "file not found"}), 404
    try:
        filepath.unlink()
        favorites = _load_favorites(source)
        if filename in favorites:
            favorites.discard(filename)
            _save_favorites(source, favorites)
        return jsonify({"ok": True})
    except OSError as exc:
        return jsonify({"error": f"delete failed: {exc}"}), 500


# ── Module asset files ────────────────────────────────────────────────────────
# Read-only access to files inside an installed module's directory.  Used by
# dashboard widgets that need a module-supplied asset — e.g. the orientation3d
# widget's STL model (manifest ui key "model", a path relative to the module
# root): /api/modules/<name>/files/gimbal.stl
@app.route("/api/modules/<name>/files/<path:relpath>")
def module_file(name, relpath):
    if name not in load_registry().get("modules", {}):
        return jsonify({"error": f"module not installed: {name}"}), 404
    root = (Path(cfg["packages_dir"]) / name).resolve()
    filepath = (root / relpath).resolve()
    if not filepath.is_relative_to(root):
        abort(400)
    if not filepath.is_file():
        abort(404)
    return send_file(filepath)

# ── Module manager ────────────────────────────────────────────────────────────
# Modules are GitHub repositories containing an eventide-module.json manifest
# (see docs/MODULES.md).  Installing a module is a background job:
#   clone → validate → deps → build → artifacts → conf.d → supervisor apply
# Each module gets its own /etc/supervisor/conf.d/module-<name>.conf, applied
# with supervisorctl reread/update.  Installed modules are tracked in a JSON
# registry (default /usr/local/eventide/modules.json).

MANIFEST_FILENAME = "eventide-module.json"
MODULE_INSTALL_DIR = "/usr/local/eventide/code"
MODULE_CONFIG_DIR  = "/usr/local/eventide/config"

_MODULE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_PROGRAM_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_ARG_NAME_RE = re.compile(r"^[a-z0-9_]+$")
_ARG_TYPES = ("str", "int", "float")
_UI_COMPONENT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_UI_TYPES = ("mjpeg", "form", "telemetry", "joystick", "table", "map", "recording", "orientation3d", "features", "master-record")
_UI_REGIONS = ("sidebar", "center")
_UI_FIELD_KINDS = ("number", "slider", "toggle", "text", "select", "nudge")
_KNOWN_PLACEHOLDERS = (
    "install_dir", "config_dir", "module_dir", "recordings_dir",
    "recordings_subdir", "venv_dir", "venv_python",
)
_VENV_DIR_NAME = ".venv"
_SUPERVISOR_STATES = {
    "RUNNING", "STOPPED", "STARTING", "BACKOFF",
    "STOPPING", "EXITED", "FATAL", "UNKNOWN",
}
_JOB_LOG_CAP = 500

module_jobs: dict[str, dict] = {}
module_jobs_lock = threading.Lock()
install_lock = threading.Lock()   # one install job at a time


def _job_log(job: dict | None, line: str) -> None:
    if job is None:
        return  # instance operations outside install jobs have no job log
    with module_jobs_lock:
        job["log"].append(str(line))
        if len(job["log"]) > _JOB_LOG_CAP:
            del job["log"][: len(job["log"]) - _JOB_LOG_CAP]


def _job_status(job: dict, status: str) -> None:
    with module_jobs_lock:
        job["status"] = status


def _job_snapshot(job: dict) -> dict:
    with module_jobs_lock:
        return {
            "id": job["id"],
            "status": job["status"],
            "source_type": job.get("source_type", "git"),
            "repo_url": job["repo_url"],
            "module": job["module"],
            "created_at": job["created_at"],
            "finished_at": job["finished_at"],
            "error": job["error"],
            "warnings": list(job["warnings"]),
            "log": list(job["log"]),
        }


# ── Registry ──────────────────────────────────────────────────────────────────

def load_registry() -> dict:
    path = Path(cfg.get("modules_registry", "/usr/local/eventide/modules.json"))
    if not path.exists():
        return {"modules": {}}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict) and isinstance(data.get("modules"), dict):
            # Backfill implicit default instances for entries written before
            # the instance system existed (persisted on the next save).
            for name, entry in data["modules"].items():
                _migrate_entry(name, entry)
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {"modules": {}}


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def save_registry(registry: dict) -> None:
    path = Path(cfg["modules_registry"])
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(registry, indent=2) + "\n")


# ── Manifest validation ───────────────────────────────────────────────────────

def _is_str_list(value) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def validate_manifest(m) -> list[str]:
    """Structural validation of an eventide-module.json manifest."""
    errors: list[str] = []
    if not isinstance(m, dict):
        return ["manifest is not a JSON object"]

    name = m.get("name")
    if not isinstance(name, str) or not _MODULE_NAME_RE.match(name):
        errors.append("'name' is required and must match ^[a-z0-9][a-z0-9-]*$")
    for field in ("version", "description"):
        if not isinstance(m.get(field), str) or not m.get(field):
            errors.append(f"'{field}' is required and must be a non-empty string")

    deps = m.get("dependencies", {})
    if not isinstance(deps, dict):
        errors.append("'dependencies' must be an object")
    else:
        for key in ("apt", "pip", "commands"):
            if key in deps and not _is_str_list(deps[key]):
                errors.append(f"'dependencies.{key}' must be a list of strings")
        req = deps.get("requirements")
        if req is not None and (
            not isinstance(req, str)
            or os.path.isabs(req)
            or ".." in Path(req).parts
        ):
            errors.append(
                "'dependencies.requirements' must be a relative path inside the repo"
            )
        ssp = deps.get("system_site_packages")
        if ssp is not None and not isinstance(ssp, bool):
            errors.append("'dependencies.system_site_packages' must be a boolean")

    inst = m.get("install", {})
    if not isinstance(inst, dict):
        errors.append("'install' must be an object")
    else:
        if "commands" in inst and not _is_str_list(inst["commands"]):
            errors.append("'install.commands' must be a list of strings")
        arts = inst.get("artifacts", {})
        if not isinstance(arts, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in arts.items()
        ):
            errors.append("'install.artifacts' must map source path → destination path")
        else:
            for dst in arts.values():
                # Destinations support the same placeholders as program
                # commands ({install_dir}, {module_dir}, ...) — expand
                # before checking for an absolute path.
                expanded = render_placeholders(
                    dst, m, name if isinstance(name, str) else "module"
                )
                if not os.path.isabs(expanded):
                    errors.append(f"artifact destination must be an absolute path: {dst}")

    sub = m.get("recordings_subdir")
    if sub is not None and (
        not isinstance(sub, str) or not sub or "/" in sub or sub in (".", "..")
    ):
        errors.append("'recordings_subdir' must be a single directory name")

    arg_names: set[str] = set()
    args = m.get("arguments", [])
    if not isinstance(args, list):
        errors.append("'arguments' must be a list")
    else:
        for a in args:
            if not isinstance(a, dict):
                errors.append("each argument must be an object")
                continue
            an = a.get("name")
            if not isinstance(an, str) or not _ARG_NAME_RE.match(an):
                errors.append("argument 'name' must match ^[a-z0-9_]+$")
                continue
            if an in arg_names:
                errors.append(f"duplicate argument name '{an}'")
            arg_names.add(an)
            if not isinstance(a.get("flag"), str) or not a.get("flag"):
                errors.append(f"argument '{an}' needs a 'flag' (e.g. \"--port\")")
            atype = a.get("type")
            if atype not in _ARG_TYPES:
                errors.append(f"argument '{an}' type must be one of {_ARG_TYPES}")
            if "default" not in a:
                errors.append(f"argument '{an}' needs a 'default'")
            elif atype == "int" and not isinstance(a["default"], int):
                errors.append(f"argument '{an}' default must be an int")
            elif atype == "float" and not isinstance(a["default"], (int, float)):
                errors.append(f"argument '{an}' default must be a number")
            elif atype == "str" and not isinstance(a["default"], str):
                errors.append(f"argument '{an}' default must be a string")

    # Instanceable modules opt in by naming the argument that identifies a
    # unique copy (e.g. the serial port).  Without this key the module runs
    # exactly one implicit instance and nothing about its behaviour changes.
    instanceable = False
    inst = m.get("instance")
    if inst is not None:
        if not isinstance(inst, dict):
            errors.append("'instance' must be an object")
        else:
            ia = inst.get("argument")
            if not isinstance(ia, str) or not ia:
                errors.append("'instance.argument' is required and must name an argument")
            else:
                idecl = next(
                    (a for a in args if isinstance(a, dict) and a.get("name") == ia),
                    None,
                )
                if idecl is None:
                    errors.append(
                        f"'instance.argument' references undeclared argument '{ia}'"
                    )
                elif idecl.get("type") not in ("str", "int"):
                    errors.append(
                        f"'instance.argument' '{ia}' must be of type str or int"
                    )
                else:
                    instanceable = True
            if "label" in inst and not isinstance(inst["label"], str):
                errors.append("'instance.label' must be a string")

    sock_names: set[str] = set()
    sockets = m.get("sockets", [])
    if not isinstance(sockets, list):
        errors.append("'sockets' must be a list")
    else:
        seen_sock_names: set[str] = set()
        for s in sockets:
            if not isinstance(s, dict):
                errors.append("each socket must be an object")
                continue
            sn = s.get("name")
            if not isinstance(sn, str) or not sn:
                errors.append("each socket needs a 'name'")
                continue
            if sn in seen_sock_names:
                errors.append(f"duplicate socket name '{sn}'")
            if sn == "inst":
                # Reserved: /proxy/<module>/inst/<iid>/<socket>/... would clash.
                errors.append("socket name 'inst' is reserved")
            seen_sock_names.add(sn)
            sock_names.add(sn)
            stype = s.get("type")
            if stype not in ("tcp", "unix"):
                errors.append(f"socket '{sn}' type must be 'tcp' or 'unix'")
            elif stype == "tcp":
                # 'port' is optional: when omitted, eventide allocates one
                # from --port-pool (per instance for instanceable modules,
                # otherwise at install time).
                port = s.get("port")
                if port is not None and (
                    not isinstance(port, int) or not (1 <= port <= 65535)
                ):
                    errors.append(f"socket '{sn}' tcp 'port' must be 1-65535 when given")
            else:
                # 'path' is optional: when omitted, eventide generates one
                # (per instance for instanceable modules).  Instanceable
                # modules must NOT pin a path — copies would collide.
                spath = s.get("path")
                if spath is not None:
                    if not isinstance(spath, str) or not spath.startswith("/"):
                        errors.append(
                            f"socket '{sn}' unix 'path' must be absolute when given"
                        )
                    elif instanceable:
                        errors.append(
                            f"socket '{sn}': instanceable modules must not declare "
                            "a unix 'path' — paths are generated per instance"
                        )

    programs = m.get("programs")
    if not isinstance(programs, list) or not programs:
        errors.append("'programs' must be a non-empty list")
    else:
        seen_prog_names: set[str] = set()
        for p in programs:
            if not isinstance(p, dict):
                errors.append("each program must be an object")
                continue
            pn = p.get("name")
            if not isinstance(pn, str) or not _PROGRAM_NAME_RE.match(pn):
                errors.append("each program needs a 'name' matching ^[a-z0-9][a-z0-9_-]*$")
                continue
            if pn in seen_prog_names:
                errors.append(f"duplicate program name '{pn}'")
            seen_prog_names.add(pn)
            cmd = p.get("command")
            if not isinstance(cmd, str) or not cmd:
                errors.append(f"program '{pn}' needs a 'command'")
                continue
            for ph in re.findall(r"\{([^}]*)\}", cmd):
                if ph in _KNOWN_PLACEHOLDERS:
                    if ph == "recordings_subdir" and not m.get("recordings_subdir"):
                        errors.append(
                            f"program '{pn}' uses '{{recordings_subdir}}' but the "
                            "manifest declares no 'recordings_subdir'"
                        )
                    continue
                if ph.startswith("arg:"):
                    if ph[4:] not in arg_names:
                        errors.append(
                            f"program '{pn}' uses undeclared argument '{{{ph}}}'"
                        )
                elif ph.startswith("socket:"):
                    if ph[7:] not in sock_names:
                        errors.append(
                            f"program '{pn}' uses undeclared socket '{{{ph}}}'"
                        )
                else:
                    errors.append(f"program '{pn}' uses unknown placeholder '{{{ph}}}'")

    ui = m.get("ui", [])
    if not isinstance(ui, list):
        errors.append("'ui' must be a list")
    else:
        tcp_sock_names = {
            s.get("name") for s in (m.get("sockets") or [])
            if isinstance(s, dict) and s.get("type") == "tcp"
        }

        def _check_sock(cid, value, where):
            if not isinstance(value, str) or not value:
                errors.append(f"ui '{cid}' needs a socket name for '{where}'")
            elif value not in tcp_sock_names:
                errors.append(
                    f"ui '{cid}' references undeclared tcp socket '{value}' ({where})"
                )

        def _is_path(value):
            return isinstance(value, str) and value.startswith("/")

        seen_ui_ids: set[str] = set()
        for comp in ui:
            if not isinstance(comp, dict):
                errors.append("each ui component must be an object")
                continue
            cid = comp.get("id")
            if not isinstance(cid, str) or not _UI_COMPONENT_ID_RE.match(cid):
                errors.append(
                    "each ui component needs an 'id' matching ^[a-z0-9][a-z0-9_-]*$"
                )
                continue
            if cid in seen_ui_ids:
                errors.append(f"duplicate ui component id '{cid}'")
            seen_ui_ids.add(cid)
            ctype = comp.get("type")
            if ctype not in _UI_TYPES:
                errors.append(f"ui '{cid}' type must be one of {_UI_TYPES}")
                continue
            if comp.get("region", "sidebar") not in _UI_REGIONS:
                errors.append(f"ui '{cid}' region must be one of {_UI_REGIONS}")
            if "default" in comp and not isinstance(comp["default"], bool):
                errors.append(f"ui '{cid}' default must be a boolean")
            if "title" in comp and not isinstance(comp["title"], str):
                errors.append(f"ui '{cid}' title must be a string")

            if ctype == "mjpeg":
                _check_sock(cid, comp.get("socket"), "socket")
                if not _is_path(comp.get("path")):
                    errors.append(f"ui '{cid}' needs a 'path' like '/stream'")
            elif ctype == "form":
                _check_sock(cid, comp.get("socket"), "socket")
                if "method" in comp and comp["method"] not in ("PUT", "POST"):
                    errors.append(f"ui '{cid}' method must be PUT or POST")
                fields = comp.get("fields")
                if not isinstance(fields, list) or not fields:
                    errors.append(f"ui '{cid}' needs a non-empty 'fields' list")
                else:
                    for f in fields:
                        if not isinstance(f, dict) or not isinstance(f.get("key"), str):
                            errors.append(f"ui '{cid}' field needs a 'key'")
                            continue
                        kind = f.get("kind", "number")
                        if kind not in _UI_FIELD_KINDS:
                            errors.append(
                                f"ui '{cid}' field '{f.get('key')}' kind must be "
                                f"one of {_UI_FIELD_KINDS}"
                            )
                        if kind == "select" and not _is_str_list(f.get("options")):
                            errors.append(
                                f"ui '{cid}' select field '{f['key']}' needs "
                                "'options' (list of strings)"
                            )
            elif ctype == "telemetry":
                _check_sock(cid, comp.get("socket"), "socket")
                if not _is_path(comp.get("get")):
                    errors.append(f"ui '{cid}' needs a 'get' path")
                rows = comp.get("rows")
                if not isinstance(rows, list) or not rows:
                    errors.append(f"ui '{cid}' needs a non-empty 'rows' list")
                elif not all(
                    isinstance(r, dict)
                    and isinstance(r.get("label"), str)
                    and isinstance(r.get("path"), str)
                    for r in rows
                ):
                    errors.append(f"ui '{cid}' rows must all have {{label, path}}")
            elif ctype == "recording":
                _check_sock(cid, comp.get("socket"), "socket")
                if not _is_path(comp.get("get")):
                    errors.append(f"ui '{cid}' needs a 'get' path")
                if not _is_path(comp.get("put")):
                    errors.append(f"ui '{cid}' needs a 'put' path")
                dur = comp.get("duration")
                if dur is not None:
                    if not isinstance(dur, dict):
                        errors.append(f"ui '{cid}' duration must be an object")
                    else:
                        for dk in ("default", "min", "max"):
                            if dk in dur and not isinstance(dur[dk], (int, float)):
                                errors.append(
                                    f"ui '{cid}' duration.{dk} must be a number"
                                )
                        if "key" in dur and not isinstance(dur["key"], str):
                            errors.append(f"ui '{cid}' duration.key must be a string")
                        if "label" in dur and not isinstance(dur["label"], str):
                            errors.append(f"ui '{cid}' duration.label must be a string")
            elif ctype == "joystick":
                _check_sock(cid, comp.get("socket"), "socket")
                if not _is_path(comp.get("put")):
                    errors.append(f"ui '{cid}' needs a 'put' path")
            elif ctype == "table":
                _check_sock(cid, comp.get("socket"), "socket")
                if not _is_path(comp.get("get")):
                    errors.append(f"ui '{cid}' needs a 'get' path")
                cols = comp.get("columns")
                if not isinstance(cols, list) or not cols:
                    errors.append(f"ui '{cid}' needs a non-empty 'columns' list")
                ra = comp.get("row_action")
                if ra is not None and (
                    not isinstance(ra, dict)
                    or not _is_path(ra.get("path"))
                    or not isinstance(ra.get("key"), str)
                ):
                    errors.append(f"ui '{cid}' row_action needs {{path, key}}")
            elif ctype == "map":
                for binding in ("track", "adsb"):
                    b = comp.get(binding)
                    if b is None:
                        continue
                    if not isinstance(b, dict):
                        errors.append(f"ui '{cid}' {binding} must be an object")
                        continue
                    _check_sock(cid, b.get("socket"), f"{binding}.socket")
                    if not _is_path(b.get("get")):
                        errors.append(f"ui '{cid}' {binding} needs a 'get' path")
    return errors


def conflict_errors(manifest: dict, registry: dict) -> list[str]:
    """Check the manifest against already-installed modules."""
    errors: list[str] = []
    installed = registry.get("modules", {})
    name = manifest["name"]
    if name in installed:
        errors.append(
            f"module '{name}' is already installed (uninstall it first to reinstall)"
        )
    existing_programs = {
        pn for e in installed.values() for pn in _entry_program_names(e)
    }
    for p in manifest.get("programs", []):
        if p["name"] in existing_programs:
            errors.append(
                f"program name '{p['name']}' is already used by another installed module"
            )
    existing_subs = {
        e["manifest"].get("recordings_subdir"): ename
        for ename, e in installed.items()
        if e["manifest"].get("recordings_subdir")
    }
    sub = manifest.get("recordings_subdir")
    if sub and sub in existing_subs:
        errors.append(
            f"recordings_subdir '{sub}' is already used by module "
            f"'{existing_subs[sub]}'"
        )
    existing_ports: dict[int, str] = {}
    for ename, e in installed.items():
        for s in e["manifest"].get("sockets", []):
            if s.get("type") == "tcp" and isinstance(s.get("port"), int):
                existing_ports.setdefault(s["port"], ename)
        for ei in (e.get("instances") or {}).values():
            for s in (ei or {}).get("sockets", []):
                if isinstance(s, dict) and s.get("type") == "tcp" and isinstance(s.get("port"), int):
                    existing_ports.setdefault(s["port"], ename)
    for s in manifest.get("sockets", []):
        if (
            s.get("type") == "tcp"
            and isinstance(s.get("port"), int)
            and s["port"] in existing_ports
        ):
            errors.append(
                f"socket port {s['port']} is already used by module "
                f"'{existing_ports[s['port']]}'"
            )
    return errors


# ── Port allocation ───────────────────────────────────────────────────────────
# TCP ports are platform-assigned: a module socket may request an explicit
# 'port' (honoured if free), but the default is to leave it out and let
# eventide pick one from --port-pool.  For non-instanceable modules the
# allocated port is written back into the manifest at install time (so
# modules.json, /api/modules, and every later re-render all see the same
# number); for instanceable modules ports are allocated per instance and live
# on the registry's instance entries instead.

def _port_pool() -> range:
    try:
        start_s, end_s = str(cfg.get("port_pool", "8100-8199")).split("-", 1)
        return range(int(start_s), int(end_s) + 1)
    except ValueError:
        raise RuntimeError(
            f"invalid --port-pool {cfg.get('port_pool')!r} (expected 'start-end')"
        )


def _port_free(port: int) -> bool:
    # A bind test catches anything the registry doesn't know about
    # (base services, other software on the payload).
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def _used_ports(registry: dict, extra_manifest: dict | None = None) -> set[int]:
    """Every TCP port already claimed: baked manifests + all instance sockets."""
    used: set[int] = set()
    manifests = [e.get("manifest", {}) for e in registry.get("modules", {}).values()]
    if extra_manifest is not None:
        manifests.append(extra_manifest)
    for m in manifests:
        for s in m.get("sockets", []):
            if isinstance(s, dict) and isinstance(s.get("port"), int):
                used.add(s["port"])
    for entry in registry.get("modules", {}).values():
        for inst in (entry.get("instances") or {}).values():
            for s in (inst or {}).get("sockets", []):
                if isinstance(s, dict) and isinstance(s.get("port"), int):
                    used.add(s["port"])
    return used


def _allocate_one_port(used: set[int], job: dict | None, sock_name: str) -> int:
    for port in _port_pool():
        if port in used or not _port_free(port):
            continue
        _job_log(job, f"allocated tcp port {port} to socket '{sock_name}'")
        return port
    raise RuntimeError(
        f"no free port in pool {cfg.get('port_pool')} "
        f"for socket '{sock_name}'"
    )


def _allocate_ports(manifest: dict, registry: dict, job: dict) -> None:
    """Assign TCP ports to the manifest's sockets that don't declare one."""
    used = _used_ports(registry, extra_manifest=manifest)
    for s in manifest.get("sockets", []):
        if not (isinstance(s, dict) and s.get("type") == "tcp" and s.get("port") is None):
            continue
        s["port"] = _allocate_one_port(used, job, s.get("name"))
        used.add(s["port"])


def _generate_unix_paths(manifest: dict, module_name: str, job: dict | None) -> None:
    """Fill in generated paths for unix sockets that don't declare one
    (non-instanceable modules — instanceable modules get paths per instance)."""
    for s in manifest.get("sockets", []):
        if isinstance(s, dict) and s.get("type") == "unix" and not s.get("path"):
            s["path"] = f"/tmp/eventide-{module_name}-{s.get('name')}.sock"
            _job_log(job, f"generated unix path {s['path']} for socket '{s.get('name')}'")


# ── Module instances ──────────────────────────────────────────────────────────
# A manifest that declares `instance: {"argument": "<arg>"}` is instanceable:
# eventide runs one copy of its programs per instance value (e.g. the serial
# port).  Everything about an instance — its argument values, allocated TCP
# ports, generated unix socket paths, supervisor program names — is stored on
# the registry entry's `instances` map.  Non-instanceable modules get a single
# implicit "default" instance mirroring the manifest, so every consumer can
# treat modules uniformly.

def _instance_id(value) -> str | None:
    """Slug identifying an instance, derived from its value
    (/dev/ttyS1 → dev-ttys1).  None when the value has no usable characters."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return slug or None


def _default_instance(manifest: dict, name: str, conf_file: str | None,
                      created_at) -> dict:
    """The implicit instance of a non-instanceable module — mirrors the
    (port-baked) manifest exactly."""
    return {
        "id": "default",
        "value": None,
        "args": {
            a["name"]: a.get("default")
            for a in manifest.get("arguments", [])
            if isinstance(a, dict) and "name" in a
        },
        "sockets": manifest.get("sockets", []),
        "programs": [
            p["name"] for p in manifest.get("programs", [])
            if isinstance(p, dict) and "name" in p
        ],
        "conf_file": conf_file or str(
            Path(cfg.get("supervisor_conf_d", "/etc/supervisor/conf.d"))
            / f"module-{name}.conf"
        ),
        "created_at": created_at,
    }


def _migrate_entry(name: str, entry: dict) -> None:
    """Backfill the default instance for pre-instances registry entries."""
    if not isinstance(entry, dict) or not isinstance(entry.get("manifest"), dict):
        return
    # Only entries written before the instance system existed lack the key;
    # an empty dict is a valid state (all instances removed) — leave it alone.
    if isinstance(entry.get("instances"), dict):
        return
    entry["instances"] = {
        "default": _default_instance(
            entry["manifest"], name, entry.get("conf_file"),
            entry.get("installed_at"),
        )
    }


def _entry_program_names(entry: dict) -> list[str]:
    """All supervisor program names a registry entry owns (every instance)."""
    names: list[str] = []
    for inst in (entry.get("instances") or {}).values():
        names.extend((inst or {}).get("programs") or [])
    return names


def _program_names(manifest: dict, inst: dict | None = None) -> list[str]:
    """Program names for one instance — suffixed with -<iid> when the module
    is instanceable (<program>-<iid>)."""
    suffix = f"-{inst['id']}" if inst and manifest.get("instance") else ""
    return [
        p["name"] + suffix
        for p in manifest.get("programs", [])
        if isinstance(p, dict) and "name" in p
    ]


def _instance_recordings_subdir(manifest: dict, inst: dict) -> str | None:
    """Recordings subdir for one instance: <sub>-<iid> for instanceable
    modules (each copy records to its own directory), <sub> otherwise."""
    sub = manifest.get("recordings_subdir")
    if not isinstance(sub, str) or not sub:
        return None
    if manifest.get("instance"):
        return f"{sub}-{inst['id']}"
    return sub


def _instance_manifest(manifest: dict, inst: dict) -> dict:
    """The manifest overlaid with an instance's values, for rendering: the
    instance's sockets (allocated ports / generated paths) replace the
    manifest's, its argument values become the defaults, and an instanceable
    module's recordings subdir gains the -<iid> suffix."""
    eff = dict(manifest)
    eff["sockets"] = inst.get("sockets", [])
    values = inst.get("args", {})
    eff["arguments"] = [
        {**a, "default": values[a["name"]]}
        if isinstance(a, dict) and a.get("name") in values else a
        for a in manifest.get("arguments", [])
    ]
    sub = _instance_recordings_subdir(manifest, inst)
    if sub:
        eff["recordings_subdir"] = sub
    return eff


def _setup_instance_sockets(manifest: dict, module_name: str, registry: dict,
                            inst: dict, job: dict | None) -> None:
    """Resolve an instance's sockets: allocate a TCP port per tcp socket
    (honouring an explicit 'port' request when free) and generate a unix
    path per unix socket."""
    used = _used_ports(registry)
    resolved = []
    for s in manifest.get("sockets", []):
        if not isinstance(s, dict):
            continue
        s2 = dict(s)
        if s2.get("type") == "tcp":
            port = s2.get("port")
            if isinstance(port, int):
                if port in used or not _port_free(port):
                    raise RuntimeError(
                        f"tcp port {port} for socket '{s2.get('name')}' "
                        "is already in use"
                    )
            else:
                port = _allocate_one_port(used, job, s2.get("name"))
            s2["port"] = port
            used.add(port)
        else:
            s2["path"] = (
                f"/tmp/eventide-{module_name}-{inst['id']}-{s2.get('name')}.sock"
            )
        resolved.append(s2)
    inst["sockets"] = resolved


def _build_instance(manifest: dict, name: str, registry: dict, value,
                    args: dict | None, job: dict | None) -> dict:
    """Validate and assemble a new instance of an instanceable module
    (no side effects: no conf written, registry untouched)."""
    spec = manifest.get("instance") or {}
    iid = _instance_id(value)
    if iid is None:
        raise RuntimeError(f"instance value {value!r} produces an empty instance id")
    entry = registry.get("modules", {}).get(name) or {}
    for existing in (entry.get("instances") or {}).values():
        if (existing or {}).get("value") == value:
            raise RuntimeError(f"an instance with value {value!r} already exists")
        if (existing or {}).get("id") == iid:
            raise RuntimeError(
                f"instance value {value!r} collides with existing instance "
                f"'{iid}' ({(existing or {}).get('value')!r})"
            )

    programs = _program_names(manifest, {"id": iid})
    taken = {
        pn
        for e in registry.get("modules", {}).values()
        for pn in _entry_program_names(e)
    }
    clashes = [pn for pn in programs if pn in taken]
    if clashes:
        raise RuntimeError(
            f"program name(s) already used by another module: {', '.join(clashes)}"
        )

    inst = {
        "id": iid,
        "value": value,
        "args": {
            a["name"]: a.get("default")
            for a in manifest.get("arguments", [])
            if isinstance(a, dict) and "name" in a
        },
        "sockets": [],
        "programs": programs,
        "conf_file": str(
            Path(cfg["supervisor_conf_d"]) / f"module-{name}-{iid}.conf"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    inst["args"][spec.get("argument")] = value
    if args:
        inst["args"].update(args)
        inst["args"][spec.get("argument")] = value
    _setup_instance_sockets(manifest, name, registry, inst, job)
    return inst


def _render_instance_conf(name: str, entry: dict, inst: dict) -> None:
    """Render and write an instance's supervisor conf file."""
    manifest = entry["manifest"]
    suffix = f"-{inst['id']}" if manifest.get("instance") else ""
    conf_file = inst.get("conf_file") or str(
        Path(cfg["supervisor_conf_d"]) / f"module-{name}{suffix}.conf"
    )
    _atomic_write(
        Path(conf_file),
        render_module_conf(
            _instance_manifest(manifest, inst), name,
            entry.get("repo_url", ""), program_suffix=suffix,
        ),
    )


# ── Config rendering ──────────────────────────────────────────────────────────

def render_placeholders(text: str, manifest: dict, module_name: str) -> str:
    venv_dir = Path(cfg["packages_dir"]) / module_name / _VENV_DIR_NAME
    out = text
    for arg in manifest.get("arguments") or []:
        if not isinstance(arg, dict) or "name" not in arg:
            continue  # tolerate malformed manifests during validation
        out = out.replace("{arg:%s}" % arg["name"], str(arg.get("default", "")))
    for sock in manifest.get("sockets") or []:
        if not isinstance(sock, dict) or "name" not in sock:
            continue  # tolerate malformed manifests during validation
        # tcp → the allocated/declared port, unix → the socket path.
        value = sock.get("path") if sock.get("type") == "unix" else sock.get("port")
        out = out.replace("{socket:%s}" % sock["name"], str(value or ""))
    out = out.replace("{install_dir}", MODULE_INSTALL_DIR)
    out = out.replace("{config_dir}", MODULE_CONFIG_DIR)
    out = out.replace("{module_dir}", str(Path(cfg["packages_dir"]) / module_name))
    out = out.replace("{recordings_dir}", cfg["recordings_dir"])
    sub = manifest.get("recordings_subdir")
    if isinstance(sub, str) and sub:
        out = out.replace(
            "{recordings_subdir}", str(Path(cfg["recordings_dir"]) / sub)
        )
    out = out.replace("{venv_dir}", str(venv_dir))
    out = out.replace("{venv_python}", str(venv_dir / "bin" / "python3"))
    return out


def render_module_conf(manifest: dict, module_name: str, repo_url: str,
                       program_suffix: str = "") -> str:
    lines = [
        f"; Generated by the eventide module manager from {repo_url}",
        "; Do not edit by hand — changes are lost on reinstall.",
    ]
    for p in manifest["programs"]:
        # program_suffix is "-<iid>" for instances of instanceable modules.
        lines.append(f"[program:{p['name']}{program_suffix}]")
        lines.append(f"command={render_placeholders(p['command'], manifest, module_name)}")
        lines.append(
            "directory="
            + render_placeholders(p.get("directory", "{install_dir}"), manifest, module_name)
        )
        lines.append(f"autostart={'true' if p.get('autostart', True) else 'false'}")
        lines.append(f"autorestart={'true' if p.get('autorestart', True) else 'false'}")
        lines.append(f"startretries={int(p.get('startretries', 10000))}")
        lines.append(f"priority={int(p.get('priority', 10))}")
        user = p.get("user", "root")
        lines.append(f"user={user}")
        # supervisord gives programs a bare environment — unlike systemd it
        # does not set HOME, which breaks tools that need a per-user dir
        # (e.g. MAVProxy's ~/.mavproxy).  Provide it explicitly.
        home = "/root" if user == "root" else f"/home/{user}"
        lines.append(f'environment=HOME="{home}"')
        lines.append("stdout_logfile=/var/log/supervisor/%(program_name)s.log")
        lines.append("")
    return "\n".join(lines)


# ── Per-module Python virtual environments ────────────────────────────────────
# Each module that uses Python gets its own venv at <module_dir>/.venv so
# modules can never break each other's dependencies.  Created with
# --system-site-packages by default so apt-provided Python libraries
# (e.g. python3-picamera2) stay visible; opt out per module with
# dependencies.system_site_packages = false.  Supervisor programs use the
# {venv_python} / {venv_dir} placeholders to run inside the venv.

def _module_requirements_path(deps: dict, module_dir: Path) -> Path | None:
    """Resolve the module's pip requirements file, or None if it has none."""
    declared = deps.get("requirements")
    if declared:
        path = module_dir / declared
        if not path.exists():
            raise RuntimeError(f"requirements file not found: {declared}")
        return path
    default = module_dir / "requirements.txt"
    return default if default.exists() else None


def _create_venv(job: dict, venv_dir: Path, system_site_packages: bool) -> None:
    if (venv_dir / "bin" / "python3").exists():
        _job_log(job, f"venv already exists at {venv_dir} — reusing")
        return
    argv = ["python3", "-m", "venv"]
    if system_site_packages:
        argv.append("--system-site-packages")
    argv.append(str(venv_dir))
    _run_logged_argv(job, argv, timeout=300)
    if not (venv_dir / "bin" / "pip").exists():
        raise RuntimeError(f"venv creation did not produce pip: {venv_dir}")


# ── Shell helpers ─────────────────────────────────────────────────────────────

def _module_env() -> dict:
    """Environment for module job commands.

    The dashboard runs under systemd with a minimal PATH, so toolchains
    installed per-user (rustup → ~/.cargo/bin) are invisible to install
    commands even though they exist on the device.  Re-add well-known
    cargo bin dirs to PATH, and point CARGO_HOME/RUSTUP_HOME at the same
    user's dirs so rustup's shim binaries find their toolchains when the
    toolchain belongs to a non-root user.
    """
    env = os.environ.copy()
    candidates = [Path.home() / ".cargo", Path("/usr/local/cargo")]
    candidates += sorted(Path("/home").glob("*/.cargo"))
    extra: list[str] = []
    for cargo_home in candidates:
        if not (cargo_home / "bin").is_dir():
            continue
        extra.append(str(cargo_home / "bin"))
        rustup_home = cargo_home.parent / ".rustup"
        if rustup_home.is_dir():
            env.setdefault("CARGO_HOME", str(cargo_home))
            env.setdefault("RUSTUP_HOME", str(rustup_home))
    if extra:
        env["PATH"] = os.pathsep.join([*extra, env.get("PATH", "")])
    return env


def _run_logged_argv(job: dict, argv: list[str], cwd=None, timeout=600) -> str:
    _job_log(job, "$ " + shlex.join(argv))
    try:
        proc = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, timeout=timeout,
            env=_module_env(),
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"command timed out after {timeout}s: {shlex.join(argv)}")
    except FileNotFoundError:
        raise RuntimeError(f"command not found: {argv[0]}")
    for line in (proc.stdout + proc.stderr).splitlines():
        _job_log(job, line)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed (exit {proc.returncode}): {shlex.join(argv)}")
    return proc.stdout


def _run_logged_shell(job: dict, command: str, cwd=None, timeout=600) -> str:
    return _run_logged_argv(job, ["/bin/bash", "-c", command], cwd=cwd, timeout=timeout)


def _https_to_ssh(url: str) -> str | None:
    m = re.match(r"^https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if m:
        return f"git@github.com:{m.group(1)}/{m.group(2)}.git"
    return None


def _clone_repo(repo_url: str, ref: str | None, dest: Path, job: dict) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    base = ["git", "clone", "--depth", "1"]
    if ref:
        base += ["--branch", ref]
    try:
        _run_logged_argv(job, [*base, repo_url, str(dest)], timeout=180)
        return
    except RuntimeError as exc:
        ssh_url = _https_to_ssh(repo_url)
        if not ssh_url:
            raise
        _job_log(job, f"HTTPS clone failed ({exc}); retrying over SSH: {ssh_url}")
        shutil.rmtree(dest, ignore_errors=True)
        _run_logged_argv(job, [*base, ssh_url, str(dest)], timeout=180)


def _git_rev(repo_dir: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15,
        )
        return proc.stdout.strip() or None
    except (subprocess.TimeoutExpired, OSError):
        return None


def _extract_zip(zip_path: Path, dest: Path, job: dict) -> Path:
    """Extract an uploaded module zip into `dest` and return the directory
    that contains eventide-module.json — either the zip root, or a single
    top-level folder as produced by GitHub's "Download ZIP" button."""
    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        raise RuntimeError("uploaded file is not a valid zip archive")
    with archive:
        names = archive.namelist()
        for member in names:
            parts = Path(member).parts
            if member.startswith("/") or ".." in parts:
                raise RuntimeError(f"unsafe path in zip: {member}")
        archive.extractall(dest)
    _job_log(job, f"extracted {zip_path.name} ({len(names)} entries)")

    if (dest / MANIFEST_FILENAME).exists():
        return dest
    top = [p for p in dest.iterdir() if p.name != "__MACOSX"]
    dirs = [p for p in top if p.is_dir()]
    if len(top) == 1 and len(dirs) == 1 and (dirs[0] / MANIFEST_FILENAME).exists():
        return dirs[0]
    raise RuntimeError(
        f"{MANIFEST_FILENAME} not found at the zip root or in a single "
        "top-level folder"
    )


# ── Supervisor control ────────────────────────────────────────────────────────

def _supervisorctl(*args, timeout=30) -> tuple[int, str]:
    proc = subprocess.run(
        ["supervisorctl", *args], capture_output=True, text=True, timeout=timeout
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _supervisor_apply(job: dict) -> None:
    """Apply conf.d changes via supervisorctl reread + update."""
    if not shutil.which("supervisorctl"):
        job["warnings"].append("supervisorctl not found; skipped reread/update")
        _job_log(job, "WARNING: supervisorctl not found — skipping reread/update")
        return
    for args in (("reread",), ("update",)):
        rc, out = _supervisorctl(*args)
        _job_log(job, f"$ supervisorctl {' '.join(args)}")
        for line in out.splitlines():
            _job_log(job, line)
        if rc != 0:
            raise RuntimeError(f"supervisorctl {' '.join(args)} failed: {out}")


def supervisor_statuses() -> dict[str, dict]:
    """Parse `supervisorctl status` into {program: {status, description}}."""
    if not shutil.which("supervisorctl"):
        return {}
    try:
        _, out = _supervisorctl("status")
    except (subprocess.TimeoutExpired, OSError):
        return {}
    result: dict[str, dict] = {}
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) >= 2 and parts[1] in _SUPERVISOR_STATES:
            result[parts[0]] = {
                "status": parts[1],
                "description": parts[2] if len(parts) > 2 else "",
            }
    return result


def _verify_programs(job: dict, manifest: dict, inst: dict | None = None) -> None:
    if not shutil.which("supervisorctl"):
        job["warnings"].append("supervisorctl not available; skipped program verification")
        return
    wanted = _program_names(manifest, inst)
    states: dict[str, str | None] = {}
    deadline = time.time() + 10
    while time.time() < deadline:
        statuses = supervisor_statuses()
        states = {w: statuses.get(w, {}).get("status") for w in wanted}
        if states and all(s == "RUNNING" for s in states.values()):
            _job_log(job, "all programs RUNNING")
            return
        time.sleep(1)
    for w, s in states.items():
        if s != "RUNNING":
            job["warnings"].append(
                f"program '{w}' is {s or 'unknown'} after install "
                "(check the SUPERVISOR tab — hardware may be absent)"
            )
            _job_log(job, f"WARNING: program '{w}' is {s or 'unknown'}")


# ── Install job ───────────────────────────────────────────────────────────────

def _run_install_job(job: dict) -> None:
    staging: Path | None = None
    module_dir: Path | None = None
    copied_artifacts: list[str] = []
    conf_path: Path | None = None
    name: str | None = None
    registered = False
    try:
        # ── Acquire source & validate ─────────────────────────────────────
        staging = Path(cfg["packages_dir"]) / f".staging-{job['id']}"
        if job.get("source_type") == "zip":
            _job_status(job, "extracting")
            staging.mkdir(parents=True, exist_ok=True)
            module_root = _extract_zip(Path(job["zip_path"]), staging, job)
        elif job.get("source_type") == "local":
            _job_status(job, "copying")
            local_root = Path(job["repo_url"])
            if not local_root.is_dir():
                raise RuntimeError(f"local module path does not exist: {local_root}")
            shutil.copytree(local_root, staging, dirs_exist_ok=True)
            module_root = staging
        else:
            _job_status(job, "cloning")
            _clone_repo(job["repo_url"], job.get("ref"), staging, job)
            module_root = staging

        manifest_path = module_root / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise RuntimeError(f"{MANIFEST_FILENAME} not found in repository root")
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{MANIFEST_FILENAME} is not valid JSON: {exc}")
        errors = validate_manifest(manifest)
        if errors:
            raise RuntimeError("invalid manifest: " + "; ".join(errors))

        name = manifest["name"]
        job["module"] = name
        registry = load_registry()
        errors = conflict_errors(manifest, registry)
        if errors:
            raise RuntimeError("; ".join(errors))
        # Allocate TCP ports + generate unix socket paths before anything
        # renders the manifest.  Instanceable modules skip this: ports and
        # unix paths are allocated per instance instead (manifest sockets
        # stay unallocated).
        if not manifest.get("instance"):
            _allocate_ports(manifest, registry, job)
            _generate_unix_paths(manifest, name, job)

        module_dir = Path(cfg["packages_dir"]) / name
        if module_dir.exists():
            # Not in the registry (checked above) → leftover from a failed
            # install; safe to replace.
            shutil.rmtree(module_dir)
        if module_root != staging:
            # Zip with a top-level folder: move just the module root out.
            shutil.move(str(module_root), str(module_dir))
            shutil.rmtree(staging, ignore_errors=True)
        else:
            shutil.move(str(staging), str(module_dir))
        staging = None
        commit = _git_rev(module_dir)
        _job_log(job, f"acquired {job['repo_url']} ({commit or 'unknown commit'})")

        # ── Dependencies ──────────────────────────────────────────────────
        _job_status(job, "deps")
        deps = manifest.get("dependencies", {})
        if deps.get("apt"):
            _run_logged_argv(
                job, ["apt-get", "install", "-y", *deps["apt"]], timeout=600
            )

        # Python packages always go into the module's own venv — never
        # system-wide — so modules can't break each other's dependencies.
        venv_dir = module_dir / _VENV_DIR_NAME
        req_path = _module_requirements_path(deps, module_dir)
        pip_pkgs = deps.get("pip", [])
        venv_referenced = any(
            "{venv_" in p.get("command", "") or "{venv_" in p.get("directory", "")
            for p in manifest.get("programs", [])
        ) or any(
            "{venv_" in cmd
            for cmd in (
                deps.get("commands", [])
                + manifest.get("install", {}).get("commands", [])
            )
        )
        if req_path is not None or pip_pkgs or venv_referenced:
            _create_venv(job, venv_dir, deps.get("system_site_packages", True))
            venv_pip = str(venv_dir / "bin" / "pip")
            if req_path is not None:
                _run_logged_argv(
                    job, [venv_pip, "install", "-r", str(req_path)],
                    cwd=module_dir, timeout=600,
                )
            if pip_pkgs:
                _run_logged_argv(
                    job, [venv_pip, "install", *pip_pkgs],
                    cwd=module_dir, timeout=600,
                )

        for cmd in deps.get("commands", []):
            _run_logged_shell(
                job, render_placeholders(cmd, manifest, name), cwd=module_dir
            )

        # ── Build ─────────────────────────────────────────────────────────
        _job_status(job, "building")
        inst = manifest.get("install", {})
        for cmd in inst.get("commands", []):
            # Compilations (cargo build --release & co.) legitimately take a
            # long time on-device — allow 30 min per build command.
            _run_logged_shell(
                job, render_placeholders(cmd, manifest, name),
                cwd=module_dir, timeout=1800,
            )

        # ── Artifacts & recordings dir ────────────────────────────────────
        _job_status(job, "artifacts")
        for src, dst in inst.get("artifacts", {}).items():
            src_path = module_dir / src
            if not src_path.exists():
                raise RuntimeError(f"artifact not found after build: {src}")
            dst_path = Path(render_placeholders(dst, manifest, name))
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            copied_artifacts.append(str(dst_path))
            _job_log(job, f"copied {src} → {dst_path}")
        sub = manifest.get("recordings_subdir")
        instanceable = bool(manifest.get("instance"))
        instances: dict[str, dict] = {}
        if sub and not instanceable:
            rec_dir = Path(cfg["recordings_dir"]) / sub
            rec_dir.mkdir(parents=True, exist_ok=True)
            _job_log(job, f"created recordings dir {rec_dir}")
        for s in manifest.get("sockets", []):
            # Unix socket paths may point into a directory that doesn't exist
            # yet (e.g. /run/eventide/<module>/...) — create the parent.
            if s.get("type") == "unix" and isinstance(s.get("path"), str):
                Path(s["path"]).parent.mkdir(parents=True, exist_ok=True)

        # ── Supervisor config ─────────────────────────────────────────────
        _job_status(job, "configuring")
        if instanceable:
            # Auto-create the first instance from the instance argument's
            # default value, so install behaves like any other module: its
            # programs come up immediately (wrong default? fix it from the
            # MODULES tab afterwards).
            spec = manifest["instance"]
            default_value = next(
                (
                    a.get("default")
                    for a in manifest.get("arguments", [])
                    if isinstance(a, dict) and a.get("name") == spec.get("argument")
                ),
                None,
            )
            first = _build_instance(
                manifest, name, registry, default_value, None, job
            )
            instances[first["id"]] = first
            _job_log(
                job,
                f"created instance '{first['id']}' ({spec.get('argument')}"
                f"={default_value!r})",
            )
            rec_sub = _instance_recordings_subdir(manifest, first)
            if rec_sub:
                rec_dir = Path(cfg["recordings_dir"]) / rec_sub
                rec_dir.mkdir(parents=True, exist_ok=True)
                _job_log(job, f"created recordings dir {rec_dir}")
            conf_path = Path(first["conf_file"])
            _atomic_write(
                conf_path,
                render_module_conf(
                    _instance_manifest(manifest, first), name, job["repo_url"],
                    program_suffix=f"-{first['id']}",
                ),
            )
        else:
            conf_path = Path(cfg["supervisor_conf_d"]) / f"module-{name}.conf"
            _atomic_write(
                conf_path,
                render_module_conf(manifest, name, job["repo_url"]),
            )
        _job_log(job, f"wrote {conf_path}")
        _supervisor_apply(job)

        # ── Verify & register ─────────────────────────────────────────────
        _job_status(job, "verifying")
        _verify_programs(job, manifest, next(iter(instances.values()), None))

        registry = load_registry()
        installed_at = datetime.now(timezone.utc).isoformat()
        entry = {
            "manifest": manifest,
            "repo_url": job["repo_url"],
            "ref": job.get("ref"),
            "commit": commit,
            "installed_at": installed_at,
            "artifacts": copied_artifacts,
            "conf_file": str(conf_path),
            "instances": (
                instances if instanceable else {
                    "default": _default_instance(
                        manifest, name, str(conf_path), installed_at
                    )
                }
            ),
        }
        registry.setdefault("modules", {})[name] = entry
        save_registry(registry)
        registered = True
        _job_status(job, "done")
        _job_log(job, f"module '{name}' installed successfully")

    except Exception as exc:  # noqa: BLE001 — any failure must roll back cleanly
        job["error"] = str(exc)
        _job_log(job, f"ERROR: {exc}")
        _job_log(job, "rolling back partial install")
        if conf_path is not None and conf_path.exists():
            conf_path.unlink()
            _supervisor_apply(job)
        for artifact in copied_artifacts:
            Path(artifact).unlink(missing_ok=True)
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        if module_dir is not None and module_dir.exists():
            shutil.rmtree(module_dir, ignore_errors=True)
        if name is not None and registered:
            registry = load_registry()
            if registry.get("modules", {}).pop(name, None) is not None:
                save_registry(registry)
        _job_status(job, "failed")
    finally:
        if job.get("zip_path"):
            # Uploaded zips are temporary — always remove after the job.
            Path(job["zip_path"]).unlink(missing_ok=True)
        job["finished_at"] = datetime.now(timezone.utc).isoformat()


def _run_install_job_guarded(job: dict) -> None:
    try:
        _run_install_job(job)
    finally:
        install_lock.release()


def _uninstall_module(name: str) -> dict | None:
    registry = load_registry()
    entry = registry.get("modules", {}).get(name)
    if entry is None:
        return None

    program_names = _entry_program_names(entry)
    if shutil.which("supervisorctl"):
        for pname in program_names:
            try:
                _supervisorctl("stop", pname)
            except (subprocess.TimeoutExpired, OSError):
                pass  # best-effort stop

    conf_paths = {
        (inst or {}).get("conf_file")
        for inst in (entry.get("instances") or {}).values()
    }
    conf_paths.add(entry.get("conf_file"))
    conf_paths.add(str(Path(cfg["supervisor_conf_d"]) / f"module-{name}.conf"))
    for conf in conf_paths:
        if conf:
            Path(conf).unlink(missing_ok=True)
    if shutil.which("supervisorctl"):
        for args in (("reread",), ("update",)):
            try:
                _supervisorctl(*args)
            except (subprocess.TimeoutExpired, OSError):
                pass  # best-effort apply

    for artifact in entry.get("artifacts", []):
        Path(artifact).unlink(missing_ok=True)
    shutil.rmtree(Path(cfg["packages_dir"]) / name, ignore_errors=True)

    del registry["modules"][name]
    save_registry(registry)
    return entry


# ── Module API ────────────────────────────────────────────────────────────────

@app.route("/api/modules")
def api_modules_list():
    registry = load_registry()
    statuses = supervisor_statuses()
    modules = []
    for name, entry in sorted(registry.get("modules", {}).items()):
        m = entry["manifest"]
        prog_templates = {
            p["name"]: p for p in m.get("programs", []) if isinstance(p, dict)
        }

        def _prog_info(pname: str, inst: dict | None = None) -> dict:
            template = pname
            if inst is not None and m.get("instance"):
                suffix = f"-{inst['id']}"
                if pname.endswith(suffix):
                    template = pname[: -len(suffix)]
            t = prog_templates.get(template, {})
            return {
                "name": pname,
                "status": statuses.get(pname, {}).get("status", "UNKNOWN"),
                "status_detail": statuses.get(pname, {}).get("description", ""),
                # Supervisor settings (editable via the args form)
                "autostart": t.get("autostart", True),
                "autorestart": t.get("autorestart", True),
                "startretries": int(t.get("startretries", 10000)),
                "priority": int(t.get("priority", 10)),
            }

        instances = []
        for iid, inst in sorted((entry.get("instances") or {}).items()):
            inst = inst or {}
            instances.append({
                "id": iid,
                "value": inst.get("value"),
                "args": inst.get("args", {}),
                "sockets": inst.get("sockets", []),
                "recordings_subdir": _instance_recordings_subdir(m, inst),
                "created_at": inst.get("created_at"),
                "programs": [
                    _prog_info(pname, inst) for pname in inst.get("programs", [])
                ],
            })
        modules.append({
            "name": name,
            "version": m.get("version"),
            "description": m.get("description"),
            "author": m.get("author"),
            "repo_url": entry.get("repo_url"),
            "installed_at": entry.get("installed_at"),
            "recordings_subdir": m.get("recordings_subdir"),
            "instance": m.get("instance"),
            "instances": instances,
            "ui": m.get("ui", []),
            "arguments": m.get("arguments", []),
            "sockets": m.get("sockets", []),
            # Flat program list across instances — for non-instanceable
            # modules this is exactly the manifest's programs, as before.
            "programs": [pi for inst in instances for pi in inst["programs"]],
        })
    return jsonify({"modules": modules})


@app.route("/api/modules/<name>")
def api_modules_detail(name):
    entry = load_registry().get("modules", {}).get(name)
    if entry is None:
        return jsonify({"error": f"module not installed: {name}"}), 404
    return jsonify(entry)


_REPO_URL_RE = re.compile(r"^(https?://|git@|file://).+")

# Module zips are source trees — 100 MB is generous.
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024


@app.errorhandler(413)
def upload_too_large(_):
    return jsonify({"error": "uploaded zip is too large (max 100 MB)"}), 413


def _new_install_job(*, source_type: str, repo_url: str,
                     ref: str | None = None, zip_path: str | None = None) -> dict:
    """Register a new install job (does not start it — see _launch_install_job).
    The caller must hold install_lock; the job wrapper releases it."""
    job = {
        "id": uuid.uuid4().hex[:12],
        "status": "pending",
        "source_type": source_type,
        "repo_url": repo_url,
        "ref": ref,
        "zip_path": zip_path,
        "module": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "error": None,
        "warnings": [],
        "log": [],
    }
    with module_jobs_lock:
        module_jobs[job["id"]] = job
    return job


def _launch_install_job(job: dict) -> None:
    threading.Thread(target=_run_install_job_guarded, args=(job,), daemon=True).start()


def _install_local_module(path: str | Path) -> dict:
    """Synchronous install of a local module directory. Used by install.sh for
    built-in default modules. Returns the finished job dict."""
    local_path = Path(path).resolve()
    install_lock.acquire()
    try:
        job = _new_install_job(source_type="local", repo_url=str(local_path))
        _run_install_job(job)
        return job
    finally:
        install_lock.release()


@app.route("/api/modules/install", methods=["POST"])
def api_modules_install():
    data = request.get_json() or {}
    repo_url = (data.get("repo_url") or "").strip()
    if not repo_url:
        return jsonify({"error": "repo_url is required"}), 400
    if not _REPO_URL_RE.match(repo_url):
        return jsonify({
            "error": "repo_url must start with https://, git@ or file://"
        }), 400
    ref = (data.get("ref") or "").strip() or None

    if not install_lock.acquire(blocking=False):
        return jsonify({"error": "another module install is already running"}), 409

    job = _new_install_job(source_type="git", repo_url=repo_url, ref=ref)
    _launch_install_job(job)
    return jsonify({"job_id": job["id"], "status": "pending"}), 202


@app.route("/api/modules/install-upload", methods=["POST"])
def api_modules_install_upload():
    """Install a module from an uploaded zip file (multipart field 'file').

    The zip must contain eventide-module.json at its root or inside a single
    top-level folder (as GitHub's "Download ZIP" produces).  From there the
    install follows the same pipeline as a git clone.
    """
    upload = request.files.get("file")
    if upload is None or not (upload.filename or "").strip():
        return jsonify({"error": "no file uploaded (multipart field 'file')"}), 400
    filename = Path(upload.filename).name  # strip any client-side path
    if not filename.lower().endswith(".zip"):
        return jsonify({"error": "uploaded file must be a .zip"}), 400
    if not install_lock.acquire(blocking=False):
        return jsonify({"error": "another module install is already running"}), 409

    job = _new_install_job(source_type="zip", repo_url=f"zip://{filename}")
    uploads_dir = Path(cfg["packages_dir"]) / ".uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    zip_path = uploads_dir / f"{job['id']}.zip"
    try:
        upload.save(zip_path)
    except OSError as exc:
        with module_jobs_lock:
            module_jobs.pop(job["id"], None)
        install_lock.release()
        return jsonify({"error": f"failed to store upload: {exc}"}), 500

    job["zip_path"] = str(zip_path)
    _launch_install_job(job)
    return jsonify({"job_id": job["id"], "status": "pending"}), 202


@app.route("/api/modules/jobs/<job_id>")
def api_modules_job(job_id):
    job = module_jobs.get(job_id)
    if job is None:
        return jsonify({"error": f"unknown job id: {job_id}"}), 404
    return jsonify(_job_snapshot(job))


@app.route("/api/modules/<name>/uninstall", methods=["POST"])
def api_modules_uninstall(name):
    with install_lock:  # wait for any running install to finish
        entry = _uninstall_module(name)
    if entry is None:
        return jsonify({"error": f"module not installed: {name}"}), 404
    return jsonify({"ok": True, "removed": name})


def _coerce_arg_updates(manifest: dict, new_args: dict) -> tuple[dict, list[str]]:
    """Type-check argument values against the manifest's declared arguments.
    Returns (updates, errors) — updates are safe to apply when errors is empty."""
    declared = {a["name"]: a for a in manifest.get("arguments", [])}
    updates: dict = {}
    errors: list[str] = []
    for key, value in new_args.items():
        arg = declared.get(key)
        if arg is None:
            errors.append(f"unknown argument: '{key}'")
            continue
        atype = arg.get("type")
        try:
            if atype == "int":
                if isinstance(value, bool):
                    raise ValueError
                value = int(str(value).strip())
            elif atype == "float":
                if isinstance(value, bool):
                    raise ValueError
                value = float(str(value).strip())
            else:
                value = str(value)
        except (TypeError, ValueError):
            errors.append(f"argument '{key}' must be of type {atype}")
            continue
        # Values land verbatim in an ini-style supervisor conf — a newline
        # would inject extra directives, so reject it outright.
        if isinstance(value, str) and ("\n" in value or "\r" in value):
            errors.append(f"argument '{key}' must be a single line")
            continue
        updates[key] = value
    return updates, errors


def _coerce_prog_updates(manifest: dict, new_progs: dict) -> tuple[dict, list[str]]:
    """Validate per-program supervisor settings (autostart, autorestart,
    startretries, priority).  Returns (updates by program name, errors)."""
    prog_by_name = {p["name"]: p for p in manifest.get("programs", [])}
    prog_updates: dict[str, dict] = {}
    errors: list[str] = []
    for pname, params in new_progs.items():
        prog = prog_by_name.get(pname)
        if prog is None:
            errors.append(f"unknown program: '{pname}'")
            continue
        if not isinstance(params, dict):
            errors.append(f"program '{pname}' settings must be an object")
            continue
        clean: dict = {}
        for key, value in params.items():
            if key in ("autostart", "autorestart"):
                if not isinstance(value, bool):
                    errors.append(f"program '{pname}' {key} must be a boolean")
                    continue
                clean[key] = value
            elif key in ("startretries", "priority"):
                try:
                    if isinstance(value, bool):
                        raise ValueError
                    iv = int(str(value).strip())
                    if iv < 0:
                        raise ValueError
                    clean[key] = iv
                except (TypeError, ValueError):
                    errors.append(
                        f"program '{pname}' {key} must be a non-negative integer"
                    )
            else:
                errors.append(f"program '{pname}' unknown setting '{key}'")
        prog_updates[pname] = clean
    return prog_updates, errors


def _apply_prog_updates(manifest: dict, prog_updates: dict) -> None:
    prog_by_name = {p["name"]: p for p in manifest.get("programs", [])}
    for pname, clean in prog_updates.items():
        if pname in prog_by_name:
            prog_by_name[pname].update(clean)


def _supervisor_reread_update() -> str | None:
    """supervisorctl reread && update; an error message on failure, None on
    success or when supervisorctl is unavailable (caller decides whether the
    latter deserves a warning)."""
    if not shutil.which("supervisorctl"):
        return None
    for sargs in (("reread",), ("update",)):
        rc, out = _supervisorctl(*sargs)
        if rc != 0:
            return f"supervisorctl {' '.join(sargs)} failed: {out}"
    return None


@app.route("/api/modules/<name>/args", methods=["POST"])
def api_modules_args(name):
    """Update a module's argument values and reload its supervisor config.

    The new values become the stored defaults in the registry manifest, the
    module's conf.d file is re-rendered from them, and supervisor is told to
    reread/update — which restarts any running programs whose command line
    changed.  An optional "programs" object updates supervisor settings
    (autostart, autorestart, startretries, priority) per program the same way.
    Instanceable modules are edited per instance instead — see
    /api/modules/<name>/instances/<iid>/args.
    """
    registry = load_registry()
    entry = registry.get("modules", {}).get(name)
    if entry is None:
        return jsonify({"error": f"module not installed: {name}"}), 404
    manifest = entry["manifest"]
    if manifest.get("instance"):
        return jsonify({
            "error": f"module '{name}' is instanceable — edit arguments per "
                     f"instance via /api/modules/{name}/instances/<iid>/args"
        }), 400
    data = request.get_json() or {}
    new_args = data.get("args")
    if not isinstance(new_args, dict):
        return jsonify({
            "error": "args must be an object mapping argument name → value"
        }), 400
    new_progs = data.get("programs", {})
    if not isinstance(new_progs, dict):
        return jsonify({
            "error": "programs must be an object mapping program name → settings"
        }), 400

    updates, errors = _coerce_arg_updates(manifest, new_args)
    prog_updates, prog_errors = _coerce_prog_updates(manifest, new_progs)
    errors += prog_errors
    if errors:
        return jsonify({"error": "; ".join(errors)}), 400

    _apply_prog_updates(manifest, prog_updates)
    for arg in manifest.get("arguments", []):
        if arg["name"] in updates:
            arg["default"] = updates[arg["name"]]

    # Keep the implicit default instance in sync with the manifest, then
    # re-render its conf (equivalent to rendering the manifest directly).
    inst = (entry.get("instances") or {}).get("default")
    if inst is None:
        inst = _default_instance(
            manifest, name, entry.get("conf_file"), entry.get("installed_at")
        )
        entry.setdefault("instances", {})["default"] = inst
    inst.setdefault("args", {}).update(updates)
    try:
        _render_instance_conf(name, entry, inst)
    except OSError as exc:
        return jsonify({"error": f"failed to write {inst.get('conf_file')}: {exc}"}), 500
    save_registry(registry)

    err = _supervisor_reread_update()
    if err:
        return jsonify({"error": err}), 500
    resp = {"ok": True, "arguments": manifest.get("arguments", [])}
    if not shutil.which("supervisorctl"):
        resp["warning"] = ("supervisorctl not found — config written but "
                           "supervisor not reloaded")
    return jsonify(resp)


# ── Module instances API ──────────────────────────────────────────────────────

@app.route("/api/modules/<name>/instances", methods=["POST"])
def api_instances_create(name):
    """Create (and start) an instance of an instanceable module.

    Body: {"value": "/dev/ttyS1", "args": {...}?} — value is the instance
    identifier substituted into the instance argument (and slugified into the
    instance id); args optionally overrides other argument defaults for this
    instance.  TCP ports are allocated and unix paths generated now.
    """
    data = request.get_json() or {}
    if "value" not in data:
        return jsonify({"error": "value is required"}), 400
    value = data.get("value")
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return jsonify({"error": "value must be a string or integer"}), 400
    if isinstance(value, str) and (not value.strip() or "\n" in value or "\r" in value):
        return jsonify({"error": "value must be a non-empty single line"}), 400
    extra_args = data.get("args", {})
    if not isinstance(extra_args, dict):
        return jsonify({
            "error": "args must be an object mapping argument name → value"
        }), 400

    with install_lock:  # serialize with installs and other instance changes
        registry = load_registry()
        entry = registry.get("modules", {}).get(name)
        if entry is None:
            return jsonify({"error": f"module not installed: {name}"}), 404
        manifest = entry["manifest"]
        spec = manifest.get("instance")
        if not spec:
            return jsonify({
                "error": f"module '{name}' is not instanceable (its manifest "
                         "declares no 'instance' key)"
            }), 400
        ia = spec.get("argument")
        coerced, errors = _coerce_arg_updates(manifest, {**extra_args, ia: value})
        if errors:
            return jsonify({"error": "; ".join(errors)}), 400
        value = coerced[ia]
        arg_updates = {k: v for k, v in coerced.items() if k != ia}
        try:
            inst = _build_instance(manifest, name, registry, value, arg_updates, None)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 400

        rec_sub = _instance_recordings_subdir(manifest, inst)
        if rec_sub:
            try:
                (Path(cfg["recordings_dir"]) / rec_sub).mkdir(
                    parents=True, exist_ok=True
                )
            except OSError as exc:
                return jsonify({
                    "error": f"failed to create recordings dir {rec_sub}: {exc}"
                }), 500
        entry.setdefault("instances", {})[inst["id"]] = inst
        try:
            _render_instance_conf(name, entry, inst)
        except OSError as exc:
            entry["instances"].pop(inst["id"], None)
            return jsonify({
                "error": f"failed to write {inst['conf_file']}: {exc}"
            }), 500
        save_registry(registry)
        err = _supervisor_reread_update()
        if err:
            return jsonify({"error": err}), 500
    return jsonify({"ok": True, "instance": inst}), 201


@app.route("/api/modules/<name>/instances/<iid>", methods=["DELETE"])
def api_instances_delete(name, iid):
    """Stop and remove an instance: its programs stop, its conf file is
    deleted, and supervisor is reloaded.  Deleting the last instance leaves
    the module installed but running nothing."""
    with install_lock:
        registry = load_registry()
        entry = registry.get("modules", {}).get(name)
        if entry is None:
            return jsonify({"error": f"module not installed: {name}"}), 404
        instances = entry.get("instances") or {}
        inst = instances.get(iid)
        if inst is None:
            return jsonify({"error": f"no such instance: {name}/{iid}"}), 404

        if shutil.which("supervisorctl"):
            for pname in (inst or {}).get("programs", []):
                try:
                    _supervisorctl("stop", pname)
                except (subprocess.TimeoutExpired, OSError):
                    pass  # best-effort stop
        conf = (inst or {}).get("conf_file")
        if conf:
            Path(conf).unlink(missing_ok=True)
        del instances[iid]
        save_registry(registry)
        err = _supervisor_reread_update()
    resp = {"ok": True, "removed": f"{name}/{iid}"}
    if err:
        resp["warning"] = err
    return jsonify(resp)


@app.route("/api/modules/<name>/instances/<iid>/args", methods=["POST"])
def api_instance_args(name, iid):
    """Update one instance's argument values and reload its supervisor config.

    Same body as /api/modules/<name>/args.  The instance argument itself is
    pinned (remove and re-add the instance to change it).  Supervisor program
    settings live on the shared program templates, so "programs" updates are
    applied to every instance's conf.  Program names in "programs" may be
    given as template names or this instance's suffixed names.
    """
    with install_lock:
        registry = load_registry()
        entry = registry.get("modules", {}).get(name)
        if entry is None:
            return jsonify({"error": f"module not installed: {name}"}), 404
        manifest = entry["manifest"]
        instances = entry.get("instances") or {}
        inst = instances.get(iid)
        if inst is None:
            return jsonify({"error": f"no such instance: {name}/{iid}"}), 404
        data = request.get_json() or {}
        new_args = data.get("args")
        if not isinstance(new_args, dict):
            return jsonify({
                "error": "args must be an object mapping argument name → value"
            }), 400
        new_progs = data.get("programs", {})
        if not isinstance(new_progs, dict):
            return jsonify({
                "error": "programs must be an object mapping program name → settings"
            }), 400

        spec = manifest.get("instance")
        if spec:
            # Accept this instance's suffixed program names as aliases for
            # the shared program templates.
            suffix = f"-{iid}"
            new_progs = {
                (k[: -len(suffix)] if k.endswith(suffix) else k): v
                for k, v in new_progs.items()
            }
        updates, errors = _coerce_arg_updates(manifest, new_args)
        prog_updates, prog_errors = _coerce_prog_updates(manifest, new_progs)
        errors += prog_errors
        if spec and spec.get("argument") in updates and (
            updates[spec["argument"]] != inst.get("value")
        ):
            errors.append(
                f"argument '{spec['argument']}' identifies this instance — "
                "remove and re-add the instance to change it"
            )
        if errors:
            return jsonify({"error": "; ".join(errors)}), 400

        _apply_prog_updates(manifest, prog_updates)
        inst.setdefault("args", {}).update(updates)
        if spec:
            inst["args"][spec["argument"]] = inst.get("value")  # pinned
            rec_sub = _instance_recordings_subdir(manifest, inst)
            if rec_sub:
                try:
                    (Path(cfg["recordings_dir"]) / rec_sub).mkdir(
                        parents=True, exist_ok=True
                    )
                except OSError as exc:
                    return jsonify({
                        "error": f"failed to create recordings dir {rec_sub}: {exc}"
                    }), 500
        else:
            for arg in manifest.get("arguments", []):
                if arg["name"] in updates:
                    arg["default"] = updates[arg["name"]]

        # Program settings are template-level: re-render every instance conf
        # so they take effect module-wide; otherwise only this conf changed.
        try:
            for other in instances.values():
                _render_instance_conf(name, entry, other or {})
        except OSError as exc:
            return jsonify({"error": f"failed to write conf file: {exc}"}), 500
        save_registry(registry)
        err = _supervisor_reread_update()
        if err:
            return jsonify({"error": err}), 500
    resp = {"ok": True, "args": inst.get("args", {})}
    if not shutil.which("supervisorctl"):
        resp["warning"] = ("supervisorctl not found — config written but "
                           "supervisor not reloaded")
    return jsonify(resp)

# ── SSH deploy key ────────────────────────────────────────────────────────────
# Private GitHub repos are cloned over SSH (see _clone_repo's HTTPS→SSH
# fallback).  These endpoints manage a single passphrase-less ed25519 keypair
# for the user this service runs as; the public key is displayed in the UI so
# the operator can add it to GitHub as a deploy key or account SSH key.
# github.com's host key is pinned into known_hosts on generation so the first
# SSH clone doesn't die on host-key verification.

_SSH_DIR = Path(os.environ.get("EVENTIDE_SSH_DIR", str(Path.home() / ".ssh")))
_SSH_KEY_PATH = _SSH_DIR / "id_ed25519"
_SSH_PUB_PATH = Path(str(_SSH_KEY_PATH) + ".pub")


def _ssh_key_info() -> dict:
    info = {
        "exists": False,
        "public_key": None,
        "fingerprint": None,
        "path": str(_SSH_KEY_PATH),
    }
    if not _SSH_PUB_PATH.exists():
        return info
    try:
        info["public_key"] = _SSH_PUB_PATH.read_text().strip()
    except OSError:
        return info
    info["exists"] = True
    try:
        proc = subprocess.run(
            ["ssh-keygen", "-lf", str(_SSH_PUB_PATH)],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode == 0:
            parts = proc.stdout.split()
            if len(parts) >= 2:
                info["fingerprint"] = parts[1]
    except (subprocess.TimeoutExpired, OSError):
        pass  # fingerprint is cosmetic — the key itself is what matters
    return info


@app.route("/api/ssh-key")
def api_ssh_key_get():
    return jsonify(_ssh_key_info())


@app.route("/api/ssh-key/generate", methods=["POST"])
def api_ssh_key_generate():
    data = request.get_json(silent=True) or {}
    if _SSH_PUB_PATH.exists() and not data.get("force"):
        return jsonify({
            "error": "an SSH key already exists (resubmit with force to overwrite)",
            **_ssh_key_info(),
        }), 409
    try:
        _SSH_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(_SSH_DIR, 0o700)
        _SSH_KEY_PATH.unlink(missing_ok=True)
        _SSH_PUB_PATH.unlink(missing_ok=True)
        proc = subprocess.run(
            [
                "ssh-keygen", "-t", "ed25519", "-N", "",
                "-C", f"eventide-{socket.gethostname()}",
                "-f", str(_SSH_KEY_PATH),
            ],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            return jsonify({"error": f"ssh-keygen failed: {proc.stderr.strip()}"}), 500
        os.chmod(_SSH_KEY_PATH, 0o600)
    except FileNotFoundError:
        return jsonify({"error": "ssh-keygen not found on this device"}), 500
    except (subprocess.TimeoutExpired, OSError) as exc:
        return jsonify({"error": f"key generation failed: {exc}"}), 500

    # Pin github.com's host key so the first SSH clone doesn't prompt/fail on
    # host verification.  Best-effort: if the device is offline the key still
    # generates, we just warn.
    warning = None
    try:
        scan = subprocess.run(
            ["ssh-keyscan", "github.com"],
            capture_output=True, text=True, timeout=20,
        )
        entries = scan.stdout.strip()
        if entries:
            known_hosts = _SSH_DIR / "known_hosts"
            existing = known_hosts.read_text() if known_hosts.exists() else ""
            if "github.com" not in existing:
                with known_hosts.open("a") as fh:
                    fh.write(entries + "\n")
                os.chmod(known_hosts, 0o600)
        else:
            warning = ("ssh-keyscan github.com returned nothing — "
                       "first SSH clone may fail host verification")
    except (subprocess.TimeoutExpired, OSError):
        warning = ("ssh-keyscan unavailable — "
                   "first SSH clone may fail host verification")

    resp = _ssh_key_info()
    if warning:
        resp["warning"] = warning
    return jsonify(resp)

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global _SUPERVISOR_RPC_URL
    parser = argparse.ArgumentParser(description="Eventide Backend — API Server")
    parser.add_argument("--recordings-dir",        default="/usr/local/eventide/recordings",
                        help="Root recordings directory (sub-dirs: evk/, picam/, ircam/)")
    parser.add_argument("--viewfinder-bin",        default="./target/release/viewfinder")
    parser.add_argument("--live-events-socket",    default="/tmp/evk4_events.sock")
    parser.add_argument("--live-port",             type=int, default=8081,
                        help="Port the legacy built-in live viewfinder binds to")
    parser.add_argument("--supervisor-rpc-url",    default=_SUPERVISOR_RPC_URL,
                        help="URL of supervisord's XML-RPC endpoint "
                             "(default: http://127.0.0.1:9001/RPC2)")
    parser.add_argument("--modules-registry",      default="/usr/local/eventide/modules.json",
                        help="Path to the installed-modules registry JSON file")
    parser.add_argument("--packages-dir",          default="/usr/local/eventide/packages",
                        help="Directory where module repositories are cloned")
    parser.add_argument("--supervisor-conf-d",     default="/etc/supervisor/conf.d",
                        help="Directory where per-module supervisor configs are written")
    parser.add_argument("--port-pool",             default="8100-8199",
                        help="Range of TCP ports allocated to module sockets that "
                             "don't request an explicit port (start-end)")
    parser.add_argument("--host",                  default="0.0.0.0")
    parser.add_argument("--port",                  type=int, default=5000)
    parser.add_argument("--html-file",             default=str(Path(__file__).resolve().with_name("dashboard.html")),
                        help="Path to the dashboard HTML file served at / "
                             "(default: dashboard.html alongside this script)")
    parser.add_argument("--settings-file",         default="/usr/local/eventide/data/settings.json",
                        help="Path to persisted system settings JSON")
    parser.add_argument("--install-local",         default=None, metavar="PATH",
                        help="Install a module from a local directory and exit")
    args = parser.parse_args()

    cfg.update(vars(args))

    # Allow CLI override of the supervisor URL module-level variable.
    _SUPERVISOR_RPC_URL = args.supervisor_rpc_url

    Path(args.packages_dir).mkdir(parents=True, exist_ok=True)

    # Load persisted system settings.
    global _settings
    _settings = load_settings()

    if args.install_local:
        job = _install_local_module(args.install_local)
        if job.get("status") == "done":
            print(f"[install-local] module '{job['module']}' installed")
            for line in job.get("log", []):
                print(f"[install-local] {line}")
            sys.exit(0)
        else:
            print(f"[install-local] failed: {job.get('error')}")
            for line in job.get("log", []):
                print(f"[install-local] {line}")
            sys.exit(1)

    print(f"[backend]  Recordings dir:    {args.recordings_dir}")
    print(f"[backend]  Viewfinder binary: {args.viewfinder_bin}")
    print(f"[backend]  Live VF port:      {args.live_port}  (legacy built-in viewfinder)")
    print(f"[backend]  Supervisor RPC:    {_SUPERVISOR_RPC_URL}  (proxied at /supervisor/)")
    print(f"[backend]  Modules registry:  {args.modules_registry}")
    print(f"[backend]  Packages dir:      {args.packages_dir}")
    print(f"[backend]  Supervisor conf.d: {args.supervisor_conf_d}")
    print(f"[backend]  Module port pool:  {args.port_pool}")
    print(f"[backend]  CORS origin:       {_ALLOWED_ORIGIN}")
    print(f"[backend]  Dashboard HTML:    {args.html_file}  (served at /)")
    print(f"[backend]  Tile proxy:        /tiles/<z>/<x>/<y>.png  →  {_OSM_BASE}")
    print(f"[backend]  API at:            http://{args.host}:{args.port}")

    # Auto-start live EVK viewfinder
    proc, err = start_viewfinder("live", vf_configs["live"])
    if err:
        print(f"[backend] WARNING: could not auto-start live viewfinder: {err}")
    else:
        viewfinders["live"] = proc
        print("[backend] Auto-started live viewfinder.")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()