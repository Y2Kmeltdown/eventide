"""
Eventide Backend
================
The backbone process of an eventide payload: serves the dashboard UI and
OSM tile proxy, exposes the control/file APIs, hosts the module manager
(/api/modules/*) and the graph compiler (/api/graph/*) that turns a
litegraph-built graph into the running supervisor config — see
docs/GRAPH_SUPERVISOR_PLAN.md — and proxies supervisord's XML-RPC endpoint.

This process CAN serve the full frontend itself (HTML at / plus an OSM
tile proxy at /tiles/), which is handy when talking to the device
directly (field laptop, local network).  For data-constrained links the
decoupled frontend server (frontend.py) is still preferred: it keeps
HTML/tile bandwidth off the link — see eventide.nginx.

Network locations are NOT hardcoded in nginx.  A module only stages its
code (clone/build/deps) — nothing runs, and no address is allocated, until
its programs are placed as nodes in a graph and that graph is submitted.
TCP ports for a node's tcp/http output sockets are then allocated by this
server (see --port-pool), and any HTTP service a node exposes is reachable
through the generic proxy

  /proxy/node/<node id>/<socket>/<upstream path>   →  127.0.0.1:<allocated port>/<upstream path>

resolved from the active graph's resolved addresses at request time.  nginx
only fronts this process (location /) and the base playback server
(/playback/).

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
import zoneinfo
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


# Standalone touchscreen kiosk UI — a separate, purpose-built page for the
# local digicam-style display (see config/setup-display.sh), hard-wired to
# the evk-datalogger and basler-camera modules. Does not replace / is not
# linked from the main dashboard; served independently at its own path.
@app.route("/kiosk")
def kiosk_page():
    html_path = Path(cfg.get("kiosk_html_file", "kiosk.html")).resolve()
    if not html_path.exists():
        return (
            "kiosk.html not found. "
            "Pass --kiosk-html-file or place it alongside eventide.py."
        ), 404
    return send_file(html_path, mimetype="text/html")


# Vendored third-party JS/CSS the dashboard needs (leaflet, gridstack,
# litegraph.js) — served locally rather than from a CDN, so the dashboard
# works with no internet access. Defaults to the vendor/ dir shipped
# alongside eventide.py (see code/vendor/).
@app.route("/vendor/<path:relpath>")
def vendor_file(relpath):
    vendor_dir = Path(cfg.get("vendor_dir") or Path(__file__).resolve().with_name("vendor")).resolve()
    filepath = (vendor_dir / relpath).resolve()
    if not filepath.is_relative_to(vendor_dir):
        abort(400)
    if not filepath.is_file():
        abort(404)
    return send_file(filepath)


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
    # timezone and hostname are NOT stored here — the OS (timedatectl /
    # hostnamectl) is the single source of truth, so GET /api/settings
    # reports them live (current_timezone()/current_hostname()) and POST
    # applies changes directly via those tools instead of persisting a
    # value that could drift from whatever the system actually has. See
    # _set_timezone()/_set_hostname() below.
    # None = no override, use the --recordings-dir the backend was started
    # with (cfg["recordings_dir"]). See current_recordings_dir().
    "recordings_dir": None,
    # Free-space percent below which GET /api/settings reports
    # recordings_dir_low_disk: true.
    "low_disk_warn_pct": 10,
    # Age-based retention: recordings_days is the cutoff (None = no cutoff
    # configured); retention_enabled is the separate, explicit opt-in that
    # actually turns on automatic deletion — see retention_sweep_loop().
    # GET /api/retention/preview works regardless of retention_enabled, so a
    # dry run never requires opting in first.
    "retention_days": None,
    "retention_enabled": False,
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


def current_recordings_dir() -> str:
    """The effective recordings root: the user's saved override if one is
    set, else the --recordings-dir this process was started with. Read this
    (not cfg["recordings_dir"] directly) anywhere recordings actually get
    written or listed, so a settings change takes effect without a backend
    restart."""
    override = (_settings or {}).get("recordings_dir")
    return override if isinstance(override, str) and override else cfg["recordings_dir"]


def _validate_recordings_dir(value) -> str | None:
    """None if `value` is a usable recordings directory, else an error
    message. Deliberately strict (must already exist and be writable) —
    see the SETTINGS tab: refusing an unmounted/typo'd path here is safer
    than silently creating it wherever the path happens to resolve."""
    if not isinstance(value, str) or not value.strip():
        return "recordings_dir must be a non-empty path"
    p = Path(value)
    if not p.is_absolute():
        return "recordings_dir must be an absolute path"
    if not p.is_dir():
        return f"directory does not exist: {value}"
    if not os.access(p, os.W_OK):
        return f"directory is not writable: {value}"
    return None


def _redirect_modules_to(new_dir: str) -> list[str]:
    """After the recordings directory changes: if a graph is currently
    active, regenerate it (full stop-and-regenerate, §2/§7) so every node's
    command picks up the new {recordings_dir}/{recordings_subdir} path —
    including recreating each recording node's directory under the new
    root, which _apply_graph's _mkdir_graph_paths already does. Returns the
    active graph's node ids when it was regenerated, else an empty list.

    NOTE: forward reference — validate_graph()/_apply_graph()/etc. are
    defined further down in this file (the graph compiler section); that's
    fine, since this function is only ever *called* at request time, well
    after module load has finished defining everything.
    """
    registry = load_registry()
    state = load_graph_state()
    graph = state.get("active")
    if not graph or not graph.get("nodes"):
        return []
    errors = validate_graph(graph, registry)
    if errors:
        raise OSError("; ".join(errors))
    try:
        resolved, _warnings = _apply_graph(graph, registry)
    except RuntimeError as exc:
        raise OSError(str(exc))
    state["resolved"] = {
        _resolved_key(nid, sname): addr for (nid, sname), addr in resolved.items()
    }
    save_graph_state(state)
    return [n["id"] for n in graph.get("nodes", []) if isinstance(n, dict)]


# ── Hostname / timezone ────────────────────────────────────────────────────────
# Both hook into systemd tools (hostnamectl/timedatectl) rather than being
# stored settings — the OS is the single source of truth, so GET always
# reports live values and POST applies changes directly, the same "OS state,
# not a settings.json value" shape as current_recordings_dir() above.

_HOSTNAME_RE = re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$")


def current_hostname() -> str:
    try:
        name = Path("/etc/hostname").read_text().strip()
        if name:
            return name
    except OSError:
        pass
    return socket.gethostname()


def _validate_hostname(value) -> str | None:
    if not isinstance(value, str) or not value:
        return "hostname must be a non-empty string"
    if len(value) > 63:
        return "hostname must be 63 characters or fewer"
    if not _HOSTNAME_RE.match(value):
        return "hostname may only contain letters, digits and hyphens, and can't start or end with a hyphen"
    return None


def _set_hostname(new_name: str) -> tuple[bool, str | None]:
    """Apply a new static hostname via hostnamectl, and best-effort fix up
    the 127.0.1.1 line in /etc/hosts — hostnamectl doesn't touch it, but a
    stale entry there is a classic cause of a slow/broken `sudo` on
    Debian-family systems afterward. Returns (True, None) on full success,
    (True, warning) if the hostname changed but the /etc/hosts tidy-up
    failed, or (False, error) if hostnamectl itself failed."""
    if not shutil.which("hostnamectl"):
        return False, "hostnamectl not found — is this a systemd-based OS?"
    old_name = current_hostname()
    try:
        subprocess.run(["hostnamectl", "set-hostname", new_name],
                        check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        return False, f"hostnamectl failed: {(exc.stderr or '').strip() or exc}"

    try:
        hosts_path = Path("/etc/hosts")
        if hosts_path.exists() and old_name:
            lines = hosts_path.read_text().splitlines()
            changed = False
            for i, line in enumerate(lines):
                parts = line.split()
                if parts and parts[0] == "127.0.1.1" and old_name in parts:
                    lines[i] = " ".join(new_name if p == old_name else p for p in parts)
                    changed = True
            if changed:
                _atomic_write(hosts_path, "\n".join(lines) + "\n")
    except OSError as exc:
        return True, f"hostname changed, but failed to update /etc/hosts (may need a manual fix): {exc}"
    return True, None


def current_timezone() -> str:
    try:
        tz = Path("/etc/timezone").read_text().strip()
        if tz:
            return tz
    except OSError:
        pass
    try:
        link = Path("/etc/localtime").resolve()
        zoneinfo_root = Path("/usr/share/zoneinfo").resolve()
        return str(link.relative_to(zoneinfo_root))
    except (OSError, ValueError):
        return "UTC"


def _validate_timezone(value) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return "timezone must be a non-empty string"
    try:
        known = zoneinfo.available_timezones()
    except Exception:
        known = None  # tzdata not available for some reason — don't hard-block
    if known and value not in known:
        return f"unknown timezone: {value} (expected an IANA name, e.g. Australia/Sydney)"
    return None


def _set_timezone(new_tz: str) -> tuple[bool, str | None]:
    if not shutil.which("timedatectl"):
        return False, "timedatectl not found — is this a systemd-based OS?"
    try:
        subprocess.run(["timedatectl", "set-timezone", new_tz],
                        check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        return False, f"timedatectl failed: {(exc.stderr or '').strip() or exc}"
    return True, None


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
    """Resolve the playback server from the active graph's eventide-core
    'playback_server' node, when the graph currently has one running. This
    only ever finds anything once a graph has been submitted — before that
    (or if eventide-core hasn't been placed), fall back to the
    PLAYBACK_UPSTREAM_URL environment variable / default so the proxy keeps
    working regardless.
    """
    found = _find_active_socket_port(
        module="eventide-core", program="playback_server", socket_name="playback",
    )
    if found:
        return f"http://127.0.0.1:{found[1]}"
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

# ── Node socket proxy ─────────────────────────────────────────────────────────
# Generic reverse proxy for HTTP services exposed by nodes in the *active*
# graph. The dashboard resolves node/socket names from /api/graph(/status)
# and calls
#   /proxy/node/<node id>/<socket>/<upstream path>
# which is forwarded to 127.0.0.1:<allocated port>/<upstream path>, the
# address the graph compiler allocated for that node's socket at submit time
# (see allocate_graph_addresses()). This replaces per-module nginx locations
# the same way the old per-module proxy did: nginx only fronts eventide.py,
# and a node's network location is derived from the graph's resolved
# addresses at request time, so nodes can come and go across submits without
# any web-server config changes.
#
# Responses are streamed (requests stream=True + a generator Response) so
# long-lived MJPEG streams work; Flask's dev server is threaded by default,
# so an open stream does not block ordinary API calls.

_HOP_BY_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "host", "content-length",
}


def _active_resolved_port(node_id: str, socket_name: str) -> int | None:
    """The allocated TCP port of one active-graph node's output socket, or
    None if that node/socket isn't part of the active graph (or resolves to
    a unix path rather than a port)."""
    state = load_graph_state()
    entry = (state.get("resolved") or {}).get(_resolved_key(node_id, socket_name))
    if entry and entry.get("kind") == "port":
        port = entry.get("value")
        return port if isinstance(port, int) else None
    return None


def _find_active_socket_port(*, module: str | None = None, program: str | None = None,
                             socket_name: str) -> tuple[str, int] | None:
    """The first active-graph node (optionally restricted to a module and/or
    program template) exposing an output socket named `socket_name`,
    resolved to its live port. Returns (node_id, port), or None if no such
    node is currently in the active graph."""
    state = load_graph_state()
    graph = state.get("active") or {}
    resolved = state.get("resolved") or {}
    for n in graph.get("nodes", []):
        if not isinstance(n, dict):
            continue
        if module is not None and n.get("module") != module:
            continue
        if program is not None and n.get("program") != program:
            continue
        entry = resolved.get(_resolved_key(n.get("id"), socket_name))
        if entry and entry.get("kind") == "port" and isinstance(entry.get("value"), int):
            return n.get("id"), entry["value"]
    return None


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


@app.route("/proxy/node/<node_id>/<socket_name>/", defaults={"rest": ""},
           methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
@app.route("/proxy/node/<node_id>/<socket_name>/<path:rest>",
           methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
def node_socket_proxy(node_id, socket_name, rest):
    port = _active_resolved_port(node_id, socket_name)
    if port is None:
        return jsonify({
            "error": f"no such active graph node tcp/http socket: {node_id}/{socket_name}"
        }), 404

    if request.method == "OPTIONS":
        return _proxy_options_response()
    return _proxy_to_port(port, rest, f"{node_id}/{socket_name}")

# ── Global recording trigger ───────────────────────────────────────────────────
# Headless fan-out to every installed module's `recording`-type UI component —
# the server-side counterpart of the dashboard's master-record widget, which
# used to do this as N independent fetch() PUTs from browser JS (see
# dashboard.html buildRecordingRow()/WIDGET_RENDERERS['master-record']) and so
# only ever ran while a tab was open. This lives here (rather than in a
# module) because it needs load_registry()/port-resolution access. The
# eventide-core module's scheduler program calls this over loopback when a
# cron job fires.

def _recording_components() -> list[dict]:
    """Every 'recording'-type ui component belonging to a program currently
    placed as a node in the *active* graph (see docs/GRAPH_SUPERVISOR_PLAN.md
    §10: ui[] is per-program, and the palette/aggregation source is now the
    running graph rather than the module registry)."""
    out = []
    state = load_graph_state()
    graph = state.get("active") or {}
    registry = load_registry()
    for n in graph.get("nodes", []):
        if not isinstance(n, dict):
            continue
        _, prog = _resolve_node(n, registry)
        if prog is None:
            continue
        for c in prog.get("ui", []):
            if not isinstance(c, dict) or c.get("type") != "recording":
                continue
            if not c.get("socket") or not c.get("put"):
                continue
            out.append({"node": n["id"], **c})
    return out


def _trigger_recording(want: bool, duration: float | None = None) -> list[dict]:
    """PUT {"<key>": want} to every recording component from
    _recording_components(). When starting with a duration, schedules a stop
    fan-out after `duration` seconds — a server-side replacement for
    buildRecordingRow's client-side scheduleAutoStop() setTimeout, so it
    still fires with no browser open. Returns a per-source result list."""
    results = []
    for comp in _recording_components():
        key = comp.get("key") or "recording"
        port = _active_resolved_port(comp["node"], comp["socket"])
        entry = {"node": comp["node"]}
        if port is None:
            entry.update(ok=False, error="offline")
            results.append(entry)
            continue
        body = {key: want}
        if want and duration:
            dur_key = (comp.get("duration") or {}).get("key") or "duration_seconds"
            body[dur_key] = duration
        try:
            r = _http.put(f"http://127.0.0.1:{port}{comp['put']}", json=body, timeout=5)
            entry.update(ok=r.ok, status=r.status_code)
        except _http.exceptions.RequestException as exc:
            entry.update(ok=False, error=str(exc))
        results.append(entry)

    if want and duration:
        threading.Timer(duration, _trigger_recording, kwargs={"want": False}).start()
    return results


@app.route("/api/recording/trigger", methods=["POST"])
def api_recording_trigger():
    data = request.get_json(silent=True) or {}
    want = bool(data.get("recording", True))
    duration = data.get("duration_seconds")
    duration = float(duration) if isinstance(duration, (int, float)) and duration > 0 else None
    results = _trigger_recording(want, duration)
    return jsonify({"ok": True, "count": len(results), "results": results})


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
        resp = dict(_settings)
    # Live-computed, not persisted: lets the dashboard show a health badge
    # for the recordings directory (e.g. "card removed") without the user
    # needing to touch the setting again to find out it's broken.
    rd = current_recordings_dir()
    resp["recordings_dir_effective"] = rd
    resp["recordings_dir_exists"] = Path(rd).is_dir()
    resp["recordings_dir_is_mount"] = os.path.ismount(rd) if resp["recordings_dir_exists"] else False
    if resp["recordings_dir_exists"]:
        try:
            usage = shutil.disk_usage(rd)
            resp["recordings_dir_free_bytes"] = usage.free
            resp["recordings_dir_free_pct"] = round(usage.free / usage.total * 100, 1) if usage.total else 0
        except OSError:
            resp["recordings_dir_free_bytes"] = None
            resp["recordings_dir_free_pct"] = None
    else:
        resp["recordings_dir_free_bytes"] = None
        resp["recordings_dir_free_pct"] = None
    warn_pct = resp.get("low_disk_warn_pct")
    resp["recordings_dir_low_disk"] = (
        resp["recordings_dir_free_pct"] is not None
        and isinstance(warn_pct, (int, float))
        and resp["recordings_dir_free_pct"] < warn_pct
    )
    resp["hostname"] = current_hostname()
    resp["timezone"] = current_timezone()
    return jsonify(resp)


@app.route("/api/settings", methods=["POST"])
def api_settings_post():
    global _settings
    data = request.get_json() or {}
    if not isinstance(data, dict):
        return jsonify({"error": "settings body must be a JSON object"}), 400

    # hostname/timezone are OS state, not persisted settings.json values —
    # pulled out of data before the generic merge below (see current_
    # hostname()/current_timezone()) and applied directly via systemd tools.
    new_hostname = data.pop("hostname", None)
    if new_hostname is not None:
        err = _validate_hostname(new_hostname)
        if err:
            return jsonify({"error": err}), 400

    new_timezone = data.pop("timezone", None)
    if new_timezone is not None:
        err = _validate_timezone(new_timezone)
        if err:
            return jsonify({"error": err}), 400

    if "recordings_dir" in data:
        err = _validate_recordings_dir(data["recordings_dir"])
        if err:
            return jsonify({"error": err}), 400

    if "retention_days" in data and data["retention_days"] is not None:
        days = data["retention_days"]
        if not isinstance(days, (int, float)) or isinstance(days, bool) or days <= 0:
            return jsonify({"error": "retention_days must be a positive number, or null to disable"}), 400

    warnings: list[str] = []
    if new_hostname is not None and new_hostname != current_hostname():
        ok, msg = _set_hostname(new_hostname)
        if not ok:
            return jsonify({"error": msg}), 500
        if msg:
            warnings.append(msg)

    if new_timezone is not None and new_timezone != current_timezone():
        ok, msg = _set_timezone(new_timezone)
        if not ok:
            return jsonify({"error": msg}), 500
        if msg:
            warnings.append(msg)

    with _settings_lock:
        _settings = load_settings()
        old_dir = current_recordings_dir()
        _settings.update(data)
        save_settings(_settings)

        redirected: list[str] = []
        if "recordings_dir" in data and data["recordings_dir"] != old_dir:
            try:
                redirected = _redirect_modules_to(data["recordings_dir"])
            except OSError as exc:
                resp = dict(_settings)
                resp["error"] = f"settings saved, but failed to redirect modules: {exc}"
                return jsonify(resp), 500

        resp = dict(_settings)
        resp["_modules_redirected"] = redirected
        resp["hostname"] = current_hostname()
        resp["timezone"] = current_timezone()
        if warnings:
            resp["_warnings"] = warnings
        return jsonify(resp)


# ── Recordings ────────────────────────────────────────────────────────────────
# Recording sources are graph-node-driven (docs/GRAPH_SUPERVISOR_PLAN.md §9):
# every node in the *active* graph whose program references
# {recordings_subdir} gets a recordings directory
# (<recordings_dir>/<recordings_subdir>-<node id>) and shows up as a source
# here. There's no more module-level or "base program" source distinct from
# a node — installing a module alone adds nothing here; a node has to be
# placed and the graph submitted before anything can record. The
# dashboard's PLAYBACK tab builds one inner tab per source — sources with
# no active node simply do not exist.

def _recording_sources() -> list[dict]:
    """One source per *active graph* node whose program references
    {recordings_subdir}, named `<subdir>-<node id>` — see module docstring
    above."""
    sources = []
    registry = load_registry()
    state = load_graph_state()
    for n in (state.get("active") or {}).get("nodes", []):
        if not isinstance(n, dict):
            continue
        manifest, prog = _resolve_node(n, registry)
        if prog is None:
            continue
        sub = _node_recordings_subdir(manifest, prog, n["id"])
        if sub:
            sources.append({
                "name": sub,
                "module": f"{n['id']} — {n.get('module')}/{n.get('program')}",
            })
    sources.sort(key=lambda s: s["name"])
    return sources


def _recordings_dir(source: str) -> Path:
    return Path(current_recordings_dir()) / source


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
    """List the recording sources — one per active-graph node that records
    (see _recording_sources())."""
    return jsonify({"sources": _recording_sources()})


@app.route("/api/recordings/<source>")
def list_recordings_source(source):
    if source not in {s["name"] for s in _recording_sources()}:
        return jsonify({"error": f"unknown recording source: {source}"}), 404
    return _list_recordings_for(source)


RECORDING_EXTENSIONS = ("*.raw", "*.mp4", "*.h264", "*.jsonl", "*.basler")

def _scan_recordings(source: str) -> list[dict]:
    """Every recording file for `source`: {name, size, ext, favorited,
    mtime}. Shared by the file-list endpoint and retention (preview +
    sweep) below, so both agree on exactly what counts as a recording."""
    recordings_dir = _recordings_dir(source)
    if not recordings_dir.exists():
        return []
    favorites = _load_favorites(source)
    seen = set()
    entries = []
    for pattern in RECORDING_EXTENSIONS:
        for f in recordings_dir.glob(pattern):
            if f.is_file() and f.name not in seen:
                seen.add(f.name)
                st = f.stat()
                entries.append({
                    "name": f.name,
                    "size": st.st_size,
                    "ext": f.suffix.lstrip("."),
                    "favorited": f.name in favorites,
                    "mtime": st.st_mtime,
                })
    return entries


def _list_recordings_for(source: str):
    entries = _scan_recordings(source)
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


# ── Retention ────────────────────────────────────────────────────────────────
# Age-based pruning of old recordings. retention_days (the cutoff) and
# retention_enabled (the actual-deletion opt-in) are separate settings on
# purpose: /api/retention/preview always works off retention_days alone, so
# the dashboard can show exactly what a cutoff would delete before the user
# ever turns automatic deletion on. Favorited recordings are never touched.

def _retention_candidates(days: float) -> list[dict]:
    cutoff = time.time() - days * 86400
    candidates = []
    for source in _recording_sources():
        for f in _scan_recordings(source["name"]):
            if not f["favorited"] and f["mtime"] < cutoff:
                candidates.append({
                    "source": source["name"],
                    "name": f["name"],
                    "size": f["size"],
                    "age_days": round((time.time() - f["mtime"]) / 86400, 1),
                })
    candidates.sort(key=lambda x: x["age_days"], reverse=True)
    return candidates


@app.route("/api/retention/preview")
def api_retention_preview():
    days = request.args.get("days", type=float)
    if days is None:
        days = _settings.get("retention_days")
    if not isinstance(days, (int, float)) or days <= 0:
        return jsonify({"error": "no retention_days configured or provided (?days=N)"}), 400
    items = _retention_candidates(days)
    return jsonify({
        "days": days,
        "count": len(items),
        "total_bytes": sum(i["size"] for i in items),
        "items": items,
    })


def retention_sweep_loop() -> None:
    """Background sweep — checked hourly, only acts while retention_enabled
    is on. Runs as a daemon thread started from main(); errors are logged
    and never crash the loop, matching the eventide-core scheduler's own
    firing-loop convention (modules/eventide-core/scheduler.py)."""
    while True:
        time.sleep(3600)
        try:
            with _settings_lock:
                enabled = _settings.get("retention_enabled")
                days = _settings.get("retention_days")
            if not enabled or not isinstance(days, (int, float)) or days <= 0:
                continue
            for item in _retention_candidates(days):
                try:
                    (_recordings_dir(item["source"]) / item["name"]).unlink()
                    print(f"[retention] deleted {item['source']}/{item['name']} "
                          f"(age {item['age_days']}d, {item['size']} bytes)", flush=True)
                except OSError as exc:
                    print(f"[retention] failed to delete {item['source']}/{item['name']}: {exc}", flush=True)
        except Exception as exc:  # keep the loop alive no matter what
            print(f"[retention] sweep error: {exc}", flush=True)


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
# (see docs/GRAPH_SUPERVISOR_PLAN.md).  Installing a module is a background
# job: clone → validate → deps → build → artifacts → register. That's as far
# as it goes — no supervisor config is written and nothing starts. It just
# makes the module's programs[] available as node types in the GRAPH tab's
# palette; what actually runs (and its conf.d file) is driven entirely by
# the graph compiler (/api/graph/submit) once a graph is built and submitted.
# Installed modules are tracked in a JSON registry (default
# /usr/local/eventide/modules.json).

MANIFEST_FILENAME = "eventide-module.json"
MODULE_INSTALL_DIR = "/usr/local/eventide/code"
MODULE_CONFIG_DIR  = "/usr/local/eventide/config"

_MODULE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_PROGRAM_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_ARG_NAME_RE = re.compile(r"^[a-z0-9_]+$")
_ARG_TYPES = ("str", "int", "float")
_UI_COMPONENT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_UI_TYPES = ("mjpeg", "form", "telemetry", "joystick", "table", "map", "recording", "orientation3d", "features", "master-record", "schedule-table")
_UI_REGIONS = ("sidebar", "center")
_UI_FIELD_KINDS = ("number", "slider", "toggle", "text", "select", "nudge")
_SOCKET_DIRECTIONS = ("output", "input")
_SOCKET_TRANSPORTS = ("unix", "tcp", "http")
_SOCKET_PATTERNS = ("stream", "request-reply")
_SOCKET_STREAM_KINDS = ("irregular", "regular", "framed")
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
        return  # copy operations outside install jobs have no job log
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


# ── Graph state ───────────────────────────────────────────────────────────────
# The graph the litegraph-based GRAPH tab edits and submits — see
# docs/GRAPH_SUPERVISOR_PLAN.md §7. Two graphs are kept: "draft" (whatever the
# editor last autosaved, with no effect on the running system) and "active"
# (the last successfully submitted graph, which is what's actually rendered
# into supervisor's conf.d). "resolved" caches the addresses allocated for
# "active" at submit time, keyed "<node id>::<socket name>" (JSON object keys
# must be strings) — it is NOT recomputed on every request, because port
# allocation depends on what else is free on the system at submit time and
# must match whatever was actually baked into the generated conf.

def _graph_state_path() -> Path:
    return Path(cfg.get("graph_file", "/usr/local/eventide/graph.json"))


def load_graph_state() -> dict:
    path = _graph_state_path()
    if not path.exists():
        return {"active": None, "draft": None, "resolved": {}}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return {
                "active": data.get("active"),
                "draft": data.get("draft"),
                "resolved": data.get("resolved") or {},
            }
    except (json.JSONDecodeError, OSError):
        pass
    return {"active": None, "draft": None, "resolved": {}}


def save_graph_state(state: dict) -> None:
    path = _graph_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(state, indent=2) + "\n")


def _resolved_key(node_id: str, socket_name: str) -> str:
    return f"{node_id}::{socket_name}"


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

    # NOTE: `arguments`, `sockets`, and `ui` are no longer module-level —
    # each lives on the program that owns it (see docs/GRAPH_SUPERVISOR_PLAN.md
    # §8). A program's `{arg:...}`/`{socket:...}` placeholders and `ui[]`
    # entries may only reference that same program's own declarations.
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

            arg_names: set[str] = set()
            args = p.get("arguments", [])
            if not isinstance(args, list):
                errors.append(f"program '{pn}': 'arguments' must be a list")
                args = []
            else:
                for a in args:
                    if not isinstance(a, dict):
                        errors.append(f"program '{pn}': each argument must be an object")
                        continue
                    an = a.get("name")
                    if not isinstance(an, str) or not _ARG_NAME_RE.match(an):
                        errors.append(f"program '{pn}': argument 'name' must match ^[a-z0-9_]+$")
                        continue
                    if an in arg_names:
                        errors.append(f"program '{pn}': duplicate argument name '{an}'")
                    arg_names.add(an)
                    if not isinstance(a.get("flag"), str) or not a.get("flag"):
                        errors.append(f"program '{pn}': argument '{an}' needs a 'flag' (e.g. \"--port\")")
                    atype = a.get("type")
                    if atype not in _ARG_TYPES:
                        errors.append(f"program '{pn}': argument '{an}' type must be one of {_ARG_TYPES}")
                    if "default" not in a:
                        errors.append(f"program '{pn}': argument '{an}' needs a 'default'")
                    elif atype == "int" and not isinstance(a["default"], int):
                        errors.append(f"program '{pn}': argument '{an}' default must be an int")
                    elif atype == "float" and not isinstance(a["default"], (int, float)):
                        errors.append(f"program '{pn}': argument '{an}' default must be a number")
                    elif atype == "str" and not isinstance(a["default"], str):
                        errors.append(f"program '{pn}': argument '{an}' default must be a string")

            sock_names: set[str] = set()
            sockets = p.get("sockets", [])
            if not isinstance(sockets, list):
                errors.append(f"program '{pn}': 'sockets' must be a list")
                sockets = []
            else:
                for s in sockets:
                    if not isinstance(s, dict):
                        errors.append(f"program '{pn}': each socket must be an object")
                        continue
                    sn = s.get("name")
                    if not isinstance(sn, str) or not sn:
                        errors.append(f"program '{pn}': each socket needs a 'name'")
                        continue
                    if sn in sock_names:
                        errors.append(f"program '{pn}': duplicate socket name '{sn}'")
                    sock_names.add(sn)

                    direction = s.get("direction")
                    if direction not in _SOCKET_DIRECTIONS:
                        errors.append(
                            f"program '{pn}': socket '{sn}' direction must be "
                            f"one of {_SOCKET_DIRECTIONS}"
                        )

                    transport = s.get("transport")
                    if transport not in _SOCKET_TRANSPORTS:
                        errors.append(
                            f"program '{pn}': socket '{sn}' transport must be "
                            f"one of {_SOCKET_TRANSPORTS}"
                        )
                        transport = None

                    pattern = s.get("pattern")
                    if transport == "http":
                        if pattern is not None and pattern != "request-reply":
                            errors.append(
                                f"program '{pn}': socket '{sn}' is transport 'http', so "
                                "'pattern' must be omitted or 'request-reply'"
                            )
                        pattern = "request-reply"
                    elif transport in ("unix", "tcp"):
                        if pattern not in _SOCKET_PATTERNS:
                            errors.append(
                                f"program '{pn}': socket '{sn}' pattern is required and "
                                f"must be one of {_SOCKET_PATTERNS}"
                            )

                    stream_kind = s.get("stream_kind")
                    if pattern == "stream":
                        if stream_kind not in _SOCKET_STREAM_KINDS:
                            errors.append(
                                f"program '{pn}': socket '{sn}' is pattern 'stream', so "
                                f"'stream_kind' is required and must be one of "
                                f"{_SOCKET_STREAM_KINDS}"
                            )
                    elif stream_kind is not None:
                        errors.append(
                            f"program '{pn}': socket '{sn}' has 'stream_kind' set but "
                            "isn't pattern 'stream' — remove it"
                        )

                    capped = s.get("capped")
                    if capped is not None:
                        if not isinstance(capped, bool):
                            errors.append(f"program '{pn}': socket '{sn}' 'capped' must be a boolean")
                        elif not (pattern == "stream" and transport != "http" and direction == "output"):
                            errors.append(
                                f"program '{pn}': socket '{sn}' 'capped' only applies to a "
                                "stream-pattern, non-http output socket — remove it"
                            )

                    if transport in ("tcp", "http"):
                        # 'port' is optional: when omitted, eventide allocates
                        # one from --port-pool at graph-submit time.
                        port = s.get("port")
                        if port is not None and (
                            not isinstance(port, int) or not (1 <= port <= 65535)
                        ):
                            errors.append(f"program '{pn}': socket '{sn}' 'port' must be 1-65535 when given")
                        if s.get("path") is not None:
                            errors.append(
                                f"program '{pn}': socket '{sn}' is {transport}, so 'path' doesn't apply"
                            )
                    elif transport == "unix":
                        # 'path' is optional: when omitted, eventide generates
                        # one at graph-submit time.
                        spath = s.get("path")
                        if spath is not None and (
                            not isinstance(spath, str) or not spath.startswith("/")
                        ):
                            errors.append(f"program '{pn}': socket '{sn}' unix 'path' must be absolute when given")
                        if s.get("port") is not None:
                            errors.append(f"program '{pn}': socket '{sn}' is unix, so 'port' doesn't apply")

                    # Optional: names an int argument that sets how many
                    # numbered copies of this output slot are generated
                    # (e.g. a fan-out node's output count). Schema-only for
                    # now — no program uses it yet.
                    count_arg = s.get("count_arg")
                    if count_arg is not None:
                        if not isinstance(count_arg, str) or not count_arg:
                            errors.append(f"program '{pn}': socket '{sn}' 'count_arg' must name an argument")
                        else:
                            cdecl = next(
                                (a for a in args
                                 if isinstance(a, dict) and a.get("name") == count_arg),
                                None,
                            )
                            if cdecl is None:
                                errors.append(
                                    f"program '{pn}': socket '{sn}' count_arg references "
                                    f"undeclared argument '{count_arg}'"
                                )
                            elif cdecl.get("type") != "int":
                                errors.append(
                                    f"program '{pn}': socket '{sn}' count_arg '{count_arg}' "
                                    "must be of type int"
                                )

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

            ui = p.get("ui", [])
            if not isinstance(ui, list):
                errors.append(f"program '{pn}': 'ui' must be a list")
            else:
                http_sock_names = {
                    s.get("name") for s in sockets
                    if isinstance(s, dict) and s.get("transport") == "http"
                }

                def _check_sock(cid, value, where):
                    if not isinstance(value, str) or not value:
                        errors.append(f"program '{pn}': ui '{cid}' needs a socket name for '{where}'")
                    elif value not in http_sock_names:
                        errors.append(
                            f"program '{pn}': ui '{cid}' references undeclared http "
                            f"socket '{value}' ({where})"
                        )

                def _is_path(value):
                    return isinstance(value, str) and value.startswith("/")

                seen_ui_ids: set[str] = set()
                for comp in ui:
                    if not isinstance(comp, dict):
                        errors.append(f"program '{pn}': each ui component must be an object")
                        continue
                    cid = comp.get("id")
                    if not isinstance(cid, str) or not _UI_COMPONENT_ID_RE.match(cid):
                        errors.append(
                            f"program '{pn}': each ui component needs an 'id' matching "
                            "^[a-z0-9][a-z0-9_-]*$"
                        )
                        continue
                    if cid in seen_ui_ids:
                        errors.append(f"program '{pn}': duplicate ui component id '{cid}'")
                    seen_ui_ids.add(cid)
                    ctype = comp.get("type")
                    if ctype not in _UI_TYPES:
                        errors.append(f"program '{pn}': ui '{cid}' type must be one of {_UI_TYPES}")
                        continue
                    if comp.get("region", "sidebar") not in _UI_REGIONS:
                        errors.append(f"program '{pn}': ui '{cid}' region must be one of {_UI_REGIONS}")
                    if "default" in comp and not isinstance(comp["default"], bool):
                        errors.append(f"program '{pn}': ui '{cid}' default must be a boolean")
                    if "title" in comp and not isinstance(comp["title"], str):
                        errors.append(f"program '{pn}': ui '{cid}' title must be a string")

                    if ctype == "mjpeg":
                        _check_sock(cid, comp.get("socket"), "socket")
                        if not _is_path(comp.get("path")):
                            errors.append(f"program '{pn}': ui '{cid}' needs a 'path' like '/stream'")
                    elif ctype == "form":
                        _check_sock(cid, comp.get("socket"), "socket")
                        if "method" in comp and comp["method"] not in ("PUT", "POST"):
                            errors.append(f"program '{pn}': ui '{cid}' method must be PUT or POST")
                        fields = comp.get("fields")
                        if not isinstance(fields, list) or not fields:
                            errors.append(f"program '{pn}': ui '{cid}' needs a non-empty 'fields' list")
                        else:
                            for f in fields:
                                if not isinstance(f, dict) or not isinstance(f.get("key"), str):
                                    errors.append(f"program '{pn}': ui '{cid}' field needs a 'key'")
                                    continue
                                kind = f.get("kind", "number")
                                if kind not in _UI_FIELD_KINDS:
                                    errors.append(
                                        f"program '{pn}': ui '{cid}' field '{f.get('key')}' kind "
                                        f"must be one of {_UI_FIELD_KINDS}"
                                    )
                                if kind == "select" and not _is_str_list(f.get("options")):
                                    errors.append(
                                        f"program '{pn}': ui '{cid}' select field '{f['key']}' "
                                        "needs 'options' (list of strings)"
                                    )
                    elif ctype == "telemetry":
                        _check_sock(cid, comp.get("socket"), "socket")
                        if not _is_path(comp.get("get")):
                            errors.append(f"program '{pn}': ui '{cid}' needs a 'get' path")
                        rows = comp.get("rows")
                        if not isinstance(rows, list) or not rows:
                            errors.append(f"program '{pn}': ui '{cid}' needs a non-empty 'rows' list")
                        elif not all(
                            isinstance(r, dict)
                            and isinstance(r.get("label"), str)
                            and isinstance(r.get("path"), str)
                            for r in rows
                        ):
                            errors.append(f"program '{pn}': ui '{cid}' rows must all have {{label, path}}")
                    elif ctype == "recording":
                        _check_sock(cid, comp.get("socket"), "socket")
                        if not _is_path(comp.get("get")):
                            errors.append(f"program '{pn}': ui '{cid}' needs a 'get' path")
                        if not _is_path(comp.get("put")):
                            errors.append(f"program '{pn}': ui '{cid}' needs a 'put' path")
                        dur = comp.get("duration")
                        if dur is not None:
                            if not isinstance(dur, dict):
                                errors.append(f"program '{pn}': ui '{cid}' duration must be an object")
                            else:
                                for dk in ("default", "min", "max"):
                                    if dk in dur and not isinstance(dur[dk], (int, float)):
                                        errors.append(
                                            f"program '{pn}': ui '{cid}' duration.{dk} must be a number"
                                        )
                                if "key" in dur and not isinstance(dur["key"], str):
                                    errors.append(f"program '{pn}': ui '{cid}' duration.key must be a string")
                                if "label" in dur and not isinstance(dur["label"], str):
                                    errors.append(f"program '{pn}': ui '{cid}' duration.label must be a string")
                    elif ctype == "joystick":
                        _check_sock(cid, comp.get("socket"), "socket")
                        if not _is_path(comp.get("put")):
                            errors.append(f"program '{pn}': ui '{cid}' needs a 'put' path")
                    elif ctype == "table":
                        _check_sock(cid, comp.get("socket"), "socket")
                        if not _is_path(comp.get("get")):
                            errors.append(f"program '{pn}': ui '{cid}' needs a 'get' path")
                        cols = comp.get("columns")
                        if not isinstance(cols, list) or not cols:
                            errors.append(f"program '{pn}': ui '{cid}' needs a non-empty 'columns' list")
                        ra = comp.get("row_action")
                        if ra is not None and (
                            not isinstance(ra, dict)
                            or not _is_path(ra.get("path"))
                            or not isinstance(ra.get("key"), str)
                        ):
                            errors.append(f"program '{pn}': ui '{cid}' row_action needs {{path, key}}")
                    elif ctype == "map":
                        for binding in ("track", "adsb"):
                            b = comp.get(binding)
                            if b is None:
                                continue
                            if not isinstance(b, dict):
                                errors.append(f"program '{pn}': ui '{cid}' {binding} must be an object")
                                continue
                            _check_sock(cid, b.get("socket"), f"{binding}.socket")
                            if not _is_path(b.get("get")):
                                errors.append(f"program '{pn}': ui '{cid}' {binding} needs a 'get' path")
    return errors


def conflict_errors(manifest: dict, registry: dict) -> list[str]:
    """Check the manifest against already-installed modules.

    Program names no longer need to be globally unique — a program is only
    a *template* now (§5 of docs/GRAPH_SUPERVISOR_PLAN.md); what actually
    runs is one supervisor program per graph *node*, named after the node's
    id, which is enforced unique within the graph at submit time
    (validate_graph()), not at module-install time. Explicit TCP/HTTP ports
    are still checked here, since — unlike pool-allocated ports — they're
    hardcoded by the module author and can never coexist if two different
    modules claim the same one, in any graph.
    """
    errors: list[str] = []
    installed = registry.get("modules", {})
    name = manifest["name"]
    if name in installed:
        errors.append(
            f"module '{name}' is already installed (uninstall it first to reinstall)"
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
        for p in e["manifest"].get("programs", []):
            for s in (p or {}).get("sockets", []):
                if (
                    isinstance(s, dict)
                    and s.get("transport") in ("tcp", "http")
                    and isinstance(s.get("port"), int)
                ):
                    existing_ports.setdefault(s["port"], ename)
    for p in manifest.get("programs", []):
        for s in (p or {}).get("sockets", []):
            if (
                isinstance(s, dict)
                and s.get("transport") in ("tcp", "http")
                and isinstance(s.get("port"), int)
                and s["port"] in existing_ports
            ):
                errors.append(
                    f"socket port {s['port']} is already used by module "
                    f"'{existing_ports[s['port']]}'"
                )
    return errors


# ── Port allocation ───────────────────────────────────────────────────────────
# TCP ports are platform-assigned: a socket may request an explicit 'port'
# (honoured if free), but the default is to leave it out and let eventide
# pick one from --port-pool. Unlike the old per-module model, nothing is
# allocated at module-install time any more — addresses are resolved once
# per graph *node* output socket, at graph-submit time (see
# allocate_graph_addresses() below).

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


# ── Config rendering ──────────────────────────────────────────────────────────
# render_placeholders() is still used for module-level, install-time text —
# dependencies.commands, install.commands, and install.artifacts destinations
# — which only ever reference {install_dir}/{module_dir}/{venv_...}/etc, never
# {arg:...}/{socket:...} (those are per-program now — see render_node_command
# below, used by the graph compiler to render each node's actual command).

def render_placeholders(text: str, manifest: dict, module_name: str) -> str:
    venv_dir = Path(cfg["packages_dir"]) / module_name / _VENV_DIR_NAME
    out = text
    out = out.replace("{install_dir}", MODULE_INSTALL_DIR)
    out = out.replace("{config_dir}", MODULE_CONFIG_DIR)
    out = out.replace("{module_dir}", str(Path(cfg["packages_dir"]) / module_name))
    out = out.replace("{recordings_dir}", current_recordings_dir())
    sub = manifest.get("recordings_subdir")
    if isinstance(sub, str) and sub:
        out = out.replace(
            "{recordings_subdir}", str(Path(current_recordings_dir()) / sub)
        )
    out = out.replace("{venv_dir}", str(venv_dir))
    out = out.replace("{venv_python}", str(venv_dir / "bin" / "python3"))
    return out


# ── The graph compiler ────────────────────────────────────────────────────────
# Turns a submitted graph (nodes = placed program instances, edges = socket
# wiring) into a single rendered supervisor conf. See
# docs/GRAPH_SUPERVISOR_PLAN.md §5-§7 for the full design; summary:
#   - a node is {"id", "module", "program", "args": {...}, ...} — "id" is
#     both the node's identity in the graph *and* its supervisor program
#     name (so no separate copies/instance mechanism is needed any more:
#     placing another node of the same program template just is another
#     instance).
#   - an edge is {"from": {"node", "socket"}, "to": {"node", "socket"}}.
#   - one address (a pool TCP port, or a generated/explicit unix path) is
#     resolved per (node, output-socket) pair, regardless of how many edges
#     (0, 1 for a stream socket, or N for a request-reply/http one) use it —
#     see allocate_graph_addresses().

_NODE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _resolve_node(node: dict, registry: dict) -> tuple[dict | None, dict | None]:
    """(module manifest, program template) a node refers to, or (None, None)
    if its module isn't installed or no longer declares that program."""
    if not isinstance(node, dict):
        return None, None
    entry = registry.get("modules", {}).get(node.get("module"))
    if entry is None:
        return None, None
    manifest = entry["manifest"]
    prog = next(
        (p for p in manifest.get("programs", [])
         if isinstance(p, dict) and p.get("name") == node.get("program")),
        None,
    )
    if prog is None:
        return None, None
    return manifest, prog


def _socket_pattern(sock: dict) -> str | None:
    """Effective pattern, treating http as always request-reply (§6) even
    when a socket entry omits the field."""
    if sock.get("transport") == "http":
        return "request-reply"
    return sock.get("pattern")


def _node_arg_values(prog: dict, node: dict) -> dict:
    """A node's effective argument values: the program template's declared
    defaults, overridden by whatever the node itself sets."""
    values = {
        a["name"]: a.get("default")
        for a in prog.get("arguments", []) if isinstance(a, dict) and "name" in a
    }
    values.update(node.get("args") or {})
    return values


def _expand_program_sockets(prog: dict, node_args: dict) -> list[dict]:
    """A program's socket list, with any `count_arg` socket expanded into N
    numbered sockets (`<name>1`..`<name>N`, 1-indexed) — N being that
    node's resolved value of the named argument. This is what lets the
    built-in Stream Fan-out program declare a variable number of outputs
    (docs/GRAPH_SUPERVISOR_PLAN.md §9), and works for any module socket
    that sets `count_arg`. Every expanded socket keeps all of the base
    socket's fields and gains `_base_name`, so callers that need to treat
    them as one group again (render_graph_conf's `{socket:<base>}`
    substitution) can."""
    expanded: list[dict] = []
    for s in prog.get("sockets", []):
        if not isinstance(s, dict) or not s.get("name"):
            continue
        count_arg = s.get("count_arg")
        if not count_arg:
            expanded.append(s)
            continue
        try:
            count = max(int(node_args.get(count_arg, 0)), 0)
        except (TypeError, ValueError):
            count = 0
        base = s["name"]
        for i in range(1, count + 1):
            expanded.append({**s, "name": f"{base}{i}", "_base_name": base})
    return expanded


def validate_graph(graph, registry: dict) -> list[str]:
    """Structural + compatibility validation of a graph document, per
    docs/GRAPH_SUPERVISOR_PLAN.md §6-§7. Unconnected stream-socket outputs are
    NOT an error here — see graph_warnings()."""
    errors: list[str] = []
    if not isinstance(graph, dict):
        return ["graph must be an object"]

    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        errors.append("'nodes' must be a list")
        nodes = []
    edges = graph.get("edges")
    if not isinstance(edges, list):
        errors.append("'edges' must be a list")
        edges = []

    node_by_id: dict[str, dict] = {}
    node_progs: dict[str, dict] = {}
    node_sockets: dict[str, list[dict]] = {}
    for n in nodes:
        if not isinstance(n, dict):
            errors.append("each node must be an object")
            continue
        nid = n.get("id")
        if not isinstance(nid, str) or not _NODE_ID_RE.match(nid):
            errors.append("each node needs an 'id' matching ^[a-z0-9][a-z0-9_-]*$")
            continue
        if nid in node_by_id:
            errors.append(f"duplicate node id '{nid}'")
            continue
        node_by_id[nid] = n
        _, prog = _resolve_node(n, registry)
        if prog is None:
            errors.append(
                f"node '{nid}' references an unknown module/program "
                f"'{n.get('module')}'/'{n.get('program')}'"
            )
            continue
        node_progs[nid] = prog
        declared = {
            a["name"] for a in prog.get("arguments", [])
            if isinstance(a, dict) and "name" in a
        }
        for key in (n.get("args") or {}):
            if key not in declared:
                errors.append(f"node '{nid}' sets unknown argument '{key}'")
        node_sockets[nid] = _expand_program_sockets(prog, _node_arg_values(prog, n))

    edge_uses: dict[tuple[str, str, str], list[str]] = {}
    seen_edge_ids: set[str] = set()
    for i, e in enumerate(edges):
        if not isinstance(e, dict):
            errors.append("each edge must be an object")
            continue
        eid = e.get("id") if isinstance(e.get("id"), str) and e.get("id") else f"#{i}"
        if eid in seen_edge_ids:
            errors.append(f"duplicate edge id '{eid}'")
        seen_edge_ids.add(eid)
        frm, to = e.get("from"), e.get("to")
        if not (isinstance(frm, dict) and isinstance(to, dict)):
            errors.append(f"edge '{eid}' needs 'from' and 'to' objects")
            continue
        fn, fs, tn, ts = frm.get("node"), frm.get("socket"), to.get("node"), to.get("socket")
        if fn not in node_by_id or tn not in node_by_id:
            errors.append(f"edge '{eid}' references a node not in this graph")
            continue
        if node_progs.get(fn) is None or node_progs.get(tn) is None:
            continue  # already reported above
        fsock = next((s for s in node_sockets.get(fn, [])
                      if isinstance(s, dict) and s.get("name") == fs), None)
        tsock = next((s for s in node_sockets.get(tn, [])
                      if isinstance(s, dict) and s.get("name") == ts), None)
        if fsock is None:
            errors.append(f"edge '{eid}': node '{fn}' has no socket '{fs}'")
            continue
        if tsock is None:
            errors.append(f"edge '{eid}': node '{tn}' has no socket '{ts}'")
            continue
        if fsock.get("direction") != "output":
            errors.append(f"edge '{eid}': '{fn}.{fs}' is not an output socket")
        if tsock.get("direction") != "input":
            errors.append(f"edge '{eid}': '{tn}.{ts}' is not an input socket")
        if fsock.get("transport") != tsock.get("transport"):
            errors.append(
                f"edge '{eid}': transport mismatch "
                f"({fsock.get('transport')} → {tsock.get('transport')})"
            )
        elif _socket_pattern(fsock) != _socket_pattern(tsock):
            errors.append(
                f"edge '{eid}': pattern mismatch "
                f"({_socket_pattern(fsock)} → {_socket_pattern(tsock)})"
            )
        elif _socket_pattern(fsock) == "stream" and fsock.get("stream_kind") != tsock.get("stream_kind"):
            errors.append(
                f"edge '{eid}': stream_kind mismatch "
                f"({fsock.get('stream_kind')} → {tsock.get('stream_kind')}) "
                "— irregular/regular/framed don't mix"
            )
        edge_uses.setdefault((fn, fs, "out"), []).append(eid)
        edge_uses.setdefault((tn, ts, "in"), []).append(eid)

    for (nid, sname, side), eids in edge_uses.items():
        if len(eids) <= 1:
            continue
        if side == "in":
            errors.append(
                f"input socket '{nid}.{sname}' has more than one incoming "
                f"edge: {eids}"
            )
            continue
        sock = next(
            (s for s in node_sockets.get(nid, [])
             if isinstance(s, dict) and s.get("name") == sname),
            None,
        )
        if sock is not None and _socket_pattern(sock) == "stream":
            errors.append(
                f"output stream socket '{nid}.{sname}' has more than one "
                f"connected edge: {eids} (stream sockets are 1:1 — insert a "
                "fan-out node instead)"
            )
        # request-reply/http outputs: any number of edges is fine (§2/§6).

    return errors


def graph_warnings(graph, registry: dict) -> list[dict]:
    """Persistent (not one-shot) warnings for the GRAPH tab: every
    stream-pattern, non-http output socket with no connected edge, unless
    its manifest entry sets "capped": true (§6-§7)."""
    if not isinstance(graph, dict):
        return []
    connected_outputs: set[tuple[str, str]] = set()
    for e in graph.get("edges", []):
        frm = e.get("from") if isinstance(e, dict) else None
        if isinstance(frm, dict):
            connected_outputs.add((frm.get("node"), frm.get("socket")))

    warnings: list[dict] = []
    for n in graph.get("nodes", []):
        if not isinstance(n, dict):
            continue
        nid = n.get("id")
        _, prog = _resolve_node(n, registry)
        if prog is None:
            continue
        for s in _expand_program_sockets(prog, _node_arg_values(prog, n)):
            if not isinstance(s, dict) or s.get("direction") != "output":
                continue
            if s.get("transport") == "http" or _socket_pattern(s) != "stream":
                continue
            if s.get("capped") or (nid, s.get("name")) in connected_outputs:
                continue
            warnings.append({
                "node": nid,
                "socket": s.get("name"),
                "message": (
                    "no consumer connected — may block or crash when nothing "
                    "connects unless the producer is capped"
                ),
            })
    return warnings


def allocate_graph_addresses(graph: dict, registry: dict,
                             job: dict | None = None) -> dict[tuple[str, str], dict]:
    """One address per (node id, output socket name): a TCP port (explicit,
    or pool-allocated) for tcp/http sockets, a unix path (explicit, or
    generated) for unix sockets. Input sockets never get their own address —
    they always resolve to whatever output they're wired to (see
    render_graph_conf). Raises RuntimeError on an explicit port collision or
    pool exhaustion."""
    resolved: dict[tuple[str, str], dict] = {}
    used_ports: set[int] = set()
    for n in graph.get("nodes", []):
        if not isinstance(n, dict):
            continue
        _, prog = _resolve_node(n, registry)
        if prog is None:
            continue
        for s in _expand_program_sockets(prog, _node_arg_values(prog, n)):
            if not (isinstance(s, dict) and s.get("direction") == "output"):
                continue
            if s.get("transport") in ("tcp", "http"):
                port = s.get("port")
                if isinstance(port, int):
                    if port in used_ports:
                        raise RuntimeError(
                            f"socket port {port} ({n['id']}.{s.get('name')}) "
                            "collides with another node's explicit port"
                        )
                else:
                    port = _allocate_one_port(used_ports, job, f"{n['id']}.{s.get('name')}")
                used_ports.add(port)
                resolved[(n["id"], s["name"])] = {"kind": "port", "value": port}
            else:  # unix
                path = s.get("path") or f"/tmp/eventide-{n['id']}-{s.get('name')}.sock"
                resolved[(n["id"], s["name"])] = {"kind": "path", "value": path}
    return resolved


def render_node_command(text: str, module_name: str, node_args: dict,
                        sock_values: dict, recordings_subdir: str | None) -> str:
    """render_placeholders(), but keyed to one graph node's resolved
    argument values and socket addresses instead of a whole manifest —
    {module_dir}/{venv_...} still resolve to the *module's* shared install
    location (one venv per module, shared by every node instance)."""
    venv_dir = Path(cfg["packages_dir"]) / module_name / _VENV_DIR_NAME
    out = text
    for key, value in node_args.items():
        out = out.replace("{arg:%s}" % key, str(value))
    for key, value in sock_values.items():
        out = out.replace("{socket:%s}" % key, str(value if value is not None else ""))
    out = out.replace("{install_dir}", MODULE_INSTALL_DIR)
    out = out.replace("{config_dir}", MODULE_CONFIG_DIR)
    out = out.replace("{module_dir}", str(Path(cfg["packages_dir"]) / module_name))
    out = out.replace("{recordings_dir}", current_recordings_dir())
    if recordings_subdir:
        out = out.replace(
            "{recordings_subdir}", str(Path(current_recordings_dir()) / recordings_subdir)
        )
    out = out.replace("{venv_dir}", str(venv_dir))
    out = out.replace("{venv_python}", str(venv_dir / "bin" / "python3"))
    return out


def _node_recordings_subdir(manifest: dict, prog: dict, node_id: str) -> str | None:
    """A node records to <recordings_subdir>-<node id> — but only when its
    program actually references the recordings dir (§9)."""
    sub = manifest.get("recordings_subdir")
    if not isinstance(sub, str) or not sub:
        return None
    text = str(prog.get("command", "")) + str(prog.get("directory", ""))
    if "{recordings_subdir}" not in text:
        return None
    return f"{sub}-{node_id}"


def _input_socket_address(node_id: str, socket_name: str, graph: dict,
                          resolved: dict[tuple[str, str], dict]) -> dict | None:
    """The address feeding one node's input socket: whatever output socket
    its (at most one, per validate_graph) incoming edge is wired to."""
    for e in graph.get("edges", []):
        to = e.get("to") if isinstance(e, dict) else None
        if isinstance(to, dict) and to.get("node") == node_id and to.get("socket") == socket_name:
            frm = e.get("from")
            if isinstance(frm, dict):
                return resolved.get((frm.get("node"), frm.get("socket")))
    return None


def graph_conf_path() -> Path:
    return Path(cfg["supervisor_conf_d"]) / "eventide-graph.conf"


def render_graph_conf(graph: dict, registry: dict,
                      resolved: dict[tuple[str, str], dict]) -> str:
    lines = [
        "; Generated by the eventide graph compiler from the submitted graph.",
        "; Do not edit by hand — regenerated in full on every submit.",
    ]
    for n in graph.get("nodes", []):
        if not isinstance(n, dict):
            continue
        nid = n["id"]
        manifest, prog = _resolve_node(n, registry)
        if prog is None:
            continue
        module_name = n["module"]
        node_args = _node_arg_values(prog, n)

        # Group expanded (count_arg) sockets back under their base name: a
        # normal socket's {socket:<name>} is a single value exactly as
        # before, but a count_arg socket's is every one of its N resolved
        # addresses joined with commas (see stream_fanout.py, which expects
        # exactly that for its own --out flag).
        sock_values: dict[str, object] = {}
        groups: dict[str, list] = {}
        joined_bases: set[str] = set()
        for s in _expand_program_sockets(prog, node_args):
            if not isinstance(s, dict) or not s.get("name"):
                continue
            base = s.get("_base_name", s["name"])
            if "_base_name" in s:
                joined_bases.add(base)
            if s.get("direction") == "output":
                addr = resolved.get((nid, s["name"]))
            else:
                addr = _input_socket_address(nid, s["name"], graph, resolved)
            groups.setdefault(base, []).append(addr["value"] if addr else None)
        for base, values in groups.items():
            if base in joined_bases:
                sock_values[base] = ",".join(
                    str(v) for v in values if v is not None
                )
            else:
                sock_values[base] = values[0]

        recordings_subdir = _node_recordings_subdir(manifest, prog, nid)

        lines.append(f"[program:{nid}]")
        lines.append("command=" + render_node_command(
            prog["command"], module_name, node_args, sock_values, recordings_subdir
        ))
        lines.append("directory=" + render_node_command(
            prog.get("directory", "{install_dir}"), module_name, node_args,
            sock_values, recordings_subdir,
        ))
        lines.append(f"autostart={'true' if prog.get('autostart', True) else 'false'}")
        lines.append(f"autorestart={'true' if prog.get('autorestart', True) else 'false'}")
        lines.append(f"startretries={int(prog.get('startretries', 10000))}")
        lines.append(f"priority={int(prog.get('priority', 10))}")
        user = prog.get("user", "root")
        lines.append(f"user={user}")
        home = "/root" if user == "root" else f"/home/{user}"
        lines.append(f'environment=HOME="{home}"')
        lines.append("stdout_logfile=/var/log/supervisor/%(program_name)s.log")
        lines.append("")
    return "\n".join(lines)


def _mkdir_graph_paths(graph: dict, registry: dict,
                       resolved: dict[tuple[str, str], dict]) -> None:
    """Create every node's recordings subdir (if its program uses one) and
    the parent directory of every resolved unix socket path."""
    for n in graph.get("nodes", []):
        if not isinstance(n, dict):
            continue
        manifest, prog = _resolve_node(n, registry)
        if prog is None:
            continue
        sub = _node_recordings_subdir(manifest, prog, n["id"])
        if sub:
            (Path(current_recordings_dir()) / sub).mkdir(parents=True, exist_ok=True)
    for addr in resolved.values():
        if addr.get("kind") == "path":
            Path(addr["value"]).parent.mkdir(parents=True, exist_ok=True)


def _apply_graph(graph: dict, registry: dict) -> tuple[dict[tuple[str, str], dict], list[dict]]:
    """Validate, allocate, render and apply one graph as the running system:
    full stop-and-regenerate, per docs/GRAPH_SUPERVISOR_PLAN.md §2/§7.
    Raises RuntimeError (pool exhaustion / port collision) or OSError
    (conf write / supervisorctl) on failure — callers decide the HTTP
    status. On success, returns (resolved addresses, persistent warnings);
    the caller is responsible for persisting the new graph state.
    """
    resolved = allocate_graph_addresses(graph, registry)
    conf_text = render_graph_conf(graph, registry, resolved)
    _mkdir_graph_paths(graph, registry, resolved)
    _atomic_write(graph_conf_path(), conf_text)
    err = _supervisor_reread_update()
    if err:
        raise OSError(err)
    return resolved, graph_warnings(graph, registry)


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


def _clone_repo(repo_url: str, ref: str | None, dest: Path, job: dict,
                commit: str | None = None) -> None:
    """Clone repo_url at ref into dest. When `commit` is given (graph
    import re-installing a module at the exact commit it was exported
    with — docs/GRAPH_SUPERVISOR_PLAN.md §12), this does a full (non-shallow)
    clone and checks it out: a shallow --depth 1 clone only has `ref`'s
    current tip, which may no longer be that commit if the branch has since
    moved."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    base = ["git", "clone"] if commit else ["git", "clone", "--depth", "1"]
    if ref:
        base += ["--branch", ref]
    timeout = 300 if commit else 180
    try:
        _run_logged_argv(job, [*base, repo_url, str(dest)], timeout=timeout)
    except RuntimeError as exc:
        ssh_url = _https_to_ssh(repo_url)
        if not ssh_url:
            raise
        _job_log(job, f"HTTPS clone failed ({exc}); retrying over SSH: {ssh_url}")
        shutil.rmtree(dest, ignore_errors=True)
        _run_logged_argv(job, [*base, ssh_url, str(dest)], timeout=timeout)
    if commit:
        _run_logged_argv(job, ["git", "-C", str(dest), "checkout", commit], timeout=60)


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


# ── Install job ───────────────────────────────────────────────────────────────
# Installing a module only clones/builds/stages it — deps, build, artifacts —
# and makes its programs[] available as node types in the GRAPH tab's
# palette. Nothing is rendered into supervisor conf or started here any
# more: that only happens once a graph referencing those programs is
# submitted (see the graph compiler above and /api/graph/submit below).

def _run_install_job(job: dict) -> None:
    staging: Path | None = None
    module_dir: Path | None = None
    copied_artifacts: list[str] = []
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
            _clone_repo(job["repo_url"], job.get("ref"), staging, job, commit=job.get("commit"))
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
        # recordings_subdir's actual directories are created per graph node
        # (<subdir>-<node id>), at graph-submit time (_mkdir_graph_paths) —
        # not here. Nothing runs, and nothing records, until a node using
        # this module's program is placed and the graph is submitted.

        # ── Register ────────────────────────────────────────────────────────
        # No supervisor config is written here any more (§7): a module only
        # makes its programs[] available in the GRAPH tab's palette. Nothing
        # runs until they're placed as nodes and a graph is submitted.
        _job_status(job, "configuring")
        registry = load_registry()
        registry.setdefault("modules", {})[name] = {
            "manifest": manifest,
            "repo_url": job["repo_url"],
            "ref": job.get("ref"),
            "commit": commit,
            "installed_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": copied_artifacts,
        }
        save_registry(registry)
        registered = True
        _job_status(job, "done")
        _job_log(job, f"module '{name}' installed successfully — its programs "
                       "are now available in the GRAPH tab's palette")

    except Exception as exc:  # noqa: BLE001 — any failure must roll back cleanly
        job["error"] = str(exc)
        _job_log(job, f"ERROR: {exc}")
        _job_log(job, "rolling back partial install")
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
    """Remove an installed module. Per docs/GRAPH_SUPERVISOR_PLAN.md §7,
    uninstalling a module in use by the graph is *allowed*, not rejected:
    any node using one of its programs is dropped from both the draft and
    the active graph, and — only for the active graph — what's left is
    re-validated and, if that introduces no new hard error, applied (full
    stop-and-regenerate, which is also what actually stops the removed
    nodes' supervisor programs). If it does introduce an error, the
    previous active graph is left running untouched and the error is
    reported on the returned entry (graph_error) — the module is removed
    either way. The caller (the dashboard) is expected to have already
    warned the user which nodes this will remove before calling."""
    registry = load_registry()
    entry = registry.get("modules", {}).get(name)
    if entry is None:
        return None

    del registry["modules"][name]
    save_registry(registry)

    for artifact in entry.get("artifacts", []):
        Path(artifact).unlink(missing_ok=True)
    shutil.rmtree(Path(cfg["packages_dir"]) / name, ignore_errors=True)

    def _trim(graph):
        if not isinstance(graph, dict):
            return graph, set()
        removed = {
            n["id"] for n in graph.get("nodes", [])
            if isinstance(n, dict) and n.get("module") == name and isinstance(n.get("id"), str)
        }
        if not removed:
            return graph, removed
        nodes = [n for n in graph.get("nodes", []) if n.get("id") not in removed]
        edges = [
            e for e in graph.get("edges", [])
            if isinstance(e, dict)
            and (e.get("from") or {}).get("node") not in removed
            and (e.get("to") or {}).get("node") not in removed
        ]
        return {**graph, "nodes": nodes, "edges": edges}, removed

    state = load_graph_state()
    new_draft, draft_removed = _trim(state.get("draft"))
    new_active, active_removed = _trim(state.get("active"))

    entry["graph_removed_nodes"] = sorted(draft_removed | active_removed)
    entry["graph_regenerated"] = False
    entry["graph_error"] = None

    state["draft"] = new_draft
    if active_removed:
        errors = validate_graph(new_active, registry)
        if errors:
            entry["graph_error"] = "; ".join(errors)
            # Leave state["active"] (and the running supervisor conf) exactly
            # as it was — regenerating a graph we know is now invalid would
            # tear down the working system for no benefit.
        else:
            try:
                resolved, _warnings = _apply_graph(new_active, registry)
                state["active"] = new_active
                state["resolved"] = {
                    _resolved_key(nid, sname): addr
                    for (nid, sname), addr in resolved.items()
                }
                entry["graph_regenerated"] = True
            except (RuntimeError, OSError) as exc:
                entry["graph_error"] = str(exc)
    save_graph_state(state)
    return entry


# ── Module API ────────────────────────────────────────────────────────────────

@app.route("/api/modules")
def api_modules_list():
    """Installed modules — the palette the GRAPH tab's node types are built
    from (docs/GRAPH_SUPERVISOR_PLAN.md §5). A module has no live
    status/instances of its own any more: `arguments`/`sockets`/`ui` are
    per-program templates, not running things — see /api/graph/status for
    what's actually executing. `active_node_count` is a convenience count of
    how many nodes in the *active* graph currently use each program, mostly
    so the MODULES tab can warn before an uninstall (§7)."""
    registry = load_registry()
    state = load_graph_state()
    active_nodes = (state.get("active") or {}).get("nodes") or []
    counts: dict[tuple[str, str], int] = {}
    for n in active_nodes:
        if isinstance(n, dict):
            key = (n.get("module"), n.get("program"))
            counts[key] = counts.get(key, 0) + 1

    modules = []
    for name, entry in sorted(registry.get("modules", {}).items()):
        m = entry["manifest"]
        modules.append({
            "name": name,
            "version": m.get("version"),
            "description": m.get("description"),
            "author": m.get("author"),
            "repo_url": entry.get("repo_url"),
            "installed_at": entry.get("installed_at"),
            "recordings_subdir": m.get("recordings_subdir"),
            "programs": [
                {
                    "name": p["name"],
                    "command": p.get("command"),
                    "arguments": p.get("arguments", []),
                    "sockets": p.get("sockets", []),
                    "ui": p.get("ui", []),
                    "autostart": p.get("autostart", True),
                    "autorestart": p.get("autorestart", True),
                    "startretries": int(p.get("startretries", 10000)),
                    "priority": int(p.get("priority", 10)),
                    "active_node_count": counts.get((name, p.get("name")), 0),
                }
                for p in m.get("programs", [])
            ],
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


def _new_install_job(*, source_type: str, repo_url: str, ref: str | None = None,
                     zip_path: str | None = None, commit: str | None = None) -> dict:
    """Register a new install job (does not start it — see _launch_install_job).
    The caller must hold install_lock; the job wrapper releases it. `commit`
    is only ever set by the graph import flow (§12), pinning the clone to an
    exact commit rather than just the tip of `ref` — see _clone_repo."""
    job = {
        "id": uuid.uuid4().hex[:12],
        "status": "pending",
        "source_type": source_type,
        "repo_url": repo_url,
        "ref": ref,
        "commit": commit,
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
    """Uninstall a module. Per docs/GRAPH_SUPERVISOR_PLAN.md §7 this always
    proceeds even if the module is in use by the graph — see
    _uninstall_module for the removed/regenerated/error fields below, which
    the dashboard should show the user (it should already have confirmed
    the removal with them before calling this, listing the affected nodes
    from GET /api/graph)."""
    with install_lock:  # wait for any running install to finish
        entry = _uninstall_module(name)
    if entry is None:
        return jsonify({"error": f"module not installed: {name}"}), 404
    return jsonify({
        "ok": True,
        "removed": name,
        "graph_removed_nodes": entry.get("graph_removed_nodes", []),
        "graph_regenerated": entry.get("graph_regenerated", False),
        "graph_error": entry.get("graph_error"),
    })


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


# ── Graph API ──────────────────────────────────────────────────────────────────
# The litegraph-driven supervisor (docs/GRAPH_SUPERVISOR_PLAN.md). Per-node
# start/stop/restart and log tailing reuse the existing generic /supervisor/
# XML-RPC proxy unchanged — a graph node is just a supervisor program named
# after its node id, so the same superStart('<id>')-style calls the MODULES
# tab already makes for module programs work here too, with no new
# endpoints needed for that part.

@app.route("/api/graph")
def api_graph_get():
    state = load_graph_state()
    return jsonify({"active": state.get("active"), "draft": state.get("draft")})


@app.route("/api/graph/draft", methods=["PUT"])
def api_graph_draft_put():
    """Autosave the in-progress graph. No validation beyond basic shape —
    the editor is allowed to be mid-edit (dangling wires, etc.); real
    validation happens at submit time."""
    data = request.get_json(silent=True)
    if not (
        isinstance(data, dict)
        and isinstance(data.get("nodes"), list)
        and isinstance(data.get("edges"), list)
    ):
        return jsonify({
            "error": "graph must be an object with 'nodes' and 'edges' lists"
        }), 400
    state = load_graph_state()
    state["draft"] = data
    save_graph_state(state)
    return jsonify({"ok": True})


@app.route("/api/graph/submit", methods=["POST"])
def api_graph_submit():
    """Validate and apply a graph as the running system (full
    stop-and-regenerate — docs/GRAPH_SUPERVISOR_PLAN.md §2/§7). Submits the
    request body if one is given, else the current draft."""
    graph = request.get_json(silent=True)
    if graph is None:
        state = load_graph_state()
        graph = state.get("draft")
        if graph is None:
            return jsonify({"error": "no draft graph to submit"}), 400

    registry = load_registry()
    errors = validate_graph(graph, registry)
    if errors:
        return jsonify({"error": "; ".join(errors), "errors": errors}), 400

    with install_lock:  # serialize with module installs/uninstalls
        try:
            resolved, warnings = _apply_graph(graph, registry)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 400
        except OSError as exc:
            return jsonify({"error": str(exc)}), 500

        state = load_graph_state()
        state["active"] = graph
        state["draft"] = graph
        state["resolved"] = {
            _resolved_key(nid, sname): addr for (nid, sname), addr in resolved.items()
        }
        save_graph_state(state)

    return jsonify({"ok": True, "warnings": warnings})


@app.route("/api/graph/status")
def api_graph_status():
    """Live per-node supervisor status for the active graph, plus the
    persistent unconnected-stream-socket warnings (§7)."""
    state = load_graph_state()
    graph = state.get("active") or {"nodes": [], "edges": []}
    statuses = supervisor_statuses()
    nodes = []
    for n in graph.get("nodes", []):
        if not (isinstance(n, dict) and isinstance(n.get("id"), str)):
            continue
        st = statuses.get(n["id"], {})
        nodes.append({
            "id": n["id"],
            "status": st.get("status", "UNKNOWN"),
            "status_detail": st.get("description", ""),
        })
    return jsonify({
        "nodes": nodes,
        "warnings": graph_warnings(graph, load_registry()),
    })


# ── Graph export / import ─────────────────────────────────────────────────────
# Full-system duplication (docs/GRAPH_SUPERVISOR_PLAN.md §12): export bundles
# a graph together with enough per-module info to reinstall it identically
# elsewhere; import re-clones those modules (pinned to the exact commit that
# was running, not just whatever `ref` currently points to) and loads the
# result as the *draft* on the destination — never the active graph — so
# ports/paths get allocated fresh and the operator reviews before SUBMIT.
#
# Only git-sourced modules (a real repo_url) can be reinstalled this way. A
# module installed from an uploaded zip, or one of the base platform's own
# --install-local modules (eventide-core, installed straight from this repo's
# checkout by install.sh — see _install_local_module), has no repo_url worth
# re-cloning. Those go in the bundle's "preinstalled_modules" list instead of
# "modules": the export still succeeds (blocking it outright would make
# export useless for the overwhelmingly common case of a graph using
# eventide-core), but import refuses up front if any of them aren't already
# present on the destination, rather than silently producing a graph with
# missing node types.

_GIT_REPO_URL_RE = re.compile(r"^(https?://|git@|git://).+")


@app.route("/api/graph/export")
def api_graph_export():
    source = request.args.get("source", "active")
    if source not in ("active", "draft"):
        return jsonify({"error": "source must be 'active' or 'draft'"}), 400
    state = load_graph_state()
    graph = state.get(source)
    if not graph or not graph.get("nodes"):
        return jsonify({"error": f"no {source} graph to export"}), 400

    registry = load_registry()
    referenced = sorted({
        n.get("module") for n in graph.get("nodes", [])
        if isinstance(n, dict) and n.get("module")
    })
    missing = [m for m in referenced if m not in registry.get("modules", {})]
    if missing:
        return jsonify({
            "error": "graph references modules that are not installed: " + ", ".join(missing)
        }), 400

    modules = []
    preinstalled = []
    for name in referenced:
        entry = registry["modules"][name]
        repo_url = entry.get("repo_url") or ""
        if _GIT_REPO_URL_RE.match(repo_url):
            modules.append({
                "name": name,
                "repo_url": repo_url,
                "ref": entry.get("ref"),
                "commit": entry.get("commit"),
            })
        else:
            preinstalled.append(name)

    return jsonify({
        "eventide_export_version": 1,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "graph": graph,
        "modules": modules,
        "preinstalled_modules": preinstalled,
    })


import_jobs: dict[str, dict] = {}
import_jobs_lock = threading.Lock()


def _new_import_job() -> dict:
    job = {
        "id": uuid.uuid4().hex[:12],
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "error": None,
        "warnings": [],
        "log": [],
        "modules_installed": [],
        "modules_skipped": [],
    }
    with import_jobs_lock:
        import_jobs[job["id"]] = job
    return job


def _import_job_log(job: dict, line: str) -> None:
    with import_jobs_lock:
        job["log"].append(str(line))


def _import_job_status(job: dict, status: str) -> None:
    with import_jobs_lock:
        job["status"] = status


def _import_job_snapshot(job: dict) -> dict:
    with import_jobs_lock:
        return {
            "id": job["id"], "status": job["status"],
            "created_at": job["created_at"], "finished_at": job["finished_at"],
            "log": list(job["log"]), "error": job["error"],
            "warnings": list(job["warnings"]),
            "modules_installed": list(job["modules_installed"]),
            "modules_skipped": list(job["modules_skipped"]),
        }


def _run_import_job(job: dict, bundle: dict) -> None:
    try:
        graph = bundle.get("graph")
        if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list):
            raise RuntimeError("bundle 'graph' is missing or malformed")
        modules = bundle.get("modules") or []
        preinstalled = bundle.get("preinstalled_modules") or []
        if not isinstance(modules, list) or not isinstance(preinstalled, list):
            raise RuntimeError("bundle 'modules'/'preinstalled_modules' must be lists")

        registry = load_registry()
        missing_pre = [m for m in preinstalled if m not in registry.get("modules", {})]
        if missing_pre:
            raise RuntimeError(
                "this device is missing module(s) the export assumed were already "
                "installed (base/zip-sourced modules aren't reinstalled by import): "
                + ", ".join(missing_pre)
            )

        _import_job_status(job, "installing")
        for m in modules:
            if not isinstance(m, dict) or not m.get("name"):
                raise RuntimeError(f"malformed module entry in bundle: {m!r}")
            name = m["name"]
            registry = load_registry()
            if name in registry.get("modules", {}):
                job["modules_skipped"].append(name)
                _import_job_log(job, f"module '{name}' already installed — skipping")
                continue
            repo_url = m.get("repo_url") or ""
            if not _GIT_REPO_URL_RE.match(repo_url):
                raise RuntimeError(f"module '{name}' has no usable repo_url in the bundle")
            _import_job_log(job, f"installing module '{name}' from {repo_url}"
                                  + (f" @ {m['commit']}" if m.get("commit") else ""))
            with install_lock:
                sub_job = _new_install_job(
                    source_type="git", repo_url=repo_url,
                    ref=m.get("ref"), commit=m.get("commit"),
                )
                _run_install_job(sub_job)
            job["log"].extend(sub_job["log"])
            if sub_job["status"] != "done":
                raise RuntimeError(f"failed to install module '{name}': {sub_job.get('error')}")
            job["modules_installed"].append(name)

        _import_job_status(job, "validating")
        registry = load_registry()
        errors = validate_graph(graph, registry)
        if errors:
            job["warnings"].append(
                "the imported graph has validation problem(s) — fix them in the GRAPH "
                "tab before submitting: " + "; ".join(errors)
            )

        state = load_graph_state()
        state["draft"] = graph
        save_graph_state(state)
        _import_job_status(job, "done")
        _import_job_log(job, "import complete — loaded as the draft graph; "
                              "review it in the GRAPH tab, then SUBMIT")
    except Exception as exc:  # noqa: BLE001 — reported on the job, not raised
        job["error"] = str(exc)
        _import_job_log(job, f"ERROR: {exc}")
        _import_job_status(job, "failed")
    finally:
        job["finished_at"] = datetime.now(timezone.utc).isoformat()


@app.route("/api/graph/import", methods=["POST"])
def api_graph_import():
    bundle = request.get_json(silent=True)
    if not isinstance(bundle, dict) or not isinstance(bundle.get("graph"), dict):
        return jsonify({"error": "invalid export bundle: missing 'graph'"}), 400
    if not isinstance(bundle.get("modules", []), list) or \
       not isinstance(bundle.get("preinstalled_modules", []), list):
        return jsonify({
            "error": "invalid export bundle: 'modules'/'preinstalled_modules' must be lists"
        }), 400

    job = _new_import_job()
    threading.Thread(target=_run_import_job, args=(job, bundle), daemon=True).start()
    return jsonify({"job_id": job["id"], "status": "pending"}), 202


@app.route("/api/graph/import/jobs/<job_id>")
def api_graph_import_job(job_id):
    job = import_jobs.get(job_id)
    if job is None:
        return jsonify({"error": f"unknown job id: {job_id}"}), 404
    return jsonify(_import_job_snapshot(job))


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
    parser.add_argument("--graph-file",            default="/usr/local/eventide/graph.json",
                        help="Path to the persisted graph state (active/draft/resolved — "
                             "see docs/GRAPH_SUPERVISOR_PLAN.md §7)")
    parser.add_argument("--packages-dir",          default="/usr/local/eventide/packages",
                        help="Directory where module repositories are cloned")
    parser.add_argument("--supervisor-conf-d",     default="/etc/supervisor/conf.d",
                        help="Directory where the generated graph supervisor config is written")
    parser.add_argument("--port-pool",             default="8100-8199",
                        help="Range of TCP ports allocated to graph node sockets that "
                             "don't request an explicit port (start-end)")
    parser.add_argument("--host",                  default="0.0.0.0")
    parser.add_argument("--port",                  type=int, default=5000)
    parser.add_argument("--html-file",             default=str(Path(__file__).resolve().with_name("dashboard.html")),
                        help="Path to the dashboard HTML file served at / "
                             "(default: dashboard.html alongside this script)")
    parser.add_argument("--kiosk-html-file",       default=str(Path(__file__).resolve().with_name("kiosk.html")),
                        help="Path to the touchscreen kiosk HTML file served at /kiosk "
                             "(default: kiosk.html alongside this script)")
    parser.add_argument("--vendor-dir",            default=str(Path(__file__).resolve().with_name("vendor")),
                        help="Directory of vendored dashboard JS/CSS (leaflet, gridstack, "
                             "litegraph.js), served at /vendor/... (default: vendor/ alongside this script)")
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
    print(f"[backend]  Graph state:       {args.graph_file}")
    print(f"[backend]  Packages dir:      {args.packages_dir}")
    print(f"[backend]  Supervisor conf.d: {args.supervisor_conf_d}")
    print(f"[backend]  Node port pool:    {args.port_pool}")
    print(f"[backend]  CORS origin:       {_ALLOWED_ORIGIN}")
    print(f"[backend]  Dashboard HTML:    {args.html_file}  (served at /)")
    print(f"[backend]  Kiosk HTML:        {args.kiosk_html_file}  (served at /kiosk)")
    print(f"[backend]  Vendor JS/CSS:     {args.vendor_dir}  (served at /vendor/...)")
    print(f"[backend]  Tile proxy:        /tiles/<z>/<x>/<y>.png  →  {_OSM_BASE}")
    print(f"[backend]  API at:            http://{args.host}:{args.port}")

    # Auto-start live EVK viewfinder
    proc, err = start_viewfinder("live", vf_configs["live"])
    if err:
        print(f"[backend] WARNING: could not auto-start live viewfinder: {err}")
    else:
        viewfinders["live"] = proc
        print("[backend] Auto-started live viewfinder.")

    threading.Thread(target=retention_sweep_loop, daemon=True).start()

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()