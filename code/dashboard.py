"""
EVK4 Dashboard
==============
Serves the standalone dashboard.html and exposes API endpoints for
controlling viewfinder / replay processes and listing recordings.

This process CAN serve the full frontend itself (HTML at / plus an OSM
tile proxy at /tiles/), which is handy when talking to the device
directly (field laptop, local network).  For data-constrained links the
decoupled frontend server (frontend.py) is still preferred: it keeps
HTML/tile bandwidth off the link — see vehicle.nginx.

Streams are proxied through nginx — this server only handles control
and file APIs; the MJPEG streams themselves are served by nginx at:

  /stream/evk/        → 127.0.0.1:8081
  /stream/picam/      → 127.0.0.1:8082/stream
  /stream/ircam/      → 127.0.0.1:8083
  /playback/

Usage:
    pip install flask
    python dashboard.py \\
        --recordings-dir /tmp/evk4_raw \\
        --viewfinder-bin ./target/release/viewfinder \\
        --replay-bin     ./target/release/replay

Then open http://localhost:5000  (or via nginx at http://<host>/)
"""

import argparse
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests as _http

from flask import (
    Flask,
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
# dashboard.py can serve the whole UI itself for direct/standalone access.
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
            "Pass --html-file or place it alongside dashboard.py."
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

# ── Recordings ────────────────────────────────────────────────────────────────
# Each camera has its own sub-directory under recordings_dir:
#   <recordings_dir>/evk/
#   <recordings_dir>/picam/
#   <recordings_dir>/ircam/
# Falls back to the root dir for backwards-compatibility (evk only).

def _recordings_dir(cam: str) -> Path:
    base = Path(cfg["recordings_dir"])
    sub  = base / cam
    return sub if sub.exists() else base


@app.route("/api/recordings")
def list_recordings_legacy():
    """Legacy endpoint — returns EVK recordings from the root dir."""
    return _list_recordings_for("evk")


@app.route("/api/recordings/<cam>")
def list_recordings_cam(cam):
    if cam not in ("evk", "picam", "ircam", "telemetry"):
        abort(400)
    return _list_recordings_for(cam)


RECORDING_EXTENSIONS = ("*.raw", "*.mp4", "*.h264", "*.jsonl")

def _list_recordings_for(cam: str):
    recordings_dir = _recordings_dir(cam)
    if not recordings_dir.exists():
        return jsonify({"files": []})
    seen = set()
    entries = []
    for pattern in RECORDING_EXTENSIONS:
        for f in recordings_dir.glob(pattern):
            if f.is_file() and f.name not in seen:
                seen.add(f.name)
                entries.append({"name": f.name, "size": f.stat().st_size, "ext": f.suffix.lstrip(".")})
    entries.sort(key=lambda x: x["name"], reverse=True)
    return jsonify({"files": entries})


@app.route("/api/recordings/<cam>/<filename>/download")
def download_recording(cam, filename):
    if cam not in ("evk", "picam", "ircam", "telemetry"):
        abort(400)
    recordings_dir = _recordings_dir(cam)
    filepath = (recordings_dir / filename).resolve()
    if filepath.parent != recordings_dir.resolve():
        abort(400)
    if not filepath.exists():
        abort(404)
    return send_file(filepath, as_attachment=True, download_name=filename)


# Legacy download route (evk, root dir)
@app.route("/api/recordings/<filename>/download")
def download_recording_legacy(filename):
    return download_recording("evk", filename)

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
_KNOWN_PLACEHOLDERS = (
    "install_dir", "config_dir", "module_dir", "recordings_dir",
    "venv_dir", "venv_python",
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


def _job_log(job: dict, line: str) -> None:
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
                if not os.path.isabs(dst):
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
            seen_sock_names.add(sn)
            stype = s.get("type")
            if stype not in ("tcp", "unix"):
                errors.append(f"socket '{sn}' type must be 'tcp' or 'unix'")
            elif stype == "tcp":
                port = s.get("port")
                if not isinstance(port, int) or not (1 <= port <= 65535):
                    errors.append(f"socket '{sn}' needs a valid tcp 'port' (1-65535)")
            else:
                spath = s.get("path")
                if not isinstance(spath, str) or not spath.startswith("/"):
                    errors.append(f"socket '{sn}' needs an absolute unix 'path'")

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
                    continue
                if ph.startswith("arg:"):
                    if ph[4:] not in arg_names:
                        errors.append(
                            f"program '{pn}' uses undeclared argument '{{{ph}}}'"
                        )
                else:
                    errors.append(f"program '{pn}' uses unknown placeholder '{{{ph}}}'")
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
        p["name"]
        for e in installed.values()
        for p in e["manifest"].get("programs", [])
    }
    for p in manifest.get("programs", []):
        if p["name"] in existing_programs:
            errors.append(
                f"program name '{p['name']}' is already used by another installed module"
            )
    existing_ports: dict[int, str] = {}
    for ename, e in installed.items():
        for s in e["manifest"].get("sockets", []):
            if s.get("type") == "tcp":
                existing_ports.setdefault(s["port"], ename)
    for s in manifest.get("sockets", []):
        if s.get("type") == "tcp" and s.get("port") in existing_ports:
            errors.append(
                f"socket port {s['port']} is already used by module "
                f"'{existing_ports[s['port']]}'"
            )
    return errors


# ── Config rendering ──────────────────────────────────────────────────────────

def render_placeholders(text: str, manifest: dict, module_name: str) -> str:
    venv_dir = Path(cfg["packages_dir"]) / module_name / _VENV_DIR_NAME
    out = text
    for arg in manifest.get("arguments", []):
        out = out.replace("{arg:%s}" % arg["name"], str(arg.get("default", "")))
    out = out.replace("{install_dir}", MODULE_INSTALL_DIR)
    out = out.replace("{config_dir}", MODULE_CONFIG_DIR)
    out = out.replace("{module_dir}", str(Path(cfg["packages_dir"]) / module_name))
    out = out.replace("{recordings_dir}", cfg["recordings_dir"])
    out = out.replace("{venv_dir}", str(venv_dir))
    out = out.replace("{venv_python}", str(venv_dir / "bin" / "python3"))
    return out


def render_module_conf(manifest: dict, module_name: str, repo_url: str) -> str:
    lines = [
        f"; Generated by the eventide module manager from {repo_url}",
        "; Do not edit by hand — changes are lost on reinstall.",
    ]
    for p in manifest["programs"]:
        lines.append(f"[program:{p['name']}]")
        lines.append(f"command={render_placeholders(p['command'], manifest, module_name)}")
        lines.append(
            "directory="
            + render_placeholders(p.get("directory", "{install_dir}"), manifest, module_name)
        )
        lines.append(f"autostart={'true' if p.get('autostart', True) else 'false'}")
        lines.append(f"autorestart={'true' if p.get('autorestart', True) else 'false'}")
        lines.append(f"startretries={int(p.get('startretries', 10000))}")
        lines.append(f"priority={int(p.get('priority', 10))}")
        lines.append(f"user={p.get('user', 'root')}")
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


def _verify_programs(job: dict, manifest: dict) -> None:
    if not shutil.which("supervisorctl"):
        job["warnings"].append("supervisorctl not available; skipped program verification")
        return
    wanted = [p["name"] for p in manifest.get("programs", [])]
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
        # ── Clone & validate ──────────────────────────────────────────────
        _job_status(job, "cloning")
        staging = Path(cfg["packages_dir"]) / f".staging-{job['id']}"
        _clone_repo(job["repo_url"], job.get("ref"), staging, job)

        manifest_path = staging / MANIFEST_FILENAME
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
        errors = conflict_errors(manifest, load_registry())
        if errors:
            raise RuntimeError("; ".join(errors))

        module_dir = Path(cfg["packages_dir"]) / name
        if module_dir.exists():
            # Not in the registry (checked above) → leftover from a failed
            # install; safe to replace.
            shutil.rmtree(module_dir)
        shutil.move(str(staging), str(module_dir))
        staging = None
        commit = _git_rev(module_dir)
        _job_log(job, f"cloned {job['repo_url']} ({commit or 'unknown commit'})")

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
            _run_logged_shell(job, cmd, cwd=module_dir, timeout=1800)

        # ── Artifacts & recordings dir ────────────────────────────────────
        _job_status(job, "artifacts")
        for src, dst in inst.get("artifacts", {}).items():
            src_path = module_dir / src
            if not src_path.exists():
                raise RuntimeError(f"artifact not found after build: {src}")
            dst_path = Path(dst)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            copied_artifacts.append(str(dst_path))
            _job_log(job, f"copied {src} → {dst}")
        sub = manifest.get("recordings_subdir")
        if sub:
            rec_dir = Path(cfg["recordings_dir"]) / sub
            rec_dir.mkdir(parents=True, exist_ok=True)
            _job_log(job, f"created recordings dir {rec_dir}")

        # ── Supervisor config ─────────────────────────────────────────────
        _job_status(job, "configuring")
        conf_text = render_module_conf(manifest, name, job["repo_url"])
        conf_path = Path(cfg["supervisor_conf_d"]) / f"module-{name}.conf"
        _atomic_write(conf_path, conf_text)
        _job_log(job, f"wrote {conf_path}")
        _supervisor_apply(job)

        # ── Verify & register ─────────────────────────────────────────────
        _job_status(job, "verifying")
        _verify_programs(job, manifest)

        registry = load_registry()
        registry.setdefault("modules", {})[name] = {
            "manifest": manifest,
            "repo_url": job["repo_url"],
            "ref": job.get("ref"),
            "commit": commit,
            "installed_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": copied_artifacts,
            "conf_file": str(conf_path),
        }
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
    manifest = entry["manifest"]

    if shutil.which("supervisorctl"):
        for p in manifest.get("programs", []):
            try:
                _supervisorctl("stop", p["name"])
            except (subprocess.TimeoutExpired, OSError):
                pass  # best-effort stop

    conf_path = Path(
        entry.get("conf_file")
        or Path(cfg["supervisor_conf_d"]) / f"module-{name}.conf"
    )
    if conf_path.exists():
        conf_path.unlink()
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
        modules.append({
            "name": name,
            "version": m.get("version"),
            "description": m.get("description"),
            "author": m.get("author"),
            "repo_url": entry.get("repo_url"),
            "installed_at": entry.get("installed_at"),
            "arguments": m.get("arguments", []),
            "sockets": m.get("sockets", []),
            "programs": [
                {
                    "name": p["name"],
                    "status": statuses.get(p["name"], {}).get("status", "UNKNOWN"),
                    "status_detail": statuses.get(p["name"], {}).get("description", ""),
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

    job = {
        "id": uuid.uuid4().hex[:12],
        "status": "pending",
        "repo_url": repo_url,
        "ref": ref,
        "module": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "error": None,
        "warnings": [],
        "log": [],
    }
    with module_jobs_lock:
        module_jobs[job["id"]] = job
    threading.Thread(target=_run_install_job_guarded, args=(job,), daemon=True).start()
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
    parser = argparse.ArgumentParser(description="EVK4 Dashboard — Backend API Server")
    parser.add_argument("--recordings-dir",        default="/usr/local/eventide/recordings",
                        help="Root recordings directory (sub-dirs: evk/, picam/, ircam/)")
    parser.add_argument("--viewfinder-bin",        default="./target/release/viewfinder")
    parser.add_argument("--live-events-socket",    default="/tmp/evk4_events.sock")
    parser.add_argument("--live-port",             type=int, default=8081,
                        help="Port the live viewfinder binds to (nginx proxies /stream/evk/)")
    parser.add_argument("--supervisor-rpc-url",    default=_SUPERVISOR_RPC_URL,
                        help="URL of supervisord's XML-RPC endpoint "
                             "(default: http://127.0.0.1:9001/RPC2)")
    parser.add_argument("--modules-registry",      default="/usr/local/eventide/modules.json",
                        help="Path to the installed-modules registry JSON file")
    parser.add_argument("--packages-dir",          default="/usr/local/eventide/packages",
                        help="Directory where module repositories are cloned")
    parser.add_argument("--supervisor-conf-d",     default="/etc/supervisor/conf.d",
                        help="Directory where per-module supervisor configs are written")
    parser.add_argument("--host",                  default="0.0.0.0")
    parser.add_argument("--port",                  type=int, default=5000)
    parser.add_argument("--html-file",             default=str(Path(__file__).resolve().with_name("dashboard.html")),
                        help="Path to the dashboard HTML file served at / "
                             "(default: dashboard.html alongside this script)")
    args = parser.parse_args()

    cfg.update(vars(args))

    # Allow CLI override of the supervisor URL module-level variable.
    _SUPERVISOR_RPC_URL = args.supervisor_rpc_url

    Path(args.packages_dir).mkdir(parents=True, exist_ok=True)

    print(f"[backend]  Recordings dir:    {args.recordings_dir}")
    print(f"[backend]  Viewfinder binary: {args.viewfinder_bin}")
    print(f"[backend]  Live VF port:      {args.live_port}  (nginx → /stream/evk/)")
    print(f"[backend]  Supervisor RPC:    {_SUPERVISOR_RPC_URL}  (proxied at /supervisor/)")
    print(f"[backend]  Modules registry:  {args.modules_registry}")
    print(f"[backend]  Packages dir:      {args.packages_dir}")
    print(f"[backend]  Supervisor conf.d: {args.supervisor_conf_d}")
    print(f"[backend]  CORS origin:       {_ALLOWED_ORIGIN}")
    print(f"[backend]  Dashboard HTML:    {args.html_file}  (served at /)")
    print(f"[backend]  Tile proxy:        /tiles/<z>/<x>/<y>.png  →  {_OSM_BASE}")
    print(f"[backend]  API at:            http://{args.host}:{args.port}")

    # Auto-start live EVK viewfinder
    proc, err = start_viewfinder("live", vf_configs["live"])
    if err:
        print(f"[dashboard] WARNING: could not auto-start live viewfinder: {err}")
    else:
        viewfinders["live"] = proc
        print("[dashboard] Auto-started live viewfinder.")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()