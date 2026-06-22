"""
EVK4 Dashboard
==============
Serves the standalone dashboard.html and exposes API endpoints for
controlling viewfinder / replay processes and listing recordings.

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
import os
import subprocess
import threading
from pathlib import Path

import requests as _http

from flask import (
    Flask,
    abort,
    jsonify,
    request,
    send_file,
    send_from_directory,
)

app = Flask(__name__, static_folder=None)

# ── CORS — allow the frontend server (any origin) to call the API ─────────────
# The dashboard HTML is now served from a separate host, so the browser will
# make cross-origin requests to this server.  We allow all origins here because
# the frontend host/port is not known at deploy time.  If you want to restrict
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
    parser.add_argument("--host",                  default="0.0.0.0")
    parser.add_argument("--port",                  type=int, default=5000)
    args = parser.parse_args()

    cfg.update(vars(args))

    # Allow CLI override of the supervisor URL module-level variable.
    _SUPERVISOR_RPC_URL = args.supervisor_rpc_url

    print(f"[backend]  Recordings dir:    {args.recordings_dir}")
    print(f"[backend]  Viewfinder binary: {args.viewfinder_bin}")
    print(f"[backend]  Live VF port:      {args.live_port}  (nginx → /stream/evk/)")
    print(f"[backend]  Supervisor RPC:    {_SUPERVISOR_RPC_URL}  (proxied at /supervisor/)")
    print(f"[backend]  CORS origin:       {_ALLOWED_ORIGIN}")
    print(f"[backend]  API at:            http://{args.host}:{args.port}")
    print(f"[backend]  NOTE: HTML is now served by frontend_server.py, not this process.")

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