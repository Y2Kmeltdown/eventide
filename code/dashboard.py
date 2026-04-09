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
  /playback/evk/      → 127.0.0.1:8084
  /playback/picam/    → 127.0.0.1:8085
  /playback/ircam/    → 127.0.0.1:8086

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

from flask import (
    Flask,
    abort,
    jsonify,
    request,
    send_file,
    send_from_directory,
)

app = Flask(__name__, static_folder=None)

# ── Global process state ──────────────────────────────────────────────────────

proc_lock = threading.Lock()

# Managed viewfinder processes keyed by mode
viewfinders: dict[str, subprocess.Popen | None] = {"live": None, "replay": None}

# One managed replay process
replay_proc: subprocess.Popen | None = None

# Current configs (persisted so the UI can reflect them)
vf_configs: dict[str, dict] = {
    "live":   {"fps": 50, "quality": 80, "width": 1280, "height": 720},
    "replay": {"fps": 50, "quality": 80, "width": 1280, "height": 720},
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

# ── Static HTML ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the standalone dashboard HTML file."""
    html_path = Path(cfg.get("html_file", "dashboard.html")).resolve()
    if not html_path.exists():
        return "dashboard.html not found. Place it alongside dashboard.py.", 404
    return send_file(html_path, mimetype="text/html")

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
    if cam not in ("evk", "picam", "ircam"):
        abort(400)
    return _list_recordings_for(cam)


def _list_recordings_for(cam: str):
    recordings_dir = _recordings_dir(cam)
    if not recordings_dir.exists():
        return jsonify({"files": []})
    files = sorted(
        [{"name": f.name, "size": f.stat().st_size}
         for f in recordings_dir.glob("*") if f.is_file()],
        key=lambda x: x["name"],
        reverse=True,
    )
    return jsonify({"files": files})


@app.route("/api/recordings/<cam>/<filename>/download")
def download_recording(cam, filename):
    if cam not in ("evk", "picam", "ircam"):
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

# ── Replay API ────────────────────────────────────────────────────────────────

@app.route("/api/replay/start", methods=["POST"])
def api_replay_start():
    global replay_proc
    data     = request.get_json() or {}
    filename = data.get("filename", "")
    speed    = float(data.get("speed", 1.0))
    cam      = data.get("cam", "evk")

    if not filename:
        return jsonify({"error": "No filename provided"}), 400
    if cam not in ("evk", "picam", "ircam"):
        return jsonify({"error": "Unknown camera"}), 400

    recordings_dir = _recordings_dir(cam)
    filepath = (recordings_dir / filename).resolve()
    if filepath.parent != recordings_dir.resolve():
        return jsonify({"error": "Invalid filename"}), 400
    if not filepath.exists():
        return jsonify({"error": "File not found"}), 404

    with proc_lock:
        kill_proc(replay_proc)
        cmd = [
            cfg["replay_bin"],
            str(filepath),
            "--events-socket", cfg["replay_events_socket"],
            "--speed",         str(speed),
        ]
        try:
            replay_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            return jsonify({"error": f"replay binary not found: {cfg['replay_bin']}"}), 500

    return jsonify({"ok": True, "filename": filename, "speed": speed, "cam": cam})


@app.route("/api/replay/stop", methods=["POST"])
def api_replay_stop():
    global replay_proc
    with proc_lock:
        kill_proc(replay_proc)
        replay_proc = None
    return jsonify({"ok": True})


@app.route("/api/replay/status")
def api_replay_status():
    with proc_lock:
        running = replay_proc is not None and replay_proc.poll() is None
    return jsonify({"running": running})

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EVK4 Dashboard")
    parser.add_argument("--recordings-dir",        default="/tmp/evk4_raw",
                        help="Root recordings directory (sub-dirs: evk/, picam/, ircam/)")
    parser.add_argument("--viewfinder-bin",        default="./target/release/viewfinder")
    parser.add_argument("--replay-bin",            default="./target/release/replay")
    parser.add_argument("--live-events-socket",    default="/tmp/evk4_events.sock")
    parser.add_argument("--replay-events-socket",  default="/tmp/evk4_replay_events.sock")
    parser.add_argument("--live-port",             type=int, default=8081,
                        help="Port the live viewfinder binds to (nginx proxies /stream/evk/)")
    parser.add_argument("--replay-port",           type=int, default=8084,
                        help="Port the replay viewfinder binds to (nginx proxies /playback/evk/)")
    parser.add_argument("--html-file",             default="dashboard.html",
                        help="Path to the standalone HTML dashboard file")
    parser.add_argument("--host",                  default="0.0.0.0")
    parser.add_argument("--port",                  type=int, default=5000)
    args = parser.parse_args()

    cfg.update(vars(args))

    print(f"[dashboard] Recordings dir:       {args.recordings_dir}")
    print(f"[dashboard] Viewfinder binary:    {args.viewfinder_bin}")
    print(f"[dashboard] Replay binary:        {args.replay_bin}")
    print(f"[dashboard] Live VF port:         {args.live_port}  (nginx → /stream/evk/)")
    print(f"[dashboard] Replay VF port:       {args.replay_port} (nginx → /playback/evk/)")
    print(f"[dashboard] HTML file:            {args.html_file}")
    print(f"[dashboard] Serving API at:       http://{args.host}:{args.port}")

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
