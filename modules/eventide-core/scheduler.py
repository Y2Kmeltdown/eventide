#!/usr/bin/env python3
"""
scheduler.py — eventide-core scheduled-recording service.

Serves a small CRUD API for cron-triggered recording jobs (GET/POST
/api/schedules, PATCH/DELETE /api/schedules/<id>) and runs a background
thread that fires each enabled job on its schedule by POSTing to the base
eventide backend's /api/recording/trigger — the same headless fan-out the
dashboard's master-record RECORD ALL / STOP ALL button now uses. This is
what lets a schedule fire recording with no browser open.

Cron expressions are the standard 5-field form (minute hour dom month dow),
matched with a small stdlib-only implementation — no external scheduling
library is used.
"""

import argparse
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

_lock = threading.Lock()
_data_file: Path
_backend_url: str

# ── Cron matching ────────────────────────────────────────────────────────────
# Standard 5-field cron (minute hour day-of-month month day-of-week), each
# field one of: "*", "*/n", "a", "a-b", "a-b/n", or a comma-separated list of
# any of the above. day-of-week is 0=Sunday..6=Saturday (cron convention).

def _field_matches(field: str, value: int, lo: int, hi: int) -> bool:
    for part in field.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if "/" in part:
            part, step_s = part.split("/", 1)
            step = int(step_s)
        if part == "*":
            start, end = lo, hi
        elif "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
        else:
            start = end = int(part)
        if start <= value <= end and (value - start) % step == 0:
            return True
    return False


def validate_cron(expr: str) -> str | None:
    """Returns an error string if `expr` is not a valid 5-field cron
    expression, else None."""
    parts = expr.split()
    if len(parts) != 5:
        return "cron expression must have 5 fields: minute hour dom month dow"
    ranges = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 6)]
    try:
        now = datetime.now()
        probe = (now.minute, now.hour, now.day, now.month, now.isoweekday() % 7)
        for field, (lo, hi), value in zip(parts, ranges, probe):
            _field_matches(field, value, lo, hi)
    except (ValueError, ZeroDivisionError) as exc:
        return f"invalid cron field: {exc}"
    return None


def cron_matches(expr: str, now: datetime) -> bool:
    parts = expr.split()
    if len(parts) != 5:
        return False
    minute, hour, dom, month, dow = parts
    return (
        _field_matches(minute, now.minute, 0, 59)
        and _field_matches(hour, now.hour, 0, 23)
        and _field_matches(dom, now.day, 1, 31)
        and _field_matches(month, now.month, 1, 12)
        and _field_matches(dow, now.isoweekday() % 7, 0, 6)
    )


# ── Persistence ──────────────────────────────────────────────────────────────
# Same atomic-write-then-replace pattern as the base backend's
# load_settings()/save_settings() (code/eventide.py).

def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def load_schedules() -> list[dict]:
    if not _data_file.exists():
        return []
    try:
        data = json.loads(_data_file.read_text())
        return data.get("items", []) if isinstance(data, dict) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_schedules(items: list[dict]) -> None:
    _atomic_write(_data_file, json.dumps({"items": items}, indent=2) + "\n")


# ── API ──────────────────────────────────────────────────────────────────────

@app.route("/api/schedules")
def api_list():
    with _lock:
        return jsonify({"items": load_schedules()})


@app.route("/api/schedules", methods=["POST"])
def api_create():
    data = request.get_json(silent=True) or {}
    label = str(data.get("label") or "").strip() or "Untitled schedule"
    cron = str(data.get("cron") or "").strip()
    err = validate_cron(cron)
    if err:
        return jsonify({"error": err}), 400
    try:
        duration = float(data.get("duration_seconds") or 0)
    except (TypeError, ValueError):
        return jsonify({"error": "duration_seconds must be a number"}), 400
    if duration < 0:
        return jsonify({"error": "duration_seconds must be >= 0"}), 400

    job = {
        "id": uuid.uuid4().hex[:10],
        "label": label,
        "cron": cron,
        "duration_seconds": duration or None,
        "enabled": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_run": None,
        "last_result": None,
    }
    with _lock:
        items = load_schedules()
        items.append(job)
        save_schedules(items)
    return jsonify(job), 201


@app.route("/api/schedules/<job_id>", methods=["PATCH"])
def api_update(job_id):
    data = request.get_json(silent=True) or {}
    with _lock:
        items = load_schedules()
        job = next((j for j in items if j["id"] == job_id), None)
        if job is None:
            return jsonify({"error": f"no such schedule: {job_id}"}), 404
        if "enabled" in data:
            job["enabled"] = bool(data["enabled"])
        save_schedules(items)
        return jsonify(job)


@app.route("/api/schedules/<job_id>", methods=["DELETE"])
def api_delete(job_id):
    with _lock:
        items = load_schedules()
        remaining = [j for j in items if j["id"] != job_id]
        if len(remaining) == len(items):
            return jsonify({"error": f"no such schedule: {job_id}"}), 404
        save_schedules(remaining)
        return jsonify({"ok": True})


# ── Firing loop ──────────────────────────────────────────────────────────────
# Polls every 20s (sub-minute granularity) and fires each enabled job at most
# once per matching minute. No catch-up for minutes missed while the process
# was down (e.g. across a reboot) — same limitation as a normal crontab.

def _fire(job: dict) -> None:
    result = "ok"
    try:
        body = {"recording": True}
        if job.get("duration_seconds"):
            body["duration_seconds"] = job["duration_seconds"]
        r = requests.post(f"{_backend_url}/api/recording/trigger", json=body, timeout=10)
        data = r.json() if r.ok else {}
        if not r.ok:
            result = f"error: backend returned {r.status_code}"
        else:
            oks = sum(1 for item in data.get("results", []) if item.get("ok"))
            result = f"ok ({oks}/{data.get('count', 0)} sources)"
    except requests.exceptions.RequestException as exc:
        result = f"error: {exc}"
    except Exception as exc:  # keep the loop alive no matter what
        result = f"error: {exc}"

    with _lock:
        items = load_schedules()
        for j in items:
            if j["id"] == job["id"]:
                j["last_run"] = datetime.now(timezone.utc).isoformat()
                j["last_result"] = result
                break
        save_schedules(items)


def scheduler_loop() -> None:
    fired_this_minute: dict[str, str] = {}
    while True:
        now = datetime.now()
        stamp = now.strftime("%Y-%m-%dT%H:%M")
        with _lock:
            items = load_schedules()
        for job in items:
            if not job.get("enabled", True):
                continue
            if fired_this_minute.get(job["id"]) == stamp:
                continue
            try:
                if cron_matches(job["cron"], now):
                    fired_this_minute[job["id"]] = stamp
                    threading.Thread(target=_fire, args=(job,), daemon=True).start()
            except Exception as exc:
                print(f"[scheduler] error evaluating job {job.get('id')}: {exc}", flush=True)
        # Bound the dedupe map so it doesn't grow forever.
        if len(fired_this_minute) > 1000:
            fired_this_minute = {k: v for k, v in fired_this_minute.items() if v == stamp}
        time.sleep(20)


def main() -> None:
    global _data_file, _backend_url
    parser = argparse.ArgumentParser(description="Eventide scheduled-recording service")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--backend-url", default="http://127.0.0.1:5000")
    args = parser.parse_args()

    _data_file = Path(args.data_file)
    _backend_url = args.backend_url.rstrip("/")

    threading.Thread(target=scheduler_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
