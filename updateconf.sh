#!/bin/bash
# updateconf.sh — push the eventide base supervisor config to the payload's
# supervisord and reload it. Per-module configs (module-<name>.conf) are
# generated and applied by the dashboard backend at module install time —
# see docs/MODULES.md. Run from anywhere; paths resolve relative to this script.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

fail() { echo "[FAIL] $*" >&2; exit 1; }

echo "[INFO] copying base supervisor config"
sudo cp "$REPO_DIR/config/supervisor-base.conf" /etc/supervisor/conf.d/00-eventide-base.conf
# playback.conf contains SEDPLACEHOLDER for the recordings dir (substituted by
# install.sh) — carry over the value already in use on this device.
RECORDINGS_DIR=$(sed -n 's/.*--recordings \([^ ]*\).*/\1/p' /etc/supervisor/conf.d/playback.conf 2>/dev/null | grep -v SEDPLACEHOLDER || true)
if [ -z "$RECORDINGS_DIR" ]; then
    RECORDINGS_DIR=$(sed -n 's/.*--recordings-dir \([^ ]*\).*/\1/p' /lib/systemd/system/dashboard.service 2>/dev/null | grep -v SEDPLACEHOLDER || true)
fi
sudo cp "$REPO_DIR/config/playback.conf" /etc/supervisor/conf.d/playback.conf
if [ -n "$RECORDINGS_DIR" ]; then
    sudo sed -i "s@SEDPLACEHOLDER@$RECORDINGS_DIR@g" /etc/supervisor/conf.d/playback.conf
else
    echo "[WARN] could not determine recordings dir — set --recordings in /etc/supervisor/conf.d/playback.conf manually"
fi
echo "[INFO] removing stale monolithic supervisor configs (pre-module installs)"
sudo rm -f /etc/supervisor/conf.d/supervisor.conf /etc/supervisor/conf.d/supervisord.conf
echo "[INFO] DONE"

echo "[INFO] applying supervisor config (reread + update)"
sudo supervisorctl reread
sudo supervisorctl update
echo "[INFO] DONE"

if [ -f "$REPO_DIR/config/streams" ]; then
    echo "[INFO] copying nginx streams config"
    sudo cp "$REPO_DIR/config/streams" /etc/nginx/sites-available/streams
    sudo nginx -t || fail "nginx config test failed"
    echo "[INFO] reloading nginx daemon"
    sudo systemctl reload nginx
    echo "[INFO] DONE"
else
    echo "[INFO] no config/streams file — skipping nginx streams config"
fi

echo "[INFO] updateconf complete"
