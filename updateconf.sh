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
sudo cp "$REPO_DIR/config/playback.conf" /etc/supervisor/conf.d/playback.conf
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
