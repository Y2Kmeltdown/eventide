#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Eventide base platform installer
#
# Installs the MINIMAL base platform on a Raspberry Pi (Raspbian Lite 64-bit):
#   OS config, Python + Flask, supervisord, nginx, the dashboard backend,
#   watchdog / RTC / MAVProxy services, the playback server, and the Rust
#   toolchain (so Rust modules can build on-device).
#
# Everything else — camera dataloggers, MJPEG streamers, the gimbal
# controller — is an installable module, installed afterwards from the
# dashboard's MODULES tab. See docs/MODULES.md.
#
# Usage:   ./install.sh [recordings_dir]
# Logging: /tmp/eventide-install.log
# Errors:  any failing command aborts the install with a clear message.
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

LOG_FILE=/tmp/eventide-install.log
exec > >(tee -a "$LOG_FILE") 2>&1

EVENTIDE_USER="${SUDO_USER:-$USER}"

step() { echo; echo "==> $*"; }
fail() { echo; echo "[FAIL] $*" >&2; echo "[FAIL] full log: $LOG_FILE" >&2; exit 1; }
trap 'fail "installation aborted at line $LINENO (command: $BASH_COMMAND)"' ERR

check_file() { [ -e "$1" ] || fail "expected file missing: $1"; }

install_service() {
    local name=$1
    check_file "/usr/local/eventide/config/$name.service"
    sudo cp "/usr/local/eventide/config/$name.service" "/lib/systemd/system/$name.service"
    sudo chmod 644 "/lib/systemd/system/$name.service"
    sudo systemctl daemon-reload
    sudo systemctl enable "$name.service"
    echo "[OK] $name.service installed and enabled"
}

## PREFLIGHT
step "Preflight checks"
[ "$EUID" -ne 0 ] || fail "run as a normal user, not root — the script uses sudo where needed"
command -v sudo    > /dev/null || fail "sudo not found"
command -v apt-get > /dev/null || fail "apt-get not found — this installer targets Debian/Raspbian"
command -v git     > /dev/null || fail "git not found — run: sudo apt update && sudo apt install -y git"
command -v curl    > /dev/null || fail "curl not found — run: sudo apt update && sudo apt install -y curl"
[ -d code ] && [ -d config ] || fail "run this script from the eventide repository root"
EVENTIDE_HOME=$(getent passwd "$EVENTIDE_USER" | cut -d: -f6)
[ -n "$EVENTIDE_HOME" ] || fail "could not determine home directory for user '$EVENTIDE_USER'"
echo "[INFO] installing for user: $EVENTIDE_USER (home: $EVENTIDE_HOME)"
echo "[INFO] logging to $LOG_FILE"

## RASPBERRY PI CONFIGURATION
step "Raspberry Pi configuration"
sudo raspi-config nonint do_spi 0
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_serial_hw 0
sudo raspi-config nonint do_serial_cons 1
sudo raspi-config nonint do_change_timezone Australia/Sydney

sudo sed -i 's/dtparam=i2c_arm=on/dtparam=i2c_arm=on,i2c_arm_baudrate=400000/g' /boot/firmware/config.txt
echo "usb_max_current_enable=1" | sudo tee -a /boot/firmware/config.txt > /dev/null
echo "dtoverlay=i2c-rtc,ds3231" | sudo tee -a /boot/firmware/config.txt > /dev/null

sudo sed -i 's/#HandlePowerKey=poweroff/HandlePowerKey=ignore/g' /etc/systemd/logind.conf

echo "RuntimeWatchdogSec=15" | sudo tee -a /etc/systemd/system.conf > /dev/null

## DIRECTORY SETUP
step "Directory setup"
if [ -z "${1:-}" ]; then
    EVENTIDE_DIR=$EVENTIDE_HOME/recordings
else
    EVENTIDE_DIR=$1
    sudo mkdir -p /usr/local/eventide/data
    sudo touch /usr/local/eventide/data/where_are_my_files.txt
    echo "Data Files have been set to $1 during installation" | sudo tee -a /usr/local/eventide/data/where_are_my_files.txt > /dev/null
fi
sudo mkdir -p "$EVENTIDE_DIR"
# Module recordings sub-directories (evk/, picam/, ...) are created by each
# module's install (recordings_subdir in eventide-module.json).

sudo mkdir -p /usr/local/eventide
sudo mkdir -p /usr/local/eventide/packages
sudo cp -a code /usr/local/eventide/code
sudo cp -a config /usr/local/eventide/config

sudo chown -R "$EVENTIDE_USER:$EVENTIDE_USER" /usr/local/eventide
sudo chown -R "$EVENTIDE_USER:$EVENTIDE_USER" "$EVENTIDE_DIR"

sudo sed -i "s@SEDPLACEHOLDER@$EVENTIDE_DIR@g" /usr/local/eventide/config/dashboard.service

## SYSTEM PACKAGES
step "System packages"
sudo apt update
sudo apt install -y \
    i2c-tools \
    util-linux-extra \
    ffmpeg \
    python3 \
    python3-pip \
    python3-venv \
    python3-flask \
    nginx \
    supervisor

## RUST TOOLCHAIN (kept in the base install so Rust modules build on-device)
step "Rust toolchain"
if [ -x "$EVENTIDE_HOME/.cargo/bin/cargo" ]; then
    echo "[INFO] cargo already installed — skipping rustup"
else
    curl https://sh.rustup.rs -sSf | bash -s -- -y
fi
check_file "$EVENTIDE_HOME/.cargo/bin/cargo"
grep -q 'cargo/bin' "$EVENTIDE_HOME/.bashrc" 2> /dev/null || \
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> "$EVENTIDE_HOME/.bashrc"

## PYTHON PACKAGES (base only — module deps live in each module's manifest)
step "Python packages (base)"
sudo pip3 install --break-system-packages -r /usr/local/eventide/config/requirements.txt

## SYSTEMD SERVICES
step "systemd services"
install_service watchdog
install_service rtc
install_service dashboard

## PLAYBACK SERVER (in-repo component)
step "Playback server build"
cd /usr/local/eventide/code/playback
"$EVENTIDE_HOME/.cargo/bin/cargo" build --release
check_file /usr/local/eventide/code/playback/target/release/playback
cd - > /dev/null

## NGINX
step "nginx configuration"
sudo cp /usr/local/eventide/config/vehicle.nginx /etc/nginx/sites-available/vehicle.nginx
sudo ln -sf /etc/nginx/sites-available/vehicle.nginx /etc/nginx/sites-enabled/
sudo cp /usr/local/eventide/config/nginx.conf /etc/nginx/nginx.conf
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx

## MAVPROXY
step "MAVProxy"
sudo apt install -y \
    python3-dev \
    python3-opencv \
    python3-matplotlib \
    python3-lxml \
    python3-pygame
sudo pip3 install --break-system-packages future PyYAML mavproxy
install_service mavproxy

## SUPERVISOR BASE CONFIG
step "supervisord base configuration"
sudo mkdir -p /etc/supervisor/conf.d
# Remove stale monolithic configs from pre-module installs.
sudo rm -f /etc/supervisor/conf.d/supervisor.conf /etc/supervisor/conf.d/supervisord.conf
sudo cp /usr/local/eventide/config/supervisor-base.conf /etc/supervisor/conf.d/00-eventide-base.conf
sudo cp /usr/local/eventide/config/playback.conf /etc/supervisor/conf.d/playback.conf
sudo systemctl restart supervisor

## VERIFICATION
step "Verification"
check_file /usr/local/eventide/code/dashboard.py
check_file /usr/local/eventide/code/playback/target/release/playback
check_file /etc/supervisor/conf.d/00-eventide-base.conf
check_file /etc/supervisor/conf.d/playback.conf
for svc in watchdog rtc dashboard mavproxy supervisor; do
    systemctl is-enabled --quiet "$svc.service" || fail "$svc.service is not enabled"
    echo "[OK] $svc.service enabled"
done
sudo nginx -t > /dev/null 2>&1 || fail "nginx config test failed"
echo "[OK] nginx config valid"
command -v supervisord > /dev/null || fail "supervisord not installed"
echo "[OK] supervisord installed"
"$EVENTIDE_HOME/.cargo/bin/cargo" --version > /dev/null || fail "cargo not working"
echo "[OK] rust toolchain working"

## DONE
step "Eventide base platform installed successfully"
sudo chmod -R 777 "$EVENTIDE_DIR"
echo "Base config: /etc/supervisor/conf.d/00-eventide-base.conf"
echo "To view running processes visit http://$HOSTNAME.local or run: supervisorctl status"
echo "Install modules (cameras, gimbal, ...) from the dashboard MODULES tab — see docs/MODULES.md."
echo "Rebooting in 10 seconds (Ctrl-C to cancel)."
sleep 10
sudo reboot
