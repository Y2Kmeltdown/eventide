#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Eventide base platform installer
#
# Installs the MINIMAL base platform: OS config, Python + Flask, supervisord,
# nginx, the dashboard backend, base systemd services, the playback server,
# and the Rust toolchain (so Rust modules can build on-device).
#
# Everything else — camera dataloggers, MJPEG streamers, the gimbal
# controller — is an installable module, installed afterwards from the
# dashboard's MODULES tab. See docs/MODULES.md.
#
# The install is split into a GENERIC part (runs on any supported OS) and
# PER-OS hooks (run only when that OS is detected) — see "OS SUPPORT" below.
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

INSTALLED_SERVICES=()
install_service() {
    local name=$1
    check_file "/usr/local/eventide/config/$name.service"
    sudo cp "/usr/local/eventide/config/$name.service" "/lib/systemd/system/$name.service"
    sudo chmod 644 "/lib/systemd/system/$name.service"
    sudo systemctl daemon-reload
    sudo systemctl enable "$name.service"
    INSTALLED_SERVICES+=("$name")
    echo "[OK] $name.service installed and enabled"
}

# ═══════════════════════════════════════════════════════════════════════════
# OS SUPPORT
# ───────────────────────────────────────────────────────────────────────────
# The install runs GENERIC steps on every OS, plus PER-OS hooks only for the
# detected OS. To add support for a new OS (e.g. Orange Pi, x86 Ubuntu):
#
#   1. Extend detect_os() so the OS is identified — via its /etc/os-release
#      ID, or a /proc/device-tree/model match for ARM boards.
#   2. Implement whichever OPTIONAL hooks it needs:
#        os_configure_<os>()  — system configuration (boot files, interfaces)
#        os_packages_<os>()   — extra apt packages
#        os_services_<os>()   — extra systemd services from config/
#        os_verify_<os>()     — extra verification checks
#
# A detected OS with no hooks gets the generic install only; hooks run via
# run_os_hook, which silently skips any hook the OS doesn't define.
# ═══════════════════════════════════════════════════════════════════════════

# Overridable for testing:  OS_RELEASE_FILE=/tmp/fake-os-release ./install.sh
OS_RELEASE_FILE="${OS_RELEASE_FILE:-/etc/os-release}"
DEVICE_TREE_MODEL="${DEVICE_TREE_MODEL:-/proc/device-tree/model}"

detect_os() {
    # 1. Identify from the os-release ID.
    local id=""
    if [ -r "$OS_RELEASE_FILE" ]; then
        id=$(bash -c ". '$OS_RELEASE_FILE'; echo \"\${ID:-}\"")
    fi
    case "$id" in
        raspbian) echo "raspbian"; return ;;
        ubuntu)   echo "ubuntu";   return ;;
        debian)   echo "debian";   return ;;
    esac
    # 2. Fall back to the device-tree model (ARM boards).
    if [ -r "$DEVICE_TREE_MODEL" ]; then
        case "$(tr -d '\0' < "$DEVICE_TREE_MODEL")" in
            *"Raspberry Pi"*) echo "raspbian"; return ;;
            *"Orange Pi"*)    echo "orangepi"; return ;;
        esac
    fi
    echo "unknown"
}

run_os_hook() {
    local hook="os_${1}_${OS}"
    if declare -F "$hook" > /dev/null; then
        "$hook"
    else
        echo "[INFO] no $1 steps for OS '$OS' — skipping"
    fi
}

# ── Raspberry Pi OS (Raspbian) ──────────────────────────────────────────────
os_configure_raspbian() {
    echo "[INFO] configuring raspberry pi interfaces and boot options"
    sudo raspi-config nonint do_spi 0
    sudo raspi-config nonint do_i2c 0
    sudo raspi-config nonint do_serial_hw 0
    sudo raspi-config nonint do_serial_cons 1

    local boot_config=/boot/firmware/config.txt
    [ -f "$boot_config" ] || boot_config=/boot/config.txt
    [ -f "$boot_config" ] || fail "no raspberry pi boot config.txt found"

    sudo sed -i 's/dtparam=i2c_arm=on/dtparam=i2c_arm=on,i2c_arm_baudrate=400000/g' "$boot_config"
    echo "usb_max_current_enable=1" | sudo tee -a "$boot_config" > /dev/null
    echo "dtoverlay=i2c-rtc,ds3231" | sudo tee -a "$boot_config" > /dev/null

    # Pi hardware watchdog (used by watchdog.service).
    echo "RuntimeWatchdogSec=15" | sudo tee -a /etc/systemd/system.conf > /dev/null
}

os_services_raspbian() {
    # Hardware-specific services: Pi watchdog + DS3231 I2C RTC.
    install_service watchdog
    install_service rtc
}

# ── Example for future OSes ─────────────────────────────────────────────────
# os_configure_orangepi() {
#     echo "[INFO] configuring orange pi"
#     # e.g. enable overlays via /boot/orangepiEnv.txt, different serial device…
# }

## PREFLIGHT
step "Preflight checks"
[ "$EUID" -ne 0 ] || fail "run as a normal user, not root — the script uses sudo where needed"
command -v sudo    > /dev/null || fail "sudo not found"
command -v apt-get > /dev/null || fail "apt-get not found — this installer targets Debian-family systems"
command -v git     > /dev/null || fail "git not found — run: sudo apt update && sudo apt install -y git"
command -v curl    > /dev/null || fail "curl not found — run: sudo apt update && sudo apt install -y curl"
[ -d code ] && [ -d config ] || fail "run this script from the eventide repository root"
EVENTIDE_HOME=$(getent passwd "$EVENTIDE_USER" | cut -d: -f6)
[ -n "$EVENTIDE_HOME" ] || fail "could not determine home directory for user '$EVENTIDE_USER'"
echo "[INFO] installing for user: $EVENTIDE_USER (home: $EVENTIDE_HOME)"
echo "[INFO] logging to $LOG_FILE"

## OS DETECTION
step "OS detection"
OS=$(detect_os)
echo "[INFO] detected OS: $OS"
if [ "$OS" = "unknown" ]; then
    echo "[WARN] unrecognized OS — running the generic install only (no OS-specific steps)."
    echo "[WARN] to add support for this OS, extend detect_os() and add os_* hooks in this script."
fi

## GENERIC SYSTEM CONFIGURATION
step "System configuration (generic)"
sudo timedatectl set-timezone Australia/Sydney
sudo sed -i 's/#HandlePowerKey=poweroff/HandlePowerKey=ignore/g' /etc/systemd/logind.conf

## OS-SPECIFIC CONFIGURATION
step "OS-specific configuration ($OS)"
run_os_hook configure

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
step "System packages (generic)"
sudo apt update
sudo apt install -y \
    i2c-tools \
    util-linux-extra \
    ffmpeg \
    python3 \
    python3-pip \
    python3-venv \
    python3-flask \
    python3-requests \
    nginx \
    supervisor

step "OS-specific packages ($OS)"
run_os_hook packages

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

## SYSTEMD SERVICES
step "systemd services (generic)"
install_service dashboard

step "OS-specific services ($OS)"
run_os_hook services

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

# ## MAVPROXY
# step "MAVProxy"
# sudo apt install -y \
#     python3-dev \
#     python3-opencv \
#     python3-matplotlib \
#     python3-lxml \
#     python3-pygame
# sudo pip3 install --break-system-packages future PyYAML mavproxy
# install_service mavproxy

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
for svc in "${INSTALLED_SERVICES[@]}" supervisor; do
    systemctl is-enabled --quiet "$svc.service" || fail "$svc.service is not enabled"
    echo "[OK] $svc.service enabled"
done
sudo nginx -t > /dev/null 2>&1 || fail "nginx config test failed"
echo "[OK] nginx config valid"
command -v supervisord > /dev/null || fail "supervisord not installed"
echo "[OK] supervisord installed"
"$EVENTIDE_HOME/.cargo/bin/cargo" --version > /dev/null || fail "cargo not working"
echo "[OK] rust toolchain working"
run_os_hook verify

## DONE
step "Eventide base platform installed successfully"
sudo chmod -R 777 "$EVENTIDE_DIR"
echo "Detected OS: $OS"
echo "Base config: /etc/supervisor/conf.d/00-eventide-base.conf"
echo "To view running processes visit http://$HOSTNAME.local or run: supervisorctl status"
echo "Install modules (cameras, gimbal, ...) from the dashboard MODULES tab — see docs/MODULES.md."
echo "Rebooting in 10 seconds (Ctrl-C to cancel)."
sleep 10
sudo reboot
