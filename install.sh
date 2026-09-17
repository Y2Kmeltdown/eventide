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

# System timezone (IANA name, e.g. Australia/Sydney, America/New_York, UTC).
# Can also be changed later from the dashboard SETTINGS tab without
# re-running this installer.
INSTALL_TIMEZONE="${INSTALL_TIMEZONE:-Australia/Sydney}"

# Optional: auto-mount a dedicated recordings SD card by filesystem label
# (not by /dev/mmcblkN or /dev/sdN — those aren't guaranteed stable across
# reboots or reader swaps). Leave RECORDINGS_SD_LABEL empty to skip this
# entirely — most installs just record to main storage. Format the card
# once yourself first: sudo mkfs.ext4 -L "$RECORDINGS_SD_LABEL" /dev/<part>
# then set the label below and re-run (or run just this installer again).
RECORDINGS_SD_LABEL="${RECORDINGS_SD_LABEL:-}"
RECORDINGS_SD_MOUNTPOINT="${RECORDINGS_SD_MOUNTPOINT:-/media/eventide}"

# Optional: set up a local touchscreen kiosk (config/setup-display.sh) as
# part of this install — boots straight into full-screen Chromium showing
# the dashboard (or /kiosk, the touchscreen-specific UI) instead of a login
# prompt. Leave SETUP_KIOSK_DISPLAY unset/empty to skip (most installs are
# headless or only ever accessed remotely). When set, config/setup-display.sh
# runs with its own defaults (see that file's CONFIGURATION block) — override
# any of KIOSK_URL, HDMI_OUTPUT, ROTATION, SCALE_FACTOR, WINDOW_SIZE,
# TOUCH_DEVICE, CHROMIUM_BIN, FORCE_MODELINE, FORCE_KMSDEV, KIOSK_WAIT_SECS
# the same way, as environment variables set before running this installer,
# e.g.:
#   SETUP_KIOSK_DISPLAY=1 KIOSK_URL=http://localhost/kiosk TOUCH_DEVICE="wch.cn USB2IIC_CTP_CONTROL" ./install.sh
SETUP_KIOSK_DISPLAY="${SETUP_KIOSK_DISPLAY:-}"

# Optional (Raspberry Pi 5 only): configure the USB-C port as a functioning
# USB 2.0 host port (dtoverlay=dwc2,dr_mode=host), in addition to its normal
# role as power input — the two coexist since power negotiation goes through
# a separate PMIC, not the dwc2 data role. Off by default: most installs
# don't need it, and it's specific to the Pi 5's USB-C controller (harmless
# to leave set on an older Pi, but it won't do anything there either).
ENABLE_USBC_HOST="${ENABLE_USBC_HOST:-}"

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

# watchdog.service is shared by every board (code/watchdog.py is one script
# for all of them, see its own header) — the only per-board difference is
# which BOARDS entry it should use, passed in via EVENTIDE_BOARD. Rather
# than templating $EVENTIDE_DIR-style with sed on the checked-out file, this
# stamps the value into the copy under /lib/systemd/system at install time.
install_watchdog_service() {
    local board=$1
    check_file "/usr/local/eventide/config/watchdog.service"
    sudo sed "s/BOARD_PLACEHOLDER/$board/" /usr/local/eventide/config/watchdog.service \
        | sudo tee /lib/systemd/system/watchdog.service > /dev/null
    sudo chmod 644 /lib/systemd/system/watchdog.service
    sudo systemctl daemon-reload
    sudo systemctl enable watchdog.service
    INSTALLED_SERVICES+=("watchdog")
    echo "[OK] watchdog.service installed and enabled (board=$board)"
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
ARMBIAN_RELEASE_FILE="${ARMBIAN_RELEASE_FILE:-/etc/armbian-release}"

detect_os() {
    # 1. Identify the Radxa Cubie A7Z specifically via Armbian's own board-id
    #    file — the same check config/setup_cubie_a7z_ports.sh uses itself.
    #    This must come before the device-tree/os-release checks below:
    #    Armbian on this SoC reports a generic model string and ID=debian
    #    like several other boards, so neither check on its own can tell a
    #    Cubie A7Z apart from them.
    if [ -r "$ARMBIAN_RELEASE_FILE" ] && grep -q '^BOARD=cubie-a7z$' "$ARMBIAN_RELEASE_FILE"; then
        echo "cubie"; return
    fi
    # 2. Identify other ARM boards from the device-tree model. This must come
    #    before the os-release ID check: newer Raspberry Pi OS releases
    #    report ID=debian in os-release, so the ID check alone can't tell a
    #    Pi from a generic Debian machine. The device-tree file only exists
    #    on ARM boards, so x86/other Debian and Ubuntu systems fall through
    #    to the ID check below.
    if [ -r "$DEVICE_TREE_MODEL" ]; then
        case "$(tr -d '\0' < "$DEVICE_TREE_MODEL")" in
            *"Raspberry Pi"*) echo "raspbian"; return ;;
            *"Orange Pi"*)    echo "orangepi"; return ;;
        esac
    fi
    # 3. Identify from the os-release ID.
    local id=""
    if [ -r "$OS_RELEASE_FILE" ]; then
        id=$(bash -c ". '$OS_RELEASE_FILE'; echo \"\${ID:-}\"")
    fi
    case "$id" in
        raspbian) echo "raspbian"; return ;;
        ubuntu)   echo "ubuntu";   return ;;
        debian)   echo "debian";   return ;;
    esac
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

    if [ -n "$ENABLE_USBC_HOST" ]; then
        echo "[INFO] enabling USB-C port as a USB 2.0 host port (dtoverlay=dwc2,dr_mode=host)"
        echo "dtoverlay=dwc2,dr_mode=host" | sudo tee -a "$boot_config" > /dev/null
    else
        echo "[INFO] ENABLE_USBC_HOST not set — USB-C port stays power-input-only"
    fi

    # Pi hardware watchdog (used by watchdog.service).
    echo "RuntimeWatchdogSec=15" | sudo tee -a /etc/systemd/system.conf > /dev/null
}

os_services_raspbian() {
    # Hardware-specific services: watchdog (code/watchdog.py, "raspbian"
    # entry in its BOARDS dict) + DS3231 I2C RTC.
    install_watchdog_service raspbian
    install_service rtc
}

# ── Orange Pi (edge kernel, e.g. Orange Pi 5 Max) ───────────────────────────
# The watchdog IC and RTC live on the same I2C bus as the Pi builds (same
# addresses, 0x67 and 0x68), but this board's newer/edge kernel doesn't
# expose i2c2 or the watchdog's GPIO feed pin the same way a Pi does:
#   - i2c2 needs an explicit device-tree overlay (not enabled by default).
#   - gpiozero's default pin factory doesn't support this board at all —
#     the watchdog feed uses libgpiod (python3 'gpiod' package) instead.
#   - the RTC isn't auto-registered by a device-tree overlay like the Pi's
#     `dtoverlay=i2c-rtc,ds3231`, so it needs an explicit driver bind at boot.
# Bus/chip/line numbers below are confirmed on the current Orange Pi 5 Max
# deployment (`i2cdetect -y 2` finds the watchdog at 0x67, RTC at 0x68) —
# reconfirm on any new/different board before trusting them.
os_configure_orangepi() {
    echo "[INFO] enabling i2c2 device-tree overlay (rockchip,rk3588 i2c2m0)"
    local dts=/tmp/rk3588-i2c2-m0-upstream.dts
    cat << 'DTS' > "$dts"
/dts-v1/;
/plugin/;

/ {
    compatible = "rockchip,rk3588";

    fragment@0 {
        target = <&i2c2>;
        __overlay__ {
            status = "okay";
            #address-cells = <1>;
            #size-cells = <0>;
            pinctrl-names = "default";
            pinctrl-0 = <&i2c2m0_xfer>;
        };
    };
};
DTS
    sudo mkdir -p /boot/overlay-user
    sudo dtc -@ -I dts -O dtb -o /boot/overlay-user/i2c2-m0-upstream.dtbo "$dts"
    rm -f "$dts"

    local env_file=/boot/armbianEnv.txt
    if [ -f "$env_file" ]; then
        if grep -q "user_overlays=" "$env_file"; then
            grep -q "i2c2-m0-upstream" "$env_file" || \
                sudo sed -i 's/user_overlays=/user_overlays=i2c2-m0-upstream /' "$env_file"
        else
            echo "user_overlays=i2c2-m0-upstream" | sudo tee -a "$env_file" > /dev/null
        fi
    else
        echo "[WARN] no $env_file found — add 'user_overlays=i2c2-m0-upstream' to your board's boot env manually"
    fi

    sudo modprobe i2c-dev
    grep -q "^i2c-dev$" /etc/modules 2> /dev/null || echo "i2c-dev" | sudo tee -a /etc/modules > /dev/null
    echo "[INFO] i2c2 overlay installed — takes effect after reboot"
}

os_packages_orangepi() {
    sudo apt-get install -y gpiod libgpiod-dev python3-dev
    # `python3 -m pip install` rather than bare `pip3 install` — confirmed
    # live that the latter can silently land in the invoking user's
    # ~/.local site-packages even under sudo (root then can't import it;
    # watchdog.service runs as root). See the same fix in
    # config/setup_cubie_a7z_ports.sh.
    sudo python3 -m pip install gpiod smbus2 --break-system-packages
}

os_services_orangepi() {
    # Hardware-specific services: watchdog (code/watchdog.py, "orangepi"
    # entry in its BOARDS dict) + DS1307-compatible RTC.
    install_watchdog_service orangepi

    # The RTC (DS1307-compatible, 0x68 on i2c-2) isn't bound by a
    # device-tree overlay like the Pi's, so bind it explicitly before the
    # generic rtc.service (hwclock -s -f /dev/rtc1) can find it.
    install_service orangepi-i2c-rtc
    install_service rtc
}

# ── Radxa Cubie A7Z (Allwinner A733 / sun60iw2, Armbian) ────────────────────
# UART/I2C pin routing and DS3231 RTC bring-up for this exact board are
# fully handled by the standalone, already-hardware-verified
# config/setup_cubie_a7z_ports.sh (derived and verified against real
# hardware — see that script's own header for details and re-verification
# notes) rather than duplicated inline here; this hook just runs it
# non-interactively (-y). No RTC install_service() call needed — the ports
# script installs its own ds3231-hwclock.service directly.
os_configure_cubie() {
    echo "[INFO] running Cubie A7Z port/RTC bring-up (config/setup_cubie_a7z_ports.sh)"
    sudo bash config/setup_cubie_a7z_ports.sh -y
}

os_packages_cubie() {
    # smbus2 for the watchdog IC's I2C registers (code/watchdog.py's
    # "cubie" BOARDS entry) — its GPIO feed pin uses python-periphery
    # instead of gpiod, which setup_cubie_a7z_ports.sh already installs
    # (in os_configure_cubie above, which runs before this hook).
    # `python3 -m pip install` rather than bare `pip3 install` — see the
    # note on the same fix in os_packages_orangepi above.
    sudo python3 -m pip install smbus2 --break-system-packages
}

os_services_cubie() {
    # Watchdog IC confirmed present on TWI7 (i2c bus 7) via a register-read
    # probe, feed pin confirmed as pin 13 / PL6 (gpiochip1 line 6) via the
    # kernel's own pinctrl debugfs table — see the "cubie" entry's comment
    # in code/watchdog.py's BOARDS dict for how these were verified.
    install_watchdog_service cubie
}

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
sudo timedatectl set-timezone "$INSTALL_TIMEZONE"
sudo sed -i 's/#HandlePowerKey=poweroff/HandlePowerKey=ignore/g' /etc/systemd/logind.conf

# eventide.service only needs loopback (nginx proxies to it on localhost) —
# it never needs an assigned IP or external connectivity — but on a stock
# image, NetworkManager-wait-online.service/systemd-networkd-wait-online.service
# block multi-user.target (and therefore every WantedBy=multi-user.target
# service, eventide.service included) until DHCP actually succeeds. That's a
# real, user-visible delay whenever the DHCP source itself is slow to come up
# (e.g. a WiFi bridge/AP powering on alongside the board). Neither wait-online
# unit provides anything eventide needs, so disable both defensively — a
# harmless no-op on any image that doesn't have them.
sudo systemctl disable NetworkManager-wait-online.service 2> /dev/null || true
sudo systemctl disable systemd-networkd-wait-online.service 2> /dev/null || true

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
sudo cp -a modules /usr/local/eventide/modules

sudo chown -R "$EVENTIDE_USER:$EVENTIDE_USER" /usr/local/eventide
sudo chown -R "$EVENTIDE_USER:$EVENTIDE_USER" "$EVENTIDE_DIR"

sudo sed -i "s@SEDPLACEHOLDER@$EVENTIDE_DIR@g" /usr/local/eventide/config/eventide.service

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

## RECORDINGS SD CARD (optional, generic — not an OS-specific hook; the
## same labeled-mount approach works identically on any board)
step "Recordings SD card auto-mount"
if [ -z "$RECORDINGS_SD_LABEL" ]; then
    echo "[INFO] RECORDINGS_SD_LABEL not set — skipping SD card auto-mount setup"
else
    # Never format automatically — the card may already hold data from a
    # previous use. Require it to already exist and be labeled.
    if ! sudo blkid -L "$RECORDINGS_SD_LABEL" > /dev/null 2>&1; then
        fail "no filesystem labeled '$RECORDINGS_SD_LABEL' found. Format the card first, e.g.: sudo mkfs.ext4 -L $RECORDINGS_SD_LABEL /dev/<the card's partition> — then re-run this installer."
    fi
    sudo tee "/etc/systemd/system/media-eventide.mount" > /dev/null << EOF
[Unit]
Description=Eventide recordings SD card

[Mount]
What=LABEL=$RECORDINGS_SD_LABEL
Where=$RECORDINGS_SD_MOUNTPOINT
Options=defaults,nofail,x-systemd.device-timeout=10

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable media-eventide.mount
    # Best-effort: nofail already means boot/install must not block on the
    # card being present, so a failure here is a warning, not a fail().
    if sudo systemctl start media-eventide.mount; then
        echo "[OK] SD card mounted at $RECORDINGS_SD_MOUNTPOINT"
    else
        echo "[WARN] media-eventide.mount enabled but did not mount now — check the card is inserted; it will retry on next boot/access"
    fi
fi

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
# Remove the pre-rename backend service (dashboard.service → eventide.service).
if systemctl list-unit-files 2> /dev/null | grep -q '^dashboard\.service'; then
    echo "[INFO] removing stale dashboard.service (renamed to eventide.service)"
    sudo systemctl disable --now dashboard.service 2> /dev/null || true
    sudo rm -f /lib/systemd/system/dashboard.service
    sudo systemctl daemon-reload
fi
install_service eventide

step "OS-specific services ($OS)"
run_os_hook services

## NGINX
step "nginx configuration"
sudo cp /usr/local/eventide/config/eventide.nginx /etc/nginx/sites-available/eventide.nginx
sudo ln -sf /etc/nginx/sites-available/eventide.nginx /etc/nginx/sites-enabled/
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
# playback.conf is now managed by the eventide-core default module.
sudo rm -f /etc/supervisor/conf.d/playback.conf
sudo cp /usr/local/eventide/config/supervisor-base.conf /etc/supervisor/conf.d/00-eventide-base.conf
sudo systemctl restart supervisor

## DEFAULT EVENTIDE MODULE
step "Default eventide-core module"
# Install the built-in module that provides the playback server and master-record UI.
sudo /usr/bin/python3 /usr/local/eventide/code/eventide.py \
    --recordings-dir "$EVENTIDE_DIR" \
    --packages-dir /usr/local/eventide/packages \
    --supervisor-conf-d /etc/supervisor/conf.d \
    --modules-registry /usr/local/eventide/modules.json \
    --settings-file /usr/local/eventide/data/settings.json \
    --install-local /usr/local/eventide/modules/eventide-core
# The module install runs as root; ensure the eventide user still owns the tree.
sudo chown -R "$EVENTIDE_USER:$EVENTIDE_USER" /usr/local/eventide

## VERIFICATION
step "Verification"
check_file /usr/local/eventide/code/eventide.py
check_file /usr/local/eventide/code/playback
check_file /etc/supervisor/conf.d/00-eventide-base.conf
check_file /etc/supervisor/conf.d/module-eventide-core.conf
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

## TOUCHSCREEN KIOSK DISPLAY (optional, generic — config/setup-display.sh
## manages its own packages/autologin/.xinitrc; nothing else in this
## installer depends on it, and it depends only on the base platform
## already being up, hence running last)
step "Touchscreen kiosk display"
if [ -z "$SETUP_KIOSK_DISPLAY" ]; then
    echo "[INFO] SETUP_KIOSK_DISPLAY not set — skipping kiosk display setup (run config/setup-display.sh manually any time)"
else
    bash /usr/local/eventide/config/setup-display.sh
fi

## DONE
step "Eventide base platform installed successfully"
sudo chmod -R 777 "$EVENTIDE_DIR"
echo "Detected OS: $OS"
echo "Base config: /etc/supervisor/conf.d/00-eventide-base.conf"
echo "To view running processes visit http://$HOSTNAME.local or run: supervisorctl status"
echo "Install modules (cameras, gimbal, ...) from the dashboard MODULES tab — see docs/MODULES.md."
if [ -n "$RECORDINGS_SD_LABEL" ]; then
    echo "SD card mounted at $RECORDINGS_SD_MOUNTPOINT — set this as the Recordings directory in the dashboard SETTINGS tab to actually use it."
fi
if [ -n "$SETUP_KIOSK_DISPLAY" ]; then
    echo "Touchscreen kiosk configured — will boot straight into it after this reboot. Re-run config/setup-display.sh any time to adjust display/touch settings."
fi
if [ "$OS" = "cubie" ]; then
    echo "Cubie A7Z UART/I2C/RTC overlays installed — take effect after this reboot. Verify with the commands config/setup_cubie_a7z_ports.sh printed above."
fi
if [ -n "$ENABLE_USBC_HOST" ]; then
    echo "USB-C port configured as a USB 2.0 host port — takes effect after this reboot."
fi
echo "Rebooting in 10 seconds (Ctrl-C to cancel)."
sleep 10
sudo reboot
