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

# SD card auto-mount: any SD card inserted is mounted at
# RECORDINGS_SD_MOUNTPOINT (and unmounted when removed). Cards are detected
# by their kernel "mmc" type, not by label or device number, so no
# formatting/labeling is needed first — and USB storage is deliberately never
# matched. Assumes ONE card slot in use for external storage (a second card
# is left alone), and never touches the disk the OS boots from. Set
# AUTOMOUNT_SD=0 to skip this entirely. Doesn't change where recordings go:
# set the mountpoint as the Recordings directory in the dashboard SETTINGS
# tab to actually record to the card.
AUTOMOUNT_SD="${AUTOMOUNT_SD:-1}"
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
DEVICE_TREE_COMPATIBLE="${DEVICE_TREE_COMPATIBLE:-/proc/device-tree/compatible}"
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
        local model model_key compat=""
        model=$(tr -d '\0' < "$DEVICE_TREE_MODEL")
        if [ -r "$DEVICE_TREE_COMPATIBLE" ]; then
            compat=$(tr '\0' ' ' < "$DEVICE_TREE_COMPATIBLE")
        fi
        # Lowercased with spaces stripped, so "Orange Pi Zero 3W" and
        # "OrangePi Zero3W" both become "orangepizero3w".
        model_key=$(echo "$model" | tr '[:upper:]' '[:lower:]' | tr -d ' ')
        case "$model" in
            *"Raspberry Pi"*) echo "raspbian"; return ;;
        esac
        # "Orange Pi" is a brand, not a SoC: the boards under it use different
        # chips (RK3588, Allwinner A733, ...) that need different I2C/overlay/
        # boot-file setup, so the per-board hooks can't be keyed on the brand
        # alone. Zero 3W is matched on the model name (it's unique to that
        # board); the RK3588 hooks additionally require an RK3588 SoC.
        case "$model_key" in
            *orangepi*zero3w*) echo "orangepi_zero3w"; return ;;
            *orangepi*)
                case "$compat" in
                    *rk3588*) echo "orangepi"; return ;;
                esac
                echo "[WARN] Orange Pi board '$model' is not a recognised RK3588 or Zero 3W model — not applying the RK3588 setup to it; falling back to the generic install." >&2
                ;;
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

# ── Orange Pi, RK3588 boards only (edge kernel, e.g. Orange Pi 5 Max) ───────
# Everything in this section is RK3588-specific (the i2c2m0 overlay targets
# rockchip,rk3588; the bus/GPIO numbers are that SoC's) — detect_os() only
# selects "orangepi" when the device tree reports an RK3588, and other Orange
# Pi boards (see orangepi_zero3w below) get their own id instead.
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

# ── Orange Pi Zero 3W (Allwinner A733) ───────────────────────────────────────
# Deliberately NO os_configure/packages/services hooks yet — it gets the
# generic install only. Its I2C bus, watchdog feed pin and RTC wiring are
# Allwinner-specific and none of them have been confirmed on this board, so
# nothing is guessed here (the RK3588 hooks above must not be reused, and the
# Cubie A7Z script below is specific to that board's own pin routing even
# though it's the same SoC). To add them, first confirm on the device:
#   i2cdetect -l ; i2cdetect -y <bus> -r   (bus with 0x67 watchdog / 0x68 RTC)
#   gpioinfo                               (feed pin's gpiochip + line)
# then add os_configure_orangepi_zero3w / os_services_orangepi_zero3w hooks
# here and a matching entry in code/watchdog.py's BOARDS (and the bus in
# code/eventide.py's _WATCHDOG_I2C_BUS_BY_BOARD).

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
# smbus2 for eventide.py's own watchdog I2C register access (SETTINGS tab
# watchdog config) — previously only OS-specific hooks installed this for
# watchdog.py's sake; the base backend now needs it on every board.
# `python3 -m pip install` rather than bare `pip3 install` — confirmed live
# elsewhere in this installer that the latter can silently land in the
# invoking user's ~/.local site-packages even under sudo, where root
# (eventide.service's user) can't import it.
sudo python3 -m pip install smbus2 --break-system-packages

step "OS-specific packages ($OS)"
run_os_hook packages

## TAILSCALE (optional, generic — board-agnostic, same reasoning as nginx/
## supervisor already being generic). Installs and enables tailscaled only;
## deliberately never runs `tailscale up` here — no auth key exists at
## install time, connecting is entirely a SETTINGS-tab action afterward.
step "Tailscale"
if command -v tailscale > /dev/null; then
    echo "[INFO] tailscale already installed — skipping"
else
    curl -fsSL https://tailscale.com/install.sh | sh
fi
sudo systemctl enable --now tailscaled

## SD CARD AUTO-MOUNT (generic — not an OS-specific hook; udev/systemd work
## identically on any board). A udev rule selects SD cards by their kernel
## "mmc" type and starts a per-partition service that mounts/unmounts them —
## see config/99-eventide-sd.rules and config/eventide-sd-mount.sh.
step "SD card auto-mount"
if [ "$AUTOMOUNT_SD" = "0" ]; then
    echo "[INFO] AUTOMOUNT_SD=0 — skipping SD card auto-mount setup"
else
    # Older installs mounted a card by filesystem label via a .mount unit at
    # this same mountpoint; that would fight the udev-driven mount below.
    if [ -e /etc/systemd/system/media-eventide.mount ]; then
        echo "[INFO] removing the old label-based media-eventide.mount (replaced by SD-type detection)"
        sudo systemctl disable --now media-eventide.mount 2> /dev/null || true
        sudo rm -f /etc/systemd/system/media-eventide.mount
    fi

    sudo mkdir -p "$RECORDINGS_SD_MOUNTPOINT"
    # While no card is mounted, make the EMPTY mountpoint immutable so nothing
    # can be written into it — otherwise a recorder pointed here with the card
    # absent would silently fill the OS drive instead. A mounted card covers
    # the directory, so this doesn't affect it. Best-effort (needs a
    # filesystem that supports chattr). Skipped if a card is mounted right now
    # (the flag would land on the card's own root directory) and if the
    # directory already has content: that means recordings have already been
    # written into it with no card mounted (and may still be in use), and
    # locking it would break them — leave that for the user to resolve.
    if mountpoint -q "$RECORDINGS_SD_MOUNTPOINT"; then
        :
    elif [ -n "$(sudo ls -A "$RECORDINGS_SD_MOUNTPOINT")" ]; then
        echo "[WARN] $RECORDINGS_SD_MOUNTPOINT already contains files on the OS drive (recordings written while no card was mounted?) — NOT locking it. A card mounted there will hide them; move them first if you want them on the card."
    else
        sudo chattr +i "$RECORDINGS_SD_MOUNTPOINT" 2> /dev/null \
            || echo "[INFO] could not mark $RECORDINGS_SD_MOUNTPOINT immutable (filesystem doesn't support it) — skipping"
    fi

    # Helper goes in root-owned /usr/local/sbin (not the eventide-user-owned
    # /usr/local/eventide tree) since a root service runs it.
    check_file /usr/local/eventide/config/eventide-sd-mount.sh
    sudo install -m 0755 -o root -g root /usr/local/eventide/config/eventide-sd-mount.sh /usr/local/sbin/eventide-sd-mount
    sudo sed "s|MOUNTPOINT_PLACEHOLDER|$RECORDINGS_SD_MOUNTPOINT|" /usr/local/eventide/config/eventide-sd-mount@.service \
        | sudo tee /etc/systemd/system/eventide-sd-mount@.service > /dev/null
    sudo chmod 644 /etc/systemd/system/eventide-sd-mount@.service
    sudo install -m 0644 -o root -g root /usr/local/eventide/config/99-eventide-sd.rules /etc/udev/rules.d/99-eventide-sd.rules
    sudo systemctl daemon-reload
    sudo udevadm control --reload
    # Replay "add" for block devices so a card that's already inserted is
    # picked up now, not only after the next insertion/boot.
    sudo udevadm trigger --action=add --subsystem-match=block
    sudo udevadm settle --timeout=10 || true
    if mountpoint -q "$RECORDINGS_SD_MOUNTPOINT"; then
        echo "[OK] SD card mounted at $RECORDINGS_SD_MOUNTPOINT"
    else
        echo "[OK] SD auto-mount installed — no card mounted right now; insert one and it will appear at $RECORDINGS_SD_MOUNTPOINT"
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
if [ "$AUTOMOUNT_SD" != "0" ]; then
    echo "SD cards auto-mount at $RECORDINGS_SD_MOUNTPOINT (SD only — USB storage is ignored) — set this as the Recordings directory in the dashboard SETTINGS tab to actually use it."
fi
if [ -n "$SETUP_KIOSK_DISPLAY" ]; then
    echo "Touchscreen kiosk configured — will boot straight into it after this reboot. Re-run config/setup-display.sh any time to adjust display/touch settings."
fi
if [ "$OS" = "orangepi_zero3w" ]; then
    echo "Orange Pi Zero 3W: generic install only — I2C, RTC and watchdog are not configured for this board yet (see the orangepi_zero3w note in install.sh)."
fi
if [ "$OS" = "cubie" ]; then
    echo "Cubie A7Z UART/I2C/RTC overlays installed — take effect after this reboot. Verify with the commands config/setup_cubie_a7z_ports.sh printed above."
fi
if [ -n "$ENABLE_USBC_HOST" ]; then
    echo "USB-C port configured as a USB 2.0 host port — takes effect after this reboot."
fi
echo "Tailscale installed but not connected — configure it from the dashboard SETTINGS tab."
echo "Rebooting in 10 seconds (Ctrl-C to cancel)."
sleep 10
sudo reboot
