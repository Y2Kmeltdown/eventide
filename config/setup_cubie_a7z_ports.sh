#!/usr/bin/env bash
#
# Radxa Cubie A7Z (Allwinner A733 / sun60iw2, Armbian) port + RTC bring-up.
#
# Sets up, from a fresh flash of this exact board:
#   - UART0 on pin 8 (TX) / pin 10 (RX)   -- also frees the pins from the serial console
#   - UART2 on pin 7 (TX) / pin 11 (RX)
#   - UART3 on pin 27 (RX) / pin 28 (TX)  -- custom pin routing, NOT Radxa's stock pin 3/5 route
#   - UART4 on pin 16 (TX) / pin 18 (RX)
#   - I2C (TWI7) on pin 3 (SDA) / pin 5 (SCL)
#   - A DS3231 RTC at 0x68 on that I2C bus, wired into hwclock/systemd
#
# Everything here was derived and verified against real hardware (loopback tests,
# i2cdetect, smbus2 reads) on one specific Cubie A7Z unit. Re-verify after any
# vendor kernel/overlay package upgrade.
#
# Run as: sudo ./setup_cubie_a7z_ports.sh [-y]
#   -y   skip the confirmation prompt (for unattended provisioning)

set -euo pipefail

SKIP_CONFIRM=0
if [[ "${1:-}" == "-y" ]]; then
    SKIP_CONFIRM=1
fi

if [[ $EUID -ne 0 ]]; then
    echo "Must be run as root (sudo $0)" >&2
    exit 1
fi

# --- Sanity check: this script hardcodes pin routing specific to this exact board ---
if ! grep -q '^BOARD=cubie-a7z$' /etc/armbian-release 2>/dev/null; then
    echo "This does not look like a Radxa Cubie A7Z (/etc/armbian-release missing BOARD=cubie-a7z)." >&2
    echo "Refusing to run -- the pin/overlay mapping here is board-specific." >&2
    exit 1
fi

KVER="$(uname -r)"
HDR_DIR="/usr/src/linux-headers-${KVER}"
OVERLAY_DIR="/boot/dtb/allwinner/overlay"
ARMBIAN_ENV="/boot/armbianEnv.txt"
WORK_DIR="/usr/local/src/eventide-cubie-a7z-setup"
OVERLAY_PREFIX="$(grep -oP '^overlay_prefix=\K.*' "$ARMBIAN_ENV" || true)"
BASE_DTB="$(find /boot/dtb-"${KVER}"/allwinner -maxdepth 1 -name '*cubie-a7z*.dtb' 2>/dev/null | head -1)"

if [[ -z "$OVERLAY_PREFIX" || -z "$BASE_DTB" || ! -f "$BASE_DTB" ]]; then
    echo "Could not determine overlay_prefix or locate the base DTB for kernel ${KVER}." >&2
    echo "overlay_prefix='${OVERLAY_PREFIX}' base_dtb='${BASE_DTB}'" >&2
    exit 1
fi

if [[ ! -d "$HDR_DIR" ]]; then
    echo "Kernel headers not found at $HDR_DIR." >&2
    echo "Install the headers package matching 'uname -r' ($KVER) and re-run." >&2
    exit 1
fi

for tool in dtc gcc make curl fdtoverlay; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Required tool '$tool' not found. Installing device-tree-compiler/build-essential/curl..."
        DEBIAN_FRONTEND=noninteractive apt-get update -qq
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq device-tree-compiler build-essential curl
        break
    fi
done

echo "Board:          cubie-a7z"
echo "Kernel:         $KVER"
echo "Overlay prefix: $OVERLAY_PREFIX"
echo "Base DTB:       $BASE_DTB"
echo "Work dir:       $WORK_DIR"
echo
echo "This will:"
echo "  - install 5 device tree overlays (uart0-pb, uart2, uart3-pd, uart4, twi7, ds3231-rtc)"
echo "  - change armbianEnv.txt console= to 'display' (REMOVES the ttyS0 serial console -- pin 8/10 become a free UART instead)"
echo "  - disable serial-getty@ttyS0.service"
echo "  - build and install an out-of-tree rtc-ds1307.ko kernel module (this vendor kernel lacks CONFIG_RTC_DRV_DS1307)"
echo "  - install a systemd service to sync the DS3231 with the system clock at boot/shutdown"
echo "  - back up $ARMBIAN_ENV before editing it"
echo
if [[ $SKIP_CONFIRM -ne 1 ]]; then
    read -rp "Continue? [y/N] " reply
    [[ "$reply" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }
fi

mkdir -p "$WORK_DIR/overlays"
cd "$WORK_DIR"

# ---------------------------------------------------------------------------
# Overlay sources
# ---------------------------------------------------------------------------

# UART2: pin 7 (TX) / pin 11 (RX) -- stock Radxa overlay, base DT already has a
# matching pinctrl group (PB0/PB1), just needs status=okay.
cat > overlays/uart2.dtso <<'EOF'
/dts-v1/;
/plugin/;

/ {
	metadata {
		title = "Enable UART2";
		compatible = "radxa,cubie-a7a", "radxa,cubie-a7z", "radxa,cubie-a7s";
		category = "misc";
		exclusive = "PB0", "PB1", "uart2";
		description = "Enable UART2 on 40-pin header pin 7 (TX) and pin 11 (RX).";
	};
};

&pio {
	uart2_pins_active: uart2_pins@0 {
		pins = "PB0", "PB1";
		function = "uart2";
		drive-strength = <10>;
	};

	uart2_pins_sleep: uart2_pins@1 {
		pins = "PB0", "PB1";
		function = "io_disabled";
		drive-strength = <10>;
	};
};

&uart2 {
	pinctrl-names = "default", "sleep";
	pinctrl-0 = <&uart2_pins_active>;
	pinctrl-1 = <&uart2_pins_sleep>;
	status = "okay";
};
EOF

# UART4: pin 16 (TX) / pin 18 (RX) -- stock Radxa overlay (radxa-pkg/radxa-overlays,
# sun60iw2p1-uart4.dtso), explicitly lists this board and these pins.
cat > overlays/uart4.dtso <<'EOF'
/dts-v1/;
/plugin/;

/ {
	metadata {
		title = "Enable UART4";
		compatible = "radxa,cubie-a7a", "radxa,cubie-a7z", "radxa,cubie-a7s";
		category = "misc";
		exclusive = "PJ24", "PJ25", "uart4";
		description = "Enable UART4.
On Radxa Cubie A7A, this is pin 16 and pin 18.
On Radxa Cubie A7Z, this is pin 16 and pin 18.
On Radxa Cubie A7S, this is pin 16 and pin 18.
";
	};
};

&pio {
	uart4_pins_active: uart4_pins@0 {
		pins = "PJ24", "PJ25";
		function = "uart4";
		drive-strength = <10>;
	};

	uart4_pins_sleep: uart4_pins@1 {
		pins = "PJ24", "PJ25";
		function = "io_disabled";
		drive-strength = <10>;
	};
};

&uart4 {
	pinctrl-names = "default", "sleep";
	pinctrl-0 = <&uart4_pins_active>;
	pinctrl-1 = <&uart4_pins_sleep>;
	status = "okay";
};
EOF

# I2C (TWI7): pin 3 (SDA) / pin 5 (SCL) -- stock Radxa overlay. Note: i2cdetect's
# default SMBus "quick write" probe stalls on some sensors (e.g. QMC6310) that
# don't handle a bare address+STOP cleanly -- use `i2cdetect -y <bus> -r` or a
# real register read (i2cget/smbus2) to check for devices, not the default scan.
cat > overlays/twi7.dtso <<'EOF'
/dts-v1/;
/plugin/;

/ {
	metadata {
		title = "Enable TWI7";
		compatible = "radxa,cubie-a7a", "radxa,cubie-a7z", "radxa,cubie-a7s";
		category = "misc";
		exclusive = "twi7", "PJ22", "PJ23";
		description = "Enable TWI7.
On Radxa Cubie A7A this is SDA pin 3 & SCL pin 5.
On Radxa Cubie A7Z this is SDA pin 3 & SCL pin 5.
On Radxa Cubie A7S this is SDA pin 3 & SCL pin 5.
";
    };
};

&pio {
	twi7_pins_default: twi7@0 {
		pins = "PJ22", "PJ23";
		function = "twi7";
		drive-strength = <10>;
		bias-pull-up;
	};

	twi7_pins_sleep: twi7@1 {
		pins = "PJ22", "PJ23";
		function = "gpio_in";
	};
};

&twi7 {
	twi-supply = <&reg_dc1sw1>;
	clock-frequency = <400000>;
	pinctrl-0 = <&twi7_pins_default>;
	pinctrl-1 = <&twi7_pins_sleep>;
	pinctrl-names = "default", "sleep";
	twi_drv_used = <1>;
	status = "okay";
};
EOF

# UART3: pin 27 (RX) / pin 28 (TX) -- CUSTOM routing. Radxa's own stock UART3
# overlay puts UART3 on pin 3/5 instead (shared with TWI7 above), which conflicts
# with the I2C use of those pins. This overlay instead mux PD16/PD17 (pin 28/27),
# which is otherwise the default TWI2 pin pair -- do not also enable a twi2/i2c2
# overlay, they'd fight over the same physical pins.
cat > overlays/uart3-pd.dtso <<'EOF'
/dts-v1/;
/plugin/;

/ {
	metadata {
		title = "Enable UART3 on pin 27/28 (custom)";
		compatible = "radxa,cubie-a7z";
		category = "misc";
		exclusive = "PD16", "PD17", "uart3";
		description = "Custom overlay: routes UART3 to 40-pin header pin 27 (RX, PD17) and pin 28 (TX, PD16), instead of Radxa's stock pin 3/5 routing (which the upstream sun60iw2p1-uart3.dtso example uses). These pins default to TWI2 in the base device tree - do not combine this with an i2c2/twi2 overlay, they share the same physical pins.";
	};
};

&pio {
	uart3_pd_pins_active: uart3_pd_pins@0 {
		pins = "PD16", "PD17";
		function = "uart3";
		drive-strength = <10>;
	};

	uart3_pd_pins_sleep: uart3_pd_pins@1 {
		pins = "PD16", "PD17";
		function = "io_disabled";
		drive-strength = <10>;
	};
};

&uart3 {
	pinctrl-names = "default", "sleep";
	pinctrl-0 = <&uart3_pd_pins_active>;
	pinctrl-1 = <&uart3_pd_pins_sleep>;
	status = "okay";
};
EOF

# UART0: pin 8 (TX) / pin 10 (RX) -- CUSTOM, experimental, no upstream precedent.
# The base device tree ships uart0's pinctrl nodes EMPTY (pins = [00 00]) and
# instead leaves PB9/PB10 claimed by the SoC's factory pinctrl-test driver.
# This overlay fills in the real pin names and disables that competing driver.
# uart0 is also the board's serial console -- this script separately removes
# ttyS0 from the kernel console args and disables its getty, otherwise the
# console will fight with anything wired to pin 8/10.
cat > overlays/uart0-pb.dtso <<'EOF'
/dts-v1/;
/plugin/;

/ {
	metadata {
		title = "Enable UART0 on pin 8/10 (custom, experimental)";
		compatible = "radxa,cubie-a7z";
		category = "misc";
		exclusive = "PB9", "PB10", "uart0";
		description = "Custom overlay: fills in the base device tree's uart0_pins@0/@1 pinctrl nodes (shipped empty - 'pins = [00 00]') with the real PB9/PB10 pin names, and disables the SoC's factory pinctrl-test driver that otherwise claims those same pins. uart0 is the board's serial console (ttyS0) and is not routed to the 40-pin header by default. No upstream Radxa overlay does this; there is no precedent to compare against.";
	};
};

&pio {
	uart0_pins@0 {
		pins = "PB9", "PB10";
		function = "uart0";
		drive-strength = <10>;
	};

	uart0_pins@1 {
		pins = "PB9", "PB10";
		function = "gpio_in";
	};
};

&pinctrl_test {
	status = "disabled";
};
EOF

# DS3231 RTC child node on TWI7 -- needed so the rtc-ds1307 driver (built
# separately below, since this vendor kernel doesn't ship it) has something to
# bind to. Requires the twi7 overlay above to also be active.
cat > overlays/ds3231-rtc.dtso <<'EOF'
/dts-v1/;
/plugin/;

/ {
	metadata {
		title = "DS3231 RTC on TWI7 (pin 3/5)";
		compatible = "radxa,cubie-a7z";
		category = "misc";
		description = "Declares a DS3231 RTC as a child device of TWI7 (I2C on 40-pin header pin 3 SDA / pin 5 SCL), so the rtc-ds1307 kernel driver binds to it and exposes /dev/rtcN for hwclock.";
	};
};

&twi7 {
	#address-cells = <1>;
	#size-cells = <0>;

	rtc@68 {
		compatible = "maxim,ds3231";
		reg = <0x68>;
	};
};
EOF

# ---------------------------------------------------------------------------
# Compile + merge-test each overlay against the real base DTB before touching
# anything under /boot, so a bad overlay is caught here instead of at boot.
# ---------------------------------------------------------------------------

echo
echo "Compiling and verifying overlays..."
OVERLAYS="uart0-pb uart2 uart3-pd uart4 twi7 ds3231-rtc"
MERGE_CHAIN="$BASE_DTB"
for name in $OVERLAYS; do
    dtc -@ -I dts -O dtb -o "overlays/${OVERLAY_PREFIX}-${name}.dtbo" "overlays/${name}.dtso" 2>/dev/null
    fdtoverlay -i "$MERGE_CHAIN" -o "overlays/merged-${name}.dtb" "overlays/${OVERLAY_PREFIX}-${name}.dtbo"
    MERGE_CHAIN="overlays/merged-${name}.dtb"
    echo "  OK: $name"
done
rm -f overlays/merged-*.dtb

echo "Installing overlays to $OVERLAY_DIR ..."
cp overlays/"${OVERLAY_PREFIX}"-*.dtbo "$OVERLAY_DIR"/

# ---------------------------------------------------------------------------
# armbianEnv.txt: merge our overlay names into the existing overlays= line
# (without clobbering anything already there) and free the console from ttyS0.
# ---------------------------------------------------------------------------

cp "$ARMBIAN_ENV" "${ARMBIAN_ENV}.bak.$(date +%s)"

CURRENT_OVERLAYS="$(grep -oP '^overlays=\K.*' "$ARMBIAN_ENV" || true)"
NEEDED="uart0-pb uart2 uart3-pd uart4 twi7 ds3231-rtc"
MERGED="$CURRENT_OVERLAYS"
for o in $NEEDED; do
    if ! grep -qw "$o" <<<"$MERGED"; then
        MERGED="$MERGED $o"
    fi
done
MERGED="$(echo "$MERGED" | xargs)"  # trim whitespace

if grep -q '^overlays=' "$ARMBIAN_ENV"; then
    sed -i "s/^overlays=.*/overlays=${MERGED}/" "$ARMBIAN_ENV"
else
    echo "overlays=${MERGED}" >> "$ARMBIAN_ENV"
fi

if grep -q '^console=' "$ARMBIAN_ENV"; then
    sed -i 's/^console=.*/console=display/' "$ARMBIAN_ENV"
else
    echo "console=display" >> "$ARMBIAN_ENV"
fi

echo "Updated $ARMBIAN_ENV:"
grep -E '^(overlays|console)=' "$ARMBIAN_ENV"

systemctl disable --now serial-getty@ttyS0.service 2>/dev/null || true

# ---------------------------------------------------------------------------
# rtc-ds1307 out-of-tree module (CONFIG_RTC_DRV_DS1307 is not set in this
# vendor kernel). Rebuild required after any kernel package upgrade.
# ---------------------------------------------------------------------------

echo
echo "Building rtc-ds1307 kernel module for $KVER ..."
curl -sL -o rtc-ds1307.c https://raw.githubusercontent.com/torvalds/linux/v6.6/drivers/rtc/rtc-ds1307.c

cat > Makefile <<EOF
obj-m += rtc-ds1307.o
KDIR := ${HDR_DIR}
PWD := \$(shell pwd)

default:
	\$(MAKE) -C \$(KDIR) M=\$(PWD) modules

clean:
	\$(MAKE) -C \$(KDIR) M=\$(PWD) clean
EOF

make >/dev/null

install -Dm644 rtc-ds1307.ko "/lib/modules/${KVER}/extra/rtc-ds1307.ko"
depmod -a
echo "rtc-ds1307" > /etc/modules-load.d/rtc-ds1307.conf

# ---------------------------------------------------------------------------
# systemd service: sync system clock <-> DS3231 at boot/shutdown.
# Polls for /dev/rtc1 with a bounded timeout instead of a hard device
# dependency, so a future failure here degrades gracefully instead of
# stalling boot.
# ---------------------------------------------------------------------------

cat > /etc/systemd/system/ds3231-hwclock.service <<'EOF'
[Unit]
Description=Sync system clock from/to external DS3231 RTC (twi7, pin 3/5)
DefaultDependencies=no
After=systemd-modules-load.service local-fs.target
Before=sysinit.target time-sync.target

[Service]
Type=oneshot
RemainAfterExit=yes
TimeoutStartSec=20
ExecStartPre=/bin/sh -c 'for i in $(seq 1 100); do [ -e /dev/rtc1 ] && exit 0; sleep 0.2; done; echo "rtc1 never appeared" >&2; exit 1'
ExecStart=/sbin/hwclock -s -f /dev/rtc1
ExecStop=/sbin/hwclock -w -f /dev/rtc1

[Install]
WantedBy=sysinit.target
EOF

systemctl daemon-reload
systemctl enable ds3231-hwclock.service

# ---------------------------------------------------------------------------
# python-periphery: used for GPIO/UART/I2C access from Python (e.g. driving a
# header pin high/low via /dev/gpiochip0). Installed system-wide (not --user)
# since GPIO access itself needs root anyway (/dev/gpiochip* is root:root 600
# on this image, no group grants access) -- so scripts using it will
# typically be run with sudo regardless of which user owns them.
# `python3 -m pip install` rather than bare `pip3 install` -- confirmed live
# that the latter can silently land in the invoking user's ~/.local
# site-packages even under sudo (root then can't import it; anything meant
# to run as root, e.g. eventide's watchdog.service, then fails to start).
# ---------------------------------------------------------------------------

echo
echo "Installing python-periphery..."
python3 -m pip install --break-system-packages -q python-periphery

echo
echo "Done. A reboot is required to apply the overlays and console change:"
echo "    sudo reboot"
echo
echo "After reboot, verify with:"
echo "    ls /dev/ttyS0 /dev/ttyS2 /dev/ttyS3 /dev/ttyS4 /dev/rtc1"
echo "    cat /sys/class/rtc/rtc1/name"
echo "    sudo hwclock -r -f /dev/rtc1"
echo "    systemctl status ds3231-hwclock.service --no-pager"
echo "    cat /proc/cmdline | grep -o console=ttyS0   # should print nothing"
