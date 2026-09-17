#!/bin/bash
set -e

# ============================================================
# Eventide Touchscreen Kiosk Setup Script
# For Armbian (or similar minimal Debian/Ubuntu-based SBC install)
#
# Boots straight into a full-screen Chromium kiosk showing a single
# webpage, with console autologin on tty1 and (optionally) rotated touch
# input matching a rotated display. The two device-specific workarounds
# below (FORCE_MODELINE, TOUCH_DEVICE) are opt-in — leave them unset for a
# normal, well-behaved display with no touchscreen.
#
# USAGE:
#   1. Edit the CONFIGURATION block below for your setup.
#   2. Run as the target user (the script uses sudo internally where needed):
#        chmod +x setup-display.sh
#        ./setup-display.sh
#   3. Reboot: sudo reboot
# ============================================================

# ---- CONFIGURATION - edit these before running, or override any of them
#      as environment variables (e.g. from install.sh's optional
#      SETUP_KIOSK_DISPLAY step) without touching this file ----
KIOSK_USER="${KIOSK_USER:-eventide}"                # the user that will autologin and run the kiosk
KIOSK_URL="${KIOSK_URL:-http://localhost/kiosk}"          # the webpage to display
HDMI_OUTPUT="${HDMI_OUTPUT:-HDMI-1}"                # confirm with: xrandr --query (name can vary by board)
ROTATION="${ROTATION:-normal}"                      # left | right | inverted | normal
SCALE_FACTOR="${SCALE_FACTOR:-1}"                   # chromium zoom-out equivalent, e.g. 0.5 = 50%
                                              # (use <1 on a very small/high-DPI panel)
WINDOW_SIZE="${WINDOW_SIZE:-1024,768}"       # resolution AFTER rotation
TOUCH_DEVICE="${TOUCH_DEVICE:-}"             # exact device name from `DISPLAY=:0 xinput list`;
                                              # leave blank to skip touch setup entirely (no touchscreen,
                                              # or the default orientation already matches)
CHROMIUM_BIN="${CHROMIUM_BIN:-chromium}"     # binary launched in kiosk mode. Some Ubuntu/Debian images
                                              # only ship a snap-backed "chromium-browser" wrapper with no
                                              # usable local binary for --kiosk mode — if `apt install
                                              # chromium` isn't available on your image, either enable a
                                              # repo that provides it (e.g. Armbian's own repo does) or
                                              # `sudo snap install chromium` and set this to
                                              # /snap/bin/chromium (and drop "chromium" from the apt
                                              # install list below, since that package won't exist there)
FORCE_MODELINE="${FORCE_MODELINE:-}"         # leave blank for a normal display (EDID auto-detected).
                                              # Only set this if Xorg fails to start with "no screens
                                              # found" — some small panels (e.g. certain Waveshare HDMI
                                              # LCDs) report a preferred timing that violates their own
                                              # declared sync range. Format: "<mode-name> <modeline
                                              # params>", e.g.
                                              # "480x640_60 32.00 480 490 500 570 640 660 680 760 -hsync -vsync"
FORCE_KMSDEV="${FORCE_KMSDEV:-}"             # leave blank normally. Only set this if Xorg fails with
                                              # "(EE) No devices detected." / "no screens found" even
                                              # though the console/tty clearly shows a working display —
                                              # seen on Raspberry Pi 5 (BCM2712), which exposes v3d (its
                                              # 3D-only GPU, no display outputs) as its own separate DRM
                                              # card device alongside vc4 (the real display device); with
                                              # no PCI bus to rank them, Xorg's "which GPU is primary"
                                              # guess can land on v3d instead. Check the Xorg log
                                              # (~/.local/share/xorg/Xorg.0.log) for lines like
                                              # "Adding drm device (/dev/dri/cardN)" — the one whose
                                              # "Platform probe" path does NOT contain "v3d" is the one to
                                              # set here, e.g. FORCE_KMSDEV=/dev/dri/card1. Device numbering
                                              # comes from device-tree probe order, so it's stable across
                                              # reboots on a given board but can differ between boards/images.
KIOSK_WAIT_SECS="${KIOSK_WAIT_SECS:-60}"     # seconds to wait for KIOSK_URL to respond before starting X
                                              # anyway. Covers any backend-not-ready-yet case (slow boot,
                                              # a DHCP source that's itself slow to power on, ...) so
                                              # Chromium never gets a chance to render a "can't connect"
                                              # page — the console just shows a plain waiting message
                                              # instead. Set to 0 to disable and start X immediately, the
                                              # old behaviour.
# ---- end configuration ----

# Coordinate Transformation Matrix per rotation direction (only used if TOUCH_DEVICE is set)
case "$ROTATION" in
  left)     TOUCH_MATRIX="0 -1 1 1 0 0 0 0 1" ;;
  right)    TOUCH_MATRIX="0 1 0 -1 0 1 0 0 1" ;;
  inverted) TOUCH_MATRIX="-1 0 1 0 -1 1 0 0 1" ;;
  normal)   TOUCH_MATRIX="1 0 0 0 1 0 0 0 1" ;;
  *) echo "Unknown ROTATION value: $ROTATION"; exit 1 ;;
esac

echo "== Eventide kiosk setup for user: $KIOSK_USER =="

# KIOSK_USER defaults to "eventide", which usually isn't the account this
# script is actually being run as (e.g. a stock Raspberry Pi OS install's
# default user is "pi") — check now and fail with a clear message, rather
# than a confusing "~eventide/.bash_profile: No such file or directory"
# later at step 4 once ~$KIOSK_USER fails to expand for a user that doesn't
# exist.
id "$KIOSK_USER" > /dev/null 2>&1 || {
    echo "ERROR: user '$KIOSK_USER' does not exist on this system." >&2
    echo "Set KIOSK_USER to your actual login user, e.g.: KIOSK_USER=pi ./setup-display.sh" >&2
    exit 1
}

# ---- 1. Install required packages ----
echo "-- Installing packages --"
sudo apt update
PKGS="xserver-xorg xinit openbox unclutter x11-xserver-utils xinput curl"
if [ "$CHROMIUM_BIN" = "chromium" ]; then
  PKGS="$PKGS chromium"
fi
sudo apt install --no-install-recommends -y $PKGS

# ---- 2. Xorg monitor override (opt-in) ----
# Only needed for a panel whose EDID Xorg can't use as-is (see FORCE_MODELINE
# above for the symptom). Skipped by default — a normal display is detected
# and configured by Xorg automatically with no override file at all.
if [ -n "$FORCE_MODELINE" ]; then
  MODE_NAME=$(echo "$FORCE_MODELINE" | awk '{print $1}')
  MODE_PARAMS=$(echo "$FORCE_MODELINE" | cut -d' ' -f2-)
  echo "-- Writing Xorg monitor override ($MODE_NAME) --"
  sudo mkdir -p /etc/X11/xorg.conf.d
  sudo tee /etc/X11/xorg.conf.d/10-monitor.conf > /dev/null << EOF
Section "Monitor"
    Identifier "$HDMI_OUTPUT"
    Modeline "$MODE_NAME" $MODE_PARAMS
    Option "PreferredMode" "$MODE_NAME"
    Option "ModeValidation" "AllowNonEdidModes, NoHorizSyncCheck, NoVertRefreshCheck, NoMaxPClkCheck, NoEdidMaxPClkCheck"
EndSection
EOF
else
  echo "-- Skipping Xorg monitor override (FORCE_MODELINE not set; using normal EDID detection) --"
fi

# ---- 2b. Xorg GPU device pin (opt-in) ----
# Only needed when Xorg's automatic "which GPU is primary" guess picks the
# wrong DRM device (see FORCE_KMSDEV above for the symptom and how to find
# the right value). Skipped by default.
if [ -n "$FORCE_KMSDEV" ]; then
  echo "-- Writing Xorg GPU device override ($FORCE_KMSDEV) --"
  sudo mkdir -p /etc/X11/xorg.conf.d
  sudo tee /etc/X11/xorg.conf.d/20-modesetting.conf > /dev/null << EOF
Section "Device"
    Identifier "GPU"
    Driver "modesetting"
    Option "kmsdev" "$FORCE_KMSDEV"
EndSection
EOF
else
  echo "-- Skipping Xorg GPU device override (FORCE_KMSDEV not set; using normal GPU auto-detection) --"
fi

# ---- 3. Console autologin on tty1 ----
echo "-- Configuring autologin on tty1 --"
sudo mkdir -p /etc/systemd/system/getty@tty1.service.d
sudo tee /etc/systemd/system/getty@tty1.service.d/override.conf > /dev/null << EOF
[Service]
ExecStart=
ExecStart=-/sbin/agetty --autologin $KIOSK_USER --noclear %I \$TERM
EOF
sudo systemctl daemon-reload

# ---- 4. Auto-start X on tty1 login ----
# Waits (bounded by KIOSK_WAIT_SECS) for KIOSK_URL to actually respond before
# starting X — otherwise X/Chromium can win the race against a backend that's
# still starting (e.g. behind a slow-to-power-on DHCP source) and briefly
# show a "can't connect" page. The wait happens on the plain text console,
# before Chromium ever opens, so a slow backend is just a status line here
# instead of a broken-looking kiosk. KIOSK_URL/KIOSK_WAIT_SECS are baked in
# as literal values below (not read from the environment at login time), so
# re-run this script to change them rather than editing .bash_profile by hand.
echo "-- Configuring startx on login --"
USER_HOME=$(eval echo "~$KIOSK_USER")
BASH_PROFILE="$USER_HOME/.bash_profile"
if ! grep -q "startx" "$BASH_PROFILE" 2>/dev/null; then
cat >> "$BASH_PROFILE" << PROFILE_EOF

if [ -z "\$DISPLAY" ] && [ "\$(tty)" = "/dev/tty1" ]; then
  if [ "$KIOSK_WAIT_SECS" -gt 0 ]; then
    echo "Waiting up to ${KIOSK_WAIT_SECS}s for $KIOSK_URL to respond..."
    SECONDS=0
    while [ "\$SECONDS" -lt "$KIOSK_WAIT_SECS" ] && ! curl -sf -o /dev/null "$KIOSK_URL"; do
      sleep 1
    done
    if [ "\$SECONDS" -lt "$KIOSK_WAIT_SECS" ]; then
      echo "Backend responded after \${SECONDS}s."
    else
      echo "Backend still not responding after ${KIOSK_WAIT_SECS}s — starting anyway."
    fi
  fi
  startx
fi
PROFILE_EOF
fi

# ---- 5. .xinitrc: kiosk launch sequence ----
# Order matters: rotate the display, THEN apply the touch transform
# (relative to the now-rotated screen), THEN launch chromium last as
# the foreground process keeping the X session alive.
echo "-- Writing .xinitrc --"
cat > "$USER_HOME/.xinitrc" << EOF
#!/bin/bash
xset s off
xset -dpms
xset s noblank
unclutter -idle 0.5 -root &
xrandr --output $HDMI_OUTPUT --rotate $ROTATION
EOF
if [ -n "$TOUCH_DEVICE" ]; then
cat >> "$USER_HOME/.xinitrc" << EOF
xinput set-prop "$TOUCH_DEVICE" 'Coordinate Transformation Matrix' $TOUCH_MATRIX
EOF
fi
cat >> "$USER_HOME/.xinitrc" << EOF
openbox-session &
$CHROMIUM_BIN --noerrdialogs --disable-infobars --kiosk --no-first-run \\
  --force-device-scale-factor=$SCALE_FACTOR \\
  --window-size=$WINDOW_SIZE --window-position=0,0 --start-fullscreen \\
  $KIOSK_URL
EOF
chmod +x "$USER_HOME/.xinitrc"
chown "$KIOSK_USER":"$KIOSK_USER" "$USER_HOME/.xinitrc" "$BASH_PROFILE"

echo "== Setup complete =="
echo "Reboot for changes to take effect: sudo reboot"
echo ""
echo "NOTE: HDMI output name and touch device name can vary by board/panel batch."
echo "If the display or touch doesn't come up correctly after reboot, verify with:"
echo "  xrandr --query"
echo "  DISPLAY=:0 xinput list"
echo "and re-run this script with corrected values in the CONFIGURATION block."
echo "(FORCE_MODELINE and TOUCH_DEVICE are opt-in — see the comments above each.)"
