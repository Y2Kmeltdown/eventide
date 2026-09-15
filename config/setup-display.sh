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

# ---- 1. Install required packages ----
echo "-- Installing packages --"
sudo apt update
PKGS="xserver-xorg xinit openbox unclutter x11-xserver-utils xinput"
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
echo "-- Configuring startx on login --"
USER_HOME=$(eval echo "~$KIOSK_USER")
BASH_PROFILE="$USER_HOME/.bash_profile"
if ! grep -q "startx" "$BASH_PROFILE" 2>/dev/null; then
cat >> "$BASH_PROFILE" << 'PROFILE_EOF'

if [ -z "$DISPLAY" ] && [ "$(tty)" = "/dev/tty1" ]; then
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
