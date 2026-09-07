#!/bin/bash
set -e

# ============================================================
# Waveshare 2.8" HDMI LCD (H) Kiosk Setup Script
# For Armbian on Radxa Cubie A7Z (or similar minimal Debian-based SBC install)
#
# Boots straight into a full-screen Chromium kiosk showing a single
# webpage, with the Waveshare panel's buggy EDID range worked around,
# console autologin, and rotated touch input matching a rotated display.
#
# USAGE:
#   1. Edit the CONFIGURATION block below for your setup.
#   2. Run as the target user (the script uses sudo internally where needed):
#        chmod +x setup-kiosk.sh
#        ./setup-kiosk.sh
#   3. Reboot: sudo reboot
# ============================================================

# ---- CONFIGURATION - edit these before running ----
KIOSK_USER="eventide"                        # the user that will autologin and run the kiosk
KIOSK_URL="http://localhost"                 # the webpage to display
HDMI_OUTPUT="HDMI-1"                         # confirm with: xrandr --query (name can vary by board)
ROTATION="left"                              # left | right | inverted | normal
SCALE_FACTOR="0.5"                           # chromium zoom-out equivalent, e.g. 0.5 = 50%
TOUCH_DEVICE="WaveShare WS170120 Touchscreen" # confirm with: DISPLAY=:0 xinput list
WINDOW_SIZE="640,480"                        # resolution AFTER rotation (panel native is 480x640;
                                              # left/right rotate makes it 640x480)
# ---- end configuration ----

# Coordinate Transformation Matrix per rotation direction (must match ROTATION)
case "$ROTATION" in
  left)     TOUCH_MATRIX="0 -1 1 1 0 0 0 0 1" ;;
  right)    TOUCH_MATRIX="0 1 0 -1 0 1 0 0 1" ;;
  inverted) TOUCH_MATRIX="-1 0 1 0 -1 1 0 0 1" ;;
  normal)   TOUCH_MATRIX="1 0 0 0 1 0 0 0 1" ;;
  *) echo "Unknown ROTATION value: $ROTATION"; exit 1 ;;
esac

echo "== Waveshare kiosk setup for user: $KIOSK_USER =="

# ---- 1. Install required packages ----
echo "-- Installing packages --"
sudo apt update
sudo apt install --no-install-recommends -y \
  xserver-xorg xinit openbox chromium unclutter \
  x11-xserver-utils xinput

# ---- 2. Xorg monitor override ----
# The Waveshare panel's own EDID reports a preferred timing that violates
# its own declared sync range (its "preferred mode" needs ~56kHz hsync but
# it declares a 28-40kHz valid range) causing Xorg to reject all modes and
# fail to start ("no screens found"). This forces Xorg to use the correct
# timing anyway and skip that broken range check.
echo "-- Writing Xorg monitor override --"
sudo mkdir -p /etc/X11/xorg.conf.d
sudo tee /etc/X11/xorg.conf.d/10-monitor.conf > /dev/null << EOF
Section "Monitor"
    Identifier "$HDMI_OUTPUT"
    Modeline "480x640_60" 32.00 480 490 500 570 640 660 680 760 -hsync -vsync
    Option "PreferredMode" "480x640_60"
    Option "ModeValidation" "AllowNonEdidModes, NoHorizSyncCheck, NoVertRefreshCheck, NoMaxPClkCheck, NoEdidMaxPClkCheck"
EndSection
EOF

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
xinput set-prop "$TOUCH_DEVICE" 'Coordinate Transformation Matrix' $TOUCH_MATRIX
openbox-session &
chromium --noerrdialogs --disable-infobars --kiosk --no-first-run \\
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
