#!/bin/bash
export DISPLAY=:0
export XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR

# Disable screen blanking/screensaver
xset s off
xset -dpms
xset s noblank

DISPLAY=:0 xinput set-prop 6 "Coordinate Transformation Matrix" 0 -1 1 1 0 0 0 0 1

# Launch your app
# Log output for debugging
exec sudo python3 /home/tripwire/eventideGUI.py > /home/tripwire/app.log 2>&1
