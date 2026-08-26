#!/usr/bin/env bash
# Exit immediately if any command fails
set -e

echo "========================================="
echo "Starting Watchdog Setup for Orange Pi 5 Max"
echo "========================================="

# 1. Update package index and install required C compilation libraries
echo "[1/4] Installing system tools and compilation dependencies..."
sudo apt-get update
sudo apt-get install -y gpiod libgpiod-dev python3-dev python3-pip

# 2. Build the correct Python-gpiod v2 bindings natively
echo "[2/4] Installing modern gpiod Python wrapper..."
sudo pip3 install gpiod smbus2 --break-system-packages

# 3. Create the working watchdog Python script block
echo "[3/4] Generating working watchdog app script (watchdog_app.py)..."
cat << 'EOF' > watchdog_app.py
import gpiod
import smbus2
import time
from gpiod.line import Direction, Value

addr = 0x67
WATCH_ON_OFF      = 0x01
WATCH_TIME        = 0x02
WATCH_REMAIN_TIME = 0x03
WATCH_STATE       = 0x04
WATCH_FwVersion   = 0x05

WATCH_ON  = 0x03
WATCH_OFF = 0x02

WATCH_Timeout = 0x03
WATCH_NO_Timeout = 0x02

WATCH_ON_LED = 0x10
WATCH_OFF_LED = 0x00

WATCH_version = 0x01
WATCH_TIME_Restart = 5 

# Set to /dev/gpiochip1 for GPIO1 bank, line offset 7 for A7 pin
CHIP_PATH = "/dev/gpiochip1"
LINE_OFFSET = 7

try:
    bus = smbus2.SMBus(1)
    
    # Request line with context manager for perfect cleanup safety
    with gpiod.request_lines(
        CHIP_PATH,
        consumer="Watchdog_Feeder",
        config={
            LINE_OFFSET: gpiod.LineSettings(
                direction=Direction.OUTPUT,
                output_value=Value.INACTIVE
            )
        }
    ) as request:

        def read(address):
            data = bus.read_i2c_block_data(addr, address, 1)
            return data[0]

        def read_word(address):
            data = bus.read_i2c_block_data(addr, address, 2)
            return ((data[1] * 256 ) + data[0])

        def write(address,data):
            temp = [0]
            temp[0] = data & 0xFF
            bus.write_i2c_block_data(addr,address,temp)

        def write_word(address,data):
            temp = [0,0]
            temp[0] = data & 0xFF
            temp[1] =(data & 0xFF00) >> 8
            bus.write_i2c_block_data(addr,address,temp)

        # Access version checking
        if read(WATCH_FwVersion) == WATCH_version: 
            print("init succeed")
            write(WATCH_ON_OFF, WATCH_ON)  
            time.sleep(0.5)
            write(WATCH_STATE, WATCH_ON_LED | WATCH_NO_Timeout)  
            write_word(WATCH_TIME, WATCH_TIME_Restart)  
        else:
            print("init fail")

        print("Entering main loop. Feeding dog on GPIO1_A7 every 0.8s. Ctrl+C to stop.")
        while True:
            request.set_value(LINE_OFFSET, Value.ACTIVE)   # Drive pin High (3.3V)
            time.sleep(0.8)
            request.set_value(LINE_OFFSET, Value.INACTIVE) # Drive pin Low (0V)
            time.sleep(0.8)
    
except KeyboardInterrupt: 
    print("\nctrl + c: Process terminated safely. Pin state dropped.")
    exit()
EOF

# 4. Make setup verification complete
echo "[4/4] Finalizing setup..."
echo "========================================="
echo "Setup Successful!"
echo "To run your script, type: sudo python3 watchdog_app.py"
echo "========================================="