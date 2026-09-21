#!/usr/bin/env python3
# Hardware watchdog feeder — one script for every supported board.
#
# Every board so far uses the same external I2C watchdog IC (address 0x67,
# same register map), but the I2C bus number and how the feed pin is driven
# differ per board — gpiozero's default pin factory only auto-detects
# Raspberry Pi boards, so every other board drives the feed pin by chip+line
# against an explicit GPIO character device instead, via whichever library
# ("gpiod" or "periphery") that board's install step already needs for other
# reasons. BOARDS below holds the per-board numbers; EVENTIDE_BOARD selects
# which entry to use.
#
# install.sh sets EVENTIDE_BOARD in watchdog.service's environment (see
# install_watchdog_service() and the per-OS os_services_* hooks) to whichever
# board it detected — this script never tries to guess the board itself, so
# a board with no confirmed hardware numbers here simply isn't started
# rather than risk driving the wrong I2C bus or GPIO line.
import os
import sys
import time

addr = 0x67
WATCH_ON_OFF      = 0x01
WATCH_TIME        = 0x02
WATCH_REMAIN_TIME = 0x03
WATCH_STATE       = 0x04
WATCH_FwVersion   = 0x05
WATCH_CYCLE_TIME    = 0x06
WATCH_RECOVERY_TIME = 0x07

WATCH_ON  = 0x03
WATCH_OFF = 0x02

WATCH_Timeout = 0x03
WATCH_NO_Timeout = 0x02

WATCH_ON_LED = 0x10
WATCH_OFF_LED = 0x00

WATCH_version = 0x01

# Bus/chip/line numbers are all confirmed against real hardware per board
# (see each entry's comment) — never add or edit an entry here without
# reconfirming the same way (i2cdetect for the bus, a pinout diagram or the
# kernel's own pinctrl debugfs table for the feed pin's chip+line). Getting
# these wrong means driving an unintended I2C bus or GPIO line on the
# actual device. gpio_chip/gpio_line address the feed pin the same way for
# both the "gpiod" and "periphery" backends — a character device path plus
# a line offset within it — so those two keys are shared between entries;
# only gpio_backend picks which library actually opens them.
BOARDS = {
    # Raspberry Pi: gpiozero auto-detects the board's own GPIO. Feed pin is
    # BCM4 (physical header pin 7), watchdog IC on I2C1 (physical pins 3/5).
    "raspbian": {
        "i2c_bus": 1,
        "gpio_backend": "gpiozero",
        "gpiozero_pin": 4,
    },
    # Orange Pi 5 Max (edge kernel): watchdog IC confirmed live via
    # `i2cdetect -y 2` (found at 0x67); GPIO1_A7 (chip 1, line 7) is the
    # feed pin. i2c bus 2 needs its own device-tree overlay on this board
    # (see os_configure_orangepi in install.sh) — it isn't enabled by default.
    "orangepi": {
        "i2c_bus": 2,
        "gpio_backend": "gpiod",
        "gpio_chip": "/dev/gpiochip1",
        "gpio_line": 7,
    },
    # Radxa Cubie A7Z: watchdog IC confirmed live on TWI7 (i2c bus 7, same
    # bus config/setup_cubie_a7z_ports.sh enables for the DS3231 RTC) —
    # NOTE: `i2cdetect -y 7` alone shows nothing at 0x67; this IC only
    # answers a real register read, not the default quick-write probe, so
    # re-confirming this needs `i2cdetect -y 7 -r` (or i2cget/smbus2), same
    # caveat the ports script's own twi7 overlay comment calls out for the
    # QMC6310. Feed pin is 40-pin header pin 13 / PL6, confirmed against
    # /sys/kernel/debug/pinctrl/7025000.pinctrl/pins (pin 358 = PL6 = local
    # offset 6 within that controller, which is gpiochip1 per gpiodetect).
    # Uses python-periphery rather than gpiod — it's what
    # setup_cubie_a7z_ports.sh already installs on this board for GPIO/I2C
    # access, so the watchdog doesn't need a second GPIO library installed
    # alongside it.
    "cubie": {
        "i2c_bus": 7,
        "gpio_backend": "periphery",
        "gpio_chip": "/dev/gpiochip1",
        "gpio_line": 6,
    },
}


def fail(msg):
    print(f"[watchdog] {msg}", file=sys.stderr)
    sys.exit(1)


board = os.environ.get("EVENTIDE_BOARD")
if not board:
    fail("EVENTIDE_BOARD is not set — install.sh should set this in watchdog.service's "
         "Environment= based on the OS it detected (see install_watchdog_service() in "
         "install.sh). Refusing to guess a board's I2C bus/GPIO pin.")
if board not in BOARDS:
    fail(f"no hardware config for EVENTIDE_BOARD={board!r} in BOARDS — this board's watchdog "
         f"I2C bus and feed-pin numbers haven't been confirmed against real hardware yet.")

cfg = BOARDS[board]

# The dashboard SETTINGS tab's watchdog fields (enabled / wait / cycle /
# recovery) live in eventide's settings.json, and eventide.py also pushes them
# to the MCU whenever it starts. Both this script and the backend write the
# same registers at their own startup, in no guaranteed order — so this reads
# the same file with the same defaults (see DEFAULT_SETTINGS in eventide.py;
# keep the two in sync) rather than a hardcoded value that could silently
# override the user's saved config after a reboot.
SETTINGS_FILE = os.environ.get("EVENTIDE_SETTINGS_FILE", "/usr/local/eventide/data/settings.json")
WATCHDOG_SETTING_DEFAULTS = {
    "watchdog_enabled": True,
    "watchdog_wait_secs": 60,
    "watchdog_cycle_secs": 1,
    "watchdog_recovery_secs": 120,
}


def load_watchdog_settings():
    settings = dict(WATCHDOG_SETTING_DEFAULTS)
    try:
        import json
        with open(SETTINGS_FILE) as fh:
            saved = json.load(fh)
        for key in settings:
            if key in saved:
                settings[key] = saved[key]
    except (OSError, ValueError):
        pass  # no/unreadable settings file yet: the defaults above apply
    return settings


import smbus2
bus = smbus2.SMBus(cfg["i2c_bus"])


def read(address):
    return bus.read_i2c_block_data(addr, address, 1)[0]


def write(address, data):
    bus.write_i2c_block_data(addr, address, [data & 0xFF])


def write_word(address, data):
    bus.write_i2c_block_data(addr, address, [data & 0xFF, (data & 0xFF00) >> 8])


if cfg["gpio_backend"] == "gpiozero":
    import gpiozero
    _feed_pin = gpiozero.DigitalOutputDevice(cfg["gpiozero_pin"], active_high=True, initial_value=False)

    def feed_high(): _feed_pin.on()
    def feed_low():  _feed_pin.off()

elif cfg["gpio_backend"] == "gpiod":
    import gpiod
    from gpiod.line import Direction, Value
    _request = gpiod.request_lines(
        cfg["gpio_chip"],
        consumer="Watchdog_Feeder",
        config={cfg["gpio_line"]: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE)},
    )

    def feed_high(): _request.set_value(cfg["gpio_line"], Value.ACTIVE)
    def feed_low():  _request.set_value(cfg["gpio_line"], Value.INACTIVE)

elif cfg["gpio_backend"] == "periphery":
    from periphery import GPIO as PeripheryGPIO
    # 3 positional args (path, line, direction) selects periphery's
    # character-device backend rather than its legacy sysfs-by-number one.
    _periphery_gpio = PeripheryGPIO(cfg["gpio_chip"], cfg["gpio_line"], "out")

    def feed_high(): _periphery_gpio.write(True)
    def feed_low():  _periphery_gpio.write(False)

else:
    fail(f"unknown gpio_backend {cfg['gpio_backend']!r} for board {board!r}")

try:
    if read(WATCH_FwVersion) == WATCH_version:
        print(f"[watchdog] init succeed (board={board}, i2c bus={cfg['i2c_bus']})")
        ws = load_watchdog_settings()
        write(WATCH_ON_OFF, WATCH_ON if ws["watchdog_enabled"] else WATCH_OFF)
        time.sleep(0.5)
        write(WATCH_STATE, WATCH_ON_LED | WATCH_NO_Timeout)
        write_word(WATCH_TIME, int(ws["watchdog_wait_secs"]))
        write_word(WATCH_CYCLE_TIME, int(ws["watchdog_cycle_secs"]))
        write_word(WATCH_RECOVERY_TIME, int(ws["watchdog_recovery_secs"]))
    else:
        print("[watchdog] init fail — firmware version mismatch")

    print(f"[watchdog] feeding every 0.8s (board={board}, backend={cfg['gpio_backend']})")
    while True:
        feed_high()
        time.sleep(0.8)
        feed_low()
        time.sleep(0.8)

except KeyboardInterrupt:
    print("\n[watchdog] ctrl+c: exiting")
    sys.exit(0)
