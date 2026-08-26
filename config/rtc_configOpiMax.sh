#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "=== Orange Pi 5 Max RTC Setup Script ==="

# 1. Instantly register the RTC device using the verified path
echo "Registering DS3132/DS3231 on I2C-2..."
echo ds1307 0x68 | sudo tee /sys/class/i2c-dev/i2c-2/device/new_device > /dev/null

# Give the kernel a moment to register the device node
sleep 1

# 2. Automatically detect if the new RTC is mapped to rtc0 or rtc1
if [ -e /dev/rtc1 ]; then
    RTC_DEV="/dev/rtc1"
elif [ -e /dev/rtc0 ]; then
    RTC_DEV="/dev/rtc0"
else
    echo "ERROR: No RTC device node found in /dev/!"
    exit 1
fi

echo "Detected RTC device node at: $RTC_DEV"

# 3. Write the current system time to the physical battery module
echo "Writing current system time to the hardware clock..."
sudo hwclock -w -f "$RTC_DEV"

# 4. Create/Overwrite the boot automation script
echo "Configuring boot persistence in /etc/rc.local..."
sudo tee /etc/rc.local > /dev/null <<EOF
#!/bin/bash
# Dynamic initialization for DS3132/DS3231 RTC on I2C-2
echo ds1307 0x68 > /sys/class/i2c-dev/i2c-2/device/new_device
sleep 1

# Sync system time from the battery clock
if [ -e /dev/rtc1 ]; then
    hwclock -s -f /dev/rtc1
else
    hwclock -s -f /dev/rtc0
fi

exit 0
EOF

# 5. Ensure the boot script is executable
sudo chmod +x /etc/rc.local

echo "=== Setup Complete! ==="
echo "The system time is backed up, and your RTC will auto-load on every reboot."
echo "Current RTC Time:"
sudo hwclock -r -f "$RTC_DEV"