#!/bin/bash

# Raspberry Pi Set up

## CONFIGURATION
sudo raspi-config nonint do_spi 0
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_serial_hw 0
sudo raspi-config nonint do_serial_cons 0
sudo raspi-config nonint do_rgpio 0
sudo raspi-config nonint do_change_timezone Australia/Sydney

sudo sed -i 's/dtparam=i2c_arm=on/dtparam=i2c_arm=on,i2c_arm_baudrate=400000/g' /boot/firmware/config.txt
sudo echo "usb_max_current_enable=1" >> /boot/firmware/config.txt
sudo echo "dtoverlay=i2c-rtc,ds3231" >> /boot/firmware/config.txt

sudo sed -i 's/#HandlePowerKey=poweroff/HandlePowerKey=ignore/g' /etc/systemd/logind.conf

sudo echo "RuntimeWatchdogSec=15" >> /etc/systemd/system.conf

## DIRECTORY SETUP
if [ -z "${1}" ]; then
    EVENTIDE_DIR=/home/$USER/recordings
else
    EVENTIDE_DIR=$1
    mkdir -p /usr/local/eventide/data
    touch /usr/local/eventide/data/where_are_my_files.txt
    echo "Data Files have been set to $1 during installation" >> /usr/local/eventide/data/where_are_my_files.txt
fi
mkdir -p $EVENTIDE_DIR

sudo mkdir -p /usr/local/eventide
sudo cp -a code /usr/local/eventide/code
sudo cp -a config /usr/local/eventide/config

## Fix this 
##sudo sed -i "s@/usr/local/eventide/data@$EVENTIDE_DIR@g" /usr/local/eventide/config/supervisord.conf


## Get Repo Updates

sudo apt update

sudo apt install -y \
    i2c-tools \
    util-linux-extra


## RUST INSTALLATION
curl https://sh.rustup.rs -sSf | bash -s -- -y

export PATH="$HOME/.cargo/bin:${PATH}"
echo "export PATH=$HOME/.cargo/bin:${PATH}" >> ~/.bashrc

## PYTHON INSTALLATION
sudo apt install -y \
    python3 \
    python3-pip

sudo apt install -y python3-picamera2 --no-install-recommends
pip install --break-system-packages -r /usr/local/eventide/config/requirements.txt

## DRIVER INSTALLATION
sudo neuromorphic-drivers-install-udev-rules
/usr/bin/python3 /usr/local/lib/python3/dist-packages/neuromorphic_drivers/udev.py

## Service Installations

## Watchdog
sudo cp /usr/local/eventide/config/watchdog.service /lib/systemd/system/watchdog.service
sudo chmod 644 /lib/systemd/system/watchdog.service
sudo systemctl daemon-reload
sudo systemctl enable watchdog.service

## RTC
sudo cp /usr/local/eventide/config/rtc.service /lib/systemd/system/rtc.service
sudo chmod 644 /lib/systemd/system/rtc.service
sudo systemctl daemon-reload
sudo systemctl enable rtc.service

## Dashboard
sudo cp /usr/local/eventide/config/dashboard.service /lib/systemd/system/dashboard.service
sudo chmod 644 /lib/systemd/system/dashboard.service
sudo systemctl daemon-reload
sudo systemctl enable dashboard.service

## NGINX Installation

## Event Camera Data Logger

## Pi Camera Datalogger

## IR Camera Datalogger

## SUPERVISOR INSTALLATION
sudo mkdir -p /etc/supervisor/conf.d
sudo cp /usr/local/eventide/config/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

sudo apt install -y \
    supervisor

echo -e "Eventide Installed successfully to view running processes visit http://$HOSTNAME.local or enter the command supervisorctl status\nReconfiguring eth0 to static ip\nPlease Wait."
sleep 10
## Network Set up

sudo chmod -R 777 $EVENTIDE_DIR

sudo reboot

# x11 Set up
# Install Prerequisites
# Prepare /etc/X11/xorg.conf.d/90-display.conf
# Prepare ~/.bash_profile
# Set up non root use
# Set up autologin

# Set up applications
# install python
# install python packages

# Neuromorphic Drivers Set up

# Picamera Set up

# Watchdog Set up

# RTC Setup

# nginx config

# sudo cp config/streams /etc/nginx/sites-available/streams
# sudo nginx -t && sudo systemctl reload nginx

# Supervisor Set up