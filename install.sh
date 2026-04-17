#!/bin/bash

# Raspberry Pi Set up

## CONFIGURATION
sudo raspi-config nonint do_spi 0
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_serial_hw 0
sudo raspi-config nonint do_serial_cons 0
sudo raspi-config nonint do_change_timezone Australia/Sydney

sudo sed -i 's/dtparam=i2c_arm=on/dtparam=i2c_arm=on,i2c_arm_baudrate=400000/g' /boot/firmware/config.txt
echo "usb_max_current_enable=1" | sudo tee -a /boot/firmware/config.txt > /dev/null
echo "dtoverlay=i2c-rtc,ds3231" | sudo tee -a /boot/firmware/config.txt > /dev/null

sudo sed -i 's/#HandlePowerKey=poweroff/HandlePowerKey=ignore/g' /etc/systemd/logind.conf

echo "RuntimeWatchdogSec=15" | sudo tee -a /etc/systemd/system.conf > /dev/null

## DIRECTORY SETUP
if [ -z "${1}" ]; then
    EVENTIDE_DIR=/home/$USER/recordings
else
    EVENTIDE_DIR=$1
    mkdir -p /usr/local/eventide/data
    touch /usr/local/eventide/data/where_are_my_files.txt
    echo "Data Files have been set to $1 during installation" >> /usr/local/eventide/data/where_are_my_files.txt
fi
sudo mkdir -p $EVENTIDE_DIR
sudo mkdir -p $EVENTIDE_DIR/evk
sudo mkdir -p $EVENTIDE_DIR/picam
sudo mkdir -p $EVENTIDE_DIR/ircam

sudo mkdir -p /usr/local/eventide
sudo mkdir -p /usr/local/eventide/packages
sudo cp -a code /usr/local/eventide/code
sudo cp -a config /usr/local/eventide/config

sudo chown -R $USER:$USER /usr/local/eventide
sudo chown -R $USER:$USER $EVENTIDE_DIR

sudo sed -i "s@SEDPLACEHOLDER@$EVENTIDE_DIR@g" /usr/local/eventide/config/supervisor.conf
sudo sed -i "s@SEDPLACEHOLDER@$EVENTIDE_DIR@g" /usr/local/eventide/config/dashboard.service

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
/home/$USER/.local/bin/neuromorphic-drivers-install-udev-rules
sudo /usr/bin/python3.13 /home/$USER/.local/lib/python3.13/site-packages/neuromorphic_drivers/udev.py

## Service Installations

## Watchdog
sudo cp /usr/local/eventide/config/watchdog.service /lib/systemd/system/watchdog.service
sudo chmod 644 /lib/systemd/system/watchdog.service
sudo systemctl daemon-reload
sudo systemctl enable watchdog.service

## RTC
sudo hwclock -w -f /dev/rtc1
sudo timedatectl set-ntp false
sudo cp /usr/local/eventide/config/rtc.service /lib/systemd/system/rtc.service
sudo chmod 644 /lib/systemd/system/rtc.service
sudo systemctl daemon-reload
sudo systemctl enable rtc.service

## Dashboard
sudo apt install -y python3-flask
sudo cp /usr/local/eventide/config/dashboard.service /lib/systemd/system/dashboard.service
sudo chmod 644 /lib/systemd/system/dashboard.service
sudo systemctl daemon-reload
sudo systemctl enable dashboard.service

## NGINX Installation
sudo apt install -y nginx
sudo cp /usr/local/eventide/config/streams /etc/nginx/sites-available/streams
sudo ln -s /etc/nginx/sites-available/streams /etc/nginx/sites-enabled/
sudo unlink /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx

## Event Camera Data Logger
cd /usr/local/eventide/packages
git clone https://github.com/Y2Kmeltdown/evk_datalogger.git
cd evk_datalogger
cargo build --release
sudo cp target/release/evk_datalogger /usr/local/eventide/code/
sudo cp target/release/viewfinder /usr/local/eventide/code/

## Pi Camera Datalogger
cd /usr/local/eventide/packages
git clone https://github.com/Y2Kmeltdown/picam_datalogger.git
cd picam_datalogger
sudo cp camera_config.json /usr/local/eventide/config/
sudo cp camera_app.py /usr/local/eventide/code/
sudo cp mjpeg_server.py /usr/local/eventide/code/

## IR Camera Datalogger
cd /usr/local/eventide/packages
git clone https://github.com/ericltb15/aravis-ir.git
cd aravis-ir
sudo chmod +x install.sh 
/usr/bin/bash install.sh
sudo cp ircam /usr/local/eventide/code/

## SUPERVISOR INSTALLATION
sudo mkdir -p /etc/supervisor/conf.d
sudo cp /usr/local/eventide/config/supervisor.conf /etc/supervisor/conf.d/supervisor.conf

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
