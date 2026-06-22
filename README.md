# Eventide

## Quick Install

Run the following command to install eventide on a raspberry pi with a clean install of raspbian lite 64-bit

``` bash
sudo apt update && sudo apt install -y git && git clone https://github.com/Y2Kmeltdown/eventide.git && cd eventide && sudo chmod +x install.sh && ./install.sh
```

## Payload Information

### Power on procedure
1. Connect all components
2. Remove Lens caps
3. Verify Gimbal is balanced. A balanced gimbal should maintain it's position when moved to any arbitrary position while powered off. A little bit of movement after releasing it is fine as long as it doesn't have clear unstable equlibrium
4. Power on Ground station. You should see the screen go white and when finished booting you will begin seeing connected devices on the screen
5. Power on Payload. A successful boot will start with the GPS flashing and the gimbal will begin homing it will move all motors indepently to rotate to it's home position. The gimbals home position is lined up exactly with the T symbol on the base connection and the payload should be facing directly forward. After it is homed it will immediately start calibrating gimbal movement relative to the Flight Controllers Bearing. It will jitter and slowly rotate. It will do a massive jump to a specific cardinal direction. Once eventide has initialised on the payload the gimbal control through the pi will activate and the gimbal should do one final movement in which the controller will make the gimbal face true north. If this doesn't happen and on the ground station eventide is showing as offline than the pi didn't boot correctly.
6. Access the web interface. Connect your laptop to one of the black ethernet ports on the ground station and when connected to the network type the address `192.168.30.7` into your web browser. This will take you to the ground control web interface.
7. Connect the ground station to the payload. In the top right of the interface there will be a tex box which will accept another IP address. Put in the IP address of the payload which is `192.168.30.2`. Once entered and the payload is connected you should begin to see the camera feed map position and telemetry data.

## Troubleshooting

- If the system has been on for a long period of time. If you power it off the eventide system likely won't power on again. The system will need some time to cool down before it can boot again. I believe this is some quirk with the SSD we use in that it fails to boot if it is too warm.
- If the gimbal controller isn't connecting and you cannot see any map positions the MAVProxy service may have failed to start up. The best fix is to SSH into the tripwire raspberry pi and reboot it using `sudo reboot`. If this isn't possible restart the whole payload.
- If the ground station doesn't power on. It is likely due to a connector coming loose inside. You may need to open the ground station and inspect all of the power connectors.
- If you power on the system and the gimbal goes limp during power on. The internal gimbal controller may have experienced a fault and disabled the motors for safety. Restart the whole system.
- If one of the sensors isn't showing any data. You can check the supervisor tab to identify if it is failing to start. Some of the sensors will show logs in that tab if you click on them. If they don't show logs more detailed logs can be found at `192.168.30.2:8080` in the error and output links for the specific sensor.

| Name | IP Address | Username | Password |
| :--- | :--- | :--- | :--- |
| Wifi | | groundstation | groundstation |
| Router | 192.168.30.1 | | Groundstation |
| Tripwire | 192.168.30.2 | highwire | highwire |
| IR Camera | 192.168.30.3 | | |
| BluSDR_Base | 192.168.30.4 | | Tripwire |
| BluSDR_Vehicle | 192.168.30.5 | | Tripwire |
| ADSB Pi | 192.168.30.6 | pi | flightaware |
| Ground Control Pi | 192.168.30.7 | groundcontrol | groundcontrol |
