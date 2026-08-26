#!/usr/bin/env bash
set -e

echo "Initializing custom i2c2_m0 setup for Orange Pi 5 Max (Kernel v7)..."

# 1. Create the Device Tree Source
cat << 'DTS' > rk3588-i2c2-m0-upstream.dts
/dts-v1/;
/plugin/;

/ {
    compatible = "rockchip,rk3588";

    fragment@0 {
        target = <&i2c2>;
        __overlay__ {
            status = "okay";
            #address-cells = <1>;
            #size-cells = <0>;
            pinctrl-names = "default";
            pinctrl-0 = <&i2c2m0_xfer>;
        };
    };
};
DTS

# 2. Compile and place the binary into the user-overlay directory
echo "Compiling device tree overlay..."
sudo mkdir -p /boot/overlay-user
sudo dtc -@ -I dts -O dtb -o /boot/overlay-user/i2c2-m0-upstream.dtbo rk3588-i2c2-m0-upstream.dts

# Clean up local source file
rm rk3588-i2c2-m0-upstream.dts

# 3. Safely update armbianEnv.txt without wiping existing user configurations
echo "Configuring boot environment variables..."
ENV_FILE="/boot/armbianEnv.txt"

# Ensure user_overlays parameter exists or add it
if grep -q "user_overlays=" "$ENV_FILE"; then
    if ! grep -q "i2c2-m0-upstream" "$ENV_FILE"; then
        sudo sed -i 's/user_overlays=/user_overlays=i2c2-m0-upstream /' "$ENV_FILE"
    fi
else
    echo "user_overlays=i2c2-m0-upstream" | sudo tee -a "$ENV_FILE"
fi

# 4. Force driver modules to load on system boot
echo "Registering kernel modules..."
sudo modprobe i2c-dev
if ! grep -q "i2c-dev" /etc/modules; then
    echo "i2c-dev" | sudo tee -a /etc/modules
fi

echo "========================================================"
echo " Setup complete! Please run 'sudo reboot' to finish."
echo "========================================================"