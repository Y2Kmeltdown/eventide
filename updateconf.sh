#!/bin/bash

echo "[INFO] copying supervisor config"
sudo cp config/supervisor.conf /etc/supervisor/conf.d/supervisord.conf
echo "[INFO] DONE"

echo "[INFO] Reloading supervisor daemon"
sudo systemctl reload supervisor
echo "[INFO] DONE"

echo "[INFO] Copying Nginx Config"
sudo cp config/streams /etc/nginx/sites-available/streams
sudo nginx -t
echo "[INFO] DONE"

echo "[INFO] Reloading nginx daemon"
sudo systemctl reload nginx
echo "[INFO] DONE"