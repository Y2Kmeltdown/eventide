sudo apt install -y \
    python3 \
    python3-pip

## Dashboard
sudo apt install -y python3-flask
sudo cp /usr/local/eventide/config/frontend.service /lib/systemd/system/frontend.service
sudo chmod 644 /lib/systemd/system/frontend.service
sudo systemctl daemon-reload
sudo systemctl enable frontend.service

## NGINX Installation
sudo apt install -y nginx
sudo cp /usr/local/eventide/config/groundcontrol.nginx /etc/nginx/sites-available/groundcontrol.nginx
sudo ln -s /etc/nginx/sites-available/groundcontrol.nginx /etc/nginx/sites-enabled/
sudo mkdir -p /var/cache/nginx/osm_tiles
sudo chown www-data:www-data /var/cache/nginx/osm_tiles
sudo unlink /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx