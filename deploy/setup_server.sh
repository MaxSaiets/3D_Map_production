#!/usr/bin/env bash
# =============================================================
# setup_server.sh — Перше розгортання на DigitalOcean (Ubuntu 22/24)
#
# Запуск:
#   curl -fsSL https://raw.githubusercontent.com/MaxSaiets/3dMap/main/deploy/setup_server.sh | bash
# або:
#   chmod +x setup_server.sh && bash setup_server.sh
# =============================================================
set -euo pipefail

REPO_URL="https://github.com/MaxSaiets/3D_Map_production.git"
APP_DIR="/opt/3dmap"
APP_USER="www-data"   # або окремий user, тут для простоти root/www-data

echo "╔══════════════════════════════════════════════════╗"
echo "║   3dMAP — Server Setup                          ║"
echo "╚══════════════════════════════════════════════════╝"

# ─── 1. Системні пакети ──────────────────────────────────────
echo ""
echo "[1/8] Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
  git curl wget build-essential \
  python3 python3-venv python3-dev python3-pip \
  nginx certbot python3-certbot-nginx \
  libgdal-dev gdal-bin \
  libspatialindex-dev \
  libgeos-dev \
  htop

# ─── 2. Node.js 20 ───────────────────────────────────────────
echo ""
echo "[2/8] Installing Node.js 20..."
if ! command -v node &>/dev/null || [[ $(node -v) != v20* ]]; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
fi
echo "Node: $(node -v) | npm: $(npm -v)"

# ─── 3. Swap + PM2 ───────────────────────────────────────────
echo ""
echo "[3/8] Configuring swap and installing PM2..."
if ! swapon --show | grep -q '/swapfile'; then
  fallocate -l 4G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=4096
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
fi
grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
npm install -g pm2

# ─── 4. Клонування репозиторію ───────────────────────────────
echo ""
echo "[4/8] Cloning repository..."
if [ -d "$APP_DIR/.git" ]; then
  echo "  Repo already exists, pulling latest..."
  cd "$APP_DIR" && git pull origin main
else
  git clone "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR"

# ─── 5. Backend — Python venv + deps ────────────────────────
echo ""
echo "[5/8] Setting up Python backend..."
cd "$APP_DIR/backend"

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi
source venv/bin/activate

pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# Manifold3D (опційно — для boolean операцій)
pip install manifold3d --quiet 2>/dev/null || echo "  [WARN] manifold3d not installed (optional)"

deactivate

# ─── 6. Frontend — npm install + build ──────────────────────
echo ""
echo "[6/8] Building Next.js frontend..."
cd "$APP_DIR/frontend"
pm2 stop 3dmap-frontend >/dev/null 2>&1 || true
npm ci --prefer-online
rm -rf .next
npm run build

# ─── 7. Директорії та права ──────────────────────────────────
echo ""
echo "[7/8] Creating directories..."
mkdir -p /tmp/3dmap_output
mkdir -p "$APP_DIR/backend/cache/terrarium"
mkdir -p "$APP_DIR/backend/cache/osm"

# ─── 8. PM2 ecosystem ────────────────────────────────────────
echo ""
echo "[8/8] Starting services with PM2..."
cd "$APP_DIR"
pm2 startOrRestart ecosystem.config.js
pm2 save
pm2 startup systemd -u root --hp /root | tail -1 | bash || true

# ─── Nginx ───────────────────────────────────────────────────
echo ""
echo "[NGINX] Configuring nginx..."
cp "$APP_DIR/deploy/nginx.conf" /etc/nginx/sites-available/3dmap
ln -sf /etc/nginx/sites-available/3dmap /etc/nginx/sites-enabled/3dmap
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   Setup complete!                                ║"
echo "║                                                  ║"
echo "║  Next steps:                                     ║"
echo "║  1. Create /opt/3dmap/backend/.env               ║"
echo "║     (copy from backend/.env.production.example) ║"
echo "║  2. Add GitHub Secrets:                          ║"
echo "║     SERVER_IP / SERVER_USER / SSH_PRIVATE_KEY    ║"
echo "║  3. (optional) SSL:                              ║"
echo "║     certbot --nginx -d yourdomain.com            ║"
echo "╚══════════════════════════════════════════════════╝"
