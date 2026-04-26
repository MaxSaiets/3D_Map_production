#!/usr/bin/env bash
# =============================================================
# deploy.sh — Скрипт деплою (виконується на сервері через GitHub Actions)
# =============================================================
set -euo pipefail

APP_DIR="/opt/3dmap"
LOG_DIR="/var/log/3dmap"

echo "[deploy] Starting deployment at $(date)"

# ── 0. Директорії ────────────────────────────────────────────
mkdir -p "$LOG_DIR"
mkdir -p /tmp/3dmap_output

# ── 1. Git pull ───────────────────────────────────────────────
echo "[deploy] Git pull..."
cd "$APP_DIR"
git fetch origin main
git reset --hard origin/main

# ── 2. Backend deps ───────────────────────────────────────────
echo "[deploy] Updating Python deps..."
cd "$APP_DIR/backend"
source venv/bin/activate
pip install -r requirements.txt --quiet
deactivate

# ── 3. Frontend build ────────────────────────────────────────
echo "[deploy] Building Next.js..."
cd "$APP_DIR/frontend"
npm ci --prefer-offline
npm run build

# ── 4. Перезапуск ────────────────────────────────────────────
echo "[deploy] Restarting services..."
pm2 restart 3dmap-backend 3dmap-frontend --update-env
pm2 save

# ── 5. Health check ──────────────────────────────────────────
sleep 3
if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "[deploy] Backend health check: OK"
else
    echo "[deploy] WARNING: Backend health check failed (may still be starting)"
fi

if curl -sf http://127.0.0.1:3000 > /dev/null 2>&1; then
    echo "[deploy] Frontend health check: OK"
else
    echo "[deploy] WARNING: Frontend health check failed (may still be starting)"
fi

echo "[deploy] Deployment complete at $(date)"
