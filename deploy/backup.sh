#!/usr/bin/env bash
# Daily backup of monadruk.com configs + recent generated models.
# Keeps 7 days. Runs via cron.
set -e

BACKUP_DIR=/root/backups
STAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p "$BACKUP_DIR"

# Config bundle (small, critical)
tar -czf "$BACKUP_DIR/config-$STAMP.tar.gz" \
  /etc/caddy/Caddyfile \
  /etc/caddy/ssl \
  /opt/3dmap/ecosystem.config.js \
  /opt/3dmap/frontend/.env.local \
  /opt/3dmap/backend/.env \
  /etc/fail2ban/jail.local \
  /etc/ssh/sshd_config.d/99-hardening.conf \
  2>/dev/null || true

# Customer/order data (irreplaceable — orders + quota/history, not covered by
# the models bundle below since they live in backend/data, not backend/output)
tar -czf "$BACKUP_DIR/customer-data-$STAMP.tar.gz" \
  /opt/3dmap/backend/data/orders.jsonl \
  /opt/3dmap/backend/data/users.json \
  2>/dev/null || true

# Recent generated models (last 2 days only, to bound size)
if [ -d /opt/3dmap/backend/output ]; then
  find /opt/3dmap/backend/output -name '*.3mf' -mtime -2 -print0 2>/dev/null \
    | tar -czf "$BACKUP_DIR/models-$STAMP.tar.gz" --null -T - 2>/dev/null || true
fi

# Retention: delete backups older than 7 days
find "$BACKUP_DIR" -name '*.tar.gz' -mtime +7 -delete 2>/dev/null || true

echo "[backup] done $STAMP -> $BACKUP_DIR"
