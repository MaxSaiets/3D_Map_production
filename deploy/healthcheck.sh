#!/usr/bin/env bash
# 3dmap health-check + self-recovery + optional Telegram alerting.
# Runs from cron every few minutes. Probes backend + frontend; if a service is
# down it attempts a pm2 restart and (if configured) sends a Telegram alert.
#
# Configure alerting by creating /opt/3dmap/deploy/.health.env with:
#   TG_BOT_TOKEN=123456:ABC...
#   TG_CHAT_ID=123456789
# Without it the script still logs + auto-restarts (no alert is sent).

set -u

LOG="/var/log/3dmap-health.log"
ENV_FILE="/opt/3dmap/deploy/.health.env"
STATE_DIR="/tmp/3dmap-health"
mkdir -p "$STATE_DIR"

BACKEND_URL="http://127.0.0.1:8000/api/health"
FRONTEND_URL="http://127.0.0.1:3000/"

# Telegram creds (optional)
TG_BOT_TOKEN=""
TG_CHAT_ID=""
# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && . "$ENV_FILE"

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

notify() {
  # $1 = message. Only sends if creds present. De-dupes repeated alerts within 30 min.
  local msg="$1"
  local key="$2"
  local stamp_file="$STATE_DIR/alert_$key"
  local now; now=$(date +%s)
  if [ -f "$stamp_file" ]; then
    local last; last=$(cat "$stamp_file" 2>/dev/null || echo 0)
    if [ $((now - last)) -lt 1800 ]; then return; fi
  fi
  echo "$now" > "$stamp_file"
  if [ -n "$TG_BOT_TOKEN" ] && [ -n "$TG_CHAT_ID" ]; then
    curl -s --max-time 10 \
      "https://api.telegram.org/bot${TG_BOT_TOKEN}/sendMessage" \
      -d chat_id="${TG_CHAT_ID}" \
      -d parse_mode="HTML" \
      --data-urlencode text="🛑 <b>monadruk.com</b>
${msg}" >/dev/null 2>&1
  fi
}

clear_alert() { rm -f "$STATE_DIR/alert_$1" 2>/dev/null; }

check() {
  # $1 name, $2 url, $3 pm2-process, $4 expected substring (optional)
  local name="$1" url="$2" proc="$3" expect="${4:-}"
  local code body
  body=$(curl -s --max-time 12 -w "\n%{http_code}" "$url" 2>/dev/null)
  code=$(echo "$body" | tail -n1)
  body=$(echo "$body" | sed '$d')
  if [ "$code" = "200" ] && { [ -z "$expect" ] || echo "$body" | grep -q "$expect"; }; then
    clear_alert "$name"
    return 0
  fi
  log "DOWN $name (http=$code). Restarting pm2:$proc"
  pm2 restart "$proc" --update-env >/dev/null 2>&1
  sleep 8
  code=$(curl -s --max-time 12 -o /dev/null -w "%{http_code}" "$url" 2>/dev/null)
  if [ "$code" = "200" ]; then
    log "RECOVERED $name after restart"
    notify "✅ <b>${name}</b> впав і був автоматично перезапущений (відновлено)." "$name"
  else
    log "STILL DOWN $name (http=$code) after restart"
    notify "❌ <b>${name}</b> НЕ ВІДПОВІДАЄ (http=$code). Авто-перезапуск не допоміг — потрібна увага." "$name"
  fi
}

check "backend"  "$BACKEND_URL"  "3dmap-backend"  '"status"'
check "frontend" "$FRONTEND_URL" "3dmap-frontend"
