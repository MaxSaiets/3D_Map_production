#!/bin/bash
# mem_guard: рятує VM від свопінгу, НЕ вбиваючи генерації клієнтів.
# Встановлення: cp deploy/mem_guard.sh /root/mem_guard.sh; cron */3 (root).
#
# 19.10.2026: старий варіант робив `pm2 restart` при swap>1500 МБ будь-коли — посеред
# друку 200 мм. Задача з черги губилась (404 «Task not found»), а pm2 не зміг убити
# засвопований процес за 1.6 с («could not be stopped») і НЕ підняв бекенд — сайт
# лежав 15 хв, поки healthcheck не перезапустив. Тепер:
#   * swap > SOFT і бекенд роздутий, але черга порожня  → плановий рестарт;
#   * swap > HARD (реальний трешинг)                    → рестарт навіть під задачею
#     (вона в журналі data/pending_tasks і відновиться після старту);
#   * після рестарту чекаємо /api/health; якщо бекенд не піднявся — добиваємо
#     старий процес і стартуємо ще раз.
SOFT_SWAP_MB=${SOFT_SWAP_MB:-1500}
HARD_SWAP_MB=${HARD_SWAP_MB:-4000}
IDLE_RSS_MB=${IDLE_RSS_MB:-1800}
HEALTH=http://127.0.0.1:8000/api/health
export PATH=/usr/local/bin:/usr/bin:/bin:$PATH

log() { echo "$(date "+%F %T") $*"; }

# cron */3, а рестарт з очікуванням триває до 6 хв — не запускаємось удруге поверх.
exec 9>/var/lock/mem_guard.lock
flock -n 9 || exit 0

swap=$(free -m | awk '/Swap/{print $3}')
[ "${swap:-0}" -gt "$SOFT_SWAP_MB" ] || exit 0

pid=$(pgrep -f "uvicorn main:app --host 127.0.0.1 --port 8000" | head -1)
rss=0
[ -n "$pid" ] && rss=$(( $(awk '/VmRSS/{print $2}' /proc/$pid/status 2>/dev/null || echo 0) / 1024 ))
busy=$(curl -s -m 10 "$HEALTH" | python3 -c 'import json,sys
try:
    q=json.load(sys.stdin)["checks"]["queue"]; print(int(q.get("used",0))+int(q.get("running",0)))
except Exception:
    print(-1)')

if [ "$swap" -gt "$HARD_SWAP_MB" ]; then
  reason="HARD swap=${swap}MB>${HARD_SWAP_MB} rss=${rss}MB busy=${busy}"
elif [ "$busy" = "0" ] && [ "$rss" -gt "$IDLE_RSS_MB" ]; then
  reason="idle swap=${swap}MB>${SOFT_SWAP_MB} rss=${rss}MB>${IDLE_RSS_MB}"
else
  log "skip swap=${swap}MB rss=${rss}MB busy=${busy} (генерація йде або бекенд не роздутий)"
  exit 0
fi

log "$reason -> pm2 restart 3dmap-backend"
pm2 restart 3dmap-backend --update-env >/dev/null 2>&1

for i in $(seq 1 36); do  # до 3 хв
  sleep 5
  if [ "$(curl -s -o /dev/null -m 5 -w '%{http_code}' "$HEALTH")" = "200" ]; then
    log "backend up after $((i*5))s"
    exit 0
  fi
done

log "backend NOT up after 180s -> kill -9 old uvicorn + pm2 restart"
pkill -9 -f "uvicorn main:app --host 127.0.0.1 --port 8000"
sleep 5
pm2 restart 3dmap-backend --update-env >/dev/null 2>&1
for i in $(seq 1 36); do
  sleep 5
  if [ "$(curl -s -o /dev/null -m 5 -w '%{http_code}' "$HEALTH")" = "200" ]; then
    log "backend up after retry ($((i*5))s)"
    exit 0
  fi
done
log "backend STILL DOWN after retry"
