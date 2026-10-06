#!/usr/bin/env bash
# Повний бекап monadruk.com — ОДИН архів, що перезаписує сам себе (06.10.2026).
# Запуск: systemd-таймер monadruk-backup.timer щодня о 00:00 за Києвом.
# Хост-ПК о 00:30 забирає архів на C:\monadruk-backup (pull_backup.ps1) — лише якщо він змінився.
#
# Що всередині (усе, що НЕ відтворюється з git):
#   backend/data (замовлення, користувачі, підписки, гранти, аналітика), backend/output
#   (згенеровані моделі клієнтів), усі .env, Caddy, cloudflared, pm2-конфіги, cron,
#   скрипти/дані ботів, дампи баз магазину (Postgres) і CRM (MariaDB), коміт сайту.
#
# Економія: «відбиток» (шляхи+розміри+час зміни файлів і хеші дампів баз) порівнюється
# з минулим запуском; якщо нічого не змінилось — архів не перезбирається.
# Аналітика/кеші/логи у відбиток НЕ входять (вони міняються від кожного візиту), але в
# архів потрапляють, коли він збирається через реальні зміни.
set -euo pipefail

OUT_DIR=/data/backups
ARCHIVE=$OUT_DIR/monadruk-full.tar.gz
FP_FILE=$OUT_DIR/monadruk-full.fingerprint
SHA_FILE=$OUT_DIR/monadruk-full.sha256
WORK=$(mktemp -d /data/backups/.work.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
log() { echo "[backup-full] $(date -u +%FT%TZ) $*"; }
mkdir -p "$OUT_DIR"

INCLUDE=(
  /opt/3dmap/backend/data
  /opt/3dmap/backend/output
  /opt/3dmap/backend/.env
  /opt/3dmap/frontend/.env.local
  /opt/3dmap/frontend/.env.production
  /opt/3dmap/deploy/.health.env
  /opt/3dmap/deploy/golden_baseline.json
  /opt/3dmap/ecosystem.config.js
  /etc/caddy
  /etc/cloudflared
  /root/backup.sh /root/mem_guard.sh /root/start-openclaw.sh /root/bots.ecosystem.config.cjs
  /root/scripts
  /root/projects
  /root/.openclaw
  /opt/telegram-shop
)
EXISTING=()
for p in "${INCLUDE[@]}"; do [ -e "$p" ] && EXISTING+=("$p"); done
EXCLUDES=(--exclude=node_modules --exclude=__pycache__ --exclude=.venv --exclude=venv --exclude='*.tmp' --exclude=.git)

# ── службові файли: дампи баз, crontab, коміт ────────────────────────────────
mkdir -p "$WORK/meta"
crontab -l > "$WORK/meta/root.crontab" 2>/dev/null || true
git -c safe.directory='*' -C /opt/3dmap rev-parse HEAD > "$WORK/meta/site_commit.txt" 2>/dev/null || true
if docker ps --format '{{.Names}}' | grep -qx tg-shop-db; then
  docker exec tg-shop-db pg_dump -U postgres --no-owner telegram_shop > "$WORK/meta/telegram_shop.sql"
fi
if docker ps --format '{{.Names}}' | grep -qx espocrm-db; then
  # лише робоча база CRM: --all-databases зависає на системній mysql.proc (table lock)
  docker exec espocrm-db sh -c 'exec mariadb-dump -uroot -p"$MARIADB_ROOT_PASSWORD" --single-transaction --skip-dump-date --routines --databases "$MARIADB_DATABASE"' \
    > "$WORK/meta/espocrm.sql"
fi

# ── відбиток ──────────────────────────────────────────────────────────────────
FP_INPUT=$OUT_DIR/monadruk-full.fpinput   # вхід відбитку (щоб бачити, ЩО змінилось)
{
    find "${EXISTING[@]}" \( -name node_modules -o -name __pycache__ -o -name .git -o -name venv -o -name .venv \) -prune -o \
      -type f ! -name '*.tmp' ! -name 'analytics.jsonl*' ! -name 'gen_stats.json' ! -name 'result_cache.json' \
      ! -name '*.log' ! -path '/root/.openclaw/*' -printf '%p\t%s\t%T@\n' 2>/dev/null | sort
    # pg_dump (17+) пише випадковий ключ \restrict/\unrestrict у кожен дамп — не рахуємо його
    for f in "$WORK"/meta/*.sql; do
      [ -f "$f" ] && echo "$(basename "$f") $(grep -Ev '^\\(un)?restrict ' "$f" | sha256sum | cut -d' ' -f1)"
    done
    sha256sum "$WORK/meta/root.crontab" "$WORK/meta/site_commit.txt" 2>/dev/null | sed "s#$WORK##"
} > "$WORK/fpinput"
NEW_FP=$(sha256sum "$WORK/fpinput" | cut -d' ' -f1)
if [ -f "$ARCHIVE" ] && [ -f "$FP_FILE" ] && [ "$(cat "$FP_FILE")" = "$NEW_FP" ]; then
  log "без змін з минулого бекапу — пропускаю (економія)"
  exit 0
fi
if [ -f "$FP_INPUT" ]; then
  CHANGED=$(diff "$FP_INPUT" "$WORK/fpinput" | grep -c '^>' || true)
  log "змінилось записів: $CHANGED; приклади: $(diff "$FP_INPUT" "$WORK/fpinput" | grep '^>' | cut -f1 | head -3 | tr '
' ' ')"
fi

# ── архів: збираємо поруч і атомарно підміняємо старий ───────────────────────
TMP_ARCHIVE="$OUT_DIR/.monadruk-full.tar.gz.part"
tar -czf "$TMP_ARCHIVE" "${EXCLUDES[@]}" --warning=no-file-changed \
  -C / "${EXISTING[@]#/}" -C "$WORK" meta || [ $? -eq 1 ]   # 1 = файл змінився під час читання (лог) — не фатально
gzip -t "$TMP_ARCHIVE"
chown root:deploy "$TMP_ARCHIVE" && chmod 640 "$TMP_ARCHIVE"
mv -f "$TMP_ARCHIVE" "$ARCHIVE"
( cd "$OUT_DIR" && sha256sum monadruk-full.tar.gz > "$SHA_FILE" )
chmod 644 "$SHA_FILE"
echo "$NEW_FP" > "$FP_FILE"
cp "$WORK/fpinput" "$FP_INPUT"
log "оновлено: $(du -h "$ARCHIVE" | cut -f1), sha256 $(cut -c1-12 "$SHA_FILE")"
