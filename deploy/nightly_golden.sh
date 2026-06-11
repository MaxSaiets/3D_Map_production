#!/usr/bin/env bash
# Нічні golden-тести геометрії (cron 4:30). Тонка обгортка над golden_check.py:
# підтягує TG-креди з .health.env і запускає скрипт venv-пайтоном бекенда.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$DIR/.health.env" ] && . "$DIR/.health.env"
export TG_BOT_TOKEN="${TG_BOT_TOKEN:-}" TG_CHAT_ID="${TG_CHAT_ID:-}"
exec "$DIR/../backend/venv/bin/python" "$DIR/golden_check.py"
