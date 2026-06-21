#!/usr/bin/env bash
# Reliable frontend (re)build + restart for /opt/3dmap.
#
# WHY THIS EXISTS:
#   On this server `node_modules/next/dist/compiled/*` files intermittently go
#   missing BETWEEN deploys (cssnano-simple, browserslist, jest-worker, …) →
#   `next build` dies with "Cannot find module". The cause is not yet root-caused
#   (ruled out: disk/inodes/OOM/git-clean/healthcheck/crons). Empirically a clean
#   `npm ci` always restores it. So this script is SELF-HEALING: it tries the fast
#   in-place build first, and only if that build fails with a module-resolution
#   error does it fall back to `npm ci` + rebuild. One command, always converges.
#
#   It also stops the running frontend FIRST (pm2 stop + pkill + fuser) so a live
#   `next start` can never race a half-written `.next` (the old deploy_fe.sh swap
#   bug that flapped the site to 502).
set -uo pipefail
cd /opt/3dmap/frontend || exit 1
LOG=/tmp/reland.log
NEXT=node_modules/next/dist/bin/next
echo "[RELAND] $(date -u +%H:%M:%S) start" > "$LOG"

# Stop the live frontend so the build has exclusive use of .next and :3000.
pm2 stop 3dmap-frontend >/dev/null 2>&1
pkill -9 -f next-server 2>/dev/null; pkill -9 -f "next start" 2>/dev/null
fuser -k 3000/tcp 2>/dev/null; sleep 2
rm -rf .next .next-build .next-old .next-bad node_modules/.cache

build() { rm -rf .next; "$NEXT" build >> "$LOG" 2>&1; }

echo "[RELAND] building (fast path)..." >> "$LOG"
build
# Self-heal: if the build failed because node_modules lost compiled modules,
# reinstall and rebuild once. (npm ci is the proven cure on this box.)
if [ ! -f .next/BUILD_ID ] && grep -qiE "Cannot find module|MODULE_NOT_FOUND|No such file|ENOTEMPTY" "$LOG"; then
  # IMPORTANT: `npm ci` on TOP of a corrupted node_modules fails on this server
  # (ENOTEMPTY / leaves next/dist/bin/next missing). MUST rm -rf node_modules first
  # so npm ci does a clean install (this is what the proven _recover_fe2.sh does).
  echo "[RELAND] node_modules corruption detected -> cache clean + rm -rf node_modules + npm ci..." >> "$LOG"
  # The npm CACHE itself gets corrupt on this box — `rm -rf node_modules && npm ci`
  # alone kept producing an INCOMPLETE install (next/dist/bin/next missing -> pm2
  # MODULE_NOT_FOUND -> 502). `npm cache clean --force` first is what actually fixes it.
  npm cache clean --force >> "$LOG" 2>&1
  rm -rf node_modules
  npm ci >> "$LOG" 2>&1
  [ -f node_modules/next/dist/bin/next ] || { echo "[RELAND] next bin still missing -> npm install" >> "$LOG"; npm install >> "$LOG" 2>&1; }
  echo "[RELAND] rebuild after clean npm ci..." >> "$LOG"
  build
fi

if [ ! -f .next/BUILD_ID ]; then
  echo "[RELAND] BUILD FAILED" >> "$LOG"
  tail -25 "$LOG"
  exit 3
fi

pm2 start 3dmap-frontend >/dev/null 2>&1; pm2 save >/dev/null 2>&1
c=000
for i in $(seq 1 40); do
  c=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:3000/ 2>/dev/null)
  [ "$c" = 200 ] && break
  sleep 1
done
echo "[RELAND] DONE code=$c BUILD_ID=$(cat .next/BUILD_ID) after ${i}s" >> "$LOG"
