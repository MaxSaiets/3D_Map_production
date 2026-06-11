#!/usr/bin/env bash
# Enable Cloudflare Authenticated Origin Pulls (mTLS) on the Caddy side.
#
# PREREQUISITE: enable Authenticated Origin Pulls in the Cloudflare dashboard
#   SSL/TLS -> Origin Server -> Authenticated Origin Pulls  (zone-wide toggle ON)
# BEFORE running this. Otherwise Cloudflare won't present a client cert and
# Caddy will reject every request.
#
# This script is SAFE: it validates, reloads, then probes the public site.
# If the probe fails it automatically rolls back and reloads the old config,
# so a misconfiguration cannot leave the site down.

set -u
CADDYFILE="/etc/caddy/Caddyfile"
CA="/etc/caddy/ssl/cloudflare-origin-pull-ca.pem"
BACKUP="/etc/caddy/Caddyfile.pre-mtls.$(date +%s)"
PROBE_URL="https://monadruk.com/api/health"

[ -f "$CA" ] || { echo "FAIL: CA cert missing at $CA"; exit 1; }

if grep -q "client_auth" "$CADDYFILE"; then
  echo "client_auth already present in Caddyfile — nothing to do."
  exit 0
fi

cp "$CADDYFILE" "$BACKUP"
echo "Backed up Caddyfile -> $BACKUP"

# Replace the single-line tls directive with a block that adds client_auth.
python3 - "$CADDYFILE" "$CA" <<'PY'
import sys, re
path, ca = sys.argv[1], sys.argv[2]
s = open(path).read()
old = "\ttls /etc/caddy/ssl/origin-cert.pem /etc/caddy/ssl/origin-key.pem"
new = (
    "\ttls /etc/caddy/ssl/origin-cert.pem /etc/caddy/ssl/origin-key.pem {\n"
    "\t\tclient_auth {\n"
    "\t\t\tmode require_and_verify\n"
    f"\t\t\ttrust_pool file {ca}\n"
    "\t\t}\n"
    "\t}"
)
if old not in s:
    print("WARN: expected tls line not found; aborting edit", file=sys.stderr); sys.exit(2)
s = s.replace(old, new, 1)
open(path, "w").write(s)
print("Patched tls directive with client_auth")
PY
[ $? -eq 0 ] || { echo "Patch failed; restoring."; cp "$BACKUP" "$CADDYFILE"; exit 1; }

rollback() {
  echo "ROLLBACK: restoring previous Caddyfile"
  cp "$BACKUP" "$CADDYFILE"
  caddy reload --config "$CADDYFILE" >/dev/null 2>&1 || systemctl reload caddy
  echo "Rolled back."
}

echo "--- caddy validate ---"
if ! caddy validate --config "$CADDYFILE"; then rollback; exit 1; fi

echo "--- reload ---"
caddy reload --config "$CADDYFILE" 2>/dev/null || systemctl reload caddy
sleep 4

echo "--- probe public site ---"
code=$(curl -s --max-time 15 -o /dev/null -w "%{http_code}" "$PROBE_URL")
if [ "$code" = "200" ]; then
  echo "OK mTLS enabled and site healthy (http $code). Backup kept at $BACKUP"
else
  echo "Probe failed (http $code) — likely CF Authenticated Origin Pulls not enabled."
  rollback
  exit 1
fi
