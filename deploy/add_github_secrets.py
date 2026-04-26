import os
"""
Додає GitHub Secrets до репозиторію через GitHub API.
Використовує PyNaCl для шифрування (вимога GitHub).
"""
import base64
import json
import sys
import urllib.request
import urllib.error

try:
    from nacl import encoding, public
except ImportError:
    print("ERROR: pip install pynacl")
    sys.exit(1)

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")  # export GITHUB_TOKEN=ghp_...
REPO = "MaxSaiets/3D_Map_production"
API = "https://api.github.com"

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "Content-Type": "application/json",
}


def gh(method, path, data=None):
    url = f"{API}{path}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=HEADERS, method=method)
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        print(f"  HTTP {e.code}: {err}")
        raise


def encrypt_secret(public_key_b64: str, secret_value: str) -> str:
    pk = public.PublicKey(public_key_b64.encode(), encoding.Base64Encoder())
    box = public.SealedBox(pk)
    encrypted = box.encrypt(secret_value.encode())
    return base64.b64encode(encrypted).decode()


def set_secret(name: str, value: str):
    print(f"  Getting repo public key...")
    key_info = gh("GET", f"/repos/{REPO}/actions/secrets/public-key")
    key_id = key_info["key_id"]
    key_b64 = key_info["key"]

    print(f"  Encrypting secret '{name}'...")
    encrypted = encrypt_secret(key_b64, value)

    print(f"  Uploading secret '{name}'...")
    gh("PUT", f"/repos/{REPO}/actions/secrets/{name}", {
        "encrypted_value": encrypted,
        "key_id": key_id,
    })
    print(f"  OK Secret '{name}' saved.")


# ── Читаємо SSH приватний ключ ────────────────────────────────
with open(r"C:\Users\sayet\.ssh\id_ed25519", "r") as f:
    ssh_private_key = f.read().strip()

# ── Secrets ──────────────────────────────────────────────────
secrets = {
    "SERVER_IP":       "209.38.210.197",
    "SERVER_USER":     "root",
    "SSH_PRIVATE_KEY": ssh_private_key,
}

print(f"Adding {len(secrets)} secrets to {REPO}...")
for name, value in secrets.items():
    try:
        set_secret(name, value)
    except Exception as e:
        print(f"  FAILED: {e}")

print("\nDone.")
