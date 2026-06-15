"""Verify Firebase ID tokens WITHOUT a service account.

Firebase ID tokens are RS256 JWTs signed by Google. We validate them against
Google's public x509 certs (cached ~1h), checking signature, audience (project
id), issuer and expiry. This avoids shipping any Firebase Admin secret.
"""
from __future__ import annotations

import os
import time
from typing import Optional, Dict

import jwt
import requests
from cryptography.x509 import load_pem_x509_certificate

PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "monadruk")
_CERTS_URL = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"
_ISSUER = f"https://securetoken.google.com/{PROJECT_ID}"

_certs_cache: Dict[str, object] = {}
_certs_fetched_at = 0.0
_CACHE_TTL = 3600


def _admin_emails() -> set[str]:
    raw = os.getenv("ADMIN_EMAILS", "")
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def is_admin(email: Optional[str]) -> bool:
    return bool(email) and email.lower() in _admin_emails()


def _get_certs() -> Dict[str, object]:
    global _certs_cache, _certs_fetched_at
    if _certs_cache and (time.time() - _certs_fetched_at) < _CACHE_TTL:
        return _certs_cache
    resp = requests.get(_CERTS_URL, timeout=10)
    resp.raise_for_status()
    certs = {}
    for kid, pem in resp.json().items():
        try:
            certs[kid] = load_pem_x509_certificate(pem.encode()).public_key()
        except Exception:  # noqa: BLE001
            continue
    _certs_cache = certs
    _certs_fetched_at = time.time()
    return certs


def verify_token(token: str) -> Optional[Dict[str, object]]:
    """Return {uid, email, email_verified, is_admin} or None if invalid."""
    if not token:
        return None
    token = token.replace("Bearer ", "").strip()
    try:
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        certs = _get_certs()
        key = certs.get(kid)
        if key is None:
            # cache might be stale — force refresh once
            _certs_cache.clear()
            key = _get_certs().get(kid)
        if key is None:
            return None
        claims = jwt.decode(
            token, key=key, algorithms=["RS256"],
            audience=PROJECT_ID, issuer=_ISSUER,
        )
        uid = claims.get("user_id") or claims.get("sub")
        if not uid:
            return None
        email = claims.get("email")
        email_verified = bool(claims.get("email_verified"))
        # БЕЗПЕКА: адмін-права лише для ПІДТВЕРДЖЕНОЇ пошти. Інакше зловмисник міг
        # зареєструватися з адмінським email через провайдера без верифікації
        # (або підмінити claim) і отримати доступ до /api/admin/*.
        return {
            "uid": uid,
            "email": email,
            "email_verified": email_verified,
            "is_admin": email_verified and is_admin(email),
            "name": claims.get("name"),
            "phone": claims.get("phone_number"),
        }
    except Exception as e:  # noqa: BLE001
        print(f"[AUTH] token verify failed: {e}")
        return None
