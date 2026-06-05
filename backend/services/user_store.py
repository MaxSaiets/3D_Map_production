"""Tiny JSON-backed user store: download quota + saved models per Firebase uid.

Regular users get FREE_DOWNLOADS free full-model downloads; beyond that they
must pay / contact us. Admins are unlimited. No external DB — a single JSON
file keyed by uid is plenty at this scale and survives restarts.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

OUTPUT_DIR = Path("output").resolve()
USERS_FILE = OUTPUT_DIR / "users.json"
FREE_DOWNLOADS = int(os.getenv("FREE_DOWNLOADS", "5"))

_lock = threading.Lock()


def _load() -> Dict[str, Any]:
    try:
        if USERS_FILE.exists():
            return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    return {}


def _save(data: Dict[str, Any]) -> None:
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        tmp = USERS_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=0), encoding="utf-8")
        tmp.replace(USERS_FILE)
    except Exception as e:  # noqa: BLE001
        print(f"[USERS] save failed: {e}")


def _get(data: Dict[str, Any], uid: str, email: str = "") -> Dict[str, Any]:
    u = data.get(uid)
    if not u:
        u = {"email": email, "downloads": 0, "models": [], "created_at": int(time.time())}
        data[uid] = u
    if email and not u.get("email"):
        u["email"] = email
    return u


def get_quota(uid: str, email: str, is_admin: bool) -> Dict[str, Any]:
    with _lock:
        data = _load()
        u = _get(data, uid, email)
        _save(data)
        used = int(u.get("downloads", 0))
        return {
            "downloads": used,
            "limit": FREE_DOWNLOADS,
            "remaining": (10**9 if is_admin else max(0, FREE_DOWNLOADS - used)),
            "is_admin": is_admin,
            "can_download": is_admin or used < FREE_DOWNLOADS,
        }


def register_download(uid: str, email: str, is_admin: bool) -> Dict[str, Any]:
    """Increment download count if allowed. Returns {ok, quota}."""
    with _lock:
        data = _load()
        u = _get(data, uid, email)
        used = int(u.get("downloads", 0))
        if not is_admin and used >= FREE_DOWNLOADS:
            return {"ok": False, "reason": "limit", "quota": {
                "downloads": used, "limit": FREE_DOWNLOADS, "remaining": 0,
                "is_admin": False, "can_download": False,
            }}
        u["downloads"] = used + 1
        _save(data)
        return {"ok": True, "quota": {
            "downloads": u["downloads"], "limit": FREE_DOWNLOADS,
            "remaining": (10**9 if is_admin else max(0, FREE_DOWNLOADS - u["downloads"])),
            "is_admin": is_admin, "can_download": is_admin or u["downloads"] < FREE_DOWNLOADS,
        }}


def add_model(uid: str, email: str, model: Dict[str, Any]) -> None:
    """Associate a generated model with the user (most-recent-first, capped)."""
    if not model.get("task_id"):
        return
    with _lock:
        data = _load()
        u = _get(data, uid, email)
        models: List[Dict[str, Any]] = u.get("models", [])
        if any(m.get("task_id") == model["task_id"] for m in models):
            return
        models.insert(0, {**model, "ts": int(time.time())})
        u["models"] = models[:100]
        _save(data)


def list_models(uid: str) -> List[Dict[str, Any]]:
    with _lock:
        return list(_load().get(uid, {}).get("models", []))
