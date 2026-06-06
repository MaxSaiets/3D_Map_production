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


import uuid as _uuid


def save_grid(uid: str, email: str, grid: Dict[str, Any]) -> Dict[str, Any]:
    """Create or update a user's city grid (per-account, server-side).

    A grid groups a tiled city area so the user can later reopen it from history
    and generate neighbouring cells (hexagons/squares). Upserts by grid['id'];
    a new id is minted when missing."""
    with _lock:
        data = _load()
        u = _get(data, uid, email)
        grids: List[Dict[str, Any]] = u.get("grids", [])
        gid = str(grid.get("id") or "").strip() or f"grid_{_uuid.uuid4().hex[:12]}"
        grid = {**grid, "id": gid}
        now = int(time.time())
        existing_idx = next((i for i, g in enumerate(grids) if g.get("id") == gid), None)
        if existing_idx is not None:
            grid["created_at"] = grids[existing_idx].get("created_at", now)
            grid["updated_at"] = now
            # preserve already-recorded generated cells unless caller supplies them
            if "cells" not in grid:
                grid["cells"] = grids[existing_idx].get("cells", [])
            grids[existing_idx] = grid
        else:
            grid.setdefault("created_at", now)
            grid["updated_at"] = now
            grid.setdefault("cells", [])
            grids.insert(0, grid)
        u["grids"] = grids[:50]
        _save(data)
        return grid


def list_grids(uid: str) -> List[Dict[str, Any]]:
    with _lock:
        return list(_load().get(uid, {}).get("grids", []))


def get_grid(uid: str, grid_id: str) -> Dict[str, Any] | None:
    with _lock:
        for g in _load().get(uid, {}).get("grids", []):
            if g.get("id") == grid_id:
                return g
    return None


def delete_grid(uid: str, grid_id: str) -> bool:
    with _lock:
        data = _load()
        u = data.get(uid)
        if not u:
            return False
        grids = u.get("grids", [])
        new = [g for g in grids if g.get("id") != grid_id]
        if len(new) == len(grids):
            return False
        u["grids"] = new
        _save(data)
        return True


def mark_grid_cell(uid: str, grid_id: str, cell: Dict[str, Any]) -> Dict[str, Any] | None:
    """Record that a grid cell was generated (keyed by row/col). Returns the grid."""
    with _lock:
        data = _load()
        u = data.get(uid)
        if not u:
            return None
        for g in u.get("grids", []):
            if g.get("id") != grid_id:
                continue
            cells: List[Dict[str, Any]] = g.get("cells", [])
            r, c = cell.get("row"), cell.get("col")
            idx = next((i for i, x in enumerate(cells)
                        if x.get("row") == r and x.get("col") == c), None)
            entry = {**cell, "ts": int(time.time())}
            if idx is not None:
                cells[idx] = {**cells[idx], **entry}
            else:
                cells.append(entry)
            g["cells"] = cells
            g["updated_at"] = int(time.time())
            _save(data)
            return g
    return None


def list_all_users() -> List[Dict[str, Any]]:
    """Admin view: every user with their email, download count and model count."""
    with _lock:
        data = _load()
    out = []
    for uid, u in data.items():
        out.append({
            "uid": uid,
            "email": u.get("email", ""),
            "downloads": int(u.get("downloads", 0)),
            "models": len(u.get("models", [])),
            "created_at": u.get("created_at"),
        })
    out.sort(key=lambda x: x.get("created_at") or 0, reverse=True)
    return out
