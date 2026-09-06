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
from typing import Any, Dict, List, Optional

OUTPUT_DIR = Path("output").resolve()
# БЕЗПЕКА: users.json (емейли/квоти/історія) НЕ можна в OUTPUT_DIR — він віддається
# статикою на /files. Тримаємо у DATA_DIR (не монтується назовні).
DATA_DIR = Path("data").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
USERS_FILE = DATA_DIR / "users.json"
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


def register_download(uid: str, email: str, is_admin: bool, task_id: str = "") -> Dict[str, Any]:
    """Increment download count if allowed. Re-downloading a task_id that was ALREADY
    charged does NOT burn another free download (втрата файлу / повторний клік не з'їдає
    квоту → користувач реально отримує 5 РІЗНИХ моделей). Returns {ok, quota}."""
    with _lock:
        data = _load()
        u = _get(data, uid, email)
        used = int(u.get("downloads", 0))
        downloaded = u.get("downloaded", [])
        already = bool(task_id) and task_id in downloaded
        if not is_admin and used >= FREE_DOWNLOADS and not already:
            return {"ok": False, "reason": "limit", "quota": {
                "downloads": used, "limit": FREE_DOWNLOADS, "remaining": 0,
                "is_admin": False, "can_download": False,
            }}
        if not already:
            u["downloads"] = used + 1
            if task_id:
                downloaded.append(task_id)
                u["downloaded"] = downloaded[-500:]
            _save(data)
        cur = int(u.get("downloads", 0))
        return {"ok": True, "quota": {
            "downloads": cur, "limit": FREE_DOWNLOADS,
            "remaining": (10**9 if is_admin else max(0, FREE_DOWNLOADS - cur)),
            "is_admin": is_admin, "can_download": is_admin or cur < FREE_DOWNLOADS,
        }}


# Регенерація моделі з історії акаунта: тримаємо лише ЦІ ключі (примітивні
# значення, рядки ≤80 символів) — /api/account/download приймає `params` від
# клієнта, тож НІКОЛИ не довіряємо йому напряму перед записом у users.json.
_PARAMS_ALLOWED_KEYS = (
    "lat", "lon", "size_mm", "scenario", "product", "relief", "label",
    "north", "south", "east", "west",
)


def sanitize_model_params(params: Any) -> Optional[Dict[str, Any]]:
    """Filter a client-supplied regenerate-params dict down to the known
    whitelist with bounded primitive values. Returns None if nothing valid
    survives (unknown dict, wrong type, or every key dropped)."""
    if not isinstance(params, dict):
        return None
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if k not in _PARAMS_ALLOWED_KEYS:
            continue
        if isinstance(v, bool) or v is None or isinstance(v, (int, float)):
            out[k] = v
        elif isinstance(v, str) and len(v) <= 80:
            out[k] = v
        # інші типи (dict/list/...) і задовгі рядки — мовчки відкидаємо
    return out or None


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
        entry = {**model, "ts": int(time.time())}
        if "params" in entry:
            sanitized = sanitize_model_params(entry.get("params"))
            if sanitized:
                entry["params"] = sanitized
            else:
                entry.pop("params", None)
        models.insert(0, entry)
        u["models"] = models[:100]
        _save(data)


def _model_file_path(model: Dict[str, Any]) -> Path | None:
    """Resolve a stored model entry to its on-disk path (best effort)."""
    name = ""
    for key in ("download_url", "file"):
        v = model.get(key)
        if v:
            name = str(v).split("/")[-1].split("?")[0]
            if name:
                break
    if not name:
        return None
    return OUTPUT_DIR / name


def list_models(uid: str) -> List[Dict[str, Any]]:
    """User's saved models, most-recent-first. Entries whose backing file no
    longer exists on disk (retention cleanup, manual deletion, restart loss)
    are flagged `expired: True` — NOT removed, so history stays intact."""
    with _lock:
        models = list(_load().get(uid, {}).get("models", []))
    out = []
    for m in models:
        p = _model_file_path(m)
        entry = dict(m)
        if p is not None and not p.exists():
            entry["expired"] = True
        out.append(entry)
    return out


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


def delete_user(uid: str) -> Dict[str, Any]:
    """GDPR-style account deletion: removes the user's generated model files
    from disk, drops their grids/models/quota record from users.json.

    Orders are intentionally NOT touched (accounting retention — orders.jsonl
    keeps its own record independent of the account). Returns
    {"deleted_models": n, "deleted_files": m}."""
    import shutil as _shutil

    with _lock:
        data = _load()
        u = data.get(uid)
        if not u:
            return {"deleted_models": 0, "deleted_files": 0}

        models: List[Dict[str, Any]] = u.get("models", [])
        deleted_models = len(models)

        # Collect every task-id-ish token referenced by this user's models so we
        # can also catch on-disk siblings (e.g. <task>_print_*.zip) that aren't
        # directly named by download_url/file.
        tokens: set[str] = set()
        for m in models:
            tid = str(m.get("task_id") or "").strip()
            if tid:
                tokens.add(tid.lower())
                tokens.add(tid.replace("-", "").lower())
            p = _model_file_path(m)
            if p is not None:
                tokens.add(p.stem.lower())

        deleted_files = 0
        seen_paths: set[Path] = set()
        for m in models:
            p = _model_file_path(m)
            if p is not None and p.exists() and p not in seen_paths:
                seen_paths.add(p)
                try:
                    p.unlink()
                    deleted_files += 1
                except OSError:
                    pass

        if tokens and OUTPUT_DIR.exists():
            try:
                for entry in OUTPUT_DIR.iterdir():
                    if entry.name == "previews":
                        continue
                    lname = entry.name.lower()
                    if entry in seen_paths:
                        continue
                    if any(tok and tok in lname for tok in tokens):
                        seen_paths.add(entry)
                        try:
                            if entry.is_dir():
                                _shutil.rmtree(entry, ignore_errors=True)
                            else:
                                entry.unlink()
                            deleted_files += 1
                        except OSError:
                            pass
            except OSError:
                pass
            # Share OG-preview PNGs live under output/previews/<task_id>.png.
            previews_dir = OUTPUT_DIR / "previews"
            if previews_dir.exists():
                try:
                    for entry in previews_dir.iterdir():
                        if entry.suffix.lower() != ".png":
                            continue
                        if entry.stem.lower() in tokens:
                            try:
                                entry.unlink()
                                deleted_files += 1
                            except OSError:
                                pass
                except OSError:
                    pass

        data.pop(uid, None)
        _save(data)
        return {"deleted_models": deleted_models, "deleted_files": deleted_files}


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
