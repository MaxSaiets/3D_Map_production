"""Ручні гранти безліміту з адмінки (06.10.2026). Те саме, що env UNLIMITED_EMAILS,
але без правки .env і рестарту: власник у /admin вводить пошту або Firebase uid,
строк (до дати включно, за Києвом) або «безстроково», нотатку — і людина одразу
отримує безлім завантажень/генерацій (auth_service.has_unlimited_grant → quota_unlimited).

Сховище — data/grants.json; відкликання не видаляє запис, а ставить revoked_at
(історія видач лишається)."""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_FILE = Path("data").resolve() / "grants.json"
_LOCK = threading.Lock()
_cache: Dict[str, Any] = {"mtime": None, "grants": []}


def _today_kyiv() -> str:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Europe/Kyiv")).date().isoformat()
    except Exception:  # noqa: BLE001
        return time.strftime("%Y-%m-%d", time.gmtime())


def _load() -> List[Dict[str, Any]]:
    try:
        return json.loads(DATA_FILE.read_text(encoding="utf-8")).get("grants", [])
    except Exception:  # noqa: BLE001
        return []


def _save(grants: List[Dict[str, Any]]) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = DATA_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"grants": grants}, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, DATA_FILE)


def _cached() -> List[Dict[str, Any]]:
    try:
        mt = DATA_FILE.stat().st_mtime
    except FileNotFoundError:
        return []
    if _cache["mtime"] != mt:
        _cache.update(mtime=mt, grants=_load())
    return _cache["grants"]


def _is_live(g: Dict[str, Any], today: str) -> bool:
    return not g.get("revoked_at") and (not g.get("until") or today <= g["until"])


def is_granted(identifier: Optional[str]) -> bool:
    """identifier = email (без урахування регістру) або Firebase uid."""
    if not identifier:
        return False
    key = identifier.strip().lower()
    today = _today_kyiv()
    return any(g.get("who", "").lower() == key and _is_live(g, today) for g in _cached())


def add(who: str, until: Optional[str], note: str = "", by: str = "") -> Dict[str, Any]:
    who = (who or "").strip()
    if not who or len(who) > 200:
        raise ValueError("Вкажіть пошту або uid")
    if until:
        datetime.strptime(until, "%Y-%m-%d")  # ValueError на кривій даті
        if until < _today_kyiv():
            raise ValueError("Дата вже минула")
    g = {
        "id": uuid.uuid4().hex[:12], "who": who, "until": until or None,
        "note": (note or "")[:300], "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "created_by": by or "", "revoked_at": None,
    }
    with _LOCK:
        grants = _load()
        grants.append(g)
        _save(grants)
    return g


def revoke(grant_id: str, by: str = "") -> Optional[Dict[str, Any]]:
    with _LOCK:
        grants = _load()
        for g in grants:
            if g.get("id") == grant_id and not g.get("revoked_at"):
                g["revoked_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                g["revoked_by"] = by or ""
                _save(grants)
                return g
    return None


def list_all() -> List[Dict[str, Any]]:
    today = _today_kyiv()
    out = [{**g, "active": _is_live(g, today)} for g in _load()]
    out.sort(key=lambda g: g.get("created_at") or "", reverse=True)  # нові зверху
    out.sort(key=lambda g: not g["active"])                          # діючі першими (стабільно)
    return out
