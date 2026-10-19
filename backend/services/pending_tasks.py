"""
Журнал незавершених задач генерації на диску.

Інцидент 19.10.2026: mem_guard перезапустив бекенд посеред друку 200 мм, а задача
клієнта, що стояла в черзі, жила лише в памʼяті процесу → після старту
/api/status віддавав 404 «Task not found», і модель на сайті так і не зʼявилась.

Тепер кожен запит /api/generate пишеться сюди (сирий body) і видаляється, коли
задача дійшла до кінця (готово / помилка / скасовано). Після рестарту бекенд
ставить ці задачі назад у чергу з ТИМ САМИМ task_id — фронт, що досі полить
статус, просто дочекається результату.

Захист від петлі: задача, яку вже відновлювали MAX_ATTEMPTS разів (тобто вона
сама могла валити процес), більше не перезапускається — віддаємо «failed».
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

MAX_ATTEMPTS = 2  # перша спроба + одне відновлення
MAX_AGE_S = 2 * 3600  # старші за 2 год ніхто вже не чекає

_dir: Optional[Path] = None


def init(data_dir: Path) -> None:
    global _dir
    _dir = Path(data_dir) / "pending_tasks"
    _dir.mkdir(parents=True, exist_ok=True)


def _path(task_id: str) -> Optional[Path]:
    if _dir is None or not task_id or "/" in task_id or "\\" in task_id:
        return None
    return _dir / f"{task_id}.json"


def save(task_id: str, body: Dict[str, Any], polygon_coords: Any = None, attempts: int = 1) -> None:
    p = _path(task_id)
    if p is None:
        return
    try:
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "task_id": task_id,
            "body": body,
            "polygon_coords": polygon_coords,
            "attempts": int(attempts),
            "created_ts": time.time(),
        }, ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(tmp, p)
    except Exception as exc:  # noqa: BLE001 - журнал не має ламати генерацію
        print(f"[PENDING] save {task_id[:8]} failed: {exc}")


def remove(task_id: str) -> None:
    p = _path(task_id)
    if p is None:
        return
    try:
        p.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[PENDING] remove {task_id[:8]} failed: {exc}")


def load_all() -> List[Dict[str, Any]]:
    """Усі записи, від найстаріших. Биті/прострочені видаляються тут же."""
    if _dir is None:
        return []
    out: List[Dict[str, Any]] = []
    now = time.time()
    for p in sorted(_dir.glob("*.json"), key=lambda x: x.stat().st_mtime):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            p.unlink(missing_ok=True)
            continue
        if now - float(rec.get("created_ts") or 0) > MAX_AGE_S:
            p.unlink(missing_ok=True)
            continue
        out.append(rec)
    return out
