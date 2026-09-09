# -*- coding: utf-8 -*-
"""Оплачений доступ до друк-файлу конкретної моделі.

⭐Рішення власника 09.09.2026: файл для самодруку коштує 149 ₴.

Навіщо це окремо від квоти. Заміряно за 30 днів: модель створили 23 людини, і
лише 8 з України. Друк і доставка — тільки по Україні, тож дві третини тих, хто
робить усю роботу зі створення мапи, не мали що купити й забирали файл дарма.
Файл не має ні логістики, ні митниці — його можна продати куди завгодно.

Модель доступу навмисно проста і НЕ прив'язана до акаунта:
доступ дається **на конкретну задачу** (`task_id`), бо саме її людина оплатила.
Так само працює й для гостя, який купив, вийшов і повернувся за посиланням.

Журнал — той самий формат, що й решта даних проєкту: JSONL, дописування,
читання цілком. Записів очікується десятки на місяць, тож індекс не потрібен.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_DIR = Path("data").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
ACCESS_LOG = DATA_DIR / "file_access.jsonl"

_lock = threading.Lock()

__all__ = ["grant", "has_access", "list_for_email", "file_price_uah", "ACCESS_LOG"]


PRICING_PATH = Path("pricing.json").resolve()
DEFAULT_FILE_PRICE_UAH = 149


def file_price_uah() -> int:
    """Ціна друк-файлу: env → pricing.json → дефолт.

    `pricing.json` читаємо ПРЯМО (як `main._load_pricing`), а не через якийсь
    сервіс — модуля `services/pricing.py` у проєкті немає, і спроба імпорту була б
    вічно-мертвою гілкою, яка тихо повертає дефолт. Саме такий баг коштував
    два дні на цьому тижні."""
    raw = os.getenv("FILE_PRICE_UAH", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    try:
        with PRICING_PATH.open("r", encoding="utf-8") as fh:
            val = int(((json.load(fh) or {}).get("file") or {}).get("price") or 0)
        if val > 0:
            return val
    except Exception:  # noqa: BLE001
        pass
    return DEFAULT_FILE_PRICE_UAH


def grant(task_id: str, *, order_number: str = "", email: str = "", amount: Any = None) -> bool:
    """Відкриває доступ до файлу задачі. Ідемпотентно: повторний виклик не дублює.

    Повертає True, якщо доступ записано щойно; False — якщо вже був або немає
    `task_id` (без нього відкривати нема чого)."""
    tid = str(task_id or "").strip()
    if not tid:
        return False
    with _lock:
        if _has_access_unlocked(tid):
            return False
        rec = {
            "task_id": tid,
            "order_number": str(order_number or ""),
            "email": str(email or "").strip().lower(),
            "amount": amount,
            "ts": int(time.time()),
        }
        try:
            ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
            with ACCESS_LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            print(f"[FILE_ACCESS] не вдалося записати доступ: {exc}", flush=True)
            return False
    print(f"[FILE_ACCESS] відкрито доступ до {tid} (замовлення {order_number or '—'})", flush=True)
    return True


def _records() -> List[Dict[str, Any]]:
    try:
        if not ACCESS_LOG.exists():
            return []
        out: List[Dict[str, Any]] = []
        for line in ACCESS_LOG.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue          # побитий рядок не має ламати доступ решті
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"[FILE_ACCESS] журнал не прочитався: {exc}", flush=True)
        return []


def _has_access_unlocked(task_id: str) -> bool:
    tid = str(task_id or "").strip()
    return bool(tid) and any(str(r.get("task_id") or "") == tid for r in _records())


def has_access(task_id: Optional[str]) -> bool:
    """Чи оплачено файл цієї задачі."""
    return _has_access_unlocked(str(task_id or ""))


def list_for_email(email: str) -> List[str]:
    """Задачі, файли яких купила ця пошта — для сторінки «мої моделі»."""
    target = str(email or "").strip().lower()
    if not target:
        return []
    return [str(r.get("task_id")) for r in _records()
            if str(r.get("email") or "").strip().lower() == target and r.get("task_id")]
