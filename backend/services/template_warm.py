"""perf-2026-09 · Нічний прогрів кешу для 9 шаблонних плиток головної сторінки.

Чому: головна лінкує на /create?template=<id> — 9 популярних пресетів. Перший
відвідувач, що генерує ТОЧНО дефолтні параметри шаблону, чекає повний час
генерації (як і будь-хто інший); result_cache (services/result_cache.py) кешує
готовий файл на RESULT_CACHE_TTL_H (72г), тож якщо прогріти кеш заздалегідь —
уся наступна доба відвідувачів того шаблону отримує миттєвий HIT.

Потік:
  1. Фронт при генерації ТОЧНО дефолтних параметрів шаблону додає в body
     `template_id`. main.generate_model (при створенні задачі) фіксує
     СИРИЙ body ДО мутацій пайплайну на task.template_id/task.template_body.
  2. На успішному завершенні (main.generate_model_task, поруч з _rc.store)
     викликається persist_template_body(...) → мержиться в
     DATA_DIR/template_bodies.json (кап 40 записів, лише прогінь тіла).
  3. main._template_warm_loop раз на добу (TEMPLATE_WARM_HOUR_UTC, за
     замовчуванням 3:00 UTC) читає файл і для кожного шаблону: якщо кеш ще
     гарячий (result_cache.lookup) — пропускає; інакше шле сам собі
     POST /api/generate з тим самим body і чекає завершення (poll
     /api/status), щоб не спамити паралельно (rate-limit 5/хв на /api/generate
     і серіалізація gen_queue все одно допускають лише один активний прогін).

Публічний, тестований шар (без реальних HTTP/sleep) — run_template_warm();
_template_warm_loop у main.py лише підключає справжні urllib-виклики і
asyncio.sleep навколо цього шару.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Optional

DATA_DIR = Path(os.getenv("DATA_DIR") or "data").resolve()
BODIES_PATH = DATA_DIR / "template_bodies.json"
MAX_ENTRIES = 40

# Ключі, які НІКОЛИ не потрапляють у збережене тіло, навіть якщо колись
# з'являться в GenerationRequest (захист копіпасти/майбутніх полів).
_DENY_SUBSTRINGS = ("screenshot", "password", "token", "secret", "authorization", "auth_")


def _json_safe(value: Any, _depth: int = 0) -> Any:
    """Рекурсивно лишає лише JSON-примітиви; все інше (datetime, set, custom
    об'єкти) відкидається як None, щоб збережений body завжди був
    ре-постабельним чистим JSON."""
    if _depth > 6:
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                continue
            lk = k.lower()
            if any(bad in lk for bad in _DENY_SUBSTRINGS):
                continue
            out[k] = _json_safe(v, _depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, _depth + 1) for v in value]
    return None


def sanitize_body(body: Dict[str, Any]) -> Dict[str, Any]:
    """Публічна обгортка над _json_safe для тіла запиту (top-level dict)."""
    safe = _json_safe(dict(body or {}))
    return safe if isinstance(safe, dict) else {}


def load_template_bodies() -> Dict[str, Dict[str, Any]]:
    try:
        if BODIES_PATH.exists():
            data = json.loads(BODIES_PATH.read_text(encoding="utf-8")) or {}
            return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001
        print(f"[TEMPLATE_WARM] load failed (ignored): {exc}")
    return {}


def _save_template_bodies(data: Dict[str, Dict[str, Any]]) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = BODIES_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(BODIES_PATH)
    except Exception as exc:  # noqa: BLE001
        print(f"[TEMPLATE_WARM] save failed (ignored): {exc}")


def persist_template_body(template_id: str, body: Dict[str, Any], cache_key: Optional[str]) -> None:
    """Мержить/перезаписує запис шаблону. Кап MAX_ENTRIES — викидає
    найстаріший (за saved_at) при переповненні."""
    if not template_id or not isinstance(body, dict):
        return
    data = load_template_bodies()
    data[str(template_id)] = {
        "body": sanitize_body(body),
        "cache_key": cache_key,
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }
    if len(data) > MAX_ENTRIES:
        oldest = sorted(data.items(), key=lambda kv: kv[1].get("saved_at", ""))
        for k, _v in oldest[: len(data) - MAX_ENTRIES]:
            data.pop(k, None)
    _save_template_bodies(data)


def next_warm_delay(now: datetime, hour: int) -> float:
    """Секунд до наступного настання `hour:00:00` UTC (сьогодні, якщо ще не
    минуло; інакше завтра). `hour` очікується 0-23 (виклик з hour<0 —
    вимкнено, перевіряється на боці виклику ДО цього)."""
    hour = max(0, min(23, int(hour)))
    target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def run_template_warm(
    post_fn: Callable[[Dict[str, Any]], Optional[str]],
    status_fn: Callable[[str], Optional[str]],
    lookup_fn: Callable[[str], Optional[Dict[str, Any]]],
    bodies: Dict[str, Dict[str, Any]],
    sleep_fn: Callable[[float], None] = time.sleep,
    poll_interval_s: float = 10.0,
    poll_timeout_s: float = 900.0,
    between_delay_s: float = 20.0,
) -> list[tuple[str, str]]:
    """Проганяє список збережених шаблонів. Ніколи не кидає назовні — кожен
    шаблон обгорнутий у try, помилка одного не зупиняє інші. Повертає
    [(template_id, outcome)] де outcome — "skip_cached" | "completed" |
    "failed" | "cancelled" | "timeout" | "error:<msg>"."""
    results: list[tuple[str, str]] = []
    items = list((bodies or {}).items())
    for idx, (template_id, entry) in enumerate(items):
        try:
            cache_key = (entry or {}).get("cache_key")
            body = (entry or {}).get("body")
            if cache_key and lookup_fn(cache_key):
                results.append((template_id, "skip_cached"))
                print(f"[TEMPLATE_WARM] {template_id}: cache still hot — skip")
                continue
            if not isinstance(body, dict):
                results.append((template_id, "error:no_body"))
                continue
            task_id = post_fn(body)
            if not task_id:
                results.append((template_id, "error:no_task_id"))
                continue
            waited = 0.0
            status: Optional[str] = None
            while waited < poll_timeout_s:
                try:
                    status = status_fn(task_id)
                except Exception as exc:  # noqa: BLE001
                    status = None
                    print(f"[TEMPLATE_WARM] {template_id}: status poll failed (ignored): {exc}")
                if status in ("completed", "failed", "cancelled"):
                    break
                sleep_fn(poll_interval_s)
                waited += poll_interval_s
            outcome = status if status in ("completed", "failed", "cancelled") else "timeout"
            results.append((template_id, outcome))
            print(f"[TEMPLATE_WARM] {template_id}: {outcome} (task={task_id}, waited={waited:.0f}s)")
        except Exception as exc:  # noqa: BLE001
            print(f"[TEMPLATE_WARM] {template_id}: unexpected error (non-fatal): {exc}")
            results.append((template_id, f"error:{exc}"))
        # Rate-limit /api/generate = 5/хв → чекаємо між шаблонами (не після
        # останнього, щоб не тримати процес довше за потрібне).
        if idx < len(items) - 1:
            sleep_fn(between_delay_s)
    return results
