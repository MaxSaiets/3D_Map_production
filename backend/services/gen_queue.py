"""Memory-aware concurrency gate for model generation.

The backend is a single process on a small VPS (≈3.8GB RAM). A heavy terrain
generation peaks several GB, so only ONE may run at a time; light jobs (flat
maps / keychains, no terrain) are cheap enough to run a few in parallel.

We model this with a weighted counting semaphore: there are CAPACITY slots; a
heavy job takes all of them (so it runs alone), a light job takes one (so a few
run concurrently). Concurrent FastAPI background tasks — whether from different
users or a multi-zone grid — all pass through this gate, which prevents the
out-of-memory restarts that previously killed in-flight generations.

IMPORTANT — why the default is serial (CAPACITY=1):
The generation pipeline is CPU-bound *Python* running as threads inside a single
process, so the GIL pins it to one core regardless of how many jobs we start.
Running two at once just halves each one's speed (and doubles memory) — net
throughput is no better and latency is worse. Measured on the 2-vCPU VPS: two
concurrent jobs both stalled. So we default to a strict FIFO queue: one job at a
time at full speed (its core + the Blender boolean subprocess on the second
core), the rest wait as "queued". This also makes OOM impossible.
Bump GEN_CAPACITY only on a box with more cores AND RAM headroom.

All knobs are env-overridable:
  GEN_CAPACITY      total slots                 (default 1 → strict serial)
  GEN_HEAVY_WEIGHT  slots a terrain job takes    (default = CAPACITY → alone)
  GEN_LIGHT_WEIGHT  slots a light job takes      (default 1)
"""
from __future__ import annotations

import os
import threading
import time

CAPACITY = max(1, int(os.getenv("GEN_CAPACITY", "1")))
HEAVY_WEIGHT = max(1, int(os.getenv("GEN_HEAVY_WEIGHT", str(CAPACITY))))
LIGHT_WEIGHT = max(1, int(os.getenv("GEN_LIGHT_WEIGHT", "1")))

_cond = threading.Condition()
_used = 0  # slots currently in use

# ⭐09.09.2026: що САМЕ зараз рахується — щоб той, хто стоїть у черзі, бачив
# правду, а не «кілька хвилин». Прод-заміри 08.09: п'ять задач print:150 підряд,
# очікування 546 / 1050 / 1433 / 1812 / 2219 / 2609 с. Останній чекав 43 хвилини
# під написом «Ваша черга настане за кілька хвилин».
# Список (вид задачі, час старту). При CAPACITY=1 у ньому завжди не більше
# одного запису; на більшій ємності точність оцінки не критична — це підказка,
# а не обіцянка.
_running: list[tuple[str, float]] = []


def weight_for_request(request) -> int:
    """Heavy (terrain) jobs take all slots; everything else is light.

    Preview / flat-plate runs are light even with terrain because they use a
    tiny low-res heightmap (see generate_model_task PREVIEW_MODE trimming)."""
    try:
        if bool(getattr(request, "preview_mode", False)) or bool(
            getattr(request, "flat_plate_mode", False)
        ):
            return LIGHT_WEIGHT
        terrain = bool(getattr(request, "terrain_enabled", False))
    except Exception:
        terrain = False
    return HEAVY_WEIGHT if terrain else LIGHT_WEIGHT


def would_block(weight: int) -> bool:
    weight = max(1, min(weight, CAPACITY))
    with _cond:
        return _used + weight > CAPACITY


def acquire(weight: int, *, bucket: str = "", on_wait=None) -> float:
    """Block until `weight` slots are free. Returns seconds spent waiting.

    Acquiring all `weight` slots atomically (under the lock) avoids the
    multi-permit deadlock two heavy jobs would hit with a plain semaphore.

    `bucket` — вид задачі (`result_cache.eta_bucket`), щоб інші могли оцінити,
    скільки ще чекати. `on_wait(waited_s, eta_free_s)` викликається приблизно
    раз на 5 с, поки чекаємо: місце для оновлення статусу задачі.

    ⚠️ `on_wait` викликається БЕЗ утримання блокування — тож усередині можна
    спокійно звертатись і до самої черги."""
    global _used
    weight = max(1, min(weight, CAPACITY))
    t0 = time.time()
    with _cond:
        while _used + weight > CAPACITY:
            _cond.wait(timeout=5.0)
            if on_wait is not None and _used + weight > CAPACITY:
                # Колбек — поза блокуванням: він оновлює статус задачі і може
                # сам звернутись до черги.
                _cond.release()
                try:
                    on_wait(time.time() - t0, eta_free_s())
                except Exception:  # noqa: BLE001
                    pass
                finally:
                    _cond.acquire()
        _used += weight
        _running.append((str(bucket or ""), time.time()))
    return time.time() - t0


def release(weight: int) -> None:
    global _used
    weight = max(1, min(weight, CAPACITY))
    with _cond:
        _used = max(0, _used - weight)
        if _running:
            _running.pop(0)          # найстаріша з тих, що рахуються
        _cond.notify_all()


def eta_free_s() -> int | None:
    """Скільки ще секунд до звільнення слота. None — коли оцінити нічим.

    Беремо задачу, яка має завершитись найраніше: p75 для її виду мінус те, що
    вже минуло. Ніколи не обіцяємо «ось-ось»: якщо оцінка вже вичерпана, а
    задача ще йде, повертаємо 30 с — краще трохи недообіцяти, ніж показати нуль
    і залишити людину дивитись на завмерлий лічильник."""
    with _cond:
        snapshot = list(_running)
    if not snapshot:
        return None
    try:
        from services.result_cache import eta_seconds
    except Exception:  # noqa: BLE001
        return None
    now = time.time()
    best: int | None = None
    for bucket, started in snapshot:
        if not bucket:
            continue
        try:
            total = int(eta_seconds(bucket))
        except Exception:  # noqa: BLE001
            continue
        left = max(30, total - int(now - started))
        best = left if best is None else min(best, left)
    return best


def stats() -> dict:
    with _cond:
        return {"capacity": CAPACITY, "used": _used, "free": CAPACITY - _used,
                "running": len(_running)}


def reset_for_tests() -> None:
    global _used
    with _cond:
        _used = 0
        _running.clear()
        _cond.notify_all()
