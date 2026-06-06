"""Memory-aware concurrency gate for model generation.

The backend is a single process on a small VPS (≈3.8GB RAM). A heavy terrain
generation peaks several GB, so only ONE may run at a time; light jobs (flat
maps / keychains, no terrain) are cheap enough to run a few in parallel.

We model this with a weighted counting semaphore: there are CAPACITY slots; a
heavy job takes all of them (so it runs alone), a light job takes one (so a few
run concurrently). Concurrent FastAPI background tasks — whether from different
users or a multi-zone grid — all pass through this gate, which prevents the
out-of-memory restarts that previously killed in-flight generations.

All knobs are env-overridable:
  GEN_CAPACITY      total slots                 (default 2)
  GEN_HEAVY_WEIGHT  slots a terrain job takes    (default = CAPACITY → alone)
  GEN_LIGHT_WEIGHT  slots a light job takes      (default 1)
"""
from __future__ import annotations

import os
import threading
import time

CAPACITY = max(1, int(os.getenv("GEN_CAPACITY", "2")))
HEAVY_WEIGHT = max(1, int(os.getenv("GEN_HEAVY_WEIGHT", str(CAPACITY))))
LIGHT_WEIGHT = max(1, int(os.getenv("GEN_LIGHT_WEIGHT", "1")))

_cond = threading.Condition()
_used = 0  # slots currently in use


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


def acquire(weight: int) -> float:
    """Block until `weight` slots are free. Returns seconds spent waiting.

    Acquiring all `weight` slots atomically (under the lock) avoids the
    multi-permit deadlock two heavy jobs would hit with a plain semaphore."""
    global _used
    weight = max(1, min(weight, CAPACITY))
    t0 = time.time()
    with _cond:
        while _used + weight > CAPACITY:
            _cond.wait(timeout=5.0)
        _used += weight
    return time.time() - t0


def release(weight: int) -> None:
    global _used
    weight = max(1, min(weight, CAPACITY))
    with _cond:
        _used = max(0, _used - weight)
        _cond.notify_all()


def stats() -> dict:
    with _cond:
        return {"capacity": CAPACITY, "used": _used, "free": CAPACITY - _used}
