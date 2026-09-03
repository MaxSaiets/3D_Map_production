"""perf-2026-09-03 · Кеш результатів генерації + статистика тривалості (ETA) +
перевірка покриття локальної OSM-БД.

Чому: на проді кожен повтор ідентичного запиту («Оновити превʼю», перезавантаження,
друга людина на тому ж місті/шаблоні, SEO-шаблони) рахувався з нуля 50–250 с —
кешу params→файл не існувало (main._make_export_basename додає ВИПАДКОВИЙ токен,
це захист від перебору URL, не хеш параметрів). ETA на фронті був вигаданим
(«зазвичай 1–2 хвилини» при реальних 4 хв за кордоном).

Що тут:
  * request_cache_key(request, zone_polygon_coords) — sha1 нормалізованого запиту
    (усі поля моделі, float → 6 знаків) + версія кешу RESULT_CACHE_VERSION.
  * lookup(key) / store(key, task) — persistent JSON у DATA_DIR (не webroot);
    запис валідний лише поки всі файли існують і не старший за RESULT_CACHE_TTL_H.
  * record_duration(bucket, s) / eta_seconds(bucket) — медіана останніх прогонів
    за відром (kind:size), fallback-константи з виміряного прод-профілю 2026-09.
  * within_local_coverage(n, s, e, w) — чи впаде bbox у ukraine.duckdb (інакше
    fetch_source піде в Overpass і триватиме ~4 хв — чесно кажемо це наперед).

Безпечно: жодних змін геометрії; кеш віддає ТОЙ САМИЙ файл, що вже згенеровано.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

_LOCK = threading.Lock()
_DATA_DIR = Path(os.getenv("DATA_DIR") or "data").resolve()
_CACHE_PATH = _DATA_DIR / "result_cache.json"
_STATS_PATH = _DATA_DIR / "gen_stats.json"
_CACHE_VERSION = os.getenv("RESULT_CACHE_VERSION", "1")
_TTL_S = float(os.getenv("RESULT_CACHE_TTL_H", "72")) * 3600.0
_MAX_ENTRIES = int(os.getenv("RESULT_CACHE_MAX", "400"))
_STATS_KEEP = 30

_cache: Dict[str, Dict[str, Any]] = {}
_stats: Dict[str, list] = {}
_loaded = False

# Виміряний прод-профіль (VM, 2 ядра) — fallback, поки нема історії у відрі.
_ETA_DEFAULTS = {"flat": 20, "preview": 60, "print": 240, "print_relief": 420}
_FOREIGN_EXTRA_S = 220  # Overpass замість DuckDB: fetch_source 214–231 с на проді

# Покриття ukraine.duckdb (Geofabrik ukraine-latest): грубий bbox України.
_UA_BBOX = (52.45, 44.15, 40.30, 22.05)  # north, south, east, west


def enabled() -> bool:
    return os.getenv("RESULT_CACHE", "1").strip().lower() not in ("0", "false", "no", "off")


def _load() -> None:
    global _loaded, _cache, _stats
    if _loaded:
        return
    _loaded = True
    try:
        if _CACHE_PATH.exists():
            _cache = json.loads(_CACHE_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        print(f"[RESULT_CACHE] load failed (ignored): {exc}")
        _cache = {}
    try:
        if _STATS_PATH.exists():
            _stats = json.loads(_STATS_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        print(f"[RESULT_CACHE] stats load failed (ignored): {exc}")
        _stats = {}


def _save_cache() -> None:
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(_cache, ensure_ascii=False), encoding="utf-8")
        tmp.replace(_CACHE_PATH)
    except Exception as exc:  # noqa: BLE001
        print(f"[RESULT_CACHE] save failed (ignored): {exc}")


def _save_stats() -> None:
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _STATS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(_stats), encoding="utf-8")
        tmp.replace(_STATS_PATH)
    except Exception as exc:  # noqa: BLE001
        print(f"[RESULT_CACHE] stats save failed (ignored): {exc}")


def _norm(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, dict):
        return {str(k): _norm(x) for k, x in sorted(v.items())}
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    return v


def request_cache_key(request: Any, zone_polygon_coords: Any = None) -> str:
    """sha1 усіх полів запиту (нормалізованих) + полігон зони + версія."""
    try:
        data = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    except Exception:
        data = dict(getattr(request, "__dict__", {}))
    payload = {"_v": _CACHE_VERSION, "req": _norm(data), "poly": _norm(zone_polygon_coords)}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def lookup(key: str) -> Optional[Dict[str, Any]]:
    """Повертає запис, якщо він свіжий і всі файли на місці; інакше None."""
    if not enabled():
        return None
    with _LOCK:
        _load()
        entry = _cache.get(key)
        if not entry:
            return None
        if time.time() - float(entry.get("ts", 0)) > _TTL_S:
            _cache.pop(key, None)
            _save_cache()
            return None
        paths = [entry.get("output_file")] + list((entry.get("output_files") or {}).values())
        for p in paths:
            if p and not Path(p).exists():
                _cache.pop(key, None)
                _save_cache()
                return None
        return dict(entry)


def store(key: str, task: Any) -> None:
    """Запамʼятовує готовий результат задачі (лише completed з файлом)."""
    if not enabled() or not key:
        return
    output_file = getattr(task, "output_file", None)
    if getattr(task, "status", "") != "completed" or not output_file:
        return
    entry = {
        "ts": time.time(),
        "task_id": getattr(task, "task_id", None),
        "output_file": str(output_file),
        "output_files": {k: str(v) for k, v in (getattr(task, "output_files", {}) or {}).items()},
        "print_quality": getattr(task, "print_quality", None),
        "keychain_manifest": getattr(task, "keychain_manifest", None),
        "message": getattr(task, "message", "") or "",
    }
    with _LOCK:
        _load()
        _cache[key] = entry
        if len(_cache) > _MAX_ENTRIES:
            for old in sorted(_cache, key=lambda k: _cache[k].get("ts", 0))[: len(_cache) - _MAX_ENTRIES]:
                _cache.pop(old, None)
        _save_cache()


def apply_cached(task: Any, entry: Dict[str, Any]) -> None:
    """Переносить кешований результат у свіжу GenerationTask."""
    task.output_files = dict(entry.get("output_files") or {})
    try:
        task.print_quality = entry.get("print_quality")
    except Exception:
        pass
    try:
        task.keychain_manifest = entry.get("keychain_manifest")
    except Exception:
        pass
    task.complete(entry["output_file"])
    task.message = "Готово (з кешу — така сама модель уже генерувалась)"
    task.from_cache = True


# ── ETA ──────────────────────────────────────────────────────────────────────

def eta_bucket(request: Any) -> str:
    flat = bool(getattr(request, "flat_plate_mode", False))
    preview = bool(getattr(request, "preview_mode", False)) and not flat
    if flat:
        kind = "flat"
    elif preview:
        kind = "preview"
    else:
        kind = "print_relief" if bool(getattr(request, "terrain_enabled", False)) else "print"
    try:
        size = int(round(float(getattr(request, "model_size_mm", 80) or 80) / 10.0) * 10)
    except Exception:
        size = 80
    return f"{kind}:{size}"


def record_duration(bucket: str, seconds: float) -> None:
    if not bucket or seconds <= 0:
        return
    with _LOCK:
        _load()
        arr = list(_stats.get(bucket, []))
        arr.append(round(float(seconds), 1))
        _stats[bucket] = arr[-_STATS_KEEP:]
        _save_stats()


def eta_seconds(bucket: str, foreign: bool = False) -> int:
    with _LOCK:
        _load()
        arr = _stats.get(bucket) or []
    if len(arr) >= 3:
        base = statistics.median(arr[-15:])
    else:
        kind = bucket.split(":")[0] if bucket else "preview"
        base = _ETA_DEFAULTS.get(kind, 90)
    if foreign:
        base += _FOREIGN_EXTRA_S
    return int(round(base))


# ── Покриття локальної OSM-БД ────────────────────────────────────────────────

def within_local_coverage(north: float, south: float, east: float, west: float) -> bool:
    """True, якщо bbox у покритті ukraine.duckdb (DuckDB доступна і bbox в Україні)."""
    try:
        from services.local_osm_db import is_available
        if not is_available():
            return False
    except Exception:
        return False
    n, s, e, w = _UA_BBOX
    return (south >= s) and (north <= n) and (west >= w) and (east <= e)
