import warnings
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Header, Form, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Р—Р°РІР°РЅС‚Р°Р¶СѓС”РјРѕ Р·РјС–РЅРЅС– СЃРµСЂРµРґРѕРІРёС‰Р° Р· .env С„Р°Р№Р»Сѓ
load_dotenv()

from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Optional, List, Tuple, Dict, Any
import os
import re
import uuid
from pathlib import Path
import trimesh
import httpx
import numpy as np

# Manifold3D РґР»СЏ С‚РѕС‡РЅРёС… boolean РѕРїРµСЂР°С†С–Р№
try:
    import manifold3d
    from manifold3d import Manifold, Mesh
    HAS_MANIFOLD = True
    MANIFOLD_VERSION = getattr(manifold3d, '__version__', 'unknown')
    print(f"[INFO] Manifold3D library loaded successfully (version: {MANIFOLD_VERSION})")
    print(f"[INFO] Manifold3D will be used for high-precision boolean operations with sharp edges")
except ImportError as e:
    HAS_MANIFOLD = False
    MANIFOLD_VERSION = None
    print(f"[WARN] Manifold3D library not found: {e}")
    print(f"[WARN] Boolean operations will use fallback methods (may be slow or jagged)")
    print(f"[WARN] Install with: pip install manifold3d")
except Exception as e:
    HAS_MANIFOLD = False
    MANIFOLD_VERSION = None
    print(f"[WARN] Error loading Manifold3D: {e}")
    print(f"[WARN] Boolean operations will use fallback methods")


# РџСЂРёРґСѓС€РµРЅРЅСЏ deprecation warnings РІС–Рґ pandas/geopandas
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pandas')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='geopandas')

import osmnx as ox
# Configure OSMnx to allow larger query areas without warning/subdivision
# Default is 50km*50km (2.5e9). Set to effectively infinite to prevent subdivision.
ox.settings.max_query_area_size = 1e50
ox.settings.use_cache = True
ox.settings.log_console = False # Reduce noise


from services.full_generation_pipeline import run_full_generation_pipeline
from services.generation_runtime_context import prepare_generation_runtime_context

from services.generation_task import GenerationTask
from services.firebase_service import FirebaseService
from services.global_center import set_global_center, get_global_center, GlobalCenter
from services.hexagonal_grid import generate_hexagonal_grid, hexagons_to_geojson, validate_hexagonal_grid, calculate_grid_center_from_geojson
from services.elevation_sync import calculate_global_elevation_reference, calculate_optimal_base_thickness
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

app = FastAPI(title="3D Map Generator API", version="1.0.0")

# Р—Р°Р·РѕСЂ РїР°Р·Сѓ РїРѕ Р‘РћРљРђРҐ (XY): 0.15РјРј Р· РєРѕР¶РЅРѕРіРѕ Р±РѕРєСѓ вЂ” РґР»СЏ РІСЃС‚Р°РІРєРё РґРѕСЂРѕРіРё РїС–СЃР»СЏ РґСЂСѓРєСѓ
GROOVE_CLEARANCE_MM = 0.15
# РњС–РЅС–РјР°Р»СЊРЅР° С€РёСЂРёРЅР° РїСЂРѕРјС–Р¶РєСѓ (РјРј) вЂ” СЏРєС‰Рѕ РјРµРЅС€Рµ, РѕР±'С”РґРЅСѓС”РјРѕ Р· РґРѕСЂРѕРіРѕСЋ (РЅРµРїСЂС–РЅС‚Р°Р±РµР»СЊРЅРёР№ СЂРµР»СЊС”С„)
MIN_PRINTABLE_GAP_MM = 0.6  # Проміжки <0.6мм об'єднуються з дорогами, щоб не лишати непринтабельні щілини



# CORS: restrict to known origins. Wildcard "*" + allow_credentials is invalid
# and insecure. Frontend is same-origin (monadruk.com), so we allow that + local
# dev. Override/extend via env CORS_ALLOW_ORIGINS (comma-separated).
_default_origins = [
    "https://monadruk.com",
    "https://www.monadruk.com",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
_env_origins = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
_allowed_origins = (
    [o.strip() for o in _env_origins.split(",") if o.strip()]
    if _env_origins
    else _default_origins
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ── DoS / spam protection: simple in-memory per-IP rate limiter ─────────────────
# No external dependency (Redis etc.) — a per-key deque of recent request
# timestamps. Caddy sits in front and sets X-Real-IP; behind Cloudflare the real
# client IP is the FIRST hop of X-Forwarded-For. We fall back to request.client.
# This protects the expensive/abuse-prone endpoints (generation, order, contact,
# analytics, share) from a single host hammering us.
import time as _time
from collections import deque as _deque
import threading as _threading

_RATE_BUCKETS: dict[tuple[str, str], "_deque[float]"] = {}
_RATE_LOCK = _threading.Lock()
_RATE_LAST_SWEEP = [0.0]


def _client_ip(request: Request) -> str:
    """Resolve the real client IP behind Caddy + Cloudflare.

    Priority: X-Real-IP (set by our Caddy) → first hop of X-Forwarded-For
    (Cloudflare → Caddy chain) → socket peer. Never trust these blindly for
    auth, but for coarse rate-limiting they are good enough."""
    h = request.headers
    real = (h.get("x-real-ip") or "").strip()
    if real:
        return real
    xff = (h.get("x-forwarded-for") or "").strip()
    if xff:
        first = xff.split(",")[0].strip()
        if first:
            return first
    try:
        return request.client.host if request.client else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _check_rate(ip: str, scope: str, limit: int, window_s: float) -> bool:
    """Return True if the (ip, scope) bucket is still under `limit` within the
    trailing `window_s` seconds; records the hit. Thread-safe (background tasks
    + request handlers may share the process)."""
    now = _time.monotonic()
    key = (ip, scope)
    with _RATE_LOCK:
        dq = _RATE_BUCKETS.get(key)
        if dq is None:
            dq = _deque()
            _RATE_BUCKETS[key] = dq
        cutoff = now - window_s
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= limit:
            return False
        dq.append(now)
        # Opportunistic sweep of fully-stale buckets so memory can't grow
        # unbounded under a spray of distinct IPs (~once a minute).
        if now - _RATE_LAST_SWEEP[0] > 60.0:
            _RATE_LAST_SWEEP[0] = now
            stale = [k for k, d in _RATE_BUCKETS.items() if not d or d[-1] < now - 3600.0]
            for k in stale:
                _RATE_BUCKETS.pop(k, None)
        return True


def rate_limit(scope: str, limits: list[tuple[int, float]]):
    """Build a FastAPI dependency enforcing one or more (limit, window_seconds)
    rules for a named scope. Raises HTTP 429 when any rule is exceeded.

    Example: rate_limit("generate", [(5, 60), (40, 3600)]) → 5/min AND 40/hour."""
    def _dep(request: Request) -> None:
        ip = _client_ip(request)
        for limit, window in limits:
            if not _check_rate(ip, scope, limit, window):
                raise HTTPException(
                    status_code=429,
                    detail="Забагато запитів — зачекайте трохи й спробуйте ще раз.",
                )
    return _dep

# Р—Р±РµСЂС–РіР°РЅРЅСЏ Р·Р°РґР°С‡ РіРµРЅРµСЂР°С†С–С—
tasks: dict[str, GenerationTask] = {}
# Р—Р±РµСЂС–РіР°РЅРЅСЏ Р·РІ'СЏР·РєС–РІ РјС–Р¶ РјРЅРѕР¶РёРЅРЅРёРјРё Р·Р°РґР°С‡Р°РјРё (task_id -> list of task_ids)
multiple_tasks_map: dict[str, list[str]] = {}

import tempfile

# Р’РёРєРѕСЂРёСЃС‚РѕРІСѓС”РјРѕ Р»РѕРєР°Р»СЊРЅСѓ РґРёСЂРµРєС‚РѕСЂС–СЋ output РґР»СЏ СЃС‚Р°Р±С–Р»СЊРЅРѕСЃС‚С–
# Р¦Рµ РІРёСЂС–С€СѓС” РїСЂРѕР±Р»РµРјСѓ Р·РЅРёРєРЅРµРЅРЅСЏ С„Р°Р№Р»С–РІ Сѓ С‚РёРјС‡Р°СЃРѕРІРёС… РїР°РїРєР°С…
OUTPUT_DIR = Path("output").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# БЕЗПЕКА: приватні дані (users.json, orders.jsonl, analytics.jsonl, panel_batches)
# НЕ можна тримати в OUTPUT_DIR — він віддається статикою на /files. Інакше будь-хто
# міг GET /files/users.json і завантажити всю базу клієнтів (емейли/телефони/адреси).
# DATA_DIR — окрема папка, що НЕ монтується назовні.
DATA_DIR = Path("data").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

def _migrate_private_data_out_of_webroot() -> None:
    """Одноразова міграція: переносить приватні файли з OUTPUT_DIR (віддається на
    /files) у DATA_DIR (не монтується). Закриває витік users.json/orders.jsonl
    і зберігає наявні дані (move, не порожній рестарт). Безпечно при кожному старті."""
    for _name in ("users.json", "orders.jsonl", "analytics.jsonl", "panel_batches.json"):
        try:
            _old = OUTPUT_DIR / _name
            _new = DATA_DIR / _name
            if _old.exists() and not _new.exists():
                _old.replace(_new)
                print(f"[SECURITY] migrated {_name}: OUTPUT_DIR -> DATA_DIR")
            elif _old.exists():
                # обидва є (напр. analytics очищено) — лишаємо DATA_DIR, прибираємо webroot-копію
                _old.unlink()
                print(f"[SECURITY] removed webroot copy of {_name} (DATA_DIR is canonical)")
        except Exception as _e:  # noqa: BLE001
            print(f"[SECURITY] migrate {_name} failed: {_e}")

_migrate_private_data_out_of_webroot()

# D3 ПАННО: persist batch→tile mapping so a server restart doesn't lose a
# finished panel. `multiple_tasks_map` + `tasks` are in-memory; here we mirror
# the batch with each tile's row/col AND its on-disk 3MF path (filled in as
# tiles complete, while `tasks` is still alive). After a restart download_all
# rebuilds the zip from these persisted paths (the output files survive on disk).
PANEL_BATCHES_PATH = DATA_DIR / "panel_batches.json"
panel_tiles: dict[str, list[dict]] = {}  # batch_id -> [{task_id, row, col, path}]


def _save_panel_batches() -> None:
    import json as _json
    try:
        data = {
            bid: {"task_ids": multiple_tasks_map.get(bid, []), "tiles": tiles}
            for bid, tiles in panel_tiles.items()
        }
        tmp = PANEL_BATCHES_PATH.with_suffix(".tmp")
        tmp.write_text(_json.dumps(data), encoding="utf-8")
        tmp.replace(PANEL_BATCHES_PATH)
    except Exception as exc:  # noqa: BLE001
        print(f"[PANNO] save batches failed (non-fatal): {exc}")


def _record_panel_tile_paths(batch_id: str) -> None:
    """Заповнює шляхи готових плиток у panel_tiles (поки tasks живі) і зберігає
    на диск, якщо щось змінилось — щоб zip пережив рестарт сервера."""
    tiles = panel_tiles.get(batch_id)
    if not tiles:
        return
    changed = False
    for rec in tiles:
        if rec.get("path"):
            continue
        t = tasks.get(rec.get("task_id"))
        if t is None or getattr(t, "status", None) != "completed":
            continue
        ofiles = getattr(t, "output_files", {}) or {}
        path_str = ofiles.get("3mf") or getattr(t, "output_file", None)
        if path_str and Path(path_str).exists():
            rec["path"] = str(path_str)
            if rec.get("row") is None:
                rec["row"] = getattr(t, "zone_row", None)
            if rec.get("col") is None:
                rec["col"] = getattr(t, "zone_col", None)
            changed = True
    if changed:
        _save_panel_batches()


def _load_panel_batches() -> None:
    import json as _json
    try:
        if not PANEL_BATCHES_PATH.exists():
            return
        data = _json.loads(PANEL_BATCHES_PATH.read_text(encoding="utf-8"))
        for bid, rec in data.items():
            ids = rec.get("task_ids") or []
            if ids:
                multiple_tasks_map[bid] = ids
                panel_tiles[bid] = rec.get("tiles") or []
        if panel_tiles:
            print(f"[PANNO] restored {len(panel_tiles)} batch(es) from disk")
    except Exception as exc:  # noqa: BLE001
        print(f"[PANNO] load batches failed (non-fatal): {exc}")


_load_panel_batches()


def _make_export_basename(
    task_id: str,
    hex_size_m: Optional[float] = None,
    model_size_mm: Optional[float] = None,
    zone_row: Optional[int] = None,
    zone_col: Optional[int] = None,
) -> str:
    """Build descriptive file basename. 4-tier graceful fallback:

    1. Batch with row/col   → `model_<grid>_<mm>_<row>_<col>_<tok>` (e.g. model_300_80_38_40_a1b2c3d4)
    2. Batch w/o row/col    → `model_<grid>_<mm>_<uuid8>_<tok>`
    3. Single endpoint      → `model_<mm>_<uuid8>_<tok>`
    4. Nothing known        → original `task_id` (legacy UUID)

    The frontend reads basename from backend's download_url and uses it as
    the download filename, so the slicer label matches the descriptive name.

    SECURITY (#7 — model enumeration): files are served statically by basename
    (`/files/<basename>.3mf`). Without the token the grid/keychain naming is fully
    predictable (`model_300_80_38_40.3mf`), so anyone could enumerate every other
    user's model by guessing grid/mm/row/col. We append a random secrets token
    (`secrets.token_hex(4)` → 8 hex chars, 32 bits) so the URL is unguessable.
    The descriptive prefix is kept for a readable slicer label. Tile/panel lookup
    never parses the filename — it keys on stored path + task_id (panel_batches
    .json), so the token does not break the panel zip; the on-disk recovery parser
    below treats the trailing hex token as part of the identity.
    """
    import secrets
    short = task_id.replace("-", "")[:8] if task_id else "single"
    tok = secrets.token_hex(4)
    try:
        if all(v is not None for v in (hex_size_m, model_size_mm, zone_row, zone_col)):
            return (
                f"model_{int(round(float(hex_size_m)))}_"
                f"{int(round(float(model_size_mm)))}_"
                f"{int(zone_row)}_{int(zone_col)}_{tok}"
            )
        if hex_size_m is not None and model_size_mm is not None:
            return (
                f"model_{int(round(float(hex_size_m)))}_"
                f"{int(round(float(model_size_mm)))}_{short}_{tok}"
            )
        if model_size_mm is not None:
            return f"model_{int(round(float(model_size_mm)))}_{short}_{tok}"
    except (TypeError, ValueError):
        pass
    return task_id

CANONICAL_CONTROL_BUNDLE_DIR = (Path("debug") / "generated" / "final_3d_input_masks_parks_fit_v006").resolve()
CONTROL_ZONE_ID = "hex_43_38"
CONTROL_ZONE_ROW = 43
CONTROL_ZONE_COL = 38
CONTROL_ZONE_BBOX = {
    "north": 50.43091804159341,
    "south": 50.423729849284264,
    "east": 30.567171788688167,
    "west": 30.55724205709598,
}
CONTROL_ZONE_POLYGON = [
    [30.567171788688167, 50.4256289330356],
    [30.56698752180705, 50.42922304214077],
    [30.56202244378919, 50.43091804159341],
    [30.55724205709598, 50.429018743034796],
    [30.557427060274737, 50.42542465985662],
    [30.562391713859018, 50.423729849284264],
    [30.567171788688167, 50.4256289330356],
]


def _is_close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= float(tol)


def _matches_control_zone_request(
    request: "GenerationRequest",
    *,
    zone_id: Optional[str] = None,
    zone_row: Optional[int] = None,
    zone_col: Optional[int] = None,
    zone_polygon_coords: Optional[list] = None,
) -> bool:
    if zone_id == CONTROL_ZONE_ID:
        return True
    if zone_row == CONTROL_ZONE_ROW and zone_col == CONTROL_ZONE_COL:
        return True

    if zone_polygon_coords and len(zone_polygon_coords) == len(CONTROL_ZONE_POLYGON):
        try:
            if all(
                _is_close(src[0], ref[0], 1e-6) and _is_close(src[1], ref[1], 1e-6)
                for src, ref in zip(zone_polygon_coords, CONTROL_ZONE_POLYGON)
            ):
                return True
        except Exception:
            pass

    try:
        return (
            _is_close(request.north, CONTROL_ZONE_BBOX["north"], 1e-6)
            and _is_close(request.south, CONTROL_ZONE_BBOX["south"], 1e-6)
            and _is_close(request.east, CONTROL_ZONE_BBOX["east"], 1e-6)
            and _is_close(request.west, CONTROL_ZONE_BBOX["west"], 1e-6)
        )
    except Exception:
        return False


def _apply_default_canonical_bundle_if_needed(
    request: "GenerationRequest",
    *,
    zone_id: Optional[str] = None,
    zone_row: Optional[int] = None,
    zone_col: Optional[int] = None,
    zone_polygon_coords: Optional[list] = None,
) -> None:
    if getattr(request, "canonical_mask_bundle_dir", None):
        return
    if not CANONICAL_CONTROL_BUNDLE_DIR.exists():
        return
    if not _matches_control_zone_request(
        request,
        zone_id=zone_id,
        zone_row=zone_row,
        zone_col=zone_col,
        zone_polygon_coords=zone_polygon_coords,
    ):
        return
    request.canonical_mask_bundle_dir = str(CANONICAL_CONTROL_BUNDLE_DIR)
    print(f"[INFO] Auto-applied canonical mask bundle for control zone: {request.canonical_mask_bundle_dir}")


def _compute_safe_base_thickness_mm(request: "GenerationRequest") -> float:
    if bool(getattr(request, "flat_plate_mode", False)):
        try:
            # Брелок: 1.0мм = 5 шарів × 0.2мм, мінімально жорстко. Плоска кольорова
            # AMS-карта (is_ams_mode, не брелок): міцна основа 3мм (велика плитка не
            # повинна гнутись/коробитись). Магніт (is_ams=false): свій 0.2 floor +
            # окрема логіка глибини кишені нижче по пайплайну.
            if bool(getattr(request, "keychain_mode", False)):
                floor = 1.0
            elif bool(getattr(request, "is_ams_mode", False)):
                floor = 3.0
            else:
                floor = 0.2
            if bool(getattr(request, "map_connector", False)):
                # ЗʼЄДНУВАЧ (плоский режим): паз-ластівчин-хвіст ріжеться у ДНІ основи,
                # а дороги/будинки сидять ЗВЕРХУ. Якщо основа тонка — паз (глибина
                # _cd) дістає до шарів зверху → «частинки будинків/доріг» у пазу
                # (скарга власника). Лишаємо ≥1.5мм суцільної основи НАД пазом.
                _cd = float(getattr(request, "map_connector_depth_mm", 2.0) or 2.0)
                floor = max(floor, _cd + 1.5, 3.5)
            return max(float(request.terrain_base_thickness_mm), floor)
        except Exception:
            return 1.0 if bool(getattr(request, "keychain_mode", False)) else 0.2
    try:
        min_required_base_mm = max(
            0.2,
            float(request.parks_embed_mm) if getattr(request, "include_parks", False) else 0.0,
            float(request.road_embed_mm),
            float(request.water_depth),
        ) + 0.5
        if bool(getattr(request, "map_connector", False)):
            # ЗʼЄДНУВАЧ (рельєф): паз у ДНІ підложки; будинки опускаються до дна для
            # суцільного друку → у смузі пазу були частинки будинків. Основа має
            # вмістити паз + запас, щоб нижня смуга лишалась ЧИСТОЮ підложкою
            # (будинки тоді обрізаються до floor+паз, див. merge_terrain_and_buildings).
            _cd = float(getattr(request, "map_connector_depth_mm", 2.0) or 2.0)
            min_required_base_mm = max(min_required_base_mm, _cd + 1.0)
        return max(float(request.terrain_base_thickness_mm), float(min_required_base_mm))
    except Exception:
        try:
            return max(float(request.terrain_base_thickness_mm), 0.2)
        except Exception:
            return 0.2


def _normalize_request_base_thickness(request: "GenerationRequest", *, zone_prefix: str = "") -> float:
    requested_base_thickness_mm = float(getattr(request, "terrain_base_thickness_mm", 0.2) or 0.2)
    final_base_thickness_mm = _compute_safe_base_thickness_mm(request)
    if abs(final_base_thickness_mm - requested_base_thickness_mm) > 1e-9:
        print(
            f"[INFO] {zone_prefix}Adjusted terrain_base_thickness_mm: "
            f"{requested_base_thickness_mm:.2f}mm -> {final_base_thickness_mm:.2f}mm"
        )
        request.terrain_base_thickness_mm = final_base_thickness_mm
    return final_base_thickness_mm


from fastapi.staticfiles import StaticFiles
from starlette.responses import PlainTextResponse as _PlainText


class SafeStatic(StaticFiles):
    """StaticFiles, що віддає ЛИШЕ артефакти моделей за розширенням. Захист у
    глибину: навіть якщо у OUTPUT_DIR опиниться .json/.jsonl/.env/.py — він НЕ
    віддасться (раніше відкритий mount зливав users.json/orders.jsonl).

    Друкарські формати (.3mf/.stl) НАВМИСНО прибрані з allowlist (task #70):
    повноякісний друк-файл не має віддаватись напряму за basename з download_url
    в обхід квоти. Превʼю йде через /api/download (in-memory task), а реальне
    завантаження — через POST /api/account/download (стрімить server-side, квота).
    Жоден фронт-консюмер НЕ робить прямий GET /files/<name>.3mf (перевірено)."""
    _ALLOWED = {".glb", ".gltf", ".obj", ".png", ".jpg", ".jpeg", ".webp", ".zip"}

    async def get_response(self, path, scope):
        ext = os.path.splitext(path)[1].lower()
        if ext and ext not in self._ALLOWED:
            return _PlainText("Not found", status_code=404)
        return await super().get_response(path, scope)


# Mount output folder — ТІЛЬКИ файли моделей (SafeStatic). Приватні дані тепер у
# DATA_DIR (не монтується). /api/files — дзеркало для проксі-конфігів, що шлють
# лише /api/* у бекенд.
app.mount("/files", SafeStatic(directory=OUTPUT_DIR), name="files")
app.mount("/api/files", SafeStatic(directory=OUTPUT_DIR), name="api_files")


class VideoStatic(StaticFiles):
    """Публічна роздача відео + обкладинок (Instagram/Threads/Pinterest тягнуть з URL)."""
    _ALLOWED = {".mp4", ".mov", ".m4v", ".webm", ".jpg", ".jpeg", ".png", ".webp"}

    async def get_response(self, path, scope):
        ext = os.path.splitext(path)[1].lower()
        if ext and ext not in self._ALLOWED:
            return _PlainText("Not found", status_code=404)
        return await super().get_response(path, scope)


# Публічна тека відео для соц-публікації (IG/Pinterest тягнуть video_url).
PUBLIC_MEDIA_DIR = Path("public_media").resolve()
PUBLIC_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/media", VideoStatic(directory=PUBLIC_MEDIA_DIR), name="media")
app.mount("/api/public_media", VideoStatic(directory=PUBLIC_MEDIA_DIR), name="api_public_media")


@app.get("/api/tiktok/callback")
async def _tiktok_oauth_callback(code: str = "", state: str = "", error: str = ""):
    """OAuth redirect для TikTok (localhost не підтримується TikTok). Зберігає code у
    DATA_DIR/tiktok_code.json, звідки publisher-бот його забирає й обмінює на токен."""
    import json as _json
    try:
        (DATA_DIR / "tiktok_code.json").write_text(
            _json.dumps({"code": code, "state": state, "error": error}), encoding="utf-8")
    except Exception:
        pass
    body = "TikTok авторизовано — можна закрити цю вкладку." if code else f"Помилка: {error}"
    return _PlainText(body)


async def _ttl_cleanup_loop():
    """TTL: remove tasks older than 2 hours every 30 minutes."""
    import asyncio as _asyncio
    while True:
        await _asyncio.sleep(1800)
        stale = [tid for tid, t in list(tasks.items()) if t.is_stale(max_age_hours=2.0)]
        for tid in stale:
            tasks.pop(tid, None)
        if stale:
            print(f"[TTL] Removed {len(stale)} stale tasks")


@app.on_event("startup")
async def startup_event():
    import asyncio as _asyncio
    _asyncio.create_task(_ttl_cleanup_loop())

    # Прогрів локальної OSM-БД (DuckDB) у фоні — перший конект ~5с; інакше перший
    # /api/building-at (підсвітка будинку) або генерація після рестарту гальмують.
    def _warm_osm_db():
        try:
            from services.local_osm_db import is_available, _get_conn
            if is_available():
                _get_conn()
                print("[STARTUP] Local OSM DuckDB connection warmed")
        except Exception as _wexc:
            print(f"[STARTUP] OSM DB warm skipped: {_wexc}")
    _threading.Thread(target=_warm_osm_db, daemon=True).start()
    """Р’С–РґРЅРѕРІР»СЋС”РјРѕ СЃС‚Р°РЅ Р·Р°РґР°С‡ РЅР° РѕСЃРЅРѕРІС– С„Р°Р№Р»С–РІ Сѓ РґРёСЂРµРєС‚РѕСЂС–С— output С‚Р° РїРµСЂРµРІС–СЂСЏС”РјРѕ Firebase"""
    
    # Р†РЅС–С†С–Р°Р»С–Р·Р°С†С–СЏ Firebase С‚Р° РІРёРІС–Рґ СЃС‚Р°С‚СѓСЃСѓ
    print("\n" + "="*60)
    print("Checking Firebase Integration...")
    FirebaseService.initialize()
    if FirebaseService._initialized:
        print(f"[OK] Firebase Storage: ACTIVE (Bucket: {os.getenv('FIREBASE_STORAGE_BUCKET')})")
        FirebaseService.configure_cors()  # <--- Fix for Frontend Access
        print(f"[OK] Remote Path: 3dMap/")
    else:
        print("[WARN] Firebase Storage: DISABLED")
        print("   Optional cloud upload only. Set FIREBASE_STORAGE_BUCKET +")
        print("   FIREBASE_CREDENTIALS_JSON (env) to enable. Auth/login does NOT")
        print("   need this — ID tokens are verified via Google public certs.")
    print("="*60 + "\n")

    print("[INFO] Р’С–РґРЅРѕРІР»РµРЅРЅСЏ СЃРїРёСЃРєСѓ Р·Р°РґР°С‡ Р· РґРёСЃРєР°...")
    if not OUTPUT_DIR.exists():
        return
    
    # РЁСѓРєР°С”РјРѕ РІСЃС– STL/3MF С„Р°Р№Р»Рё
    for file_path in OUTPUT_DIR.glob("*"):
        if file_path.suffix.lower() not in [".stl", ".3mf"]:
            continue
        
        # Parse filename — supports two formats:
        #   legacy:      <uuid>[_<part>].<ext>             → task_id = uuid
        #   descriptive: model_<grid>_<mm>_<row>_<col>_<tok>[_<part>].<ext>
        #                (or model_<mm>_<uuid8>_<tok>[_<part>])
        #                                                  → task_id = full basename up to <part>
        # The basename now carries a random hex token (anti-enumeration, see
        # _make_export_basename), so we CANNOT assume a fixed token count. Instead
        # we split off a recognized trailing PART suffix (base/roads/.../print_layout,
        # optionally `_part_NNN`) and treat everything before it as the task_id.
        name = file_path.name
        stem = file_path.stem
        if stem.startswith("model_"):
            parts = stem.split("_")
            _PART_KEYS = {"base", "roads", "buildings", "parks", "water",
                          "print", "layout", "acceptance", "assembly", "package", "part"}
            # find the first token that begins a part-suffix; everything before is id
            cut = len(parts)
            for i in range(2, len(parts)):  # never strip "model" or the first id token
                if parts[i] in _PART_KEYS:
                    cut = i
                    break
            task_id = "_".join(parts[:cut]) if cut > 1 else stem
            part_part = "_".join(parts[cut:]) if cut < len(parts) else None
        else:
            if "_" in stem:
                task_id, part_part = stem.split("_", 1)
            else:
                task_id = stem
                part_part = None

        # Якщо такий task_id ще не в списку, створюємо "заглушку"
        if task_id not in tasks:
            tasks[task_id] = GenerationTask(
                task_id=task_id,
                request=None,  # Параметри старого запиту не відомі
                status="completed",
                progress=100,
                output_file=str(file_path)
            )

        task = tasks[task_id]
        ext = file_path.suffix.lstrip(".").lower()
        if part_part:
            key = f"{part_part}_{ext}"
            task.set_output(key, str(file_path))
        else:
            task.set_output(ext, str(file_path))
            if not task.output_file:
                task.output_file = str(file_path)
    
    print(f"[INFO] Р’С–РґРЅРѕРІР»РµРЅРѕ {len(tasks)} Р·Р°РґР°С‡.")


class GenerationRequest(BaseModel):
    """Запит на генерацію 3D моделі"""
    model_config = ConfigDict(protected_namespaces=())

    # ── Auto-clamp: замість 422 помилок — м'яко клампимо значення в межі Field(ge/le)
    @model_validator(mode='before')
    @classmethod
    def _auto_clamp_numeric_fields(cls, data):
        if not isinstance(data, dict):
            return data
        try:
            from annotated_types import Ge, Le, Gt, Lt
        except ImportError:
            return data
        for field_name, field_info in cls.model_fields.items():
            if field_name not in data:
                continue
            value = data[field_name]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            ge_val = le_val = gt_val = lt_val = None
            for meta in getattr(field_info, 'metadata', []) or []:
                if isinstance(meta, Ge):
                    ge_val = meta.ge
                elif isinstance(meta, Le):
                    le_val = meta.le
                elif isinstance(meta, Gt):
                    gt_val = meta.gt
                elif isinstance(meta, Lt):
                    lt_val = meta.lt
            orig = value
            clamped = value
            if ge_val is not None and clamped < ge_val:
                clamped = ge_val
            if gt_val is not None and clamped <= gt_val:
                clamped = gt_val + (1 if isinstance(value, int) else 0.001)
            if le_val is not None and clamped > le_val:
                clamped = le_val
            if lt_val is not None and clamped >= lt_val:
                clamped = lt_val - (1 if isinstance(value, int) else 0.001)
            if clamped != orig:
                print(f"[AUTO-CLAMP] {field_name}: {orig} -> {clamped} (limits ge={ge_val} le={le_val})")
                data[field_name] = type(orig)(clamped) if isinstance(orig, int) else float(clamped)
        return data

    north: float
    south: float
    east: float
    west: float
    # РџР°СЂР°РјРµС‚СЂРё РіРµРЅРµСЂР°С†С–С—
    road_width_multiplier: float = 1.0
    # Print-aware РїР°СЂР°РјРµС‚СЂРё (РІ РњР†Р›Р†РњР•РўР РђРҐ РЅР° С„С–РЅР°Р»СЊРЅС–Р№ РјРѕРґРµР»С–)
    # Rotated rect polygon (4 corners as [lon, lat]) — для повернутих ділянок.
    # Якщо задано, backend обрізає OSM до цього полігону, а не axis-aligned bbox.
    zone_polygon_coords: Optional[list] = None
    road_height_mm: float = Field(default=0.5, ge=0.2, le=5.0)
    road_embed_mm: float = Field(default=0.3, ge=0.0, le=2.0)
    # road_clearance_mm РІРёРґР°Р»РµРЅРѕ вЂ” Р·Р°РІР¶РґРё РІРёРєРѕСЂРёСЃС‚РѕРІСѓС”С‚СЊСЃСЏ GROOVE_CLEARANCE_MM = 0.15
    building_min_height: float = 2.0
    building_height_multiplier: float = 1.0
    building_foundation_mm: float = Field(default=0.6, ge=0.1, le=5.0)
    building_embed_mm: float = Field(default=0.2, ge=0.0, le=2.0)
    # РњР°РєСЃРёРјР°Р»СЊРЅР° РіР»РёР±РёРЅР° С„СѓРЅРґР°РјРµРЅС‚Сѓ (РјРј РќРђ Р¤Р†РќРђР›Р¬РќР†Р™ РњРћР”Р•Р›Р†).
    # Р¦Рµ "Р·Р°РїРѕР±С–Р¶РЅРёРє" РґР»СЏ РєСЂСѓС‚РёС… СЃС…РёР»С–РІ/С€СѓРјРЅРѕРіРѕ DEM: С‰РѕР± Р±СѓРґС–РІР»С– РЅРµ Р№С€Р»Рё РЅР°РґС‚Рѕ РіР»РёР±РѕРєРѕ РїС–Рґ Р·РµРјР»СЋ.
    building_max_foundation_mm: float = Field(default=2.5, ge=0.2, le=10.0)
    # Extra detail layers
    include_parks: bool = True
    parks_height_mm: float = Field(default=0.6, ge=0.1, le=5.0)
    parks_embed_mm: float = Field(default=1.0, ge=0.0, le=2.0)
    water_depth: float = 1.2  # РјРј РІ Р·РµРјР»С– (РїРѕРІРµСЂС…РЅСЏ РІРѕРґРё 0.2РјРј РЅРёР¶С‡Рµ СЂРµР»СЊС”С„Сѓ)
    terrain_enabled: bool = True
    terrain_z_scale: float = 3.0  # Р—Р±С–Р»СЊС€РµРЅРѕ РґР»СЏ РєСЂР°С‰РѕС— РІРёРґРёРјРѕСЃС‚С– СЂРµР»СЊС”С„Сѓ
    # РўРѕРЅРєР° РѕСЃРЅРѕРІР° РґР»СЏ РґСЂСѓРєСѓ: Р·Р° Р·Р°РјРѕРІС‡СѓРІР°РЅРЅСЏРј 1РјРј (РєРѕСЂРёСЃС‚СѓРІР°С‡ РјРѕР¶Рµ Р·РјС–РЅРёС‚Рё).
    terrain_base_thickness_mm: float = Field(default=0.3, ge=0.2, le=20.0)  # РўРѕРЅРєР° РїС–РґР»РѕР¶РєР°, РјС–РЅС–РјСѓРј 0.2РјРј
    # Р”РµС‚Р°Р»С–Р·Р°С†С–СЏ СЂРµР»СЊС”С„Сѓ
    # - terrain_resolution: РєС–Р»СЊРєС–СЃС‚СЊ С‚РѕС‡РѕРє РїРѕ РѕСЃС– (mesh РґРµС‚Р°Р»СЊ). Р’РёС‰Р° = РґРµС‚Р°Р»СЊРЅС–С€Рµ, РїРѕРІС–Р»СЊРЅС–С€Рµ.
    # Default 200 (was 350): 350×350 + subdivide gave ~640k vertices on a ~500m
    # zone = ~0.1mm/vertex on the printed model, ~16× finer than the 0.4mm nozzle
    # can print. Pure waste that slowed terrain build / grooves / repair / export.
    # 200 ≈ 2.5m/vertex (printable resolution) and keeps relief smooth.
    terrain_resolution: int = Field(default=200, ge=40, le=600)
    # Subdivision quadruples vertex count to add SUB-printable smoothness — off by
    # default (200 res is already smooth enough for print). Enable for hero renders.
    terrain_subdivide: bool = Field(default=False, description="Subdivision для ще плавнішого mesh (×4 вершини)")
    terrain_subdivide_levels: int = Field(default=1, ge=0, le=2, description="Р С–РІРЅС– subdivision (0-2, Р±С–Р»СЊС€Рµ = РїР»Р°РІРЅС–С€Рµ Р°Р»Рµ РїРѕРІС–Р»СЊРЅС–С€Рµ)")
    # - terrarium_zoom: Р·СѓРј DEM tiles (Terrarium). Р’РёС‰Р° = РґРµС‚Р°Р»СЊРЅС–С€Рµ, Р°Р»Рµ Р±С–Р»СЊС€Рµ С‚Р°Р№Р»С–РІ.
    terrarium_zoom: int = Field(default=15, ge=10, le=16)
    # Р—РіР»Р°РґР¶СѓРІР°РЅРЅСЏ СЂРµР»СЊС”С„Сѓ (sigma РІ РєР»С–С‚РёРЅРєР°С… heightfield). 0 = Р±РµР· Р·РіР»Р°РґР¶СѓРІР°РЅРЅСЏ.
    # Р”РѕРїРѕРјР°РіР°С” РїСЂРёР±СЂР°С‚Рё "РіСЂСѓР±С– РіСЂР°РЅС–/С€СѓРј" РЅР° DEM, РѕСЃРѕР±Р»РёРІРѕ РїСЂРё РІРёСЃРѕРєРѕРјСѓ zoom.
    terrain_smoothing_sigma: float = Field(default=2.0, ge=0.0, le=5.0)  # РћРїС‚РёРјР°Р»СЊРЅРµ Р·РіР»Р°РґР¶СѓРІР°РЅРЅСЏ РґР»СЏ С–РґРµР°Р»СЊРЅРѕРіРѕ СЂРµР»СЊС”С„Сѓ
    # Terrain-first СЃС‚Р°Р±С–Р»С–Р·Р°С†С–СЏ: РІРёРјРєРЅРµРЅРѕ Р·Р° Р·Р°РјРѕРІС‡СѓРІР°РЅРЅСЏРј, С‰РѕР± Р·Р±РµСЂРµРіС‚Рё РїСЂРёСЂРѕРґРёР№ СЂРµР»СЊС”С„.
    # Р‘СѓРґС–РІР»С– РјР°СЋС‚СЊ РІР»Р°СЃРЅС– С„СѓРЅРґР°РјРµРЅС‚Рё (building_foundation_mm), С‚РѕРјСѓ РІРёСЂС–РІРЅСЋРІР°РЅРЅСЏ Р·РµРјР»С– РЅРµ С” РєСЂРёС‚РёС‡РЅРёРј.
    flatten_buildings_on_terrain: bool = False
    # Terrain-first СЃС‚Р°Р±С–Р»С–Р·Р°С†С–СЏ РґР»СЏ РґРѕСЂС–Рі: РІРёРјРєРЅРµРЅРѕ Р·Р° Р·Р°РјРѕРІС‡СѓРІР°РЅРЅСЏРј,
    # РѕСЃРєС–Р»СЊРєРё РґР»СЏ РіСѓСЃС‚РѕС— РјРµСЂРµР¶С– РґРѕСЂС–Рі С†Рµ СЃС‚РІРѕСЂСЋС” С€С‚СѓС‡РЅС– "РїР»Р°С‚Рѕ" (С‡РµСЂРµР· Р·Р»РёС‚С‚СЏ РіРµРѕРјРµС‚СЂС–Р№),
    # С‰Рѕ РїСЃСѓС” СЂРµР»СЊС”С„ РЅР° РїР°РіРѕСЂР±Р°С…. Р”РѕСЂРѕРіРё С– С‚Р°Рє РіР°СЂРЅРѕ Р»СЏРіР°СЋС‚СЊ РїРѕ СЃРїР»Р°Р№РЅР°С….
    flatten_roads_on_terrain: bool = False
    # Fast preview mode (~30s): skip Blender groove cutting + manifold cleanup,
    # downscale terrain to 60x60. Includes ALL layers (terrain/roads/buildings/
    # parks/water) but they sit as separate components (no printability checks).
    preview_mode: bool = False
    export_format: str = "3mf"  # "stl" Р°Р±Рѕ "3mf"
    model_size_mm: float = 80.0  # Р РѕР·РјС–СЂ РјРѕРґРµР»С– РІ РјС–Р»С–РјРµС‚СЂР°С… (Р·Р° Р·Р°РјРѕРІС‡СѓРІР°РЅРЅСЏРј 80РјРј = 8СЃРј)
    # РљРѕРЅС‚РµРєСЃС‚ РЅР°РІРєРѕР»Рѕ Р·РѕРЅРё (РІ РјРµС‚СЂР°С…): Р·Р°РІР°РЅС‚Р°Р¶СѓС”РјРѕ OSM/Extras Р· Р±С–Р»СЊС€РёРј bbox,
    # Р°Р»Рµ С„С–РЅР°Р»СЊРЅС– РјРµС€С– РІСЃРµ РѕРґРЅРѕ РѕР±СЂС–Р·Р°С”РјРѕ РїРѕ РїРѕР»С–РіРѕРЅСѓ Р·РѕРЅРё.
    # РџР°СЂР°РјРµС‚СЂРё РґР»СЏ РїСЂРµРІ'СЋ (РјРѕР¶Р»РёРІС–СЃС‚СЊ РІРёРєР»СЋС‡Р°С‚Рё/РІРєР»СЋС‡Р°С‚Рё РєРѕРјРїРѕРЅРµРЅС‚Рё)
    preview_include_base: bool = True
    preview_include_roads: bool = True
    preview_include_buildings: bool = True
    preview_include_water: bool = True
    preview_include_parks: bool = True
    # Р¦Рµ РїРѕС‚СЂС–Р±РЅРѕ, С‰РѕР± РєРѕСЂРµРєС‚РЅРѕ РІРёР·РЅР°С‡Р°С‚Рё РјРѕСЃС‚Рё/РїРµСЂРµС‚РёРЅРё Р±С–Р»СЏ РєСЂР°СЋ Р·РѕРЅРё.
    context_padding_m: float = Field(default=400.0, ge=0.0, le=5000.0)
    # РўРµСЃС‚СѓРІР°РЅРЅСЏ: РіРµРЅРµСЂСѓРІР°С‚Рё С‚С–Р»СЊРєРё СЂРµР»СЊС”С„ Р±РµР· Р±СѓРґС–РІРµР»СЊ/РґРѕСЂС–Рі/РІРѕРґРё (Р·Р° Р·Р°РјРѕРІС‡СѓРІР°РЅРЅСЏРј False - РїРѕРІРЅР° РјРѕРґРµР»СЊ)
    terrain_only: bool = False  # РўРµСЃС‚РѕРІРёР№ СЂРµР¶РёРј РІРёРјРєРЅРµРЅРѕ Р·Р° Р·Р°РјРѕРІС‡СѓРІР°РЅРЅСЏРј
    # РЎРёРЅС…СЂРѕРЅС–Р·Р°С†С–СЏ РІРёСЃРѕС‚ РјС–Р¶ Р·РѕРЅР°РјРё (РґР»СЏ РіРµРєСЃР°РіРѕРЅР°Р»СЊРЅРѕС— СЃС–С‚РєРё)
    elevation_ref_m: Optional[float] = None  # Р“Р»РѕР±Р°Р»СЊРЅР° Р±Р°Р·РѕРІР° РІРёСЃРѕС‚Р° (РјРµС‚СЂРё РЅР°Рґ СЂС–РІРЅРµРј РјРѕСЂСЏ)
    baseline_offset_m: float = 0.0  # Р—РјС–С‰РµРЅРЅСЏ baseline (РјРµС‚СЂРё)
    # СЕРІЯ ЗОН: спільний перепад висот (max-ref, світові метри) для ВСІЄЇ сітки. Коли
    # >0 — кожна плитка масштабує рельєф ОДНИМ спільним gain (континуальний рельєф через
    # шов, без сходинок). 0 = per-tile lift (одиночна мапа). Рахується в /generate-zones.
    terrain_relief_range_m: float = 0.0
    # Preserve global XY coordinates (do NOT center per tile) for perfect stitching across zones/sessions.
    preserve_global_xy: bool = False
    # Explicit Grid Step (meters) for perfect stitching (avoids legacy resolution-based gaps)
    grid_step_m: Optional[float] = None
    # Explicit Hex size for grid generation
    hex_size_m: float = Field(default=300.0, ge=100.0, le=2000.0)
    # AMS / Flat Mode: Optimized for multicolor printing (Flat terrain + Fixed layers)
    is_ams_mode: bool = False
    # Layered plate mode: flat solid base plus additive water/road/park/building layers.
    flat_plate_mode: bool = False
    flat_water_layer_mm: float = Field(default=0.22, ge=0.0, le=5.0)
    flat_roads_layer_mm: float = Field(default=0.42, ge=0.0, le=5.0)
    flat_parks_layer_mm: float = Field(default=0.36, ge=0.0, le=5.0)
    flat_max_building_height_mm: float = Field(default=0.0, ge=0.0, le=20.0)
    flat_uniform_building_height: bool = False
    # Кольорова тема/палітра (#2): classic (дефолт, без змін) | sepia | noir | ocean | neon.
    # Застосовується post-export як перепатч m:colorgroup 3MF (стилістика «настрою» карти).
    color_palette: str = "classic"
    # Мапа-магніт: кругла кишеня під магніт у центрі дна (плаский режим).
    magnet_pocket: bool = False
    magnet_pocket_diameter_mm: float = Field(default=10.4, ge=4.0, le=30.0)
    magnet_pocket_depth_mm: float = Field(default=2.0, ge=1.0, le=4.0)
    # Кілька кишень (шайби Ø4×2мм): 1 = центр (старий режим), 4 = діагональне
    # кільце по кутах з відступом inset від краю. Розкладку рахує
    # build_magnet_pocket_geometry (flat_plate_pipeline).
    magnet_pocket_count: int = Field(default=1, ge=1, le=8)
    magnet_pocket_inset_mm: float = Field(default=8.0, ge=3.0, le=30.0)
    # Підпис на плоскій мапі/магніті: рельєфний текст у смузі внизу плити.
    map_label: str = Field(default="", max_length=40)
    map_label_text_height_mm: float = Field(default=5.0, ge=2.5, le=12.0)
    # З'ЄДНУВАЧ-ПАЗИ (метелик/bowtie): універсальні «ластівчин-хвіст» пази на
    # серединах граней плоскої карти + окрема деталь-ключ. Дві карти/плитки
    # стикуються пазами; паз ріжеться у ДНІ ≥3мм основи → спереду шов непомітний.
    # build_map_connector_geometry (flat_plate_pipeline); FDM-кліренс 0.2мм/бік.
    map_connector: bool = False
    map_connector_edges: str = Field(default="NSEW", max_length=4)
    map_connector_span_mm: float = Field(default=10.0, ge=4.0, le=30.0)
    map_connector_length_mm: float = Field(default=15.0, ge=6.0, le=40.0)
    map_connector_depth_mm: float = Field(default=2.0, ge=0.2, le=4.0)
    map_connector_clearance_mm: float = Field(default=0.03, ge=0.02, le=0.6)
    # Грані, для яких випускаємо КЛЮЧ (для серії — лише S/E внутрішні, 1 ключ/шов).
    # Порожнє → ключ для кожного пазу (single-tile). Паз ріжемо на всіх map_connector_edges.
    map_connector_key_edges: str = Field(default="", max_length=4)
    # НАПРЯМКОВИЙ добір граней (азимути нормалі у градусах, "30,90,150") — для серії
    # шестикутників/повернутих клітин, де NSEW (4 кардинали) не адресує 6 граней.
    # Порожнє → стара NSEW-поведінка. *_key_az = підмножина з ключем (1 на шов).
    map_connector_edge_az: str = Field(default="", max_length=200)
    map_connector_key_az: str = Field(default="", max_length=200)
    # ПРЕМІУМ-РАМКА: компас (стрілка-N), масштабна лінійка (0…N м) і координати
    # центру (lat/lon) окремою чорною деталлю «Frame», вирізаною з шарів карти.
    # build_map_frame_overlay (flat_plate_pipeline). Працює у flat_plate.
    map_frame: bool = False
    map_frame_compass: bool = True
    map_frame_scale: bool = True
    map_frame_coords: bool = True
    # СТИЛЬ ОРНАМЕНТАЛЬНОЇ РАМКИ: "classic" = поточна поведінка (компас+лінійка+
    # координати, без зовнішнього ободка); "ornate" = декоративний підведений
    # подвійний ободок по периметру + кутові мотиви; "compass" = ті ж елементи +
    # тонкий зовнішній ободок. Рендериться у build_map_frame_overlay.
    frame_style: str = "classic"
    # ВИДІЛЕНА БУДІВЛЯ на карті: користувач обирає свій будинок (highlight_point
    # [lon,lat] від кліку по карті; інакше — будинок у центрі) → ОКРЕМА ЧЕРВОНА
    # вставна деталь (паз+peg). build_highlight_insert (flat_plate_pipeline).
    map_highlight_building: bool = False
    highlight_point: Optional[List[float]] = Field(default=None, max_length=2)
    # Кілька будинків: список [[lon,lat],...] (дім, робота, орієнтири) — кожен окрема
    # ЧЕРВОНА вставна деталь. Має пріоритет над highlight_point. Кап 12 у пайплайні.
    highlight_points: Optional[List[List[float]]] = Field(default=None, max_length=12)
    keychain_mode: bool = False
    keychain_label: str = Field(default="", max_length=64)
    keychain_base_shape: str = Field(default="rounded", max_length=24)
    keychain_layout_rotation_deg: float = Field(default=0.0, ge=0.0, le=360.0)
    keychain_loop_style: str = Field(default="round", max_length=24)
    keychain_loop_angle_deg: float = Field(default=0.0, ge=0.0, le=360.0)
    keychain_body_width_mm: float = Field(default=35.0, ge=20.0, le=180.0)
    keychain_body_height_mm: float = Field(default=55.0, ge=16.0, le=140.0)
    keychain_map_x_mm: float = Field(default=2.0, ge=0.0, le=180.0)
    keychain_map_y_mm: float = Field(default=3.0, ge=0.0, le=140.0)
    keychain_map_width_mm: float = Field(default=31.0, ge=4.0, le=180.0)
    keychain_map_height_mm: float = Field(default=40.0, ge=4.0, le=140.0)
    keychain_map_rotation_deg: float = Field(default=0.0, ge=0.0, le=360.0)
    keychain_loop_center_x_mm: float = Field(default=17.5, ge=-30.0, le=210.0)
    keychain_loop_center_y_mm: float = Field(default=-4.0, ge=-40.0, le=180.0)
    keychain_label_center_x_mm: float = Field(default=17.5, ge=0.0, le=180.0)
    keychain_label_center_y_mm: float = Field(default=49.5, ge=0.0, le=140.0)
    keychain_label_angle_deg: float = Field(default=0.0, ge=0.0, le=360.0)
    keychain_loop_outer_radius_mm: float = Field(default=6.5, ge=2.4, le=18.0)
    keychain_loop_inner_radius_mm: float = Field(default=3.0, ge=1.5, le=12.0)
    keychain_corner_radius_mm: float = Field(default=4.0, ge=0.0, le=16.0)
    keychain_label_band_height_mm: float = Field(default=9.0, ge=0.0, le=30.0)
    keychain_label_raise_mm: float = Field(default=0.6, ge=0.0, le=3.0)
    keychain_label_text_height_mm: float = Field(default=4.2, ge=2.0, le=12.0)
    keychain_label_width_mm: float = Field(default=30.0, ge=4.0, le=180.0)
    keychain_label_stroke_mm: float = Field(default=0.9, ge=0.8, le=3.0)
    keychain_label_font_style: str = Field(default="block", max_length=24)
    keychain_rim_width_mm: float = Field(default=1.2, ge=0.0, le=6.0)
    keychain_rim_height_mm: float = Field(default=0.45, ge=0.0, le=3.0)
    # Другий рядок напису (дата/координати) — менший кегль, під основним.
    keychain_label2: str = Field(default="", max_length=64)
    keychain_label2_text_height_mm: float = Field(default=2.4, ge=1.5, le=8.0)
    # Напис на ЗВОРОТІ: гравіюється у нижню грань (дзеркально — читається при перевороті).
    keychain_back_label: str = Field(default="", max_length=64)
    keychain_back_text_height_mm: float = Field(default=5.0, ge=2.5, le=14.0)
    keychain_back_engrave_mm: float = Field(default=0.5, ge=0.2, le=1.2)
    # ТОПО-БРЕЛОК (C3): heightfield-рельєф висот на жетоні замість карти
    # (дороги/вода/парки/будівлі не друкуються). keychain_relief_mm — макс.
    # висота рельєфу над базою.
    keychain_topo_mode: bool = False
    keychain_relief_mm: float = Field(default=2.2, ge=0.6, le=4.0)
    # МАРКЕР «особливе місце»: піднята фігурка (heart/star/circle) у центрі карти
    # (= точка, яку шукав користувач). Окремий теракотовий шар. "" = вимкнено.
    keychain_place_marker: str = Field(default="", max_length=12)
    keychain_place_marker_size_mm: float = Field(default=6.0, ge=3.0, le=14.0)
    # ПІДСВІТКА БУДИНКУ: будинок у ЦЕНТРІ карти виноситься ОКРЕМОЮ деталлю іншого
    # кольору (друкується окремо/іншим філаментом, приклеюється/вставляється на місце).
    # v1 = окрема золота деталь "Highlight" (будинок прибрано з шару buildings → пляма
    # пласка). v2 (наступне) = паз у базі + peg для механічної вставки.
    keychain_highlight_building: bool = False
    # D4 GPX-ТРЕК: маршрут [[lon,lat],...] як підвищений шар поверх мапи
    # (фронт парсить .gpx сам). Ріжеться по зоні; на рельєфі — шапка по терейну.
    gpx_track: Optional[List[List[float]]] = Field(default=None, max_length=8000)
    gpx_width_mm: float = Field(default=1.2, ge=0.6, le=3.0)
    gpx_raise_mm: float = Field(default=0.6, ge=0.2, le=1.5)
    canonical_mask_bundle_dir: Optional[str] = None
    auto_canonicalize_masks: bool = True


class GenerationResponse(BaseModel):
    """Р’С–РґРїРѕРІС–РґСЊ Р· ID Р·Р°РґР°С‡С–"""
    task_id: str
    status: str
    message: Optional[str] = None
    all_task_ids: Optional[List[str]] = None  # Р”Р»СЏ РјРЅРѕР¶РёРЅРЅРёС… Р·РѕРЅ


@app.get("/")
async def root():
    return {"message": "3D Map Generator API", "version": "1.0.0"}


_PRICING_CACHE: Dict[str, Any] = {"mtime": 0.0, "data": None}
_PRICING_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pricing.json")


def _load_pricing() -> Dict[str, Any]:
    """pricing.json з mtime-кешем — правиться на сервері без рестарту."""
    import json
    try:
        mtime = os.path.getmtime(_PRICING_PATH)
        if _PRICING_CACHE["data"] is None or mtime != _PRICING_CACHE["mtime"]:
            with open(_PRICING_PATH, "r", encoding="utf-8") as f:
                _PRICING_CACHE["data"] = json.load(f)
            _PRICING_CACHE["mtime"] = mtime
    except Exception as e:  # noqa: BLE001
        print(f"[PRICING] load failed: {e}")
        if _PRICING_CACHE["data"] is None:
            # Аварійний fallback МУСИТЬ збігатися з pricing.json (інакше при збої
            # читання файлу клієнт переплатив би ~40-67%). Звірено з pricing.json
            # 2026-06-18: S150/M240(60мм магніт 150)/L360/XL550, рельєф +60, from 150.
            _PRICING_CACHE["data"] = {
                "currency": "UAH", "currency_symbol": "₴",
                "map": {"sizes_mm": {"55": 150, "60": 150, "80": 240, "110": 360, "150": 550},
                        "relief_addon": 60, "from": 150},
                "keychain": {"base": 120, "from": 120},
            }
    return _PRICING_CACHE["data"]


@app.get("/api/quote")
async def get_quote(product: str = "map", size_mm: Optional[float] = None, relief: bool = False):
    """Орієнтовна ціна для UI (sticky-bar, форма замовлення). НЕ оферта."""
    p = _load_pricing()
    sym = p.get("currency_symbol", "₴")
    if product == "keychain":
        price = int(p.get("keychain", {}).get("base", 120))
    else:
        sizes = {float(k): int(v) for k, v in p.get("map", {}).get("sizes_mm", {"55": 250}).items()}
        if size_mm:
            nearest = min(sizes.keys(), key=lambda k: abs(k - float(size_mm)))
            price = sizes[nearest]
        else:
            price = int(p.get("map", {}).get("from", min(sizes.values())))
        if relief:
            price += int(p.get("map", {}).get("relief_addon", 0))
    return {
        "currency": p.get("currency", "UAH"),
        "price": price,
        "formatted": f"{price} {sym}",
        "approx": False,
    }


def _nearest_map_price(sizes: Dict[float, int], size_mm: Optional[float]) -> int:
    if not sizes:
        return 0
    if size_mm is None:
        return min(sizes.values())
    nearest = min(sizes.keys(), key=lambda k: abs(k - float(size_mm)))
    return int(sizes[nearest])


def _compute_authoritative_amount(
    product_type: str,
    task: Optional["GenerationTask"],
    pricing: Dict[str, Any],
) -> Tuple[float, str]:
    """ПЛАТІЖНА ЦІЛІСНІСТЬ: рахуємо суму до сплати НА СЕРВЕРІ з pricing.json та
    РЕАЛЬНИХ параметрів задачі (model_size_mm/relief/magnet/панно-плитки), а НЕ
    з est_price, який надсилає клієнт (його можна підробити). Повертає (UAH, ccy).

    Для панно (batch) сума = ціна однієї плитки × кількість плиток.
    """
    currency = pricing.get("currency") or "UAH"
    map_cfg = pricing.get("map", {}) or {}
    sizes = {float(k): int(v) for k, v in (map_cfg.get("sizes_mm", {}) or {}).items()}
    req = getattr(task, "request", None) if task is not None else None

    if (product_type or "map") == "keychain":
        amount = float((pricing.get("keychain", {}) or {}).get("base", 120))
        return round(amount, 2), currency

    # ── map / magnet ────────────────────────────────────────────────────────
    size_mm = None
    relief = False
    is_magnet = False
    tile_count = 1
    if req is not None:
        try:
            size_mm = float(getattr(req, "model_size_mm", None) or 0.0) or None
        except Exception:  # noqa: BLE001
            size_mm = None
        try:
            # Relief addon застосовуємо лише якщо рельєф реально увімкнено й
            # модель не плоска плита (flat_plate=без 3D-рельєфу).
            relief = bool(getattr(req, "terrain_enabled", False)) and not bool(
                getattr(req, "flat_plate_mode", False)
            )
        except Exception:  # noqa: BLE001
            relief = False
        try:
            is_magnet = bool(getattr(req, "magnet_pocket", False))
        except Exception:  # noqa: BLE001
            is_magnet = False

    # Магніт має власну ціну (ключ "60" у sizes_mm = 180₴ за прайсом).
    if is_magnet and 60.0 in sizes:
        amount = float(sizes[60.0])
        return round(amount, 2), currency

    # Невідомий розмір (немає задачі) → беремо стартову ціну `from`, а НЕ
    # найдешевший рядок (інакше випадково взяли б ціну магніта "60"=180).
    if size_mm is None:
        base = float(map_cfg.get("from", 150) or 150)
    else:
        base = float(_nearest_map_price(sizes, size_mm) or map_cfg.get("from", 150))
    if relief:
        base += float(map_cfg.get("relief_addon", 0) or 0)

    # ПАННО: якщо task_id — це batch, рахуємо за всі плитки.
    return round(base * max(1, int(tile_count)), 2), currency


@app.get("/api/building-at")
async def building_at(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    _rl: None = Depends(rate_limit("building_at", [(120, 60.0)])),
):
    """Контур будівлі у точці (для підсвітки «свій будинок» на карті). Бере той
    самий локальний OSM-DuckDB, що й генератор → ШВИДКО (без Overpass) і ТОЧНО
    збігається з тим, що надрукується. Повертає {"footprint": [[lon,lat],...]} або None."""
    try:
        from services.local_osm_db import get_gdf, is_available
        if not is_available():
            return {"footprint": None}
        import math as _math
        from shapely.geometry import Point as _Pt
        _m = 70.0
        _dlat = _m / 111320.0
        _dlon = _m / (111320.0 * max(_math.cos(_math.radians(lat)), 0.05))
        gdf = get_gdf("buildings", lat + _dlat, lat - _dlat, lon + _dlon, lon - _dlon)
        if gdf is None or gdf.empty:
            return {"footprint": None}
        pt = _Pt(lon, lat)
        best, best_d = None, float("inf")
        for geom in gdf.geometry.values:
            if geom is None or getattr(geom, "is_empty", True):
                continue
            try:
                if geom.contains(pt):  # клік усередині будівлі — точний вибір
                    best = geom
                    break
                d = geom.distance(pt)
                if d < best_d:
                    best_d, best = d, geom
            except Exception:
                continue
        if best is None:
            return {"footprint": None}
        if best.geom_type == "MultiPolygon":
            best = max(best.geoms, key=lambda g: g.area)
        if best.geom_type != "Polygon":
            return {"footprint": None}
        coords = [[float(x), float(y)] for x, y in best.exterior.coords]
        return {"footprint": coords}
    except Exception as exc:
        print(f"[BUILDING-AT] failed: {exc}")
        return {"footprint": None}


@app.get("/api/health")
async def health():
    """Lightweight health probe for monitoring/alerting.

    Checks process liveness + that the output directory is writable.
    Returns 200 with status="ok" when healthy; status="degraded" otherwise
    (still HTTP 200 so probes can read the detail).
    """
    import time as _time
    checks = {}
    ok = True
    # output dir writable
    try:
        _probe = os.path.join(OUTPUT_DIR, ".health_probe")
        with open(_probe, "w") as _f:
            _f.write("ok")
        os.remove(_probe)
        checks["output_writable"] = True
    except Exception as _e:  # noqa: BLE001
        checks["output_writable"] = False
        checks["output_error"] = str(_e)
        ok = False
    # free disk space (GB) on output volume
    try:
        import shutil as _shutil
        _free = _shutil.disk_usage(OUTPUT_DIR).free / (1024 ** 3)
        checks["disk_free_gb"] = round(_free, 2)
        if _free < 1.0:
            ok = False
    except Exception:  # noqa: BLE001
        pass
    return {
        "status": "ok" if ok else "degraded",
        "version": "1.0.0",
        "ts": int(_time.time()),
        "checks": checks,
    }


# Карта код-країни → координати столиці (для авто-центрування карти /create за IP).
# Cloudflare додає заголовок CF-IPCountry на origin-запитах → читаємо його ТУТ
# (без зовнішніх викликів — тривіально й швидко). Невідома/відсутня країна → Київ.
COUNTRY_CENTER = {
    "UA": (50.4501, 30.5234, "Київ"),
    "PL": (52.2297, 21.0122, "Warszawa"),
    "DE": (52.52, 13.405, "Berlin"),
    "CZ": (50.0755, 14.4378, "Praha"),
    "ES": (40.4168, -3.7038, "Madrid"),
    "IT": (41.9028, 12.4964, "Roma"),
    "FR": (48.8566, 2.3522, "Paris"),
    "GB": (51.5074, -0.1278, "London"),
    "US": (40.7128, -74.006, "New York"),
    "CA": (45.4215, -75.6972, "Ottawa"),
    "PT": (38.7223, -9.1393, "Lisboa"),
    "NL": (52.3676, 4.9041, "Amsterdam"),
    "AT": (48.2082, 16.3738, "Wien"),
    "SK": (48.1486, 17.1077, "Bratislava"),
    "RO": (44.4268, 26.1025, "Bucuresti"),
    "LT": (54.6872, 25.2797, "Vilnius"),
    "LV": (56.9496, 24.1052, "Riga"),
    "EE": (59.437, 24.7536, "Tallinn"),
    "HU": (47.4979, 19.0402, "Budapest"),
    "SE": (59.3293, 18.0686, "Stockholm"),
    "NO": (59.9139, 10.7522, "Oslo"),
    "FI": (60.1699, 24.9384, "Helsinki"),
    "IE": (53.3498, -6.2603, "Dublin"),
    "CH": (47.3769, 8.5417, "Zurich"),
    "BE": (50.8503, 4.3517, "Brussels"),
}


@app.get("/api/geo")
async def geo(request: Request):
    """Геолокація за IP через Cloudflare CF-IPCountry → центр карти /create.

    Cloudflare проксує origin-запити із заголовком CF-IPCountry (2-літерний код).
    Мапимо його на координати столиці. Невідома/відсутня країна → Київ (UA).
    Жодних зовнішніх викликів — тривіально й швидко.
    """
    code = (request.headers.get("cf-ipcountry", "") or "").strip().upper()
    lat, lng, label = COUNTRY_CENTER.get(code, COUNTRY_CENTER["UA"])
    return {
        "country": code if code in COUNTRY_CENTER else "UA",
        "lat": lat,
        "lng": lng,
        "label": label,
    }


class OrderRequest(BaseModel):
    name: str = Field(max_length=80)
    phone: str = Field(default="", max_length=32)
    product_type: str = Field(default="map", max_length=24)   # "map" | "keychain"
    task_id: Optional[str] = Field(default=None, max_length=128)
    delivery_method: str = Field(default="", max_length=32)    # "nova" | "ukr" | "pickup" | ...
    delivery_country: str = Field(default="", max_length=80)   # "Україна" або країна ЄС
    delivery_city: str = Field(default="", max_length=120)
    delivery_branch: str = Field(default="", max_length=120)
    delivery_address: str = Field(default="", max_length=300)
    comment: str = Field(default="", max_length=2000)
    est_price: str = Field(default="", max_length=64)          # інфо-нотатка з сайту (НЕ авторитет для оплати)
    summary: Dict[str, Any] = {}
    screenshots: List[str] = []         # data:image/png;base64,... (трим до 4 у валідаторі)

    @model_validator(mode="after")
    def _cap_screenshots(self):
        # Лояльний кап (НЕ 422): беремо щонайбільше 4 перших і відкидаємо завеликі
        # (>3МБ data-URL) — захист від роздування пейлоада/памʼяті без відмови
        # клієнту з валідним замовленням.
        if self.screenshots:
            self.screenshots = [
                s for s in self.screenshots[:4]
                if isinstance(s, str) and len(s) <= 3_000_000
            ]
        return self


async def _deliver_order_model_when_ready(order_number, name: str, task_id: str) -> None:
    """Фоновий вотчер: order-now шле картку ОДРАЗУ (3MF ще генерується). Коли повна
    генерація завершиться — дошлемо оператору друкарський .3mf окремим повідомленням.
    Полимо до ~6 хв; ніколи не шлемо GLB (лише .3mf/.stl)."""
    import asyncio as _aio
    from services.order_service import send_model_document
    for _ in range(72):  # ~6 хв (72 × 5с)
        await _aio.sleep(5)
        try:
            path = None
            t = tasks.get(task_id)
            if t is not None:
                files = getattr(t, "output_files", {}) or {}
                cand = files.get("3mf") or files.get("stl")
                if not cand:
                    _of = getattr(t, "output_file", None)
                    if _of and str(_of).lower().endswith((".3mf", ".stl")):
                        cand = _of
                if cand and Path(cand).exists():
                    path = Path(cand)
            if path is None:
                p = _find_file_on_disk_by_task_id(task_id, "3mf")
                if p is not None and str(p).lower().endswith((".3mf", ".stl")) and Path(p).exists():
                    path = Path(p)
            if path is not None:
                send_model_document(order_number, name, path)
                return
            # Якщо задача впала — припиняємо чекати.
            if t is not None and getattr(t, "status", "") in ("failed", "error"):
                return
        except Exception:  # noqa: BLE001
            continue


# ── Nova Poshta: пошук міста + відділення для форми замовлення ──────────────
# Ключ читається server-side (NOVA_POSHTA_API_KEY). Без ключа → configured:false
# → фронт показує ручне введення (як було). Проксі ховає ключ від клієнта.
@app.get("/api/delivery/np/status")
async def np_status():
    from services import nova_poshta as _np
    return {"configured": _np.is_configured()}


@app.get("/api/delivery/np/cities")
async def np_cities(
    q: str = "",
    _rl: None = Depends(rate_limit("np", [(90, 60.0), (1500, 3600.0)])),
):
    from services import nova_poshta as _np
    if not _np.is_configured():
        return {"configured": False, "items": []}
    return {"configured": True, "items": _np.search_cities(q)}


@app.get("/api/delivery/np/warehouses")
async def np_warehouses(
    cityRef: str = "",
    q: str = "",
    _rl: None = Depends(rate_limit("np", [(90, 60.0), (1500, 3600.0)])),
):
    from services import nova_poshta as _np
    if not _np.is_configured():
        return {"configured": False, "items": []}
    return {"configured": True, "items": _np.search_warehouses(cityRef, q)}


@app.post("/api/order")
async def create_order_endpoint(
    order: OrderRequest,
    authorization: Optional[str] = Header(default=None),
    _rl: None = Depends(rate_limit("order", [(5, 3600.0)])),
):
    """Accept a customer order and push it to the Telegram CRM (card + file + screenshots)."""
    from services.order_service import create_order
    # Чіткі 4xx замість 500 на неповному запиті (порожні поля / невідомий продукт).
    if not (order.name or "").strip():
        raise HTTPException(status_code=422, detail="Вкажіть ім'я")
    if not (order.phone or "").strip():
        raise HTTPException(status_code=422, detail="Вкажіть номер телефону для звʼязку")
    _ptype = (order.product_type or "map").strip().lower()
    if _ptype not in ("map", "keychain"):
        raise HTTPException(status_code=422, detail="Невідомий тип виробу (очікується «map» або «keychain»).")
    # СЕРВЕР-САЙД валідація доставки (дзеркало OrderDialog.tsx) — щоб обхід форми
    # через прямий API не створював недоставних замовлень (без міста/відділення).
    _dm = (order.delivery_method or "").strip().lower()
    _dcountry = (order.delivery_country or "").strip()
    _dcity = (order.delivery_city or "").strip()
    _dbranch = (order.delivery_branch or "").strip()
    _daddress = (order.delivery_address or "").strip()
    _EU_METHODS = {"novapost_eu", "meest"}
    if _dm in _EU_METHODS:
        if not _dcountry:
            raise HTTPException(status_code=422, detail="Вкажіть країну доставки")
        if not _dcity:
            raise HTTPException(status_code=422, detail="Вкажіть місто доставки")
        if _dm == "novapost_eu" and not _dbranch:
            raise HTTPException(status_code=422, detail="Вкажіть відділення Nova Post")
        if _dm == "meest" and not _daddress:
            raise HTTPException(status_code=422, detail="Вкажіть адресу доставки")
    elif _dm != "pickup":
        # Будь-який не-самовивіз (nova/ukr/порожній метод): потрібні місто + відділення.
        if not _dcity:
            raise HTTPException(status_code=422, detail="Вкажіть місто доставки")
        if not _dbranch:
            raise HTTPException(status_code=422, detail="Вкажіть відділення пошти")
        if _dm == "ukr" and not _daddress:
            raise HTTPException(status_code=422, detail="Вкажіть адресу (вулиця/будинок) для Укрпошти")
    payload = order.model_dump()
    payload["product_type"] = _ptype
    # Мʼяка привʼязка до акаунта: якщо клієнт залогінений — замовлення видно в кабінеті.
    try:
        from services.auth_service import verify_token
        user = verify_token(authorization or "")
        if user:
            payload["uid"] = user.get("uid")
            payload["user_email"] = user.get("email")
    except Exception:  # noqa: BLE001
        pass
    # Посилання на оплату з конфігу (pricing.json → payment.url). Якщо задано —
    # клієнт побачить кнопку «Оплатити зараз», оператор — позначку в картці.
    pricing = _load_pricing()
    pay = (pricing.get("payment") or {})
    pay_url = (pay.get("url") or "").strip()
    if pay_url:
        payload["payment_url"] = pay_url
    # Resolve the on-disk model file from the in-memory task if available.
    # ЛИШЕ друкарський .3mf/.stl — НІКОЛИ GLB-прев'ю (інакше оператор отримає
    # «битий» файл .3mf із GLB-вмістом). Якщо 3MF ще генерується — output_file не
    # ставимо; фоновий вотчер нижче дошле файл оператору, щойно він буде готовий.
    try:
        t = tasks.get(order.task_id) if order.task_id else None
        if t is not None:
            files = getattr(t, "output_files", {}) or {}
            of = files.get("3mf") or files.get("stl")
            if not of:
                _of = getattr(t, "output_file", None)
                if _of and str(_of).lower().endswith((".3mf", ".stl")):
                    of = _of
            if of:
                payload["output_file"] = of
    except Exception:  # noqa: BLE001
        pass
    try:
        result = create_order(payload)
        # Фоновий вотчер: якщо друкарський 3MF ще генерувався на момент замовлення
        # (order-now), дошлемо оператору файл окремо, щойно генерація завершиться.
        if order.task_id and "output_file" not in payload:
            try:
                import asyncio as _aio
                _aio.create_task(_deliver_order_model_when_ready(
                    result.get("order_number"), order.name or "", order.task_id))
            except Exception:  # noqa: BLE001
                pass
        # ОПЛАТА: динамічний LiqPay-checkout (якщо ключі налаштовані) має пріоритет над
        # статичним посиланням з конфігу.
        # ПЛАТІЖНА ЦІЛІСНІСТЬ: суму до сплати рахуємо НА СЕРВЕРІ з pricing.json та
        # реальних параметрів задачі — НЕ довіряємо order.est_price (його можна
        # підробити в запиті). est_price лишається лише як інфо-нотатка в картці.
        try:
            from services.liqpay import is_configured, parse_amount, build_checkout
            if is_configured():
                # Authoritative amount from server-side params (не з клієнта).
                _order_task = tasks.get(order.task_id) if order.task_id else None
                amount, currency = _compute_authoritative_amount(
                    order.product_type, _order_task, pricing
                )
                # ПАННО (batch): сума × кількість плиток.
                try:
                    if order.task_id and str(order.task_id).startswith("batch_"):
                        _tile_ids = multiple_tasks_map.get(order.task_id) or []
                        if _tile_ids:
                            amount = round(amount * len(_tile_ids), 2)
                except Exception:  # noqa: BLE001
                    pass
                # parse_amount лишається в коді (фід — серверна цифра, не клієнтська).
                amount, currency = parse_amount(f"{amount:.2f}", order.product_type, pricing)
                site = (os.getenv("PUBLIC_SITE_URL") or "https://monadruk.com").rstrip("/")
                checkout = build_checkout(
                    amount=amount, currency=currency,
                    description=f"Monadruk #{result.get('order_number')} · {order.product_type}",
                    order_id=str(result.get("order_number") or order.task_id or ""),
                    result_url=f"{site}/account",
                    server_url=f"{site}/api/liqpay/callback",
                )
                if checkout:
                    result["payment"] = {**checkout, "amount": amount, "currency": currency,
                                         "label": pay.get("label_uk") or "Оплатити зараз"}
        except Exception as _pe:  # noqa: BLE001
            print(f"[liqpay] checkout build failed: {_pe}")
        if "payment" not in result and pay_url:
            result["payment"] = {"url": pay_url, "label": pay.get("label_uk") or "Оплатити зараз"}
        return {"status": "ok", **result}
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] order failed: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail="Не вдалося оформити замовлення")


@app.post("/api/liqpay/callback")
async def liqpay_callback(data: str = Form(default=""), signature: str = Form(default="")):
    """LiqPay server-callback (webhook): підтвердження статусу оплати. Перевіряємо підпис
    приватним ключем; при success — нотифікуємо оператора в Telegram + лог у журнал.
    LiqPay шле application/x-www-form-urlencoded з полями data + signature."""
    from services.liqpay import verify_callback
    info = verify_callback(data, signature)
    if info is None:
        raise HTTPException(status_code=403, detail="bad signature")
    status = str(info.get("status") or "")
    if status in ("success", "sandbox", "subscribed", "wait_accept"):
        try:
            from services.order_service import mark_order_paid
            mark_order_paid(str(info.get("order_id") or ""), info)
        except Exception as exc:  # noqa: BLE001
            print(f"[liqpay] mark_paid error: {exc}")
    return {"ok": True}


@app.get("/api/account/orders")
async def account_orders(authorization: Optional[str] = Header(default=None)):
    """Замовлення поточного користувача (з orders-журналу, новіші перші)."""
    from services.order_service import list_orders_for_uid
    u = _require_user(authorization)
    return {"orders": list_orders_for_uid(u["uid"])}


class ContactRequest(BaseModel):
    name: str = Field(default="", max_length=80)
    phone: str = Field(max_length=32)
    message: str = Field(default="", max_length=2000)
    source: str = Field(default="", max_length=200)


@app.post("/api/contact")
async def contact_endpoint(
    req: ContactRequest,
    _rl: None = Depends(rate_limit("contact", [(5, 3600.0)])),
):
    """Customer 'leave a request' → Telegram CRM."""
    from services.order_service import send_contact
    if not (req.phone or "").strip():
        raise HTTPException(status_code=422, detail="Вкажіть телефон")
    ok = send_contact(req.name, req.phone, req.message, req.source)
    return {"status": "ok" if ok else "logged"}


# ── Self-hosted analytics (free, no third party; data stays on this server) ─────
class TrackEvent(BaseModel):
    event: str = "pageview"
    path: str = ""
    locale: str = ""
    ref: str = ""
    props: Optional[Dict[str, Any]] = None


ANALYTICS_LOG = DATA_DIR / "analytics.jsonl"


@app.post("/api/track")
async def track_event(
    ev: TrackEvent,
    x_forwarded_for: Optional[str] = Header(default=None),
    user_agent: Optional[str] = Header(default=None),
    cf_ipcountry: Optional[str] = Header(default=None),
    _rl: None = Depends(rate_limit("track", [(120, 60.0)])),
):
    """Append a privacy-friendly analytics event. No raw IP is stored — only a
    daily salted hash, so we can count unique visitors without tracking people.
    Country code comes from Cloudflare (Cf-Ipcountry) — coarse geo, no raw IP."""
    import hashlib, json
    from datetime import datetime, timezone
    try:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        ip = (x_forwarded_for or "").split(",")[0].strip()
        salt = os.getenv("SECRET_KEY", "monadruk")
        visitor = hashlib.sha256(f"{ip}|{user_agent or ''}|{day}|{salt}".encode()).hexdigest()[:16]
        cc = (cf_ipcountry or "").strip().upper()[:2]
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "day": day,
            "event": (ev.event or "pageview")[:64],
            "path": (ev.path or "")[:200],
            "locale": (ev.locale or "")[:8],
            "ref": (ev.ref or "")[:200],
            "visitor": visitor,
            "cc": cc,
        }
        if ev.props:
            try:
                rec["props"] = {str(k)[:40]: str(v)[:120] for k, v in list(ev.props.items())[:10]}
            except Exception:  # noqa: BLE001
                pass
        # Обмежуємо ріст логу (клік-трекінг додає обсяг): при >25МБ ротуємо в .1
        # (одна резервна копія), щоб диск не заповнився. Адмін-статистика читає
        # лише поточний файл (останні ~25МБ подій) — цього достатньо.
        try:
            if ANALYTICS_LOG.exists() and ANALYTICS_LOG.stat().st_size > 25_000_000:
                ANALYTICS_LOG.replace(ANALYTICS_LOG.with_name("analytics.jsonl.1"))
        except Exception:  # noqa: BLE001
            pass
        with ANALYTICS_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        pass
    return {"status": "ok"}


@app.get("/api/admin/stats")
async def admin_stats(authorization: Optional[str] = Header(default=None), days: int = 30):
    """Aggregate analytics.jsonl for the admin dashboard (free, self-hosted)."""
    u = _require_user(authorization)
    if not u["is_admin"]:
        raise HTTPException(status_code=403, detail="Лише для адміністраторів")
    import json
    from collections import Counter, defaultdict
    totals = {"events": 0, "pageviews": 0, "uniqueVisitors": 0}
    by_day: Dict[str, Dict[str, Any]] = {}
    ev_counter: Counter = Counter()
    path_counter: Counter = Counter()
    locale_counter: Counter = Counter()
    country_counter: Counter = Counter()
    ref_counter: Counter = Counter()
    funnel_counter: Counter = Counter()       # крок воронки → к-сть сесій
    click_points: Dict[str, list] = defaultdict(list)  # path → [[x,y],...] для теплокарти
    click_label_counter: Counter = Counter()  # (path, label) → к-сть кліків
    visitors: set = set()
    day_visitors: Dict[str, set] = {}
    # Стрічка ВІЗИТІВ: групуємо події за анонімним visitor-хешем → бачимо кожного
    # відвідувача (анонім) ОКРЕМО: країна, ЗВІДКИ (реферер), які сторінки, коли.
    visitor_sessions: Dict[str, Dict] = {}
    FUNNEL_STEPS = ["view", "area", "generate", "order_open", "order_submit"]
    try:
        if ANALYTICS_LOG.exists():
            for line in ANALYTICS_LOG.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                totals["events"] += 1
                ev = r.get("event", "")
                ev_counter[ev] += 1
                props = r.get("props") or {}
                path = r.get("path", "")
                if ev == "pageview":
                    totals["pageviews"] += 1
                    path_counter[path] += 1
                    if r.get("locale"):
                        locale_counter[r["locale"]] += 1
                    if r.get("cc"):
                        country_counter[r["cc"]] += 1
                    _ref = (r.get("ref") or "").split("?")[0][:60]
                    if _ref and "monadruk" not in _ref:
                        ref_counter[_ref] += 1
                elif ev == "funnel":
                    step = props.get("step")
                    if step:
                        funnel_counter[step] += 1
                elif ev == "click":
                    x, y = props.get("x"), props.get("y")
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)) and len(click_points[path]) < 1200:
                        click_points[path].append([round(float(x), 1), round(float(y), 1)])
                    el = props.get("el")
                    if el:
                        click_label_counter[(path, str(el)[:48])] += 1
                d = r.get("day", "")
                vis = r.get("visitor", "")
                if vis:
                    visitors.add(vis)
                    day_visitors.setdefault(d, set()).add(vis)
                    _ts = r.get("ts", "")
                    vs = visitor_sessions.setdefault(vis, {
                        "id": vis[:6], "cc": "", "ref": "", "paths": [], "events": 0,
                        "first": _ts, "last": _ts, "locale": "",
                    })
                    vs["events"] += 1
                    if _ts > vs["last"]:
                        vs["last"] = _ts
                    if _ts and _ts < vs["first"]:
                        vs["first"] = _ts
                    if not vs["cc"] and r.get("cc"):
                        vs["cc"] = r["cc"]
                    if not vs["locale"] and r.get("locale"):
                        vs["locale"] = r["locale"]
                    if not vs["ref"]:  # реферер входу = перший непорожній не-monadruk
                        _rf = (r.get("ref") or "").split("?")[0][:60]
                        if _rf and "monadruk" not in _rf:
                            vs["ref"] = _rf
                    if ev == "pageview" and path and path not in vs["paths"] and len(vs["paths"]) < 12:
                        vs["paths"].append(path)
                bd = by_day.setdefault(d, {"day": d, "events": 0, "pageviews": 0})
                bd["events"] += 1
                if ev == "pageview":
                    bd["pageviews"] += 1
    except Exception:  # noqa: BLE001
        pass
    totals["uniqueVisitors"] = len(visitors)
    series = sorted(by_day.values(), key=lambda x: x["day"])[-days:]
    for s in series:
        s["visitors"] = len(day_visitors.get(s["day"], set()))
    # Воронка у фіксованому порядку + % від першого кроку (де відвалюються).
    first = funnel_counter.get("view", 0) or 1
    funnel = [{"step": st, "count": funnel_counter.get(st, 0),
               "pct": round(100.0 * funnel_counter.get(st, 0) / first, 1)} for st in FUNNEL_STEPS]
    # Топ-6 сторінок за к-стю кліків (для теплокарти) + топ елементів.
    top_click_paths = sorted(click_points.items(), key=lambda kv: -len(kv[1]))[:6]

    # ── Замовлення + дохід (щоб адмін бачив гроші, а не лише трафік) ────────
    # orders.jsonl містить по записі замовлення + окремі type:payment події.
    # Беремо ПОТОЧНИЙ стан кожного замовлення (останній запис з його order_number),
    # рахуємо кількість, суму est_price (перше число з рядка «≈ 390 ₴») і розбивку
    # за статусом. Оплачений дохід = сума по замовленнях у статусах paid/printed/
    # shipped/done. Best-effort: будь-яка помилка лишає нулі, не валить дашборд.
    orders_summary = {
        "count": 0, "byStatus": {}, "byProduct": {},
        "revenueEstimated": 0.0, "revenuePaid": 0.0, "currency": "₴",
    }
    try:
        import re as _re
        from services.order_service import ORDERS_LOG as _ORDERS_LOG, ORDER_STATUSES as _ORDER_STATUSES

        def _price_num(s: Any) -> float:
            m = _re.search(r"-?\d[\d\s]*[.,]?\d*", str(s or "").replace(" ", " "))
            if not m:
                return 0.0
            try:
                return float(m.group(0).replace(" ", "").replace(",", "."))
            except ValueError:
                return 0.0

        latest: Dict[str, Dict[str, Any]] = {}  # order_number -> поточний стан замовлення
        if _ORDERS_LOG.exists():
            for line in _ORDERS_LOG.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                onum = str(rec.get("order_number") or "")
                if not onum:
                    continue
                if rec.get("type") == "payment":
                    # подія оплати: позначаємо замовлення оплаченим (статус paid),
                    # якщо вже бачили саме замовлення.
                    if rec.get("paid") and onum in latest:
                        latest[onum]["_paid"] = True
                        if latest[onum].get("status") in (None, "new"):
                            latest[onum]["status"] = "paid"
                    continue
                # запис замовлення (можливо оновлений статус) — найсвіжіший виграє
                cur = latest.get(onum) or {}
                cur.update(rec)
                latest[onum] = cur

        paid_states = {"paid", "printed", "shipped", "done"}
        prod_counter: Counter = Counter()
        status_counter: Counter = Counter()
        rev_est = 0.0
        rev_paid = 0.0
        for onum, rec in latest.items():
            status = str(rec.get("status") or "new").lower()
            if status not in _ORDER_STATUSES:
                status = "new"
            status_counter[status] += 1
            prod_counter[str(rec.get("product_type") or "map")] += 1
            price = _price_num(rec.get("est_price"))
            rev_est += price
            if status in paid_states or rec.get("_paid"):
                rev_paid += price
        orders_summary = {
            "count": len(latest),
            "byStatus": dict(status_counter),
            "byProduct": dict(prod_counter),
            "revenueEstimated": round(rev_est, 2),
            "revenuePaid": round(rev_paid, 2),
            "currency": "₴",
        }
    except Exception as _oe:  # noqa: BLE001
        print(f"[admin/stats] orders aggregation failed (non-fatal): {_oe}")

    # Стрічка останніх ВІЗИТІВ (анонім): сортуємо за останньою активністю.
    recent_visitors = sorted(
        visitor_sessions.values(), key=lambda v: v.get("last", ""), reverse=True
    )[:30]
    recent_visitors = [{
        "id": v["id"],
        "cc": v["cc"] or "—",
        "ref": v["ref"] or "(прямий/закладка)",
        "paths": v["paths"][:8],
        "events": v["events"],
        "locale": v["locale"] or "",
        "first": v["first"],
        "last": v["last"],
    } for v in recent_visitors]

    return {
        "totals": totals,
        "byDay": series,
        "orders": orders_summary,
        "topEvents": ev_counter.most_common(15),
        "topPaths": path_counter.most_common(15),
        "byLocale": locale_counter.most_common(10),
        "byCountry": country_counter.most_common(15),
        "topRefs": ref_counter.most_common(10),
        "funnel": funnel,
        "recentVisitors": recent_visitors,
        "clicksByPath": {p: pts for p, pts in top_click_paths},
        "topClicks": [[f"{p or '/'} · {lbl}", c] for (p, lbl), c in click_label_counter.most_common(20)],
    }


# ── Account / auth (Firebase token verified without a service account) ──────────
def _require_user(authorization: Optional[str]) -> Dict[str, Any]:
    from services.auth_service import verify_token
    user = verify_token(authorization or "")
    if not user:
        raise HTTPException(status_code=401, detail="Потрібен вхід")
    return user


@app.get("/api/account/quota")
async def account_quota(authorization: Optional[str] = Header(default=None)):
    from services.user_store import get_quota
    u = _require_user(authorization)
    return {"user": {"email": u.get("email"), "is_admin": u["is_admin"]},
            "quota": get_quota(u["uid"], u.get("email") or "", u["is_admin"])}


@app.get("/api/account/models")
async def account_models(authorization: Optional[str] = Header(default=None)):
    from services.user_store import list_models
    u = _require_user(authorization)
    return {"models": list_models(u["uid"])}


# ── Per-user city grids (save / history / generate neighbouring cells) ──────────
class GridCellModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    row: int
    col: int
    task_id: Optional[str] = None
    download_url: Optional[str] = None


class GridSaveRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: Optional[str] = None
    name: str = ""
    city: str = ""
    center: Optional[list] = None
    grid_type: str = "hexagonal"
    hex_size_m: float = 300.0
    bounds: Optional[dict] = None
    rotation_deg: float = 0.0
    cells: Optional[list] = None


@app.get("/api/account/grids")
async def account_grids_list(authorization: Optional[str] = Header(default=None)):
    from services.user_store import list_grids
    u = _require_user(authorization)
    return {"grids": list_grids(u["uid"])}


@app.post("/api/account/grids")
async def account_grids_save(req: GridSaveRequest, authorization: Optional[str] = Header(default=None)):
    from services.user_store import save_grid
    u = _require_user(authorization)
    grid = save_grid(u["uid"], u.get("email") or "", req.model_dump(exclude_none=True))
    return {"ok": True, "grid": grid}


@app.get("/api/account/grids/{grid_id}")
async def account_grid_get(grid_id: str, authorization: Optional[str] = Header(default=None)):
    from services.user_store import get_grid
    u = _require_user(authorization)
    g = get_grid(u["uid"], grid_id)
    if not g:
        raise HTTPException(status_code=404, detail="Сітку не знайдено")
    return {"grid": g}


@app.delete("/api/account/grids/{grid_id}")
async def account_grid_delete(grid_id: str, authorization: Optional[str] = Header(default=None)):
    from services.user_store import delete_grid
    u = _require_user(authorization)
    return {"ok": delete_grid(u["uid"], grid_id)}


class GridCellMarkRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    row: int
    col: int
    task_id: Optional[str] = None
    download_url: Optional[str] = None


@app.post("/api/account/grids/{grid_id}/cells")
async def account_grid_mark_cell(grid_id: str, req: GridCellMarkRequest,
                                 authorization: Optional[str] = Header(default=None)):
    """Record a generated cell within a saved grid (so history shows progress)."""
    from services.user_store import mark_grid_cell
    u = _require_user(authorization)
    g = mark_grid_cell(u["uid"], grid_id, req.model_dump(exclude_none=True))
    if not g:
        raise HTTPException(status_code=404, detail="Сітку не знайдено")
    return {"ok": True, "grid": g}


class TrackModelRequest(BaseModel):
    task_id: str
    title: str = ""
    city: str = ""
    product_type: str = "map"
    download_url: str = ""


@app.post("/api/account/track")
async def account_track(req: TrackModelRequest, authorization: Optional[str] = Header(default=None)):
    """Associate a generated model with the signed-in user (for their history)."""
    from services.user_store import add_model
    u = _require_user(authorization)
    add_model(u["uid"], u.get("email") or "", {
        "task_id": req.task_id, "title": req.title, "city": req.city,
        "product_type": req.product_type, "download_url": req.download_url,
    })
    return {"status": "ok"}


class DownloadGrantRequest(BaseModel):
    task_id: str
    title: str = ""
    city: str = ""
    product_type: str = "map"
    download_url: str = ""
    preview: str = ""   # optional small PNG data-URL thumbnail for the account history


def _resolve_model_path(req: "DownloadGrantRequest") -> Optional[Path]:
    """Find the on-disk ДРУКАРСЬКИЙ model file (.3mf/.stl) for a download.
    КРИТИЧНО: НІКОЛИ не віддаємо GLB-прев'ю — слайсер його не відкриє, юзер бачить
    «битий файл». Якщо є лише GLB (preview-таск) і немає .3mf — повертаємо None,
    щоб віддати чесну помилку «ще готується», а не зламаний файл."""
    def _is_print(pth) -> bool:
        return bool(pth) and str(pth).lower().endswith((".3mf", ".stl"))
    # from in-memory task — лише друкарські формати
    t = tasks.get(req.task_id) if req.task_id else None
    if t is not None:
        files = getattr(t, "output_files", {}) or {}
        for cand in (files.get("3mf"), files.get("stl"), getattr(t, "output_file", None)):
            if _is_print(cand) and Path(cand).exists():
                return Path(cand)
    # from a provided /files/<name> url — лише якщо це .3mf/.stl
    if req.download_url:
        name = req.download_url.split("/")[-1].split("?")[0]
        if _is_print(name):
            p = OUTPUT_DIR / name
            if p.exists():
                return p
    # by task-id on disk — шукаємо саме .3mf
    if req.task_id:
        p = _find_file_on_disk_by_task_id(req.task_id, "3mf")
        if _is_print(p) and Path(p).exists():
            return Path(p)
    return None


@app.post("/api/account/download")
async def account_download(req: DownloadGrantRequest, authorization: Optional[str] = Header(default=None)):
    """Quota-gated download. Verifies the Firebase token, enforces the free
    limit (402 when exhausted for non-admins) and STREAMS the file so the full
    model is only ever delivered through this authenticated path."""
    from fastapi.responses import FileResponse
    from services.user_store import register_download, add_model
    u = _require_user(authorization)
    # БЕЗПЕКА: безкоштовні завантаження — лише для ПІДТВЕРДЖЕНОЇ пошти (не-адмін),
    # інакше квоту FREE_DOWNLOADS легко обнулити, реєструючи нові непідтверджені
    # акаунти. Адмін (вже гейтований email_verified у verify_token) — без обмежень.
    if not u["is_admin"] and not u.get("email_verified", False):
        raise HTTPException(
            status_code=403,
            detail="Підтвердьте email, щоб завантажувати моделі (перевірте пошту).",
        )
    # СЕРІЯ/ПАННО ZIP: download_url=/api/zones/{batch}/download_all (або task_id=batch_*).
    # _resolve_model_path знає лише .3mf/.stl → zip-архів не резолвився → 404 → «нічого
    # не качається» (скарга власника). Тут гейтимо квоту й віддаємо zip через наявний
    # download_all_zones (він будує архів + layout.png; 409 поки плитки готуються).
    _dl_url = req.download_url or ""
    _is_batch = "/download_all" in _dl_url or (req.task_id or "").startswith("batch_")
    if _is_batch:
        _bid = req.task_id if (req.task_id or "").startswith("batch_") else _dl_url.split("/zones/")[-1].split("/download_all")[0]
        _res = register_download(u["uid"], u.get("email") or "", u["is_admin"], _bid or "")
        if not _res["ok"]:
            raise HTTPException(status_code=402, detail="Вичерпано безкоштовні завантаження")
        add_model(u["uid"], u.get("email") or "", {
            "task_id": _bid, "title": req.title, "city": req.city,
            "product_type": req.product_type, "download_url": _dl_url,
            "preview": (req.preview or "")[:200000],
        })
        return await download_all_zones(_bid)
    path = _resolve_model_path(req)
    if path is None:
        raise HTTPException(status_code=404, detail="Файл моделі не знайдено")
    res = register_download(u["uid"], u.get("email") or "", u["is_admin"], req.task_id or "")
    if not res["ok"]:
        raise HTTPException(status_code=402, detail="Вичерпано безкоштовні завантаження")
    add_model(u["uid"], u.get("email") or "", {
        "task_id": req.task_id, "title": req.title, "city": req.city,
        "product_type": req.product_type, "download_url": req.download_url,
        # cap thumbnail size so users.json stays small (~a small PNG data-URL)
        "preview": (req.preview or "")[:200000],
    })
    return FileResponse(
        str(path), media_type="model/3mf", filename=path.name,
        headers={"X-Quota-Remaining": str(res["quota"]["remaining"])},
    )


@app.get("/api/admin/orders")
async def admin_orders(authorization: Optional[str] = Header(default=None)):
    u = _require_user(authorization)
    if not u["is_admin"]:
        raise HTTPException(status_code=403, detail="Лише для адміністраторів")
    orders = []
    log = DATA_DIR / "orders.jsonl"  # БЕЗПЕКА: ПДн-лог тепер у DATA_DIR (не /files)
    try:
        if log.exists():
            for line in log.read_text(encoding="utf-8").splitlines():
                try:
                    orders.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    continue
    except Exception:  # noqa: BLE001
        pass
    orders.reverse()
    return {"orders": orders}


class OrderStatusUpdate(BaseModel):
    status: str = Field(max_length=24)


@app.post("/api/admin/orders/{order_number}/status")
async def admin_set_order_status(
    order_number: str,
    body: OrderStatusUpdate,
    authorization: Optional[str] = Header(default=None),
):
    """Адмін: змінює статус замовлення в orders.jsonl.

    Дозволені статуси: new, paid, printed, shipped, done. Лише для адмінів
    (email_verified + ADMIN_EMAILS). LiqPay-оплата теж сама ставить «paid».
    """
    from services.order_service import set_order_status, ORDER_STATUSES
    u = _require_user(authorization)
    if not u["is_admin"]:
        raise HTTPException(status_code=403, detail="Лише для адміністраторів")
    status = (body.status or "").strip().lower()
    if status not in ORDER_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Невідомий статус «{body.status}». Дозволені: {', '.join(ORDER_STATUSES)}",
        )
    try:
        ok = set_order_status(order_number, status)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not ok:
        raise HTTPException(status_code=404, detail="Замовлення не знайдено")
    return {"ok": True, "order_number": order_number, "status": status}


@app.get("/api/admin/users")
async def admin_users(authorization: Optional[str] = Header(default=None)):
    from services.user_store import list_all_users
    u = _require_user(authorization)
    if not u["is_admin"]:
        raise HTTPException(status_code=403, detail="Лише для адміністраторів")
    return {"users": list_all_users()}


def _validate_keychain_print_scale(request: GenerationRequest) -> None:
    """Auto-adjust keychain scale instead of erroring.

    Якщо зона завелика для обраного розміру карти на брелку, замість 422 помилки
    розширюємо keychain_map_width_mm/keychain_map_height_mm так, щоб масштаб
    залишався друкованим (≤ 8 м/мм). Залишаємо керування користувачу — він
    може зменшити ділянку, але система не блокує генерацію.
    """
    if not (bool(getattr(request, "flat_plate_mode", False)) and bool(getattr(request, "keychain_mode", False))):
        return
    # ТОПО-БРЕЛОК: ліміт 8 м/мм існує для читабельності ВУЛИЦЬ. Рельєф висот
    # навпаки потребує великих зон (гори = кілометри) — гейт не застосовуємо.
    if bool(getattr(request, "keychain_topo_mode", False)):
        return
    map_w_mm = max(float(getattr(request, "keychain_map_width_mm", 0.0) or 0.0), 1.0)
    map_h_mm = max(float(getattr(request, "keychain_map_height_mm", 0.0) or 0.0), 1.0)
    lat_mid = (float(request.north) + float(request.south)) * 0.5
    width_m = abs(float(request.east) - float(request.west)) * 111_320.0 * max(float(np.cos(np.deg2rad(lat_mid))), 0.2)
    height_m = abs(float(request.north) - float(request.south)) * 111_320.0
    meters_per_mm = max(width_m / map_w_mm, height_m / map_h_mm)
    MAX_M_PER_MM = 8.0
    if meters_per_mm > MAX_M_PER_MM:
        # Розраховуємо мінімальну ширину/висоту карти щоб залишитися в межах
        body_w = float(getattr(request, "keychain_body_width_mm", 35.0) or 35.0)
        body_h = float(getattr(request, "keychain_body_height_mm", 55.0) or 55.0)
        # Розширюємо карту: збільшуємо обидва виміри пропорційно, не більше тіла мінус rim
        rim_w = float(getattr(request, "keychain_rim_width_mm", 1.2) or 1.2)
        max_map_w = max(body_w - 2 * rim_w, 8.0)
        max_map_h = max(body_h - 2 * rim_w - 12.0, 8.0)  # 12mm запас на label band/loop
        new_w = min(max(width_m / MAX_M_PER_MM, map_w_mm), max_map_w)
        new_h = min(max(height_m / MAX_M_PER_MM, map_h_mm), max_map_h)
        request.keychain_map_width_mm = float(new_w)
        request.keychain_map_height_mm = float(new_h)
        # Переоцінюємо
        actual_m_per_mm = max(width_m / new_w, height_m / new_h)
        print(
            f"[AUTO-ADJUST] Keychain scale: {meters_per_mm:.1f} м/мм -> "
            f"{actual_m_per_mm:.1f} м/мм (map={new_w:.1f}x{new_h:.1f}mm). "
            f"Якщо все ще завелика — деталь може злитися; рекомендуємо зменшити ділянку."
        )


@app.post("/api/generate", response_model=GenerationResponse)
async def generate_model(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    _rl: None = Depends(rate_limit("generate", [(5, 60.0), (40, 3600.0)])),
):
    """
    РЎС‚РІРѕСЂСЋС” Р·Р°РґР°С‡Сѓ РіРµРЅРµСЂР°С†С–С— 3D РјРѕРґРµР»С–
    """
    try:
        print(f"[INFO] РћС‚СЂРёРјР°РЅРѕ Р·Р°РїРёС‚ РЅР° РіРµРЅРµСЂР°С†С–СЋ: north={request.north}, south={request.south}, east={request.east}, west={request.west}")
        # ── Bbox sanity guard ───────────────────────────────────────────
        # Чітка 4xx замість 500 при невалідних координатах (None/NaN/перевернутий
        # bbox). Pydantic уже привів типи, але не ловить інверсію/нескінченність.
        try:
            _bn, _bs = float(request.north), float(request.south)
            _be, _bw = float(request.east), float(request.west)
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="Координати ділянки мають бути числами (north/south/east/west).")
        if not all(np.isfinite(v) for v in (_bn, _bs, _be, _bw)):
            raise HTTPException(status_code=422, detail="Координати ділянки недійсні (NaN/Inf).")
        if not (-90.0 <= _bs < _bn <= 90.0) or not (-180.0 <= _bw < _be <= 180.0):
            raise HTTPException(
                status_code=400,
                detail=("Невірна ділянка: north має бути більше south, east більше west, "
                        "усі в межах широти ±90° і довготи ±180°. Перемалюйте рамку."),
            )
        # ── Zone size guard (fixed 1:10000 scale) ───────────────────────
        # Max real-world zone scales with the model size at a constant 1:10000
        # scale (0.1 mm/m): 80mm model ↔ 800m zone, and +100m per +1cm. This keeps
        # printable detail consistent and generation fast.
        #   max_zone_m = model_size_mm * ZONE_M_PER_MODEL_MM   (default 10.0)
        # An absolute hard ceiling MAX_ZONE_SPAN_M still applies (0 = none).
        try:
            _m_per_mm = float(os.getenv("ZONE_M_PER_MODEL_MM", "10.0"))
        except Exception:
            _m_per_mm = 10.0
        try:
            _model_mm = float(getattr(request, "model_size_mm", None) or 80.0)
        except Exception:
            _model_mm = 80.0
        _max_span = _model_mm * _m_per_mm if _m_per_mm > 0 else 0.0
        try:
            _hard_ceiling = float(os.getenv("MAX_ZONE_SPAN_M", "0"))
        except Exception:
            _hard_ceiling = 0.0
        if _hard_ceiling > 0:
            _max_span = min(_max_span, _hard_ceiling) if _max_span > 0 else _hard_ceiling
        # ТОПО-БРЕЛОК (C3): рельєфу потрібні ВЕЛИКІ зони (гори = кілометри),
        # масштаб 1:10000 для вуличної деталізації тут не застосовний.
        # Власна стеля 30 км зі сторони — межа розумного для DEM/OSM-фетчу.
        if bool(getattr(request, "keychain_topo_mode", False)) and bool(getattr(request, "keychain_mode", False)):
            _max_span = 30000.0
        # GPX-ТРЕК: маршрути часто більші за вуличний 1:10000 (біг/вело — кілька
        # км). Даємо гнучкий масштаб до 35 м/мм (як фронт), щоб увесь трек влазив
        # без помилки «зона завелика». Деталі мапи стають дрібніші — це ок.
        if getattr(request, "gpx_track", None) and _max_span > 0:
            _max_span = max(_max_span, _model_mm * 35.0)
        if _max_span > 0:
            import math as _m
            _clat = (float(request.north) + float(request.south)) * 0.5
            _ns_m = abs(float(request.north) - float(request.south)) * 111_320.0
            _ew_m = abs(float(request.east) - float(request.west)) * 111_320.0 * max(0.05, _m.cos(_m.radians(_clat)))
            # small tolerance so a ~400.0m selection isn't rejected by rounding
            _tol = _max_span * 1.02 + 5.0
            if _ns_m > _tol or _ew_m > _tol:
                raise HTTPException(
                    status_code=400,
                    detail=(f"Зона завелика для моделі {_model_mm/10:.0f} см: "
                            f"{_ns_m:.0f}×{_ew_m:.0f} м, максимум ~{_max_span:.0f} м зі сторони "
                            f"(масштаб 1:10000). Виберіть меншу ділянку або більший розмір моделі."),
                )
        _validate_keychain_print_scale(request)
        
        # Calculate grid_step_m if not provided (for Single Mode consistency)
        if request.grid_step_m is None:
             target_res = float(request.terrain_resolution) if request.terrain_resolution else 150.0
             computed_step = float(request.hex_size_m) / target_res
             computed_step = round(computed_step * 2) / 2.0
             if computed_step < 0.5: computed_step = 0.5
             request.grid_step_m = computed_step
             print(f"[INFO] Auto-calc grid_step_m for single request: {request.grid_step_m}")

        task_id = str(uuid.uuid4())
        task = GenerationTask(task_id=task_id, request=request)
        tasks[task_id] = task
        
        # Запускаємо генерацію в фоні. Передаємо zone_polygon_coords якщо є —
        # для повернутих rect-ділянок backend обріже OSM по полігону, а не bbox.
        polygon_coords = getattr(request, "zone_polygon_coords", None)
        background_tasks.add_task(
            generate_model_task,
            task_id,
            request,
            None,  # zone_id
            polygon_coords,  # zone_polygon_coords
        )
        
        print(f"[INFO] РЎС‚РІРѕСЂРµРЅРѕ Р·Р°РґР°С‡Сѓ {task_id} РґР»СЏ РіРµРЅРµСЂР°С†С–С— РјРѕРґРµР»С–")
        return GenerationResponse(task_id=task_id, status="processing", message="Р—Р°РґР°С‡Р° СЃС‚РІРѕСЂРµРЅР°")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] РџРѕРјРёР»РєР° СЃС‚РІРѕСЂРµРЅРЅСЏ Р·Р°РґР°С‡С–: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"РџРѕРјРёР»РєР° СЃС‚РІРѕСЂРµРЅРЅСЏ Р·Р°РґР°С‡С–: {str(e)}")


class CustomWorldRequest(BaseModel):
    """Запит режиму «опиши світ» (AI/процедурна генерація, задача #5)."""
    prompt: str = Field(..., min_length=1, max_length=2000)
    size_mm: float = Field(default=120.0, ge=40.0, le=220.0)
    keychain_mode: bool = False


def generate_custom_task(task_id: str, prompt: str, size_mm: float, keychain_mode: bool):
    """Фон: промт → spec (Claude або rule-based) → процедурний світ → 3MF/STL/GLB."""
    task = tasks.get(task_id)
    if task is None:
        return
    try:
        from services.llm_orchestrator import prompt_to_spec
        from services.procedural_generator import generate_world_mesh
        task.update_status("processing", 15, "Аналізую опис світу...")
        spec, src = prompt_to_spec(prompt, size_mm)
        task.update_status("processing", 45, f"Будую {spec.get('shape', 'світ')}...")
        mesh = generate_world_mesh(spec)
        task.update_status("processing", 80, "Експортую модель...")
        basename = f"custom_{task_id[:8]}"
        p3mf = str(OUTPUT_DIR / f"{basename}.3mf")
        pstl = str(OUTPUT_DIR / f"{basename}.stl")
        pglb = str(OUTPUT_DIR / f"{basename}.glb")
        mesh.export(p3mf); mesh.export(pstl); mesh.export(pglb)
        task.set_output("3mf", p3mf); task.set_output("stl", pstl); task.set_output("glb", pglb)
        task.complete(p3mf)
        task.message = f"Світ готовий · {spec.get('shape', '')} [{src}]"
        print(f"[CUSTOM] {task_id} done: {spec.get('shape')} src={src} -> {p3mf}", flush=True)
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        task.fail(f"Помилка генерації світу: {exc}")


@app.post("/api/generate-custom", response_model=GenerationResponse)
async def generate_custom(
    request: CustomWorldRequest,
    background_tasks: BackgroundTasks,
    _rl: None = Depends(rate_limit("generate_custom", [(5, 60.0), (30, 3600.0)])),
):
    """Режим «опиши світ» (#5): вільний опис → процедурна 3D-модель рельєфу
    (heightfield+форма), готова до друку. Claude покращує промт→spec ЯКЩО заданий
    ANTHROPIC_API_KEY, інакше rule-based парсер (працює одразу). Статус — /api/status."""
    task_id = str(uuid.uuid4())
    tasks[task_id] = GenerationTask(task_id=task_id, request={"prompt": request.prompt})
    background_tasks.add_task(
        generate_custom_task, task_id, request.prompt, float(request.size_mm), bool(request.keychain_mode),
    )
    print(f"[CUSTOM] created task {task_id} for prompt: {request.prompt[:60]}", flush=True)
    return GenerationResponse(task_id=task_id, status="processing", message="Задача створена")


@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    """
    РћС‚СЂРёРјСѓС” СЃС‚Р°С‚СѓСЃ Р·Р°РґР°С‡С– РіРµРЅРµСЂР°С†С–С— Р°Р±Рѕ РјРЅРѕР¶РёРЅРЅРёС… Р·Р°РґР°С‡
    """
    # Helper to build a static URL from an absolute path. Defined at the top so
    # BOTH the batch branch and the single-task branch can use it (previously it
    # was only defined in the single-task branch, so batch status 500'd with
    # UnboundLocalError — breaking grid/zone generation polling).
    def to_static_url(path_str):
        if not path_str:
            return None
        return f"/files/{Path(path_str).name}"

    # РџРµСЂРµРІС–СЂСЏС”РјРѕ, С‡Рё С†Рµ batch Р·Р°РїРёС‚ РЅР° РјРЅРѕР¶РёРЅРЅС– Р·Р°РґР°С‡С– (С„РѕСЂРјР°С‚: batch_<uuid>)
    if task_id.startswith("batch_"):
        all_task_ids_list = multiple_tasks_map.get(task_id)
        if not all_task_ids_list:
            raise HTTPException(status_code=404, detail="Multiple tasks not found")

        # Фіксуємо шляхи готових плиток на диск (поки tasks живі) — щоб zip
        # пережив рестарт сервера.
        _record_panel_tile_paths(task_id)

        # РџРѕРІРµСЂС‚Р°С”РјРѕ СЃС‚Р°С‚СѓСЃ РІСЃС–С… Р·Р°РґР°С‡
        tasks_status = []
        for tid in all_task_ids_list:
            if tid in tasks:
                t = tasks[tid]
                output_files = getattr(t, "output_files", {}) or {}
                
                download_url = None
                if t.status == "completed":
                    if t.output_file:
                        download_url = f"/files/{Path(t.output_file).name}"
                    elif "3mf" in output_files:
                        download_url = f"/files/{Path(output_files['3mf']).name}"
                
                tasks_status.append({
                    "task_id": tid,
                    "status": t.status,
                    "progress": t.progress,
                    "message": t.message,
                    "output_file": t.output_file,
                    "output_files": output_files,
                    "keychain_manifest": getattr(t, "keychain_manifest", None),
                    "download_url": download_url,
                    "firebase_url": getattr(t, "firebase_url", None),
                    "print_quality": getattr(t, "print_quality", None),
                    "preview_3mf": to_static_url(output_files.get("preview_3mf")),
                    "firebase_preview_3mf": t.firebase_outputs.get("preview_3mf"),
                    "firebase_preview_parts": {
                        "base": t.firebase_outputs.get("base_3mf"),
                        "roads": t.firebase_outputs.get("roads_3mf"),
                        "buildings": t.firebase_outputs.get("buildings_3mf"),
                        "water": t.firebase_outputs.get("water_3mf"),
                        "parks": t.firebase_outputs.get("parks_3mf"),
                    },
                })
            else:
                # ПІСЛЯ РЕСТАРТУ: tasks порожні. Якщо плитку вже збережено на
                # диск (panel_tiles) АБО файл лежить в output — рапортуємо
                # «completed», щоб фронт показав готовність і дав завантажити zip.
                rec = next((x for x in panel_tiles.get(task_id, []) if x.get("task_id") == tid), None)
                p = rec.get("path") if rec else None
                if not (p and Path(p).exists()):
                    disk = _find_file_on_disk_by_task_id(tid, "3mf")
                    p = str(disk) if disk is not None else None
                if p and Path(p).exists():
                    tasks_status.append({
                        "task_id": tid, "status": "completed", "progress": 100,
                        "message": "Restored from disk", "output_file": p,
                        "output_files": {"3mf": p},
                        "download_url": f"/files/{Path(p).name}",
                        "keychain_manifest": None, "firebase_url": None, "print_quality": None,
                        "preview_3mf": None, "firebase_preview_3mf": None,
                        "firebase_preview_parts": {"base": None, "roads": None, "buildings": None, "water": None, "parks": None},
                    })
                else:
                    # Плитка ВТРАЧЕНА (рестарт посеред генерації, файла нема). Раніше
                    # її просто пропускали → completed НІКОЛИ не = total → фронт
                    # полив вічно «N/M». Тепер рапортуємо ТЕРМІНАЛЬНИЙ «failed»,
                    # щоб фронт зупинив полінг і дав завантажити готові плитки.
                    tasks_status.append({
                        "task_id": tid, "status": "failed", "progress": 0,
                        "message": "Плитку втрачено (перезапуск сервера) — перегенеруйте",
                        "output_file": None, "output_files": {},
                        "download_url": None, "keychain_manifest": None,
                        "firebase_url": None, "print_quality": None,
                        "preview_3mf": None, "firebase_preview_3mf": None,
                        "firebase_preview_parts": {"base": None, "roads": None, "buildings": None, "water": None, "parks": None},
                    })

        return {
            "task_id": task_id,
            "status": "multiple",
            "tasks": tasks_status,
            "total": len(all_task_ids_list),
            "completed": sum(1 for t in tasks_status if t["status"] == "completed"),
            "all_task_ids": all_task_ids_list
        }
    
    if task_id not in tasks:
        disk_file = _find_file_on_disk_by_task_id(task_id)
        if disk_file is not None:
            return {
                "task_id": task_id,
                "status": "completed",
                "progress": 100,
                "message": "Model ready (restored from disk)",
                "download_url": f"/files/{disk_file.name}",
                "firebase_url": None,
                "download_url_stl": None,
                "download_url_3mf": f"/files/{disk_file.name}" if disk_file.suffix.lower() == ".3mf" else None,
                "keychain_manifest": None,
                "preview_3mf": None,
                "preview_parts": {"base": None, "roads": None, "buildings": None, "water": None, "parks": None},
                "firebase_preview_3mf": None,
                "firebase_preview_parts": {"base": None, "roads": None, "buildings": None, "water": None, "parks": None},
            }
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    output_files = getattr(task, "output_files", {}) or {}

    # Main download logic: prefer user requested format if available
    main_download_url = None
    if task.status == "completed":
        if task.output_file:
             main_download_url = to_static_url(task.output_file)
        elif "3mf" in output_files:
             main_download_url = to_static_url(output_files["3mf"])
        elif "stl" in output_files:
             main_download_url = to_static_url(output_files["stl"])

    return {
        "task_id": task_id,
        "status": task.status,
        "progress": task.progress,
        "message": task.message,
        "download_url": main_download_url,
        "firebase_url": task.firebase_url,
        "print_quality": getattr(task, "print_quality", None),
        "download_url_stl": to_static_url(output_files.get("stl")),
        "download_url_3mf": to_static_url(output_files.get("3mf")),
        "keychain_manifest": getattr(task, "keychain_manifest", None),
        "preview_3mf": to_static_url(output_files.get("preview_3mf")),  # РћСЃРЅРѕРІРЅРµ РїСЂРµРІ'СЋ РІ 3MF
        "preview_parts": {
            "base": to_static_url(output_files.get("base_3mf")),
            "roads": to_static_url(output_files.get("roads_3mf")),
            "buildings": to_static_url(output_files.get("buildings_3mf")),
            "water": to_static_url(output_files.get("water_3mf")),
            "parks": to_static_url(output_files.get("parks_3mf")),
        },
        "firebase_preview_3mf": task.firebase_outputs.get("preview_3mf"),
        "firebase_preview_parts": {
            "base": task.firebase_outputs.get("base_3mf"),
            "roads": task.firebase_outputs.get("roads_3mf"),
            "buildings": task.firebase_outputs.get("buildings_3mf"),
            "water": task.firebase_outputs.get("water_3mf"),
            "parks": task.firebase_outputs.get("parks_3mf"),
        },
    }


class SharePreviewRequest(BaseModel):
    task_id: str
    image: str = Field(max_length=3_000_000)  # PNG data-URL, ~2МБ після base64


_SHARE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


@app.post("/api/share/preview")
async def share_preview(
    req: SharePreviewRequest,
    _rl: None = Depends(rate_limit("share_preview", [(10, 3600.0)])),
):
    """E4: зберігає рендер моделі користувача для OG-шерингу (/share/{task})."""
    import base64

    if not _SHARE_ID_RE.match(req.task_id or ""):
        raise HTTPException(status_code=400, detail="Невалідний task_id")
    # Задача має існувати (в памʼяті або файлом на диску) — анти-сміття
    if req.task_id not in tasks and _find_file_on_disk_by_task_id(req.task_id) is None:
        raise HTTPException(status_code=404, detail="Модель не знайдено")
    prefix = "data:image/png;base64,"
    if not req.image.startswith(prefix):
        raise HTTPException(status_code=400, detail="Очікую PNG data-URL")
    try:
        raw = base64.b64decode(req.image[len(prefix):], validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Битий base64")
    if len(raw) > 2_000_000 or not raw.startswith(b"\x89PNG"):
        raise HTTPException(status_code=400, detail="Не PNG або завеликий (макс 2МБ)")
    previews_dir = OUTPUT_DIR / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    (previews_dir / f"{req.task_id}.png").write_bytes(raw)
    print(f"[SHARE] preview saved: {req.task_id} ({len(raw)} bytes)")
    return {"status": "ok", "share_path": f"/share/{req.task_id}"}


@app.get("/api/og/{task_id}")
async def og_image(task_id: str):
    """E4: OG-картинка для /share/{task} — реальний рендер моделі користувача."""
    if not _SHARE_ID_RE.match(task_id or ""):
        raise HTTPException(status_code=400, detail="Невалідний task_id")
    p = OUTPUT_DIR / "previews" / f"{task_id}.png"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Превʼю не знайдено")
    return FileResponse(
        str(p),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/api/zones/{batch_id}/download_all")
async def download_all_zones(batch_id: str):
    """D3 ПАННО: zip з усіма плитками batch-генерації + layout.png (схема
    розкладки R×C з іменами файлів). 409 — поки не всі плитки готові."""
    import io
    import zipfile

    task_ids_list = multiple_tasks_map.get(batch_id)
    if not task_ids_list:
        raise HTTPException(status_code=404, detail="Batch не знайдено (можливо, сервер перезапускався — згенеруйте панно знову)")

    # Поки tasks живі — фіксуємо шляхи на диск (щоб zip пережив рестарт).
    _record_panel_tile_paths(batch_id)
    tile_meta = {x.get("task_id"): x for x in panel_tiles.get(batch_id, [])}

    items = []  # (row, col, filename, path)
    still_running = 0  # плитки, що ЩЕ генеруються (треба зачекати)
    lost = 0           # плитки failed/втрачені (рестарт) — НЕ зачекаються ніколи
    for tid in task_ids_list:
        t = tasks.get(tid)
        if t is not None and t.status == "completed":
            output_files = getattr(t, "output_files", {}) or {}
            # 3MF пріоритетний (друкований формат); preview-задачі мають .glb primary
            path_str = output_files.get("3mf") or t.output_file
            if path_str and Path(path_str).exists():
                p = Path(path_str)
                items.append((getattr(t, "zone_row", None), getattr(t, "zone_col", None), p.name, p))
                continue
        # FALLBACK 1 (рестарт): збережений шлях плитки з panel_tiles
        rec = tile_meta.get(tid)
        rec_path = rec.get("path") if rec else None
        if rec_path and Path(rec_path).exists():
            p = Path(rec_path)
            items.append((rec.get("row"), rec.get("col"), p.name, p))
            continue
        # FALLBACK 2: файл міг лишитись на диску навіть якщо task зник
        disk = _find_file_on_disk_by_task_id(tid, "3mf")
        if disk is not None and Path(disk).exists():
            items.append((rec.get("row") if rec else None, rec.get("col") if rec else None, Path(disk).name, Path(disk)))
            continue
        # Не знайдено: ще генерується (task живий, не термінальний) vs втрачено
        if t is not None and t.status not in ("completed", "failed", "cancelled"):
            still_running += 1
        else:
            lost += 1
    # Лише ЖИВІ незавершені плитки блокують zip (409 = «зачекай»). Якщо нічого
    # не генерується, а частина плиток втрачена — НЕ блокуємо вічно: віддаємо
    # zip із готових (краще, ніж нескінченне «N/M»); юзер дозамовить/перегенерує.
    if still_running > 0:
        raise HTTPException(status_code=409, detail=f"Готово {len(items)}/{len(task_ids_list)} плиток — зачекайте завершення генерації")
    if not items:
        raise HTTPException(status_code=404, detail="Файли плиток не знайдено (перезапуск сервера — згенеруйте панно знову)")
    if lost > 0:
        print(f"[PANEL] download_all_zones: {len(items)}/{len(task_ids_list)} плиток, {lost} втрачено — віддаю частковий zip")

    # layout.png — схема розкладки: сітка з підписами row/col + імʼя файлу
    layout_png: Optional[bytes] = None
    try:
        from PIL import Image, ImageDraw

        rows = sorted({r for r, c, n, p in items if r is not None})
        cols = sorted({c for r, c, n, p in items if c is not None})
        if rows and cols:
            cell_w, cell_h, pad = 300, 200, 20
            img = Image.new("RGB", (pad * 2 + cell_w * len(cols), pad * 2 + cell_h * len(rows) + 40), "#F4EFE4")
            draw = ImageDraw.Draw(img)
            draw.text((pad, 8), f"Monadruk · панно {len(rows)}x{len(cols)} — розкладка плиток (вид зверху, північ вгорі)", fill="#2E4A3A")
            for r, c, name, _p in items:
                if r is None or c is None:
                    continue
                # row 0 = ПІВНІЧ (верх); col 0 = захід (ліво)
                x0 = pad + cols.index(c) * cell_w
                y0 = pad + 40 + rows.index(r) * cell_h
                draw.rectangle([x0, y0, x0 + cell_w - 4, y0 + cell_h - 4], outline="#2E4A3A", width=3, fill="#FFFFFF")
                draw.text((x0 + 12, y0 + 12), f"ряд {r} · кол {c}", fill="#2E4A3A")
                draw.text((x0 + 12, y0 + 40), name, fill="#6B6B5E")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            layout_png = buf.getvalue()
    except Exception as exc:
        print(f"[PANNO] layout.png failed (non-fatal): {exc}")

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for r, c, name, p in items:
            arcname = f"tile_r{r}_c{c}_{name}" if (r is not None and c is not None) else name
            zf.write(p, arcname=arcname)
        if layout_png:
            zf.writestr("layout.png", layout_png)
        zf.writestr(
            "README.txt",
            "Monadruk — настінне панно з 3D-плиток\n"
            f"Плиток: {len(items)}. Схема розкладки: layout.png (північ вгорі).\n"
            "Друкуйте кожну плитку окремо; шви состиковано на бекенді (<0.1мм).\n",
        )
    zip_buf.seek(0)
    short = batch_id.replace("batch_", "")[:8]
    from fastapi.responses import StreamingResponse

    print(f"[PANNO] zip ready: {len(items)} tiles, batch={batch_id}")
    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="monadruk_panno_{short}.zip"'},
    )


def _task_owner(task: "GenerationTask") -> Optional[str]:
    """Optional uid the task was created for (set on authenticated generate),
    if we ever stored one. None = legacy/anonymous task."""
    return getattr(task, "owner_uid", None)


@app.delete("/api/task/{task_id}")
async def cancel_task(task_id: str, authorization: Optional[str] = Header(default=None)):
    """Cancel a generation task or batch. БЕЗПЕКА: змінює стан → потрібен валідний
    токен. Якщо задача привʼязана до власника — лише власник/адмін може скасувати
    (інакше будь-хто міг убивати чужі генерації за вгаданим task_id)."""
    u = _require_user(authorization)

    def _can_cancel(t: "GenerationTask") -> bool:
        owner = _task_owner(t)
        return owner is None or owner == u["uid"] or u.get("is_admin", False)

    if task_id.startswith("batch_"):
        task_ids_list = multiple_tasks_map.get(task_id, [])
        live = [tasks[tid] for tid in task_ids_list if tid in tasks]
        if not live:
            raise HTTPException(status_code=404, detail="Batch tasks not found")
        if not all(_can_cancel(t) for t in live):
            raise HTTPException(status_code=403, detail="Це не ваша генерація")
        count = sum(1 for t in live if (t.cancel() or True))
        print(f"[INFO] Cancelled batch {task_id} ({count} sub-tasks) by {u.get('email')}")
        return {"cancelled": True, "count": count}
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    if not _can_cancel(tasks[task_id]):
        raise HTTPException(status_code=403, detail="Це не ваша генерація")
    tasks[task_id].cancel()
    print(f"[INFO] Cancelled task {task_id} by {u.get('email')}")
    return {"cancelled": True}


def _find_file_on_disk_by_task_id(task_id, format=None):
    """Locate generated file on disk by task_id when task is missing from memory.
    Supports new naming model_<grid>_<mm>_<short>.<ext> and legacy <task_id>.<ext>."""
    from pathlib import Path
    if not OUTPUT_DIR.exists():
        return None
    fmt = (format or "").lower().strip(".") or "3mf"
    short = task_id.replace("-", "")[:8]
    exts = [fmt, "3mf", "stl", "glb"]
    seen = set()
    for ext in exts:
        if ext in seen:
            continue
        seen.add(ext)
        exact = OUTPUT_DIR / f"{task_id}.{ext}"
        if exact.exists():
            return exact
        for path in OUTPUT_DIR.glob(f"*{short}*.{ext}"):
            if path.is_file():
                return path
    return None


@app.get("/api/download/{task_id}")
async def download_model(
    task_id: str,
    format: Optional[str] = Query(default=None, description="Optional: stl Р°Р±Рѕ 3mf"),
    part: Optional[str] = Query(default=None, description="Optional preview part: base|roads|buildings|water"),
    # ЩЕДРИЙ анти-DoS ліміт per-IP. Ендпойнт двоїстий: окрім завантажень він віддає
    # ЖИВЕ превʼю (glb + 3mf-fallback на task) і мініатюри /account (glb на кожну модель),
    # тож легітимний сплеск = десятки запитів. 240/хв та 2000/год НЕ зачіпають реальних
    # користувачів, але обрізають масовий харвест/bandwidth-DoS. (Квоту тут НЕ гейтимо —
    # це зламало б анонімне превʼю; див. memory download-quota-gate.)
    _rl: None = Depends(rate_limit("download", [(240, 60.0), (2000, 3600.0)])),
):
    """
    Р—Р°РІР°РЅС‚Р°Р¶СѓС” Р·РіРµРЅРµСЂРѕРІР°РЅРёР№ С„Р°Р№Р» Р· Firebase С‡РµСЂРµР· РїСЂРѕРєСЃС–
    """
    if task_id not in tasks:
        disk_file = _find_file_on_disk_by_task_id(task_id, format)
        if disk_file is not None:
            print(f"[INFO] Task {task_id} not in memory, serving from disk: {disk_file}")
            lower = str(disk_file).lower()
            if lower.endswith(".3mf"):
                mt = "model/3mf"
            elif lower.endswith(".glb"):
                mt = "model/gltf-binary"
            elif lower.endswith(".stl"):
                mt = "model/stl"
            else:
                mt = "application/octet-stream"
            return FileResponse(str(disk_file), media_type=mt, filename=disk_file.name)
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    if task.status != "completed":
        raise HTTPException(status_code=400, detail="Model not ready")
    
    print(f"[DEBUG] Download request: task={task_id}, format={format}, part={part}")

    output_files = getattr(task, "output_files", {}) or {}
    preview_mode = bool(getattr(getattr(task, "request", None), "preview_mode", False))

    def _serve_local_file(local_path: str):
        print(f"[INFO] Serving local file: {local_path}")
        lower = local_path.lower()
        if lower.endswith(".3mf"):
            mt = "model/3mf"
        elif lower.endswith(".glb"):
            mt = "model/gltf-binary"
        else:
            mt = "application/octet-stream"
        return FileResponse(local_path, media_type=mt, filename=Path(local_path).name)

    if preview_mode:
        if part:
            raise HTTPException(status_code=404, detail="Preview uses a single local GLB file")

        preview_path = output_files.get("glb") or output_files.get("3mf")
        if not preview_path and task.output_file and task.output_file.lower().endswith((".glb", ".3mf")):
            preview_path = task.output_file
        if preview_path and Path(preview_path).exists():
            return _serve_local_file(preview_path)

        raise HTTPException(status_code=404, detail="Local preview file not found")
    
    # 1. Р’РёР·РЅР°С‡Р°С”РјРѕ РєР»СЋС‡ РїРѕС‚СЂС–Р±РЅРѕРіРѕ С„Р°Р№Р»Сѓ РІ Firebase
    target_key = None
    if format or part:
        fmt = (format or "stl").lower().strip(".")
        if part:
            p = part.lower()
            target_key = f"{p}_{fmt}" # e.g. "roads_stl"
        else:
            target_key = fmt # e.g. "3mf" or "stl"
    else:
        # Default logic: try primary output file
        if task.output_file:
            ext = Path(task.output_file).suffix.lstrip(".").lower()
            target_key = ext
        else:
             target_key = "3mf" # Fallback

    # 2. РЁСѓРєР°С”РјРѕ С„Р°Р№Р» РІ Firebase
    print(f"[INFO] Looking for file in Firebase: key={target_key}")
    firebase_url = getattr(task, "firebase_outputs", {}).get(target_key)
    
    # РЇРєС‰Рѕ С†Рµ РѕСЃРЅРѕРІРЅРёР№ С„Р°Р№Р», РјРѕР¶Рµ Р±СѓС‚Рё РІ task.firebase_url
    if not firebase_url and (not part) and task.firebase_url:
         requested_fmt = (format or "").lower().strip(".")
         firebase_ext = Path(str(task.firebase_url).split("?", 1)[0]).suffix.lstrip(".").lower()
         if not requested_fmt or not firebase_ext or firebase_ext == requested_fmt:
             firebase_url = task.firebase_url

    # Fallback: СЏРєС‰Рѕ РїРѕС‚СЂС–Р±РЅР° С‡Р°СЃС‚РёРЅР° (base_3mf, roads_3mf С‚РѕС‰Рѕ) РІС–РґСЃСѓС‚РЅСЏ вЂ” РІРёРєРѕСЂРёСЃС‚РѕРІСѓС”РјРѕ РѕСЃРЅРѕРІРЅРёР№ 3MF
    # (РѕРєСЂРµРјС– С‡Р°СЃС‚РёРЅРё РЅРµ Р·Р°РІР°РЅС‚Р°Р¶СѓСЋС‚СЊСЃСЏ, 3MF РјС–СЃС‚РёС‚СЊ СѓСЃС– РєРѕРјРїРѕРЅРµРЅС‚Рё РІ РѕРґРЅРѕРјСѓ С„Р°Р№Р»С–)
    if not firebase_url and part and fmt == "3mf":
        valid_parts = {"base", "roads", "buildings", "water", "parks", "green"}
        if part.lower() in valid_parts:
            firebase_url = (
                getattr(task, "firebase_outputs", {}).get("3mf")
                or getattr(task, "firebase_outputs", {}).get("preview_3mf")
                or getattr(task, "firebase_url", None)
            )
            if firebase_url:
                print(f"[INFO] Part {part}_3mf not found, using main 3MF file (contains all parts)")

    # FALLBACK: якщо немає Firebase URL (preview-mode пропускає Firebase upload,
    # або просто перший запуск без cloud) — шукаємо файл локально на диску.
    if not firebase_url:
        local_path: Optional[str] = None

        # 1. Точний ключ у task.output_files (e.g. "3mf", "base_3mf", "roads_stl")
        if target_key and target_key in output_files:
            cand = output_files[target_key]
            if cand and Path(cand).exists():
                local_path = cand

        # 2. Якщо це основний файл (без part) — пробуємо task.output_file
        if not local_path and not part and task.output_file and Path(task.output_file).exists():
            requested_fmt = (format or "").lower().strip(".")
            output_ext = Path(task.output_file).suffix.lstrip(".").lower()
            if requested_fmt and output_ext and output_ext != requested_fmt:
                local_path = None
            else:
                local_path = task.output_file

        # 3. Якщо запитали part у 3mf і немає окремої part-файлу — віддаємо основний
        #    (вся scene у одному 3mf файлі)
        if not local_path and part and (format or "").lower() == "3mf":
            valid_parts = {"base", "roads", "buildings", "water", "parks", "green"}
            if part.lower() in valid_parts:
                main_3mf = output_files.get("3mf")
                if main_3mf and Path(main_3mf).exists():
                    local_path = main_3mf
                    print(f"[INFO] Part {part}_3mf not on disk, using main 3MF (contains all parts)")
                elif task.output_file and task.output_file.lower().endswith(".3mf") and Path(task.output_file).exists():
                    local_path = task.output_file

        if local_path:
            return _serve_local_file(local_path)

        if part == "poi":
            print(f"[INFO] POI part not available (expected), returning 404")
        print(f"[WARN] File not found in Firebase or locally: key={target_key}")
        raise HTTPException(status_code=404, detail=f"File not found: {target_key}")

    # 3. Р—Р°РІР°РЅС‚Р°Р¶СѓС”РјРѕ С„Р°Р№Р» Р· Firebase С‡РµСЂРµР· РїСЂРѕРєСЃС–
    print(f"[INFO] Proxying file from Firebase: {firebase_url}")
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.get(firebase_url)
            response.raise_for_status()
            
            # Р’РёР·РЅР°С‡Р°С”РјРѕ media type Р· URL Р°Р±Рѕ Р· Content-Type Р·Р°РіРѕР»РѕРІРєР°
            # Р’Р°Р¶Р»РёРІРѕ: РІРёРєРѕСЂРёСЃС‚РѕРІСѓС”РјРѕ РїСЂР°РІРёР»СЊРЅС– MIME С‚РёРїРё РґР»СЏ Р·Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ С„Р°Р№Р»С–РІ
            if firebase_url.endswith(".3mf"):
                content_type = "model/3mf"
            elif firebase_url.endswith(".stl"):
                # Р’РёРєРѕСЂРёСЃС‚РѕРІСѓС”РјРѕ application/octet-stream РґР»СЏ STL, С‰РѕР± Р±СЂР°СѓР·РµСЂ Р·Р°РІР¶РґРё Р·Р°РІР°РЅС‚Р°Р¶СѓРІР°РІ С„Р°Р№Р»
                content_type = "application/octet-stream"
            else:
                # РЎРїСЂРѕР±СѓС”РјРѕ РѕС‚СЂРёРјР°С‚Рё Р· Р·Р°РіРѕР»РѕРІРєС–РІ Firebase, С–РЅР°РєС€Рµ application/octet-stream
                content_type = response.headers.get("Content-Type", "application/octet-stream")
            
            # Р’РёР·РЅР°С‡Р°С”РјРѕ С–Рј'СЏ С„Р°Р№Р»Сѓ Р· URL
            filename = Path(firebase_url).name or f"model.{target_key}"
            
            # Р’РёРєРѕСЂРёСЃС‚РѕРІСѓС”РјРѕ РїСЂРѕСЃС‚РёР№ С„РѕСЂРјР°С‚ Content-Disposition РґР»СЏ РєСЂР°С‰РѕС— СЃСѓРјС–СЃРЅРѕСЃС‚С– Р· Р±СЂР°СѓР·РµСЂР°РјРё
            content_disposition = f'attachment; filename="{filename}"'
            
            print(f"[DEBUG] Proxying Firebase file: {filename}, Size: {len(response.content)} bytes")
            print(f"[DEBUG] Content-Disposition: {content_disposition}")
            print(f"[DEBUG] Content-Type: {content_type}")
            
            # Р’РёРєРѕСЂРёСЃС‚РѕРІСѓС”РјРѕ Response Р· РїСЂР°РІРёР»СЊРЅРёРјРё Р·Р°РіРѕР»РѕРІРєР°РјРё РґР»СЏ Р·Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ С„Р°Р№Р»Сѓ
            from fastapi.responses import Response
            
            return Response(
                content=response.content,
                media_type=content_type,
                headers={
                    "Content-Disposition": content_disposition,
                    "Content-Length": str(len(response.content)),
                    "Access-Control-Expose-Headers": "Content-Disposition, Content-Length, Content-Type",
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )
    except httpx.TimeoutException:
        print(f"[ERROR] Timeout while downloading from Firebase: {firebase_url}")
        raise HTTPException(status_code=504, detail="Timeout downloading file from Firebase")
    except httpx.HTTPStatusError as e:
        print(f"[ERROR] HTTP error downloading from Firebase: {e.response.status_code}")
        raise HTTPException(status_code=502, detail=f"Failed to download from Firebase: {e.response.status_code}")
    except Exception as e:
        print(f"[ERROR] Error proxying Firebase file: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to proxy file from Firebase: {str(e)}")


@app.post("/api/merge-zones")
async def merge_zones_endpoint(
    task_ids: List[str] = Query(..., description="РЎРїРёСЃРѕРє task_id Р·РѕРЅ РґР»СЏ РѕР±'С”РґРЅР°РЅРЅСЏ"),
    format: str = Query(default="3mf", description="Р¤РѕСЂРјР°С‚ РІРёС…С–РґРЅРѕРіРѕ С„Р°Р№Р»Сѓ (stl Р°Р±Рѕ 3mf)"),
    authorization: Optional[str] = Header(default=None),
    # Анти-DoS: завантаження+конкатенація мешів синхронно у потоці запиту — без
    # ліміту авторизований юзер може ганяти важкі merge нескінченно (як інші важкі POST).
    _rl: None = Depends(rate_limit("merge_zones", [(10, 60.0), (60, 3600.0)])),
):
    # БЕЗПЕКА: створює файл з чужих задач → потрібен валідний токен.
    _require_user(authorization)
    # Обмежуємо кількість зон на виклик, щоб обмежити вартість конкатенації.
    if len(task_ids) > 64:
        raise HTTPException(status_code=400, detail="Too many zones (max 64)")
    """
    РћР±'С”РґРЅСѓС” РєС–Р»СЊРєР° Р·РѕРЅ РІ РѕРґРёРЅ С„Р°Р№Р» РґР»СЏ РІС–РґРѕР±СЂР°Р¶РµРЅРЅСЏ СЂР°Р·РѕРј.
    
    Args:
        task_ids: РЎРїРёСЃРѕРє task_id Р·РѕРЅ РґР»СЏ РѕР±'С”РґРЅР°РЅРЅСЏ
        format: Р¤РѕСЂРјР°С‚ РІРёС…С–РґРЅРѕРіРѕ С„Р°Р№Р»Сѓ (stl Р°Р±Рѕ 3mf)
    
    Returns:
        РћР±'С”РґРЅР°РЅРёР№ С„Р°Р№Р» РјРѕРґРµР»С–
    """
    if not task_ids or len(task_ids) == 0:
        raise HTTPException(status_code=400, detail="РќРµ РІРєР°Р·Р°РЅРѕ task_ids РґР»СЏ РѕР±'С”РґРЅР°РЅРЅСЏ")
    
    # РџРµСЂРµРІС–СЂСЏС”РјРѕ, С‡Рё РІСЃС– Р·Р°РґР°С‡С– Р·Р°РІРµСЂС€РµРЅС–
    completed_tasks = []
    for tid in task_ids:
        if tid not in tasks:
            raise HTTPException(status_code=404, detail=f"Task {tid} not found")
        task = tasks[tid]
        if task.status != "completed":
            raise HTTPException(status_code=400, detail=f"Task {tid} not completed yet")
        completed_tasks.append(task)
    
    # Р—Р°РІР°РЅС‚Р°Р¶СѓС”РјРѕ РІСЃС– РјРµС€С–
    all_meshes = []
    
    for task in completed_tasks:
        try:
            # Р—Р°РІР°РЅС‚Р°Р¶СѓС”РјРѕ STL С„Р°Р№Р» (РІС–РЅ РјС–СЃС‚РёС‚СЊ РѕР±'С”РґРЅР°РЅСѓ РјРѕРґРµР»СЊ)
            stl_file = task.output_file
            if stl_file and stl_file.endswith('.stl'):
                mesh = trimesh.load(stl_file)
                if mesh is not None:
                    all_meshes.append(mesh)
        except Exception as e:
            print(f"[WARN] РџРѕРјРёР»РєР° Р·Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РјРµС€Сѓ Р· {task.task_id}: {e}")
            continue
    
    if not all_meshes:
        raise HTTPException(status_code=400, detail="РќРµ РІРґР°Р»РѕСЃСЏ Р·Р°РІР°РЅС‚Р°Р¶РёС‚Рё Р¶РѕРґРЅРѕРіРѕ РјРµС€Сѓ")
    
    # РћР±'С”РґРЅСѓС”РјРѕ РІСЃС– РјРµС€С–
    try:
        merged_mesh = trimesh.util.concatenate(all_meshes)
        if merged_mesh is None:
            raise HTTPException(status_code=500, detail="РќРµ РІРґР°Р»РѕСЃСЏ РѕР±'С”РґРЅР°С‚Рё РјРµС€С–")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"РџРѕРјРёР»РєР° РѕР±'С”РґРЅР°РЅРЅСЏ РјРµС€С–РІ: {str(e)}")
    
    # Р—Р±РµСЂС–РіР°С”РјРѕ РѕР±'С”РґРЅР°РЅРёР№ С„Р°Р№Р»
    # Р—Р±РµСЂС–РіР°С”РјРѕ РѕР±'С”РґРЅР°РЅРёР№ С„Р°Р№Р»
    merged_id = f"merged_{uuid.uuid4()}"
    if format.lower() == "3mf":
        output_file = OUTPUT_DIR / f"{merged_id}.3mf"
        merged_mesh.export(str(output_file), file_type="3mf")
    else:
        output_file = OUTPUT_DIR / f"{merged_id}.stl"
        merged_mesh.export(str(output_file), file_type="stl")
    
    return FileResponse(
        str(output_file),
        media_type="model/3mf" if format.lower() == "3mf" else "model/stl",
        filename=output_file.name
    )


@app.get("/api/test-model")
async def get_test_model():
    """
    РџРѕРІРµСЂС‚Р°С” С‚РµСЃС‚РѕРІСѓ РјРѕРґРµР»СЊ С†РµРЅС‚СЂСѓ РљРёС”РІР° (1РєРј x 1РєРј)
    РЎРїРѕС‡Р°С‚РєСѓ РЅР°РјР°РіР°С”С‚СЊСЃСЏ РїРѕРІРµСЂРЅСѓС‚Рё STL (РЅР°РґС–Р№РЅС–С€Рµ), РїРѕС‚С–Рј 3MF
    """
    # РЎРїРѕС‡Р°С‚РєСѓ РїРµСЂРµРІС–СЂСЏС”РјРѕ STL (РЅР°РґС–Р№РЅС–С€Рµ РґР»СЏ Р·Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ)
    test_model_stl = OUTPUT_DIR / "test_model_kyiv.stl"
    if test_model_stl.exists():
        return FileResponse(
            test_model_stl,
            media_type="application/octet-stream",
            filename="test_model_kyiv.stl"
        )
    
    # РЇРєС‰Рѕ STL РЅРµРјР°С”, РїРµСЂРµРІС–СЂСЏС”РјРѕ 3MF
    test_model_3mf = OUTPUT_DIR / "test_model_kyiv.3mf"
    if test_model_3mf.exists():
        return FileResponse(
            test_model_3mf,
            media_type="application/vnd.ms-package.3dmanufacturing-3dmodel+xml",
            filename="test_model_kyiv.3mf"
        )
    
    raise HTTPException(
        status_code=404, 
        detail="Test model not found. Run generate_test_model.py first."
    )


@app.get("/api/test-model/manifest")
async def get_test_model_manifest():
    """
    РњР°РЅС–С„РµСЃС‚ STL С‡Р°СЃС‚РёРЅ РґР»СЏ РєРѕР»СЊРѕСЂРѕРІРѕРіРѕ РїСЂРµРІ'СЋ (base/roads/buildings/water/parks/poi)
    """
    parts = {}
    
    parts = {}
    for p in ["base", "roads", "buildings", "water", "parks"]:
        fp = OUTPUT_DIR / f"test_model_kyiv_{p}.stl"
        if fp.exists():
            parts[p] = f"/api/test-model/part/{p}"
    if not parts:
        raise HTTPException(status_code=404, detail="No test-model parts found. Run generate_test_model.py first.")
    return {"parts": parts}


@app.get("/api/test-model/part/{part_name}")
async def get_test_model_part(part_name: str):
    p = part_name.lower()
    file_path = OUTPUT_DIR / f"test_model_kyiv_{p}.stl"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Test model part not found")
    return FileResponse(str(file_path), media_type="model/stl", filename=file_path.name)


@app.post("/api/global-center")
async def set_global_center_endpoint(
    center_lat: float = Query(...),
    center_lon: float = Query(...),
    utm_zone: Optional[int] = Query(None),
    authorization: Optional[str] = Header(default=None),
):
    """
    Р’СЃС‚Р°РЅРѕРІР»СЋС” РіР»РѕР±Р°Р»СЊРЅРёР№ С†РµРЅС‚СЂ РєР°СЂС‚Рё РґР»СЏ СЃРёРЅС…СЂРѕРЅС–Р·Р°С†С–С— РєРІР°РґСЂР°С‚С–РІ.

    БЕЗПЕКА: цей setter мутує ГЛОБАЛЬНИЙ стан, що впливає на стикування ВСІХ
    генерацій → лише адмін (інакше будь-хто міг збити прив'язку сітки на проді).

    Args:
        center_lat: РЁРёСЂРѕС‚Р° РіР»РѕР±Р°Р»СЊРЅРѕРіРѕ С†РµРЅС‚СЂСѓ (WGS84)
        center_lon: Р”РѕРІРіРѕС‚Р° РіР»РѕР±Р°Р»СЊРЅРѕРіРѕ С†РµРЅС‚СЂСѓ (WGS84)
        utm_zone: UTM Р·РѕРЅР° (РѕРїС†С–РѕРЅР°Р»СЊРЅРѕ, РІРёР·РЅР°С‡Р°С”С‚СЊСЃСЏ Р°РІС‚РѕРјР°С‚РёС‡РЅРѕ СЏРєС‰Рѕ РЅРµ РІРєР°Р·Р°РЅРѕ)

    Returns:
        Р†РЅС„РѕСЂРјР°С†С–СЏ РїСЂРѕ РІСЃС‚Р°РЅРѕРІР»РµРЅРёР№ С†РµРЅС‚СЂ
    """
    u = _require_user(authorization)
    if not u.get("is_admin", False):
        raise HTTPException(status_code=403, detail="Лише для адміністраторів")
    try:
        global_center = set_global_center(center_lat, center_lon, utm_zone)
        center_x_utm, center_y_utm = global_center.get_center_utm()
        return {
            "status": "success",
            "center": {
                "lat": center_lat,
                "lon": center_lon,
                "utm_zone": global_center.utm_zone,
                "utm_x": center_x_utm,
                "utm_y": center_y_utm,
            },
            "message": f"Р“Р»РѕР±Р°Р»СЊРЅРёР№ С†РµРЅС‚СЂ РІСЃС‚Р°РЅРѕРІР»РµРЅРѕ: ({center_lat:.6f}, {center_lon:.6f})"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"РџРѕРјРёР»РєР° РІСЃС‚Р°РЅРѕРІР»РµРЅРЅСЏ РіР»РѕР±Р°Р»СЊРЅРѕРіРѕ С†РµРЅС‚СЂСѓ: {str(e)}")


@app.get("/api/osm/extract")
async def osm_extract_endpoint(
    north: float, south: float, east: float, west: float
):
    """
    Локальний OSM extract з DuckDB (заміна Overpass API).
    Швидкість: 50-200ms vs 2-10s Overpass.

    Повертає {"buildings":[], "roads":[], "bridges":[], "water":[], "parks":[], "source":"local|overpass"}
    Кожен елемент має {id, wkt, тип-специфічні поля}.
    Якщо локальна БД відсутня — повертає {"source": "unavailable"} і frontend має fallback на Overpass.
    """
    from services.local_osm_db import is_available, extract_bbox
    if not is_available():
        return {"buildings": [], "roads": [], "bridges": [], "water": [], "parks": [],
                "source": "unavailable", "message": "Local OSM DB not found, use Overpass API"}
    try:
        # Валідація bbox
        if not (-90 <= south <= north <= 90) or not (-180 <= west <= east <= 180):
            raise HTTPException(status_code=400, detail=f"Invalid bbox: N={north} S={south} E={east} W={west}")
        if (north - south) > 1.0 or (east - west) > 1.0:
            raise HTTPException(status_code=400, detail="bbox too large (>1°), please narrow selection")
        data = extract_bbox(north, south, east, west)
        data["source"] = "local"
        return data
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OSM extract failed: {exc}")


@app.get("/api/global-center")
async def get_global_center_endpoint():
    """
    РћС‚СЂРёРјСѓС” РїРѕС‚РѕС‡РЅРёР№ РіР»РѕР±Р°Р»СЊРЅРёР№ С†РµРЅС‚СЂ РєР°СЂС‚Рё
    
    Returns:
        Р†РЅС„РѕСЂРјР°С†С–СЏ РїСЂРѕ РїРѕС‚РѕС‡РЅРёР№ С†РµРЅС‚СЂ Р°Р±Рѕ null СЏРєС‰Рѕ РЅРµ РІСЃС‚Р°РЅРѕРІР»РµРЅРѕ
    """
    global_center = get_global_center()
    if global_center is None:
        return {"status": "not_set", "center": None}
    
    center_x_utm, center_y_utm = global_center.get_center_utm()
    return {
        "status": "set",
        "center": {
            "lat": global_center.center_lat,
            "lon": global_center.center_lon,
            "utm_zone": global_center.utm_zone,
            "utm_x": center_x_utm,
            "utm_y": center_y_utm,
        }
    }


class HexagonalGridRequest(BaseModel):
    """Р—Р°РїРёС‚ РґР»СЏ РіРµРЅРµСЂР°С†С–С— СЃС–С‚РєРё (С€РµСЃС‚РёРєСѓС‚РЅРёРєРё Р°Р±Рѕ РєРІР°РґСЂР°С‚Рё)"""
    north: float
    south: float
    east: float
    west: float
    hex_size_m: float = Field(default=300.0, ge=100.0, le=10000.0)  # 0.3 РєРј Р·Р° Р·Р°РјРѕРІС‡СѓРІР°РЅРЅСЏРј
    grid_type: str = Field(default="hexagonal", description="РўРёРї СЃС–С‚РєРё: 'hexagonal', 'square' Р°Р±Рѕ 'circle'")


class HexagonalGridResponse(BaseModel):
    """Р’С–РґРїРѕРІС–РґСЊ Р· РіРµРєСЃР°РіРѕРЅР°Р»СЊРЅРѕСЋ СЃС–С‚РєРѕСЋ"""
    geojson: dict
    hex_count: int
    is_valid: bool
    validation_errors: List[str] = []
    grid_center: Optional[dict] = None  # Р¦РµРЅС‚СЂ СЃС–С‚РєРё РґР»СЏ СЃРёРЅС…СЂРѕРЅС–Р·Р°С†С–С— РєРѕРѕСЂРґРёРЅР°С‚


@app.post("/api/hexagonal-grid", response_model=HexagonalGridResponse)
async def generate_hexagonal_grid_endpoint(request: HexagonalGridRequest):
    """
    Р“РµРЅРµСЂСѓС” РіРµРєСЃР°РіРѕРЅР°Р»СЊРЅСѓ СЃС–С‚РєСѓ РґР»СЏ Р·Р°РґР°РЅРѕС— РѕР±Р»Р°СЃС‚С–.
    РЁРµСЃС‚РёРєСѓС‚РЅРёРєРё РјР°СЋС‚СЊ СЂРѕР·РјС–СЂ hex_size_m (Р·Р° Р·Р°РјРѕРІС‡СѓРІР°РЅРЅСЏРј 0.5 РєРј).
    РљР•РЁРЈР„ СЃС–С‚РєСѓ РїС–СЃР»СЏ РїРµСЂС€РѕС— РіРµРЅРµСЂР°С†С–С— РґР»СЏ С€РІРёРґС€РѕРіРѕ РґРѕСЃС‚СѓРїСѓ.
    """
    import hashlib
    import json
    
    try:
        # РЎС‚РІРѕСЂСЋС”РјРѕ С…РµС€ РїР°СЂР°РјРµС‚СЂС–РІ РґР»СЏ С–РґРµРЅС‚РёС„С–РєР°С†С–С— СЃС–С‚РєРё
        grid_type = request.grid_type.lower() if hasattr(request, 'grid_type') else 'hexagonal'
        grid_cache_version = "v2"
        cache_key = f"{grid_cache_version}_{request.north:.6f}_{request.south:.6f}_{request.east:.6f}_{request.west:.6f}_{request.hex_size_m:.1f}_{grid_type}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        
        # РЁР»СЏС… РґРѕ РєРµС€Сѓ СЃС–С‚РѕРє
        cache_dir = Path("cache/grids")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"grid_{cache_hash}.json"
        
        # РџРµСЂРµРІС–СЂСЏС”РјРѕ С‡Рё С” Р·Р±РµСЂРµР¶РµРЅР° СЃС–С‚РєР°
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    print(f"[INFO] Р’РёРєРѕСЂРёСЃС‚РѕРІСѓС”С‚СЊСЃСЏ Р·Р±РµСЂРµР¶РµРЅР° СЃС–С‚РєР° Р· РєРµС€Сѓ: {cache_file.name}")
                    return HexagonalGridResponse(**cached_data)
            except Exception as e:
                print(f"[WARN] РџРѕРјРёР»РєР° С‡РёС‚Р°РЅРЅСЏ РєРµС€Сѓ СЃС–С‚РєРё: {e}, РіРµРЅРµСЂСѓС”РјРѕ РЅРѕРІСѓ")
        
        print(f"[INFO] Р“РµРЅРµСЂР°С†С–СЏ РЅРѕРІРѕС— СЃС–С‚РєРё: north={request.north}, south={request.south}, east={request.east}, west={request.west}, hex_size_m={request.hex_size_m}")
        
        # Перевірка валідності координат → чітка 4xx (а не 500) з українським поясненням.
        try:
            _n, _s = float(request.north), float(request.south)
            _e, _w = float(request.east), float(request.west)
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="Координати області мають бути числами (north/south/east/west).")
        if not all(np.isfinite(v) for v in (_n, _s, _e, _w)):
            raise HTTPException(status_code=422, detail="Координати області недійсні (NaN/Inf).")
        if _n <= _s or _e <= _w:
            raise HTTPException(
                status_code=400,
                detail=(f"Невірні координати області: north ({_n}) має бути більше south ({_s}), "
                        f"а east ({_e}) більше west ({_w}). Перемалюйте рамку області."),
            )
        if float(request.hex_size_m) <= 0:
            raise HTTPException(status_code=400, detail="Розмір клітинки сітки має бути додатнім числом.")
        
        # РљРѕРЅРІРµСЂС‚СѓС”РјРѕ lat/lon bbox РІ UTM РґР»СЏ РіРµРЅРµСЂР°С†С–С— СЃС–С‚РєРё
        from services.crs_utils import bbox_latlon_to_utm
        bbox_utm = bbox_latlon_to_utm(
            request.north, request.south, request.east, request.west
        )
        bbox_meters = bbox_utm[:4]  # (minx, miny, maxx, maxy)
        to_wgs84 = bbox_utm[6]  # Р¤СѓРЅРєС†С–СЏ РґР»СЏ РєРѕРЅРІРµСЂС‚Р°С†С–С— UTM -> WGS84 (С–РЅРґРµРєСЃ 6)
        
        # Р“РµРЅРµСЂСѓС”РјРѕ СЃС–С‚РєСѓ (С€РµСЃС‚РёРєСѓС‚РЅРёРєРё, РєРІР°РґСЂР°С‚Рё Р°Р±Рѕ РєСЂСѓРіРё)
        if grid_type == 'square':
            from services.hexagonal_grid import generate_square_grid
            cells = generate_square_grid(bbox_meters, square_size_m=request.hex_size_m)
            print(f"[INFO] Р—РіРµРЅРµСЂРѕРІР°РЅРѕ {len(cells)} РєРІР°РґСЂР°С‚С–РІ")
        elif grid_type == 'circle':
            from services.hexagonal_grid import generate_circular_grid
            # Р Р°РґС–СѓСЃ = РїРѕР»РѕРІРёРЅР° hex_size_m (РґС–Р°РјРµС‚СЂ = hex_size_m РґР»СЏ СЃСѓРјС–СЃРЅРѕСЃС‚С– Р· С–РЅС€РёРјРё СЃС–С‚РєР°РјРё)
            radius_m = request.hex_size_m / 2.0
            cells = generate_circular_grid(bbox_meters, radius_m=radius_m)
            print(f"[INFO] Р—РіРµРЅРµСЂРѕРІР°РЅРѕ {len(cells)} РєСЂСѓРіС–РІ")
        else:
            cells = generate_hexagonal_grid(bbox_meters, hex_size_m=request.hex_size_m)
            print(f"[INFO] Р—РіРµРЅРµСЂРѕРІР°РЅРѕ {len(cells)} С€РµСЃС‚РёРєСѓС‚РЅРёРєС–РІ")
        
        # РљРѕРЅРІРµСЂС‚СѓС”РјРѕ РІ GeoJSON Р· РєРѕРЅРІРµСЂС‚Р°С†С–С”СЋ РєРѕРѕСЂРґРёРЅР°С‚ UTM -> WGS84
        geojson = hexagons_to_geojson(cells, to_wgs84=to_wgs84)
        
        # Р’Р°Р»С–РґСѓС”РјРѕ СЃС–С‚РєСѓ (С‚С–Р»СЊРєРё РґР»СЏ С€РµСЃС‚РёРєСѓС‚РЅРёРєС–РІ; square С– circle Р·Р°РІР¶РґРё РІР°Р»С–РґРЅС–)
        is_valid = True
        errors = []
        if grid_type == 'hexagonal':
            is_valid, errors = validate_hexagonal_grid(cells)
            if errors:
                print(f"[WARN] РџРѕРјРёР»РєРё РІР°Р»С–РґР°С†С–С— СЃС–С‚РєРё: {errors}")
        
        # РћР±С‡РёСЃР»СЋС”РјРѕ С†РµРЅС‚СЂ СЃС–С‚РєРё РґР»СЏ СЃРёРЅС…СЂРѕРЅС–Р·Р°С†С–С— РєРѕРѕСЂРґРёРЅР°С‚
        grid_center = None
        try:
            center_lat, center_lon = calculate_grid_center_from_geojson(geojson, to_wgs84=to_wgs84)
            grid_center = {
                "lat": center_lat,
                "lon": center_lon
            }
            print(f"[INFO] Р¦РµРЅС‚СЂ СЃС–С‚РєРё: lat={center_lat:.6f}, lon={center_lon:.6f}")
        except Exception as e:
            print(f"[WARN] РќРµ РІРґР°Р»РѕСЃСЏ РѕР±С‡РёСЃР»РёС‚Рё С†РµРЅС‚СЂ СЃС–С‚РєРё: {e}")
        
        response = HexagonalGridResponse(
            geojson=geojson,
            hex_count=len(cells),
            is_valid=is_valid,
            validation_errors=errors,
            grid_center=grid_center
        )
        
        # Р—Р±РµСЂС–РіР°С”РјРѕ СЃС–С‚РєСѓ РІ РєРµС€
        try:
            cache_data = {
                "geojson": response.geojson,
                "hex_count": response.hex_count,
                "is_valid": response.is_valid,
                "validation_errors": response.validation_errors,
                "grid_center": response.grid_center
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            print(f"[INFO] РЎС–С‚РєР° Р·Р±РµСЂРµР¶РµРЅР° РІ РєРµС€: {cache_file.name}")
        except Exception as e:
            print(f"[WARN] РќРµ РІРґР°Р»РѕСЃСЏ Р·Р±РµСЂРµРіС‚Рё СЃС–С‚РєСѓ РІ РєРµС€: {e}")
        
        return response
    except HTTPException:
        raise  # 4xx валідації (невірні координати/розмір) не маскуємо під 500
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] РџРѕРјРёР»РєР° РіРµРЅРµСЂР°С†С–С— СЃС–С‚РєРё: {e}\n{error_trace}")
        raise HTTPException(status_code=500, detail="Не вдалося згенерувати сітку. Спробуйте меншу область або інший розмір клітинки.")


class ZoneGenerationRequest(BaseModel):
    """Р—Р°РїРёС‚ РґР»СЏ РіРµРЅРµСЂР°С†С–С— РјРѕРґРµР»РµР№ РґР»СЏ РІРёР±СЂР°РЅРёС… Р·РѕРЅ"""
    model_config = ConfigDict(protected_namespaces=())
    
    zones: List[dict] = Field(max_length=64)  # РЎРїРёСЃРѕРє Р·РѕРЅ (GeoJSON features) — кап 64 проти DoS
    # Hex grid parameters (used to reconstruct exact zone polygons in metric space for perfect stitching)
    hex_size_m: float = Field(default=300.0, ge=100.0, le=10000.0)
    # IMPORTANT: city/area bbox (WGS84) for a stable global reference across sessions.
    # If provided, global_center + DEM bbox + elevation_ref are computed/cached from this bbox,
    # so later "add more zones" runs stitch perfectly with earlier prints.
    north: Optional[float] = None
    south: Optional[float] = None
    east: Optional[float] = None
    west: Optional[float] = None
    # Р’СЃС– С–РЅС€С– РїР°СЂР°РјРµС‚СЂРё СЏРє Сѓ GenerationRequest
    model_size_mm: float = Field(default=80.0, ge=10.0, le=500.0)
    road_width_multiplier: float = Field(default=0.8, ge=0.1, le=5.0)
    road_height_mm: float = Field(default=0.5, ge=0.1, le=10.0)
    road_embed_mm: float = Field(default=0.3, ge=0.0, le=5.0)
    building_min_height: float = Field(default=5.0, ge=1.0, le=100.0)
    building_height_multiplier: float = Field(default=1.8, ge=0.1, le=10.0)
    building_foundation_mm: float = Field(default=0.6, ge=0.0, le=10.0)
    building_embed_mm: float = Field(default=0.2, ge=0.0, le=5.0)
    building_max_foundation_mm: float = Field(default=5.0, ge=0.0, le=20.0)
    water_depth: float = Field(default=1.2, ge=0.1, le=10.0)  # 1.2РјРј РІ Р·РµРјР»С–, РїРѕРІРµСЂС…РЅСЏ 0.2РјРј РЅРёР¶С‡Рµ СЂРµР»СЊС”С„Сѓ
    terrain_enabled: bool = True
    terrain_z_scale: float = Field(default=0.5, ge=0.1, le=10.0)
    terrain_base_thickness_mm: float = Field(default=0.3, ge=0.2, le=20.0)  # РџС–РґР»РѕР¶РєР° 0.3РјРј Р·Р° Р·Р°РјРѕРІС‡СѓРІР°РЅРЅСЏРј
    terrain_resolution: int = Field(default=180, ge=50, le=500)
    terrarium_zoom: int = Field(default=15, ge=10, le=18)
    terrain_smoothing_sigma: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    terrain_subdivide: bool = False
    terrain_subdivide_levels: int = Field(default=1, ge=1, le=3)
    flatten_buildings_on_terrain: bool = True
    flatten_roads_on_terrain: bool = False
    # Fast preview mode (~30s): skip Blender groove cutting + manifold cleanup,
    # downscale terrain to 60x60. Same content as full mode.
    preview_mode: bool = False
    export_format: str = Field(default="3mf", pattern="^(stl|3mf)$")
    context_padding_m: float = Field(default=400.0, ge=0.0, le=5000.0)
    # Fast mode for stitching diagnostics: generate only terrain (optionally with water depression)
    terrain_only: bool = False
    include_parks: bool = True
    parks_height_mm: float = Field(default=0.6, ge=0.1, le=5.0)
    parks_embed_mm: float = Field(default=1.0, ge=0.0, le=2.0)
    include_pois: bool = False # POI removed but keep field for compatibility
    is_ams_mode: bool = False
    flat_plate_mode: bool = False
    flat_water_layer_mm: float = Field(default=0.22, ge=0.0, le=5.0)
    flat_roads_layer_mm: float = Field(default=0.42, ge=0.0, le=5.0)
    flat_parks_layer_mm: float = Field(default=0.36, ge=0.0, le=5.0)
    flat_max_building_height_mm: float = Field(default=0.0, ge=0.0, le=20.0)
    flat_uniform_building_height: bool = False
    # Кольорова тема/палітра (#2): classic (дефолт, без змін) | sepia | noir | ocean | neon.
    # Застосовується post-export як перепатч m:colorgroup 3MF (стилістика «настрою» карти).
    color_palette: str = "classic"
    # Мапа-магніт: кругла кишеня під магніт у центрі дна (плаский режим).
    magnet_pocket: bool = False
    magnet_pocket_diameter_mm: float = Field(default=10.4, ge=4.0, le=30.0)
    magnet_pocket_depth_mm: float = Field(default=2.0, ge=1.0, le=4.0)
    # Кілька кишень (шайби Ø4×2мм): 1 = центр (старий режим), 4 = діагональне
    # кільце по кутах з відступом inset від краю. Розкладку рахує
    # build_magnet_pocket_geometry (flat_plate_pipeline).
    magnet_pocket_count: int = Field(default=1, ge=1, le=8)
    magnet_pocket_inset_mm: float = Field(default=8.0, ge=3.0, le=30.0)
    # Підпис на плоскій мапі/магніті: рельєфний текст у смузі внизу плити.
    map_label: str = Field(default="", max_length=40)
    map_label_text_height_mm: float = Field(default=5.0, ge=2.5, le=12.0)
    # З'ЄДНУВАЧ-ПАЗИ (метелик/bowtie): універсальні «ластівчин-хвіст» пази на
    # серединах граней плоскої карти + окрема деталь-ключ. Дві карти/плитки
    # стикуються пазами; паз ріжеться у ДНІ ≥3мм основи → спереду шов непомітний.
    # build_map_connector_geometry (flat_plate_pipeline); FDM-кліренс 0.2мм/бік.
    map_connector: bool = False
    map_connector_edges: str = Field(default="NSEW", max_length=4)
    map_connector_span_mm: float = Field(default=10.0, ge=4.0, le=30.0)
    map_connector_length_mm: float = Field(default=15.0, ge=6.0, le=40.0)
    map_connector_depth_mm: float = Field(default=2.0, ge=0.2, le=4.0)
    map_connector_clearance_mm: float = Field(default=0.03, ge=0.02, le=0.6)
    # Грані, для яких випускаємо КЛЮЧ (для серії — лише S/E внутрішні, 1 ключ/шов).
    # Порожнє → ключ для кожного пазу (single-tile). Паз ріжемо на всіх map_connector_edges.
    map_connector_key_edges: str = Field(default="", max_length=4)
    # НАПРЯМКОВИЙ добір граней (азимути нормалі у градусах, "30,90,150") — для серії
    # шестикутників/повернутих клітин, де NSEW (4 кардинали) не адресує 6 граней.
    # Порожнє → стара NSEW-поведінка. *_key_az = підмножина з ключем (1 на шов).
    map_connector_edge_az: str = Field(default="", max_length=200)
    map_connector_key_az: str = Field(default="", max_length=200)
    # ПРЕМІУМ-РАМКА: компас (стрілка-N), масштабна лінійка (0…N м) і координати
    # центру (lat/lon) окремою чорною деталлю «Frame», вирізаною з шарів карти.
    # build_map_frame_overlay (flat_plate_pipeline). Працює у flat_plate.
    map_frame: bool = False
    map_frame_compass: bool = True
    map_frame_scale: bool = True
    map_frame_coords: bool = True
    # СТИЛЬ ОРНАМЕНТАЛЬНОЇ РАМКИ: "classic" = поточна поведінка (компас+лінійка+
    # координати, без зовнішнього ободка); "ornate" = декоративний підведений
    # подвійний ободок по периметру + кутові мотиви; "compass" = ті ж елементи +
    # тонкий зовнішній ободок. Рендериться у build_map_frame_overlay.
    frame_style: str = "classic"
    # ВИДІЛЕНА БУДІВЛЯ на карті: користувач обирає свій будинок (highlight_point
    # [lon,lat] від кліку по карті; інакше — будинок у центрі) → ОКРЕМА ЧЕРВОНА
    # вставна деталь (паз+peg). build_highlight_insert (flat_plate_pipeline).
    map_highlight_building: bool = False
    highlight_point: Optional[List[float]] = Field(default=None, max_length=2)
    # Кілька будинків: список [[lon,lat],...] (дім, робота, орієнтири) — кожен окрема
    # ЧЕРВОНА вставна деталь. Має пріоритет над highlight_point. Кап 12 у пайплайні.
    highlight_points: Optional[List[List[float]]] = Field(default=None, max_length=12)
    keychain_mode: bool = False
    keychain_label: str = Field(default="", max_length=64)
    keychain_base_shape: str = Field(default="rounded", max_length=24)
    keychain_layout_rotation_deg: float = Field(default=0.0, ge=0.0, le=360.0)
    keychain_loop_style: str = Field(default="round", max_length=24)
    keychain_loop_angle_deg: float = Field(default=0.0, ge=0.0, le=360.0)
    keychain_body_width_mm: float = Field(default=35.0, ge=20.0, le=180.0)
    keychain_body_height_mm: float = Field(default=55.0, ge=16.0, le=140.0)
    keychain_map_x_mm: float = Field(default=2.0, ge=0.0, le=180.0)
    keychain_map_y_mm: float = Field(default=3.0, ge=0.0, le=140.0)
    keychain_map_width_mm: float = Field(default=31.0, ge=4.0, le=180.0)
    keychain_map_height_mm: float = Field(default=40.0, ge=4.0, le=140.0)
    keychain_map_rotation_deg: float = Field(default=0.0, ge=0.0, le=360.0)
    keychain_loop_center_x_mm: float = Field(default=17.5, ge=-30.0, le=210.0)
    keychain_loop_center_y_mm: float = Field(default=-4.0, ge=-40.0, le=180.0)
    keychain_label_center_x_mm: float = Field(default=17.5, ge=0.0, le=180.0)
    keychain_label_center_y_mm: float = Field(default=49.5, ge=0.0, le=140.0)
    keychain_label_angle_deg: float = Field(default=0.0, ge=0.0, le=360.0)
    keychain_loop_outer_radius_mm: float = Field(default=6.5, ge=2.4, le=18.0)
    keychain_loop_inner_radius_mm: float = Field(default=3.0, ge=1.5, le=12.0)
    keychain_corner_radius_mm: float = Field(default=4.0, ge=0.0, le=16.0)
    keychain_label_band_height_mm: float = Field(default=9.0, ge=0.0, le=30.0)
    keychain_label_raise_mm: float = Field(default=0.6, ge=0.0, le=3.0)
    keychain_label_text_height_mm: float = Field(default=4.2, ge=2.0, le=12.0)
    keychain_label_width_mm: float = Field(default=30.0, ge=4.0, le=180.0)
    keychain_label_stroke_mm: float = Field(default=0.9, ge=0.8, le=3.0)
    keychain_label_font_style: str = Field(default="block", max_length=24)
    keychain_rim_width_mm: float = Field(default=1.2, ge=0.0, le=6.0)
    keychain_rim_height_mm: float = Field(default=0.45, ge=0.0, le=3.0)
    # Другий рядок напису (дата/координати) — менший кегль, під основним.
    keychain_label2: str = Field(default="", max_length=64)
    keychain_label2_text_height_mm: float = Field(default=2.4, ge=1.5, le=8.0)
    # Напис на ЗВОРОТІ: гравіюється у нижню грань (дзеркально — читається при перевороті).
    keychain_back_label: str = Field(default="", max_length=64)
    keychain_back_text_height_mm: float = Field(default=5.0, ge=2.5, le=14.0)
    keychain_back_engrave_mm: float = Field(default=0.5, ge=0.2, le=1.2)
    # ТОПО-БРЕЛОК (C3): heightfield-рельєф на жетоні (див. GenerationRequest).
    keychain_topo_mode: bool = False
    keychain_relief_mm: float = Field(default=2.2, ge=0.6, le=4.0)
    canonical_mask_bundle_dir: Optional[str] = None
    auto_canonicalize_masks: bool = True


@app.post("/api/generate-zones", response_model=GenerationResponse)
async def generate_zones_endpoint(
    request: ZoneGenerationRequest,
    background_tasks: BackgroundTasks,
    _rl: None = Depends(rate_limit("generate_zones", [(3, 60.0)])),
):

    if not request.zones or len(request.zones) == 0:
        raise HTTPException(status_code=400, detail="РќРµ РІРёР±СЂР°РЅРѕ Р¶РѕРґРЅРѕС— Р·РѕРЅРё")
    
    # РљР РРўРР§РќРћ: Р’РёР·РЅР°С‡Р°С”РјРѕ РіР»РѕР±Р°Р»СЊРЅРёР№ С†РµРЅС‚СЂ РґР»СЏ Р’РЎР†Р„Р‡ СЃС–С‚РєРё.
    # If client provides city bbox, use it for a stable reference; otherwise fallback to selected zones bbox.
    # Р¦Рµ Р·Р°Р±РµР·РїРµС‡СѓС”, С‰Рѕ РІСЃС– Р·РѕРЅРё РІРёРєРѕСЂРёСЃС‚РѕРІСѓСЋС‚СЊ РѕРґРЅСѓ С‚РѕС‡РєСѓ РІС–РґР»С–РєСѓ (0,0)
    # С– С–РґРµР°Р»СЊРЅРѕ РїС–РґС…РѕРґСЏС‚СЊ РѕРґРЅР° РґРѕ РѕРґРЅРѕС—
    print(f"[INFO] Р’РёР·РЅР°С‡РµРЅРЅСЏ РіР»РѕР±Р°Р»СЊРЅРѕРіРѕ С†РµРЅС‚СЂСѓ РґР»СЏ РІСЃС–С”С— СЃС–С‚РєРё ({len(request.zones)} Р·РѕРЅ)...")
    
    grid_bbox = None
    # 1) Prefer explicit city bbox (stable across later zone additions)
    try:
        if request.north is not None and request.south is not None and request.east is not None and request.west is not None:
            if float(request.north) > float(request.south) and float(request.east) > float(request.west):
                grid_bbox = {
                    "north": float(request.north),
                    "south": float(request.south),
                    "east": float(request.east),
                    "west": float(request.west),
                }
    except Exception:
        grid_bbox = None

    # 2) Fallback: compute bbox from selected zones (old behavior)
    if grid_bbox is None:
        all_lons = []
        all_lats = []
        for zone in request.zones:
            geometry = zone.get('geometry', {})
            if geometry.get('type') != 'Polygon':
                continue
            coordinates = geometry.get('coordinates', [])
            if not coordinates or len(coordinates) == 0:
                continue
            all_coords = [coord for ring in coordinates for coord in ring]
            zone_lons = [coord[0] for coord in all_coords]
            zone_lats = [coord[1] for coord in all_coords]
            all_lons.extend(zone_lons)
            all_lats.extend(zone_lats)
        if len(all_lons) == 0 or len(all_lats) == 0:
            raise HTTPException(status_code=400, detail="РќРµ РІРґР°Р»РѕСЃСЏ РІРёР·РЅР°С‡РёС‚Рё РєРѕРѕСЂРґРёРЅР°С‚Рё Р·РѕРЅ")
        grid_bbox = {
            'north': max(all_lats),
            'south': min(all_lats),
            'east': max(all_lons),
            'west': min(all_lons)
        }
    
    # Р’РёР·РЅР°С‡Р°С”РјРѕ С†РµРЅС‚СЂ РІСЃС–С”С— СЃС–С‚РєРё
    grid_center_lat = (grid_bbox['north'] + grid_bbox['south']) / 2.0
    grid_center_lon = (grid_bbox['east'] + grid_bbox['west']) / 2.0
    
    print(f"[INFO] Р“Р»РѕР±Р°Р»СЊРЅРёР№ С†РµРЅС‚СЂ СЃС–С‚РєРё: lat={grid_center_lat:.6f}, lon={grid_center_lon:.6f}")
    print(f"[INFO] Bbox РІСЃС–С”С— СЃС–С‚РєРё: north={grid_bbox['north']:.6f}, south={grid_bbox['south']:.6f}, east={grid_bbox['east']:.6f}, west={grid_bbox['west']:.6f}")
    
    # Cache global city reference so future "add more zones" uses the same values.
    grid_bbox_latlon = (grid_bbox['north'], grid_bbox['south'], grid_bbox['east'], grid_bbox['west'])
    import hashlib, json
    cache_dir = Path("cache/cities")
    cache_dir.mkdir(parents=True, exist_ok=True)
    # cache version bump: v5 adds elevation_max_m (series shared relief gain) — refresh
    city_key = f"v5_{grid_bbox_latlon[0]:.6f}_{grid_bbox_latlon[1]:.6f}_{grid_bbox_latlon[2]:.6f}_{grid_bbox_latlon[3]:.6f}_z{int(request.terrarium_zoom)}_zs{float(request.terrain_z_scale):.3f}_ms{float(request.model_size_mm):.1f}"
    city_hash = hashlib.md5(city_key.encode()).hexdigest()
    city_cache_file = cache_dir / f"city_{city_hash}.json"

    cached = None
    if city_cache_file.exists():
        try:
            cached = json.loads(city_cache_file.read_text(encoding="utf-8"))
            print(f"[INFO] Р’РёРєРѕСЂРёСЃС‚РѕРІСѓС”РјРѕ РєРµС€ РјС–СЃС‚Р°: {city_cache_file.name}")
        except Exception:
            cached = None

    if cached and isinstance(cached, dict) and "center" in cached:
        try:
            c = cached.get("center") or {}
            global_center = set_global_center(float(c["lat"]), float(c["lon"]))
        except Exception:
            global_center = set_global_center(grid_center_lat, grid_center_lon)
    else:
        global_center = set_global_center(grid_center_lat, grid_center_lon)
    print(f"[INFO] Р“Р»РѕР±Р°Р»СЊРЅРёР№ С†РµРЅС‚СЂ РІСЃС‚Р°РЅРѕРІР»РµРЅРѕ: lat={global_center.center_lat:.6f}, lon={global_center.center_lon:.6f}, UTM zone={global_center.utm_zone}")

    # CRITICAL: store global DEM bbox so all zones sample elevations from the same tile set (and it is stable across sessions)
    try:
        from services.global_center import set_global_dem_bbox_latlon
        set_global_dem_bbox_latlon(grid_bbox_latlon)
    except Exception:
        pass
    
    # РљР РРўРР§РќРћ: РћР±С‡РёСЃР»СЋС”РјРѕ РіР»РѕР±Р°Р»СЊРЅРёР№ elevation_ref_m РґР»СЏ РІСЃС–С”С— СЃС–С‚РєРё
    # Р¦Рµ Р·Р°Р±РµР·РїРµС‡СѓС”, С‰Рѕ РІСЃС– Р·РѕРЅРё РІРёРєРѕСЂРёСЃС‚РѕРІСѓСЋС‚СЊ РѕРґРЅСѓ Р±Р°Р·РѕРІСѓ РІРёСЃРѕС‚Сѓ РґР»СЏ РЅРѕСЂРјР°Р»С–Р·Р°С†С–С—
    # С– С–РґРµР°Р»СЊРЅРѕ СЃС‚РёРєСѓСЋС‚СЊСЃСЏ РѕРґРЅР° Р· РѕРґРЅРѕСЋ
    print(f"[INFO] РћР±С‡РёСЃР»РµРЅРЅСЏ РіР»РѕР±Р°Р»СЊРЅРѕРіРѕ elevation_ref РґР»СЏ СЃРёРЅС…СЂРѕРЅС–Р·Р°С†С–С— РІРёСЃРѕС‚ РјС–Р¶ Р·РѕРЅР°РјРё...")
    
    # Р’РёР·РЅР°С‡Р°С”РјРѕ source_crs РґР»СЏ РѕР±С‡РёСЃР»РµРЅРЅСЏ elevation_ref
    source_crs = None
    try:
        from services.crs_utils import bbox_latlon_to_utm
        bbox_utm_result = bbox_latlon_to_utm(*grid_bbox_latlon)
        source_crs = bbox_utm_result[4]  # CRS
    except Exception as e:
        print(f"[WARN] РќРµ РІРґР°Р»РѕСЃСЏ РІРёР·РЅР°С‡РёС‚Рё source_crs РґР»СЏ elevation_ref: {e}")
    
    # РћР±С‡РёСЃР»СЋС”РјРѕ РіР»РѕР±Р°Р»СЊРЅРёР№ elevation_ref_m С‚Р° baseline_offset_m
    # Guard against corrupted/invalid cached refs (we've seen Terrarium outlier pixels produce huge negative mins).
    cached_elev = None
    if cached and isinstance(cached, dict):
        try:
            ce = cached.get("elevation_ref_m")
            if ce is not None:
                ce = float(ce)
                # Reject clearly bogus negative refs (Terrarium outliers) that create "tower bases".
                if -120.0 <= ce <= 9000.0:
                    cached_elev = ce
        except Exception:
            cached_elev = None

    global_elevation_max_m = None
    if cached_elev is not None:
        global_elevation_ref_m = float(cached.get("elevation_ref_m"))
        global_baseline_offset_m = float(cached.get("baseline_offset_m") or 0.0)
        try:
            _cmax = cached.get("elevation_max_m")
            global_elevation_max_m = float(_cmax) if _cmax is not None else None
        except Exception:
            global_elevation_max_m = None
        print(f"[INFO] Р“Р»РѕР±Р°Р»СЊРЅРёР№ elevation_ref_m (РєРµС€): {global_elevation_ref_m:.2f}Рј")
        print(f"[INFO] Р“Р»РѕР±Р°Р»СЊРЅРёР№ baseline_offset_m (РєРµС€): {global_baseline_offset_m:.3f}Рј")
    else:
        # Pass explicit bbox if available to ensure stability
        explicit_grid_bbox_tuple = None
        if grid_bbox is not None:
            explicit_grid_bbox_tuple = (
                grid_bbox['north'],
                grid_bbox['south'],
                grid_bbox['east'],
                grid_bbox['west']
            )

        # 524-guard: this DEM sampling is the only synchronous external work BEFORE the
        # response. On an UNCACHED city Terrarium can take >100s -> Cloudflare 524 (and it
        # blocks the whole event loop). Offload to a thread + timeout; on timeout fall back
        # to per-zone normalization (ref=None, supported downstream). Cached city never gets here.
        import asyncio as _aio
        try:
            global_elevation_ref_m, global_baseline_offset_m, global_elevation_max_m = await _aio.wait_for(
                _aio.to_thread(
                    calculate_global_elevation_reference,
                    zones=request.zones,
                    source_crs=source_crs,
                    terrarium_zoom=request.terrarium_zoom if hasattr(request, 'terrarium_zoom') else 15,
                    z_scale=float(request.terrain_z_scale),
                    sample_points_per_zone=25,
                    global_center=global_center,
                    explicit_bbox=explicit_grid_bbox_tuple,
                ),
                timeout=25.0,
            )
        except Exception as _elev_exc:
            print(f"[WARN] elevation_ref slow/failed ({_elev_exc}); fallback to per-zone normalization")
            global_elevation_ref_m = None
            global_baseline_offset_m = None
            global_elevation_max_m = None
    
    if global_elevation_ref_m is not None:
        print(f"[INFO] Р“Р»РѕР±Р°Р»СЊРЅРёР№ elevation_ref_m: {global_elevation_ref_m:.2f}Рј (РІРёСЃРѕС‚Р° РЅР°Рґ СЂС–РІРЅРµРј РјРѕСЂСЏ)")
        print(f"[INFO] Р“Р»РѕР±Р°Р»СЊРЅРёР№ baseline_offset_m: {global_baseline_offset_m:.3f}Рј")
    else:
        print(f"[WARN] РќРµ РІРґР°Р»РѕСЃСЏ РѕР±С‡РёСЃР»РёС‚Рё РіР»РѕР±Р°Р»СЊРЅРёР№ elevation_ref_m, РєРѕР¶РЅР° Р·РѕРЅР° РІРёРєРѕСЂРёСЃС‚РѕРІСѓРІР°С‚РёРјРµ Р»РѕРєР°Р»СЊРЅСѓ РЅРѕСЂРјР°Р»С–Р·Р°С†С–СЋ")

    # СЕРІЯ ЗОН — спільний перепад висот (світові метри) для ЄДИНОГО gain рельєфу на всіх
    # плитках. Без цього кожна плитка масштабувала рельєф власним gain (рівнинна ×3.5,
    # горбиста ×1.2) → сходинка на шві («висоти перепадають»). Передаємо у кожну зону.
    global_relief_range_m = 0.0
    try:
        if (global_elevation_ref_m is not None and global_elevation_max_m is not None
                and float(global_elevation_max_m) > float(global_elevation_ref_m)):
            global_relief_range_m = float(global_elevation_max_m) - float(global_elevation_ref_m)
            print(f"[INFO] Р“Р»РѕР±Р°Р»СЊРЅРёР№ РїРµСЂРµРїР°Рґ РІРёСЃРѕС‚ (range): {global_relief_range_m:.2f}Рј "
                  f"(СЌРґРёРЅРёР№ gain РґР»СЏ РІСЃС–С… РїР»РёС‚РѕРє СЃРµСЂС–С—)")
    except Exception:
        global_relief_range_m = 0.0

    # РћР±С‡РёСЃР»СЋС”РјРѕ РѕРїС‚РёРјР°Р»СЊРЅСѓ С‚РѕРІС‰РёРЅСѓ РїС–РґР»РѕР¶РєРё РґР»СЏ РІСЃС–С… Р·РѕРЅ
    # CRITICAL: base thickness must be stable across "add more zones", BUT ALSO must be thick enough to hold all grooves!
    # If a park embeds 1.0mm, the base MUST be more than 1.0mm, otherwise the boolean cut will punch a hole through the bottom floor!
    requested_base_thickness_mm = float(request.terrain_base_thickness_mm)
    final_base_thickness_mm = _normalize_request_base_thickness(request)
    min_required_base_mm = _compute_safe_base_thickness_mm(
        request.model_copy(update={"terrain_base_thickness_mm": 0.2})
    )
    print(
        f"[INFO] Р¤С–РЅР°Р»СЊРЅР° С‚РѕРІС‰РёРЅР° РїС–РґР»РѕР¶РєРё: {final_base_thickness_mm:.2f}РјРј "
        f"(Р·Р°РїРёС‚Р°РЅР°: {requested_base_thickness_mm:.2f}РјРј, "
        f"РјС–РЅ.РїРѕС‚СЂС–Р±РЅР° РґР»СЏ РїР°Р·С–РІ: {min_required_base_mm:.2f}РјРј)"
    )

    # Save/refresh city cache for future requests
    try:
        cache_payload = {
            "bbox": {"north": grid_bbox_latlon[0], "south": grid_bbox_latlon[1], "east": grid_bbox_latlon[2], "west": grid_bbox_latlon[3]},
            "center": {"lat": float(global_center.center_lat), "lon": float(global_center.center_lon)},
            "terrarium_zoom": int(request.terrarium_zoom),
            "terrain_z_scale": float(request.terrain_z_scale),
            "model_size_mm": float(request.model_size_mm),
            "elevation_ref_m": float(global_elevation_ref_m) if global_elevation_ref_m is not None else None,
            "baseline_offset_m": float(global_baseline_offset_m) if global_baseline_offset_m is not None else 0.0,
            "elevation_max_m": float(global_elevation_max_m) if global_elevation_max_m is not None else None,
            "terrain_base_thickness_mm": float(final_base_thickness_mm),
        }
        city_cache_file.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    
    # 3. РћР±С‡РёСЃР»СЋС”РјРѕ РіР»РѕР±Р°Р»СЊРЅРёР№ РєСЂРѕРє СЃС–С‚РєРё (Grid Step) РґР»СЏ С–РґРµР°Р»СЊРЅРѕРіРѕ СЃС‚РёРєСѓРІР°РЅРЅСЏ
    # Р—Р°РјС–СЃС‚СЊ "resolution" (СЏРєРёР№ РґР°С” СЂС–Р·РЅРёР№ РєСЂРѕРє РґР»СЏ СЂС–Р·РЅРёС… bbox), РІРёРєРѕСЂРёСЃС‚РѕРІСѓС”РјРѕ С„С–РєСЃРѕРІР°РЅРёР№ РєСЂРѕРє РІ РјРµС‚СЂР°С….
    # Р‘Р°Р·СѓС”РјРѕСЃСЊ РЅР° СЃРµСЂРµРґРЅСЊРѕРјСѓ СЂРѕР·РјС–СЂС– Р·РѕРЅРё (РЅР°РїСЂРёРєР»Р°Рґ, 400Рј) С– Р±Р°Р¶Р°РЅС–Р№ СЂРµР·РѕР»СЋС†С–С—.
    # Р¦Рµ РіР°СЂР°РЅС‚СѓС”, С‰Рѕ vertices РІСЃС–С… Р·РѕРЅ Р»РµР¶Р°С‚РёРјСѓС‚СЊ РЅР° РѕРґРЅС–Р№ РіР»РѕР±Р°Р»СЊРЅС–Р№ СЃС–С‚С†С–.
    target_res = float(request.terrain_resolution) if request.terrain_resolution else 150.0
    
    # РћРџРўРРњР†Р—РђР¦Р†РЇ: РђРґР°РїС‚РёРІРЅРёР№ grid_step_m РґР»СЏ РєСЂР°С‰РѕС— РїСЂРѕРґСѓРєС‚РёРІРЅРѕСЃС‚С–
    # Р”Р»СЏ РјРµРЅС€РёС… Р·РѕРЅ РІРёРєРѕСЂРёСЃС‚РѕРІСѓС”РјРѕ Р±С–Р»СЊС€РёР№ РєСЂРѕРє (РјРµРЅС€Р° РґРµС‚Р°Р»С–Р·Р°С†С–СЏ)
    base_size = float(getattr(request, "hex_size_m", 300.0))  # Р‘Р°Р·РѕРІРёР№ СЂРѕР·РјС–СЂ Р·РѕРЅРё
    base_grid_step = base_size / target_res
    
    # РЇРєС‰Рѕ resolution РІРёСЃРѕРєРёР№ (>150), Р·Р±С–Р»СЊС€СѓС”РјРѕ РєСЂРѕРє РґР»СЏ РѕРїС‚РёРјС–Р·Р°С†С–С—
    # Р¦Рµ Р·РјРµРЅС€СѓС” РєС–Р»СЊРєС–СЃС‚СЊ РІРµСЂС€РёРЅ Р±РµР· РІС‚СЂР°С‚Рё СЏРєРѕСЃС‚С– РґР»СЏ Р±С–Р»СЊС€РѕСЃС‚С– РІРёРїР°РґРєС–РІ
    if target_res > 150:
        # Р”Р»СЏ РІРёСЃРѕРєРѕС— СЂРµР·РѕР»СЋС†С–С—: Р·Р±С–Р»СЊС€СѓС”РјРѕ РєСЂРѕРє РЅР° 25% РґР»СЏ РѕРїС‚РёРјС–Р·Р°С†С–С—
        base_grid_step *= 1.25
        print(f"[INFO] OPTIMIZATION: Increased grid_step for resolution={target_res} (performance mode)")
    
    global_grid_step_m = base_grid_step
    # РћРєСЂСѓРіР»СЏС”РјРѕ РґРѕ СЂРѕР·СѓРјРЅРѕРіРѕ Р·РЅР°С‡РµРЅРЅСЏ (РЅР°РїСЂРёРєР»Р°Рґ, 0.5, 1.0, 2.0, 2.5, 3.0)
    global_grid_step_m = round(global_grid_step_m * 2) / 2.0
    if global_grid_step_m < 0.5: global_grid_step_m = 0.5
    print(f"[INFO] Р“Р»РѕР±Р°Р»СЊРЅРёР№ РєСЂРѕРє СЃС–С‚РєРё (grid_step_m): {global_grid_step_m}Рј (РґР»СЏ resolution={target_res})")

    task_ids = []
    for zone_idx, zone in enumerate(request.zones):
        # ... (rest of loop)
        # РћС‚СЂРёРјСѓС”РјРѕ bbox Р· Р·РѕРЅРё
        geometry = zone.get('geometry', {})
        if geometry.get('type') != 'Polygon':
            continue
        
        coordinates = geometry.get('coordinates', [])
        if not coordinates or len(coordinates) == 0:
            continue
        
        # Р—РЅР°С…РѕРґРёРјРѕ min/max РєРѕРѕСЂРґРёРЅР°С‚Рё
        all_coords = [coord for ring in coordinates for coord in ring]
        lons = [coord[0] for coord in all_coords]
        lats = [coord[1] for coord in all_coords]
        
        zone_bbox = {
            'north': max(lats),
            'south': min(lats),
            'east': max(lons),
            'west': min(lons)
        }
        
        # РЎС‚РІРѕСЂСЋС”РјРѕ GenerationRequest РґР»СЏ С†С–С”С— Р·РѕРЅРё
        # Р’РёРєРѕСЂРёСЃС‚РѕРІСѓС”РјРѕ РґРµС„РѕР»С‚РЅРµ Р·РЅР°С‡РµРЅРЅСЏ РґР»СЏ terrain_smoothing_sigma СЏРєС‰Рѕ None
        terrain_smoothing_sigma = request.terrain_smoothing_sigma if request.terrain_smoothing_sigma is not None else 2.0

        # З'ЄДНУВАЧІ СЕРІЇ: фронт вмикає request.map_connector + кладе у кожну зону
        # properties.connector_edges = ВНУТРІШНІ (спільні з сусідом) грані цієї плитки
        # (напр. "SE" для кутової). Ставимо замки ЛИШЕ на ці грані → сусідні шматки
        # стикуються, а зовнішній периметр серії лишається чистим. Без edges (зовн.
        # кутова без сусідів) — конектор пропускаємо.
        _zprops = zone.get('properties', {}) or {}
        _zone_conn_edges = str(_zprops.get('connector_edges') or '').upper()
        # Ключі лише на S/E внутрішніх гранях (фронт рахує) → 1 ключ на спільний шов.
        _zone_key_edges = str(_zprops.get('connector_key_edges') or '').upper()
        # НАПРЯМКОВІ грані (азимути) — шестикутник/повернута сітка. Фронт кладе
        # connector_edge_az/connector_key_az; коли є — бек добирає грані за нормаллю.
        _zone_edge_az = str(_zprops.get('connector_edge_az') or '')
        _zone_key_az = str(_zprops.get('connector_key_az') or '')
        _zone_conn = bool(getattr(request, 'map_connector', False)) and bool(_zone_conn_edges or _zone_edge_az)

        zone_request = GenerationRequest(
            north=zone_bbox['north'],
            south=zone_bbox['south'],
            east=zone_bbox['east'],
            west=zone_bbox['west'],
            model_size_mm=request.model_size_mm,
            road_width_multiplier=request.road_width_multiplier,
            road_height_mm=request.road_height_mm,
            road_embed_mm=request.road_embed_mm,
            building_min_height=request.building_min_height,
            building_height_multiplier=request.building_height_multiplier,
            building_foundation_mm=request.building_foundation_mm,
            building_embed_mm=request.building_embed_mm,
            building_max_foundation_mm=request.building_max_foundation_mm,
            water_depth=request.water_depth,
            terrain_enabled=request.terrain_enabled,
            terrain_z_scale=request.terrain_z_scale,
            terrain_base_thickness_mm=final_base_thickness_mm,  # Р’РёРєРѕСЂРёСЃС‚РѕРІСѓС”РјРѕ РѕРїС‚РёРјР°Р»СЊРЅСѓ С‚РѕРІС‰РёРЅСѓ
            terrain_resolution=request.terrain_resolution,
            terrarium_zoom=request.terrarium_zoom,
            terrain_smoothing_sigma=terrain_smoothing_sigma,
            terrain_subdivide=request.terrain_subdivide if request.terrain_subdivide is not None else False,
            terrain_subdivide_levels=request.terrain_subdivide_levels if request.terrain_subdivide_levels is not None else 1,
            flatten_buildings_on_terrain=request.flatten_buildings_on_terrain,
            flatten_roads_on_terrain=request.flatten_roads_on_terrain if request.flatten_roads_on_terrain is not None else False,
            export_format=request.export_format,
            # З'єднувачі-замки на ВНУТРІШНІХ гранях плитки серії (per-zone).
            map_connector=_zone_conn,
            map_connector_edges=(_zone_conn_edges or "NSEW"),
            map_connector_key_edges=_zone_key_edges,
            map_connector_edge_az=_zone_edge_az,
            map_connector_key_az=_zone_key_az,
            map_connector_span_mm=request.map_connector_span_mm,
            map_connector_length_mm=request.map_connector_length_mm,
            map_connector_depth_mm=request.map_connector_depth_mm,
            map_connector_clearance_mm=request.map_connector_clearance_mm,
            context_padding_m=request.context_padding_m,
            terrain_only=bool(getattr(request, "terrain_only", False)),
            preview_mode=bool(getattr(request, "preview_mode", False)) and not bool(getattr(request, "flat_plate_mode", False)),
            include_parks=request.include_parks,
            parks_height_mm=request.parks_height_mm,
            parks_embed_mm=request.parks_embed_mm,
            # include_pois is not in GenerationRequest anymore or hidden
            # РљР РРўРР§РќРћ: РџРµСЂРµРґР°С”РјРѕ РіР»РѕР±Р°Р»СЊРЅС– РїР°СЂР°РјРµС‚СЂРё РґР»СЏ СЃРёРЅС…СЂРѕРЅС–Р·Р°С†С–С— РІРёСЃРѕС‚
            elevation_ref_m=global_elevation_ref_m,  # Р“Р»РѕР±Р°Р»СЊРЅР° Р±Р°Р·РѕРІР° РІРёСЃРѕС‚Р° РґР»СЏ РІСЃС–С… Р·РѕРЅ
            baseline_offset_m=global_baseline_offset_m,  # Р“Р»РѕР±Р°Р»СЊРЅРµ Р·РјС–С‰РµРЅРЅСЏ baseline
            terrain_relief_range_m=float(global_relief_range_m),  # СЕРІЯ: спільний перепад → єдиний gain рельєфу
            preserve_global_xy=True,  # IMPORTANT: export in a shared coordinate frame for stitching
            grid_step_m=global_grid_step_m,  # GLOBAL GRID FIX
            is_ams_mode=request.is_ams_mode and not bool(getattr(request, "flat_plate_mode", False)),
            flat_plate_mode=bool(getattr(request, "flat_plate_mode", False)),
            flat_water_layer_mm=float(getattr(request, "flat_water_layer_mm", 0.22)),
            flat_roads_layer_mm=float(getattr(request, "flat_roads_layer_mm", 0.42)),
            flat_parks_layer_mm=float(getattr(request, "flat_parks_layer_mm", 0.36)),
            flat_max_building_height_mm=float(getattr(request, "flat_max_building_height_mm", 0.0)),
            flat_uniform_building_height=bool(getattr(request, "flat_uniform_building_height", False)),
            keychain_mode=bool(getattr(request, "keychain_mode", False)),
            keychain_label=str(getattr(request, "keychain_label", "") or ""),
            keychain_base_shape=str(getattr(request, "keychain_base_shape", "rounded") or "rounded"),
            keychain_layout_rotation_deg=float(getattr(request, "keychain_layout_rotation_deg", 0.0)),
            keychain_loop_style=str(getattr(request, "keychain_loop_style", "round") or "round"),
            keychain_loop_angle_deg=float(getattr(request, "keychain_loop_angle_deg", 0.0)),
            keychain_body_width_mm=float(getattr(request, "keychain_body_width_mm", 35.0)),
            keychain_body_height_mm=float(getattr(request, "keychain_body_height_mm", 55.0)),
            keychain_map_x_mm=float(getattr(request, "keychain_map_x_mm", 2.0)),
            keychain_map_y_mm=float(getattr(request, "keychain_map_y_mm", 3.0)),
            keychain_map_width_mm=float(getattr(request, "keychain_map_width_mm", 31.0)),
            keychain_map_height_mm=float(getattr(request, "keychain_map_height_mm", 40.0)),
            keychain_map_rotation_deg=float(getattr(request, "keychain_map_rotation_deg", 0.0)),
            keychain_loop_center_x_mm=float(getattr(request, "keychain_loop_center_x_mm", 17.5)),
            keychain_loop_center_y_mm=float(getattr(request, "keychain_loop_center_y_mm", -4.0)),
            keychain_label_center_x_mm=float(getattr(request, "keychain_label_center_x_mm", 17.5)),
            keychain_label_center_y_mm=float(getattr(request, "keychain_label_center_y_mm", 49.5)),
            keychain_label_angle_deg=float(getattr(request, "keychain_label_angle_deg", 0.0)),
            keychain_loop_outer_radius_mm=float(getattr(request, "keychain_loop_outer_radius_mm", 6.5)),
            keychain_loop_inner_radius_mm=float(getattr(request, "keychain_loop_inner_radius_mm", 3.0)),
            keychain_corner_radius_mm=float(getattr(request, "keychain_corner_radius_mm", 4.0)),
            keychain_label_band_height_mm=float(getattr(request, "keychain_label_band_height_mm", 9.0)),
            keychain_label_raise_mm=float(getattr(request, "keychain_label_raise_mm", 0.6)),
            keychain_label_text_height_mm=float(getattr(request, "keychain_label_text_height_mm", 4.2)),
            keychain_label_width_mm=float(getattr(request, "keychain_label_width_mm", 30.0)),
            keychain_label_stroke_mm=float(getattr(request, "keychain_label_stroke_mm", 0.9)),
            keychain_label_font_style=str(getattr(request, "keychain_label_font_style", "block") or "block"),
            keychain_rim_width_mm=float(getattr(request, "keychain_rim_width_mm", 1.2)),
            keychain_rim_height_mm=float(getattr(request, "keychain_rim_height_mm", 0.45)),
        )
        
        # Р“РµРЅРµСЂСѓС”РјРѕ РјРѕРґРµР»СЊ РґР»СЏ Р·РѕРЅРё
        task_id = str(uuid.uuid4())
        zone_id_str = zone.get('id', f'zone_{zone_idx}')
        props = zone.get("properties") or {}
        zone_row = props.get("row")
        zone_col = props.get("col")
        task = GenerationTask(task_id=task_id, request=zone_request)
        # row/col на задачі — для схеми розкладки панно (download_all + layout.png)
        try:
            task.zone_row = int(zone_row) if zone_row is not None else None
            task.zone_col = int(zone_col) if zone_col is not None else None
        except Exception:
            task.zone_row = task.zone_col = None
        tasks[task_id] = task
        
        # Р—Р±РµСЂС–РіР°С”РјРѕ С„РѕСЂРјСѓ Р·РѕРЅРё (РїРѕР»С–РіРѕРЅ) РґР»СЏ РѕР±СЂС–Р·Р°РЅРЅСЏ РјРµС€С–РІ
        zone_polygon_coords = coordinates[0] if coordinates and len(coordinates) > 0 else None  # Р—РѕРІРЅС–С€РЅС–Р№ ring РїРѕР»С–РіРѕРЅСѓ
        
        # РџРµСЂРµРІС–СЂРєР° РІР°Р»С–РґРЅРѕСЃС‚С– zone_polygon_coords
        if zone_polygon_coords is not None:
            if len(zone_polygon_coords) < 3:
                print(f"[WARN] Zone {zone_id_str}: zone_polygon_coords РјР°С” РјРµРЅС€Рµ 3 С‚РѕС‡РѕРє ({len(zone_polygon_coords)}), РІСЃС‚Р°РЅРѕРІР»СЋС”РјРѕ None")
                zone_polygon_coords = None
            else:
                print(f"[DEBUG] Zone {zone_id_str}: zone_polygon_coords РјР°С” {len(zone_polygon_coords)} С‚РѕС‡РѕРє")
        else:
            print(f"[WARN] Zone {zone_id_str}: zone_polygon_coords С” None, РѕР±СЂС–Р·Р°РЅРЅСЏ Р±СѓРґРµ РїРѕ bbox")
        
        print(f"[INFO] РЎС‚РІРѕСЂСЋС”РјРѕ Р·Р°РґР°С‡Сѓ {task_id} РґР»СЏ Р·РѕРЅРё {zone_id_str} (Р·РѕРЅР° {zone_idx + 1}/{len(request.zones)})")
        print(f"[DEBUG] Zone bbox: north={zone_bbox['north']:.6f}, south={zone_bbox['south']:.6f}, east={zone_bbox['east']:.6f}, west={zone_bbox['west']:.6f}")
        print(f"[DEBUG] Zone polygon coords: {'present' if zone_polygon_coords else 'missing'}, grid_bbox_latlon: {'present' if grid_bbox_latlon else 'missing'}, row/col: {zone_row}/{zone_col}")
        
        background_tasks.add_task(
            generate_model_task,
            task_id=task_id,
            request=zone_request,
            zone_id=zone_id_str,
            zone_polygon_coords=zone_polygon_coords,  # РџРµСЂРµРґР°С”РјРѕ РєРѕРѕСЂРґРёРЅР°С‚Рё РїРѕР»С–РіРѕРЅСѓ РґР»СЏ РѕР±СЂС–Р·Р°РЅРЅСЏ (fallback)
            zone_row=zone_row,
            zone_col=zone_col,
            grid_bbox_latlon=grid_bbox_latlon,
            hex_size_m=float(getattr(request, "hex_size_m", 300.0)),
        )
        
        task_ids.append(task_id)
        print(f"[DEBUG] Р—Р°РґР°С‡Р° {task_id} РґРѕРґР°РЅР° РґРѕ background_tasks. Р’СЃСЊРѕРіРѕ Р·Р°РґР°С‡: {len(task_ids)}")
    
    if len(task_ids) == 0:
        raise HTTPException(status_code=400, detail="РќРµ РІРґР°Р»РѕСЃСЏ СЃС‚РІРѕСЂРёС‚Рё Р·Р°РґР°С‡С– РґР»СЏ Р·РѕРЅ")
    
    print(f"[INFO] РЎС‚РІРѕСЂРµРЅРѕ {len(task_ids)} Р·Р°РґР°С‡ РґР»СЏ РіРµРЅРµСЂР°С†С–С— Р·РѕРЅ: {task_ids}")
    
    # Р—Р±РµСЂС–РіР°С”РјРѕ Р·РІ'СЏР·РѕРє РґР»СЏ РјРЅРѕР¶РёРЅРЅРёС… Р·Р°РґР°С‡
    # Р’РђР–Р›РР’Рћ: РіСЂСѓРїРѕРІРёР№ task_id РјР°С” Р±СѓС‚Рё СѓРЅС–РєР°Р»СЊРЅРёРј, С–РЅР°РєС€Рµ multiple_2 Р±СѓРґРµ РєРѕР»С–Р·РёС‚Рё РјС–Р¶ Р·Р°РїСѓСЃРєР°РјРё
    if len(task_ids) > 1:
        main_task_id = f"batch_{uuid.uuid4()}"
        multiple_tasks_map[main_task_id] = task_ids
        # Зберігаємо batch на диск (row/col кожної плитки; шляхи дописуються по
        # мірі готовності у status-ендпойнті) — щоб панно пережило рестарт.
        panel_tiles[main_task_id] = [
            {
                "task_id": tid,
                "row": getattr(tasks.get(tid), "zone_row", None),
                "col": getattr(tasks.get(tid), "zone_col", None),
                "path": None,
            }
            for tid in task_ids
        ]
        _save_panel_batches()
        print(f"[INFO] Batch Р·Р°РґР°С‡С–: {main_task_id} -> {task_ids}")
        print(f"[INFO] Р”Р»СЏ РІС–РґРѕР±СЂР°Р¶РµРЅРЅСЏ РІСЃС–С… Р·РѕРЅ СЂР°Р·РѕРј РІРёРєРѕСЂРёСЃС‚РѕРІСѓР№С‚Рµ all_task_ids: {task_ids}")
    else:
        main_task_id = task_ids[0]
    
    # РџРѕРІРµСЂС‚Р°С”РјРѕ СЃРїРёСЃРѕРє task_id
    # Р’РђР–Р›РР’Рћ: all_task_ids РјС–СЃС‚РёС‚СЊ РІСЃС– task_id РґР»СЏ РєРѕР¶РЅРѕС— Р·РѕРЅРё
    # Р¤СЂРѕРЅС‚РµРЅРґ РјР°С” Р·Р°РІР°РЅС‚Р°Р¶РёС‚Рё РІСЃС– С„Р°Р№Р»Рё Р· С†РёС… task_id С‚Р° РѕР±'С”РґРЅР°С‚Рё С—С…
    return GenerationResponse(
        task_id=main_task_id,
        status="processing",
        message=f"РЎС‚РІРѕСЂРµРЅРѕ {len(task_ids)} Р·Р°РґР°С‡ РґР»СЏ РіРµРЅРµСЂР°С†С–С— Р·РѕРЅ. Р’РёРєРѕСЂРёСЃС‚РѕРІСѓР№С‚Рµ all_task_ids РґР»СЏ Р·Р°РІР°РЅС‚Р°Р¶РµРЅРЅСЏ РІСЃС–С… Р·РѕРЅ.",
        all_task_ids=task_ids  # Р”РѕРґР°С”РјРѕ СЃРїРёСЃРѕРє РІСЃС–С… task_id
    )


def _friendly_generation_error(exc: Exception) -> str:
    """Map a raw pipeline exception to a short, user-friendly UA message.

    Покупець не повинен бачити Python-трейс/назву винятку. Підбираємо людську
    підказку за ключовими словами; сирий текст уже залогований у консоль вище.
    """
    raw = str(exc or "").lower()

    def _has(*subs: str) -> bool:
        return any(s in raw for s in subs)

    if _has("groove", "boolean", "manifold", "watertight", "non-manifold"):
        return ("Не вдалося зібрати друковану геометрію для цієї ділянки. "
                "Спробуйте трохи зменшити зону або вимкнути дрібні шари (дороги/парки).")
    if _has("timeout", "timed out", "deadline"):
        return ("Генерація зайняла надто довго й була перервана. "
                "Спробуйте меншу ділянку або повторіть пізніше.")
    if _has("memory", "out of memory", "killed", "alloc"):
        return ("Ділянка завелика для обробки. Виберіть меншу зону або менший розмір моделі.")
    if _has("overpass", "osm", "http", "connection", "timeouterror", "ssl", "dns", "resolve", "elevation", "terrarium", "dem", "tile"):
        return ("Тимчасова проблема із завантаженням мапи/рельєфу. "
                "Спробуйте ще раз за хвилину.")
    if _has("empty", "no data", "no roads", "no buildings", "nothing to"):
        return ("Для цієї ділянки замало даних мапи. Спробуйте іншу зону поблизу.")
    return ("Не вдалося згенерувати модель для цієї ділянки. "
            "Спробуйте іншу зону або інші параметри, або повторіть пізніше.")


def generate_model_task(
    task_id: str,
    request: GenerationRequest,
    zone_id: Optional[str] = None,
    zone_polygon_coords: Optional[list] = None,
    zone_row: Optional[int] = None,
    zone_col: Optional[int] = None,
    grid_bbox_latlon: Optional[Tuple[float, float, float, float]] = None,
    hex_size_m: Optional[float] = None,
):

    print(f"[INFO] === РџРћР§РђРўРћРљ Р“Р•РќР•Р РђР¦Р†Р‡ РњРћР”Р•Р›Р† === Task ID: {task_id}, Zone ID: {zone_id}")
    task = tasks[task_id]
    zone_prefix = f"[{zone_id}] " if zone_id else ""

    # Check if task was cancelled before starting
    if task.cancelled:
        print(f"[INFO] {zone_prefix}Task {task_id} cancelled before start - skipping")
        return

    # ULTRA-FAST PREVIEW MODE (~30s target): aggressively trim every heavy
    # input so the deep pipeline produces a buyer-friendly model fast.
    # Deep helpers in terrain_cutter / mesh_quality / full_generation_pipeline
    # / export_pipeline read the PREVIEW_MODE env var directly so we don't
    # have to thread a new param through 5+ layers.
    flat_plate_mode = bool(getattr(request, "flat_plate_mode", False))
    preview_mode = bool(getattr(request, "preview_mode", False)) and not flat_plate_mode
    # КРИТИЧНО: скидаємо env-прапори preview-режиму ПЕРЕД повною генерацією.
    # Раніше preview-генерація лишала BOOLEAN_BACKEND=noop у процесі-воркері →
    # наступна ПОВНА генерація (замовлення/завантаження/terrain з GPX) падала
    # на валідації грувів («Groove stage failed: boolean_noop»). Preview —
    # дефолт для покупців, тож це труїло кожне реальне замовлення в тому ж воркері.
    if not preview_mode:
        os.environ.pop("BOOLEAN_BACKEND", None)
        os.environ.pop("PREVIEW_MODE", None)
    if flat_plate_mode and bool(getattr(request, "keychain_mode", False)):
        try:
            if float(getattr(request, "context_padding_m", 0) or 0) > 35.0:
                request.context_padding_m = 35.0
        except Exception:
            request.context_padding_m = 35.0
    if preview_mode:
        os.environ["PREVIEW_MODE"] = "1"
        request.export_format = "glb"
        # Boolean grooves in newer pipeline go through resolve_boolean_backend
        # which reads BOOLEAN_BACKEND env var. "noop" = NoOpBooleanBackend that
        # returns terrain unchanged → grooves visible on the screenshot vanish.
        os.environ["BOOLEAN_BACKEND"] = "noop"
        try:
            # 1. Smaller terrain mesh (50×50 ≈ 2500 verts vs 350×350 = 122k)
            if int(getattr(request, "terrain_resolution", 0) or 0) > 50:
                request.terrain_resolution = 50
            # 2. Skip terrain smoothing (Gaussian) — fast pass on raw DEM
            if getattr(request, "terrain_smoothing_sigma", 0):
                request.terrain_smoothing_sigma = 0.0
            # 3. Skip subdivision (doubles vertex count each level)
            if hasattr(request, "terrain_subdivide"):
                request.terrain_subdivide = False
            # 4. Lower DEM zoom — fewer Mapbox tiles to fetch (each ~256 px)
            try:
                if int(getattr(request, "terrarium_zoom", 0) or 0) > 13:
                    request.terrarium_zoom = 13
            except Exception:
                pass
            # 5. Trim OSM context padding — less Overpass payload to download
            try:
                if float(getattr(request, "context_padding_m", 0) or 0) > 50.0:
                    request.context_padding_m = 50.0
            except Exception:
                pass
            # 6. Roads sit FLAT on terrain (no inlay embed). Without groove
            # cuts the original road inlay mesh (embedded 2m into terrain)
            # punched ragged dark slivers above the surface — looked like
            # broken textures. road_embed=0 lifts the road plate onto the
            # surface so it reads as a clean coloured ribbon.
            try:
                request.road_embed_mm = 0.0
            except Exception:
                pass
            try:
                # Very thin road plate so it can't punch through slopes.
                # 0.15mm in model ≈ 1m world — reads as a flat dark ribbon
                # without sliver artifacts on rolling terrain.
                request.road_height_mm = 0.15
            except Exception:
                pass
            # 7. Parks sit flat on terrain — embed=0. Default 1.0mm × (1/0.143)
            # = ~7m world embed, but without groove cut that embedded portion
            # cuts through terrain leaving dark slivers around park edges.
            try:
                request.parks_embed_mm = 0.0
            except Exception:
                pass
            try:
                # Thinner park top so any remaining height contrast is small.
                if float(getattr(request, "parks_height_mm", 0) or 0) > 0.3:
                    request.parks_height_mm = 0.3
            except Exception:
                pass
        except Exception:
            pass
        print(
            f"[INFO] {zone_prefix}PREVIEW MODE ON: "
            f"terrain={getattr(request, 'terrain_resolution', '?')}, "
            f"smoothing={getattr(request, 'terrain_smoothing_sigma', '?')}, "
            f"subdivide={getattr(request, 'terrain_subdivide', '?')}, "
            f"zoom={getattr(request, 'terrarium_zoom', '?')}, "
            f"padding={getattr(request, 'context_padding_m', '?')}m; "
            f"grooves/manifold/print-gate/boolean/firebase/preview-parts ALL skipped"
        )
    else:
        os.environ.pop("PREVIEW_MODE", None)
        os.environ.pop("BOOLEAN_BACKEND", None)

    _apply_default_canonical_bundle_if_needed(
        request,
        zone_id=zone_id,
        zone_row=zone_row,
        zone_col=zone_col,
        zone_polygon_coords=zone_polygon_coords,
    )
    _normalize_request_base_thickness(request, zone_prefix=zone_prefix)
    
    print(f"[DEBUG] {zone_prefix} AMS Mode: {'ENABLED' if request.is_ams_mode else 'DISABLED'}")
    print(f"[DEBUG] {zone_prefix} Flat Plate Mode: {'ENABLED' if flat_plate_mode else 'DISABLED'}")
    
    try:
        runtime_context = prepare_generation_runtime_context(
            request=request,
            zone_prefix=zone_prefix,
        )
        latlon_bbox = runtime_context.latlon_bbox
        global_center = runtime_context.global_center

        file_basename = _make_export_basename(
            task_id,
            hex_size_m=hex_size_m,
            model_size_mm=getattr(request, "model_size_mm", None),
            zone_row=zone_row,
            zone_col=zone_col,
        )
        print(f"[INFO] {zone_prefix}File basename: {file_basename}")

        # ── Memory-aware concurrency gate ──────────────────────────────
        # Single process on a small VPS: a heavy terrain job must run alone,
        # light jobs (flat maps / keychains) may run a few in parallel. This
        # serializes the memory-heavy pipeline across ALL concurrent requests
        # (multiple users or grid zones), preventing the OOM restarts that used
        # to kill in-flight generations.
        from services import gen_queue
        _gen_weight = gen_queue.weight_for_request(request)
        if gen_queue.would_block(_gen_weight):
            task.update_status("queued", 0, "У черзі на генерацію — сервер зараз зайнятий…")
            print(f"[QUEUE] {zone_prefix}Task {task_id} queued (weight={_gen_weight}, {gen_queue.stats()})")
        _gen_wait = gen_queue.acquire(_gen_weight)
        if _gen_wait > 0.5:
            print(f"[QUEUE] {zone_prefix}Task {task_id} started after {_gen_wait:.0f}s wait ({gen_queue.stats()})")
        try:
            if task.cancelled:
                print(f"[INFO] {zone_prefix}Task {task_id} cancelled while queued — skipping")
                return
            # Optional CPU profile of the whole pipeline (PIPELINE_PROFILE=1):
            # dumps top cumulative-time functions so we can target real hot spots.
            _profiler = None
            if os.environ.get("PIPELINE_PROFILE", "").lower() in ("1", "true", "yes"):
                # ПАСТКА: cProfile глобальний на потік → у СЕРІЇ (паралельні плитки в
                # одному event-loop) друга плитка падала «Another profiling tool is
                # already active» і ВСЯ плитка не генерувалась. Профіль не критичний —
                # огортаємо у try, щоб НІКОЛИ не валити генерацію.
                try:
                    import cProfile
                    _profiler = cProfile.Profile()
                    _profiler.enable()
                except Exception as _pe:
                    print(f"[PROFILE] {zone_prefix}skipped (non-fatal): {_pe}")
                    _profiler = None
            workflow_result = run_full_generation_pipeline(
                task=task,
                request=request,
                task_id=task_id,
                output_dir=OUTPUT_DIR,
                global_center=global_center,
                latlon_bbox=latlon_bbox,
                zone_polygon_coords=zone_polygon_coords,
                grid_bbox_latlon=grid_bbox_latlon,
                zone_row=zone_row,
                zone_col=zone_col,
                hex_size_m=hex_size_m,
                zone_prefix=zone_prefix,
                min_printable_gap_mm=MIN_PRINTABLE_GAP_MM,
                groove_clearance_mm=GROOVE_CLEARANCE_MM,
                file_basename=file_basename,
            )
            if _profiler is not None:
                try:
                    import pstats, io as _io
                    _profiler.disable()
                    _s = _io.StringIO()
                    pstats.Stats(_profiler, stream=_s).sort_stats("cumulative").print_stats(30)
                    print(f"[PROFILE] {zone_prefix}Top functions by cumulative time:\n" + _s.getvalue())
                except Exception:
                    pass
        finally:
            gen_queue.release(_gen_weight)
        if workflow_result.terrain_only_result is not None:
            return
        print(f"[OK] Model generation completed. Task ID: {task_id}, Zone ID: {zone_id}, File: {workflow_result.output_file_abs}")
        
        
    except Exception as e:
        print(f"[ERROR] === РџРћРњРР›РљРђ Р“Р•РќР•Р РђР¦Р†Р‡ РњРћР”Р•Р›Р† === Task ID: {task_id}, Zone ID: {zone_id}, Error: {e}")
        import traceback
        traceback.print_exc()
        # Користувачу — дружнє повідомлення українською, БЕЗ Python-трейсу/класу
        # винятку (раніше сире `str(e)` типу "BooleanError: ..." текло у фронт).
        task.fail(_friendly_generation_error(e))
        # IMPORTANT: don't re-raise from background task, otherwise Starlette logs it as ASGI error
        # and it can interrupt other tasks. The failure is already recorded in task state.
        return


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
