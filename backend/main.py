import warnings
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
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
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Optional, List, Tuple
import os
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
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
from services.site_preview import build_fast_preview

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



# CORS РЅР°Р»Р°С€С‚СѓРІР°РЅРЅСЏ РґР»СЏ С–РЅС‚РµРіСЂР°С†С–С— Р· frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Р—Р±РµСЂС–РіР°РЅРЅСЏ Р·Р°РґР°С‡ РіРµРЅРµСЂР°С†С–С—
tasks: dict[str, GenerationTask] = {}
# Р—Р±РµСЂС–РіР°РЅРЅСЏ Р·РІ'СЏР·РєС–РІ РјС–Р¶ РјРЅРѕР¶РёРЅРЅРёРјРё Р·Р°РґР°С‡Р°РјРё (task_id -> list of task_ids)
multiple_tasks_map: dict[str, list[str]] = {}

import tempfile

# Р’РёРєРѕСЂРёСЃС‚РѕРІСѓС”РјРѕ Р»РѕРєР°Р»СЊРЅСѓ РґРёСЂРµРєС‚РѕСЂС–СЋ output РґР»СЏ СЃС‚Р°Р±С–Р»СЊРЅРѕСЃС‚С–
# Р¦Рµ РІРёСЂС–С€СѓС” РїСЂРѕР±Р»РµРјСѓ Р·РЅРёРєРЅРµРЅРЅСЏ С„Р°Р№Р»С–РІ Сѓ С‚РёРјС‡Р°СЃРѕРІРёС… РїР°РїРєР°С…
OUTPUT_DIR = Path("output").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ORDERS_DIR = Path(os.getenv("ORDERS_DIR", "orders")).resolve()
ORDERS_DIR.mkdir(parents=True, exist_ok=True)

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
    try:
        min_required_base_mm = max(
            0.2,
            float(request.parks_embed_mm) if getattr(request, "include_parks", False) else 0.0,
            float(request.road_embed_mm),
            float(request.water_depth),
        ) + 0.5
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
# Mount output folder as static files
app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="files")


@app.on_event("startup")
async def startup_event():
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
        print("   Make sure FIREBASE_STORAGE_BUCKET is set in .env")
        print("   and serviceAccountKey.json exists in backend folder.")
    print("="*60 + "\n")

    print("[INFO] Р’С–РґРЅРѕРІР»РµРЅРЅСЏ СЃРїРёСЃРєСѓ Р·Р°РґР°С‡ Р· РґРёСЃРєР°...")
    if not OUTPUT_DIR.exists():
        return
    
    # РЁСѓРєР°С”РјРѕ РІСЃС– STL/3MF С„Р°Р№Р»Рё
    for file_path in OUTPUT_DIR.glob("*"):
        if file_path.suffix.lower() not in [".stl", ".3mf"]:
            continue
        
        # task_id - С†Рµ С–Рј'СЏ С„Р°Р№Р»Сѓ РґРѕ РїРµСЂС€РѕРіРѕ "_" Р°Р±Рѕ "."
        name = file_path.name
        task_id = name.split(".")[0].split("_")[0]
        
        # РЇРєС‰Рѕ С‚Р°РєРёР№ task_id С‰Рµ РЅРµ РІ СЃРїРёСЃРєСѓ, СЃС‚РІРѕСЂСЋС”РјРѕ "Р·Р°РіР»СѓС€РєСѓ"
        if task_id not in tasks:
            tasks[task_id] = GenerationTask(
                task_id=task_id,
                request=None, # РњРё РЅРµ Р·РЅР°С”РјРѕ РїР°СЂР°РјРµС‚СЂС–РІ СЃС‚Р°СЂРѕРіРѕ Р·Р°РїРёС‚Сѓ
                status="completed",
                progress=100,
                output_file=str(file_path)
            )
        
        # Р”РѕРґР°С”РјРѕ С„Р°Р№Р» РґРѕ output_files
        # Р¤РѕСЂРјР°С‚ С–РјРµРЅС–: {task_id}_{part}.stl Р°Р±Рѕ {task_id}.stl/3mf
        task = tasks[task_id]
        if "_" in name:
            part_part = name.split("_")[1].split(".")[0]
            ext = file_path.suffix.lstrip(".").lower()
            key = f"{part_part}_{ext}"
            task.set_output(key, str(file_path))
        else:
            ext = file_path.suffix.lstrip(".").lower()
            task.set_output(ext, str(file_path))
            if not task.output_file:
                task.output_file = str(file_path)
    
    print(f"[INFO] Р’С–РґРЅРѕРІР»РµРЅРѕ {len(tasks)} Р·Р°РґР°С‡.")


class GenerationRequest(BaseModel):
    """Р—Р°РїРёС‚ РЅР° РіРµРЅРµСЂР°С†С–СЋ 3D РјРѕРґРµР»С–"""
    model_config = ConfigDict(protected_namespaces=())
    
    north: float
    south: float
    east: float
    west: float
    # РџР°СЂР°РјРµС‚СЂРё РіРµРЅРµСЂР°С†С–С—
    road_width_multiplier: float = 1.0
    # Print-aware РїР°СЂР°РјРµС‚СЂРё (РІ РњР†Р›Р†РњР•РўР РђРҐ РЅР° С„С–РЅР°Р»СЊРЅС–Р№ РјРѕРґРµР»С–)
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
    terrain_resolution: int = Field(default=350, ge=80, le=600)  # Р’РёСЃРѕРєР° РґРµС‚Р°Р»С–Р·Р°С†С–СЏ РґР»СЏ РјР°РєСЃРёРјР°Р»СЊРЅРѕ РїР»Р°РІРЅРѕРіРѕ СЂРµР»СЊС”С„Сѓ
    # Subdivision: РґРѕРґР°С‚РєРѕРІР° РґРµС‚Р°Р»С–Р·Р°С†С–СЏ mesh РїС–СЃР»СЏ СЃС‚РІРѕСЂРµРЅРЅСЏ (РґР»СЏ С‰Рµ РїР»Р°РІРЅС–С€РѕРіРѕ СЂРµР»СЊС”С„Сѓ)
    terrain_subdivide: bool = Field(default=True, description="Р—Р°СЃС‚РѕСЃСѓРІР°С‚Рё subdivision РґР»СЏ РїР»Р°РІРЅС–С€РѕРіРѕ mesh")
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
    # Preserve global XY coordinates (do NOT center per tile) for perfect stitching across zones/sessions.
    preserve_global_xy: bool = False
    # Explicit Grid Step (meters) for perfect stitching (avoids legacy resolution-based gaps)
    grid_step_m: Optional[float] = None
    # Explicit Hex size for grid generation
    hex_size_m: float = Field(default=300.0, ge=100.0, le=2000.0)
    # AMS / Flat Mode: Optimized for multicolor printing (Flat terrain + Fixed layers)
    is_ams_mode: bool = False
    canonical_mask_bundle_dir: Optional[str] = None
    auto_canonicalize_masks: bool = True


class GenerationResponse(BaseModel):
    """Р’С–РґРїРѕРІС–РґСЊ Р· ID Р·Р°РґР°С‡С–"""
    task_id: str
    status: str
    message: Optional[str] = None
    all_task_ids: Optional[List[str]] = None  # Р”Р»СЏ РјРЅРѕР¶РёРЅРЅРёС… Р·РѕРЅ


class PreviewRequest(BaseModel):
    north: float
    south: float
    east: float
    west: float
    polygon_geojson: Optional[dict] = None
    include_terrain: bool = True
    include_roads: bool = True
    include_buildings: bool = True
    include_water: bool = True
    include_parks: bool = True
    road_width_multiplier: float = 0.8
    building_min_height: float = 5.0
    building_height_multiplier: float = 1.8
    model_size_mm: float = 180.0
    terrain_z_scale: float = 0.5
    terrain_resolution: int = 180


class SiteOrderRequest(BaseModel):
    name: str = ""
    contact: str = ""
    city: str = "Київ"
    bounds: dict
    polygon_geojson: Optional[dict] = None
    preview_id: Optional[str] = None
    model_size_mm: float = 180.0
    material: str = "white"
    layers: dict = Field(default_factory=dict)
    price_uah: Optional[int] = None
    comment: str = ""
    area_mode: str = "rect"
    selected_zones: List[dict] = Field(default_factory=list)
    grid_type: str = "rect"
    hex_size_m: float = 650.0
    preview_metrics: dict = Field(default_factory=dict)
    model_logic: dict = Field(default_factory=dict)
    generation_request: Optional[dict] = None


def _order_path(order_id: str) -> Path:
    safe_id = "".join(ch for ch in order_id if ch.isalnum() or ch in {"-", "_"})
    return ORDERS_DIR / f"{safe_id}.json"


def _read_order(order_id: str) -> dict[str, Any]:
    path = _order_path(order_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Заявку не знайдено")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Не вдалося прочитати заявку: {exc}")


def _write_order(order: dict[str, Any]) -> None:
    _order_path(str(order["id"])).write_text(json.dumps(order, ensure_ascii=False, indent=2), encoding="utf-8")


def _bounds_from_order(order: dict[str, Any]) -> dict[str, float]:
    bounds = order.get("bounds") or {}
    return {
        "north": float(bounds.get("north")),
        "south": float(bounds.get("south")),
        "east": float(bounds.get("east")),
        "west": float(bounds.get("west")),
    }


def _synthesize_generation_request(order: dict[str, Any]) -> dict[str, Any]:
    if isinstance(order.get("generation_request"), dict):
        request = dict(order["generation_request"])
    else:
        bounds = _bounds_from_order(order)
        layers = order.get("layers") or {}
        request = {
            **bounds,
            "road_width_multiplier": 0.8,
            "road_height_mm": 0.5,
            "road_embed_mm": 0.3,
            "building_min_height": 5.0,
            "building_height_multiplier": 1.8,
            "building_foundation_mm": 0.6,
            "building_embed_mm": 0.2,
            "building_max_foundation_mm": 5.0,
            "water_depth": 1.2,
            "terrain_enabled": bool(layers.get("terrain", True)),
            "terrain_z_scale": 0.5,
            "terrain_base_thickness_mm": 0.3,
            "terrain_resolution": 180,
            "terrarium_zoom": 15,
            "terrain_subdivide": False,
            "terrain_subdivide_levels": 1,
            "terrain_smoothing_sigma": 2.0,
            "flatten_buildings_on_terrain": True,
            "flatten_roads_on_terrain": False,
            "export_format": "3mf",
            "model_size_mm": float(order.get("model_size_mm") or 180.0),
            "context_padding_m": 400.0,
            "terrain_only": False,
            "include_parks": bool(layers.get("parks", True)),
            "parks_height_mm": 0.6,
            "parks_embed_mm": 1.0,
            "preview_include_base": bool(layers.get("terrain", True)),
            "preview_include_roads": bool(layers.get("roads", True)),
            "preview_include_buildings": bool(layers.get("buildings", True)),
            "preview_include_water": bool(layers.get("water", True)),
            "preview_include_parks": bool(layers.get("parks", True)),
            "hex_size_m": float(order.get("hex_size_m") or 650.0),
            "is_ams_mode": False,
        }
    bounds = _bounds_from_order(order)
    for key, value in bounds.items():
        request.setdefault(key, value)
    request.setdefault("model_size_mm", float(order.get("model_size_mm") or 180.0))
    request.setdefault("hex_size_m", float(order.get("hex_size_m") or 650.0))
    return request


@app.get("/")
async def root():
    return {"message": "3D Map Generator API", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/preview")
async def create_fast_preview(request: PreviewRequest):
    if request.north <= request.south or request.east <= request.west:
        raise HTTPException(status_code=400, detail="Некоректні межі ділянки")

    lat_span = abs(request.north - request.south)
    lng_span = abs(request.east - request.west)
    if lat_span * lng_span > 0.00025:
        raise HTTPException(status_code=400, detail="Ділянка завелика для швидкого preview. Зменшіть рамку або розбийте її на зони.")

    return build_fast_preview(
        bounds={
            "north": request.north,
            "south": request.south,
            "east": request.east,
            "west": request.west,
        },
        polygon_geojson=request.polygon_geojson,
        include_terrain=request.include_terrain,
        include_roads=request.include_roads,
        include_buildings=request.include_buildings,
        include_water=request.include_water,
        include_parks=request.include_parks,
        road_width_multiplier=request.road_width_multiplier,
        building_min_height=request.building_min_height,
        building_height_multiplier=request.building_height_multiplier,
        model_size_mm=request.model_size_mm,
        terrain_z_scale=request.terrain_z_scale,
        terrain_resolution=request.terrain_resolution,
    )


@app.post("/api/orders")
async def create_site_order(request: SiteOrderRequest):
    order_id = f"R-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:5].upper()}"
    payload = request.model_dump()
    payload["generation_request"] = _synthesize_generation_request(payload)
    payload.update(
        {
            "id": order_id,
            "status": "new",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _write_order(payload)
    return {"ok": True, "order_id": order_id}


@app.get("/api/admin/orders")
async def list_site_orders(token: Optional[str] = Query(default=None)):
    expected = os.getenv("ADMIN_API_TOKEN")
    if expected and token != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")
    orders = []
    for path in sorted(ORDERS_DIR.glob("*.json"), reverse=True):
        try:
            orders.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return {"orders": orders}


@app.post("/api/admin/orders/{order_id}/generate", response_model=GenerationResponse)
async def start_order_generation(order_id: str, background_tasks: BackgroundTasks, token: Optional[str] = Query(default=None)):
    expected = os.getenv("ADMIN_API_TOKEN")
    if expected and token != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    order = _read_order(order_id)
    request_payload = _synthesize_generation_request(order)
    selected_zones = order.get("selected_zones") or []

    try:
        if selected_zones:
            zone_request_payload = dict(request_payload)
            zone_request_payload["zones"] = selected_zones
            zone_request_payload["hex_size_m"] = float(order.get("hex_size_m") or request_payload.get("hex_size_m") or 650.0)
            response = await generate_zones_endpoint(ZoneGenerationRequest(**zone_request_payload), background_tasks)
        else:
            response = await generate_model(GenerationRequest(**request_payload), background_tasks)

        order["generation_request"] = request_payload
        order["generation_task_id"] = response.task_id
        order["generation_all_task_ids"] = response.all_task_ids or []
        order["generation_status"] = response.status
        order["status"] = "in_progress"
        order["generated_at"] = datetime.now(timezone.utc).isoformat()
        _write_order(order)
        return response
    except HTTPException:
        raise
    except Exception as exc:
        order["generation_status"] = "failed_to_start"
        order["generation_error"] = str(exc)
        _write_order(order)
        raise HTTPException(status_code=500, detail=f"Не вдалося запустити Blender: {exc}")


@app.post("/api/generate", response_model=GenerationResponse)
async def generate_model(request: GenerationRequest, background_tasks: BackgroundTasks):
    """
    РЎС‚РІРѕСЂСЋС” Р·Р°РґР°С‡Сѓ РіРµРЅРµСЂР°С†С–С— 3D РјРѕРґРµР»С–
    """
    try:
        print(f"[INFO] РћС‚СЂРёРјР°РЅРѕ Р·Р°РїРёС‚ РЅР° РіРµРЅРµСЂР°С†С–СЋ: north={request.north}, south={request.south}, east={request.east}, west={request.west}")
        
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
        
        # Р—Р°РїСѓСЃРєР°С”РјРѕ РіРµРЅРµСЂР°С†С–СЋ РІ С„РѕРЅС–
        background_tasks.add_task(generate_model_task, task_id, request)
        
        print(f"[INFO] РЎС‚РІРѕСЂРµРЅРѕ Р·Р°РґР°С‡Сѓ {task_id} РґР»СЏ РіРµРЅРµСЂР°С†С–С— РјРѕРґРµР»С–")
        return GenerationResponse(task_id=task_id, status="processing", message="Р—Р°РґР°С‡Р° СЃС‚РІРѕСЂРµРЅР°")
    except Exception as e:
        print(f"[ERROR] РџРѕРјРёР»РєР° СЃС‚РІРѕСЂРµРЅРЅСЏ Р·Р°РґР°С‡С–: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"РџРѕРјРёР»РєР° СЃС‚РІРѕСЂРµРЅРЅСЏ Р·Р°РґР°С‡С–: {str(e)}")


@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    """
    РћС‚СЂРёРјСѓС” СЃС‚Р°С‚СѓСЃ Р·Р°РґР°С‡С– РіРµРЅРµСЂР°С†С–С— Р°Р±Рѕ РјРЅРѕР¶РёРЅРЅРёС… Р·Р°РґР°С‡
    """
    # РџРµСЂРµРІС–СЂСЏС”РјРѕ, С‡Рё С†Рµ batch Р·Р°РїРёС‚ РЅР° РјРЅРѕР¶РёРЅРЅС– Р·Р°РґР°С‡С– (С„РѕСЂРјР°С‚: batch_<uuid>)
    if task_id.startswith("batch_"):
        all_task_ids_list = multiple_tasks_map.get(task_id)
        if not all_task_ids_list:
            raise HTTPException(status_code=404, detail="Multiple tasks not found")
        
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
                    "download_url": download_url,
                    "firebase_url": getattr(t, "firebase_url", None),
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
        
        return {
            "task_id": task_id,
            "status": "multiple",
            "tasks": tasks_status,
            "total": len(tasks_status),
            "completed": sum(1 for t in tasks_status if t["status"] == "completed"),
            "all_task_ids": all_task_ids_list
        }
    
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    output_files = getattr(task, "output_files", {}) or {}
    # Helper to build static URL from absolute path
    def to_static_url(path_str):
        if not path_str: return None
        return f"/files/{Path(path_str).name}"

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
        "download_url_stl": to_static_url(output_files.get("stl")),
        "download_url_3mf": to_static_url(output_files.get("3mf")),
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


@app.get("/api/download/{task_id}")
async def download_model(
    task_id: str,
    format: Optional[str] = Query(default=None, description="Optional: stl Р°Р±Рѕ 3mf"),
    part: Optional[str] = Query(default=None, description="Optional preview part: base|roads|buildings|water"),
):
    """
    Р—Р°РІР°РЅС‚Р°Р¶СѓС” Р·РіРµРЅРµСЂРѕРІР°РЅРёР№ С„Р°Р№Р» Р· Firebase С‡РµСЂРµР· РїСЂРѕРєСЃС–
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    if task.status != "completed":
        raise HTTPException(status_code=400, detail="Model not ready")
    
    print(f"[DEBUG] Download request: task={task_id}, format={format}, part={part}")
    
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

    if not firebase_url:
        # РЇРєС‰Рѕ С†Рµ POI С– Р№РѕРіРѕ РЅРµРјР°С” - 404
        if part == "poi":
            print(f"[INFO] POI part not available (expected), returning 404")
        
        print(f"[WARN] File not found in Firebase: key={target_key}")
        raise HTTPException(status_code=404, detail=f"File not found in Firebase: {target_key}")

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
    format: str = Query(default="3mf", description="Р¤РѕСЂРјР°С‚ РІРёС…С–РґРЅРѕРіРѕ С„Р°Р№Р»Сѓ (stl Р°Р±Рѕ 3mf)")
):
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
async def set_global_center_endpoint(center_lat: float = Query(...), center_lon: float = Query(...), utm_zone: Optional[int] = Query(None)):
    """
    Р’СЃС‚Р°РЅРѕРІР»СЋС” РіР»РѕР±Р°Р»СЊРЅРёР№ С†РµРЅС‚СЂ РєР°СЂС‚Рё РґР»СЏ СЃРёРЅС…СЂРѕРЅС–Р·Р°С†С–С— РєРІР°РґСЂР°С‚С–РІ
    
    Args:
        center_lat: РЁРёСЂРѕС‚Р° РіР»РѕР±Р°Р»СЊРЅРѕРіРѕ С†РµРЅС‚СЂСѓ (WGS84)
        center_lon: Р”РѕРІРіРѕС‚Р° РіР»РѕР±Р°Р»СЊРЅРѕРіРѕ С†РµРЅС‚СЂСѓ (WGS84)
        utm_zone: UTM Р·РѕРЅР° (РѕРїС†С–РѕРЅР°Р»СЊРЅРѕ, РІРёР·РЅР°С‡Р°С”С‚СЊСЃСЏ Р°РІС‚РѕРјР°С‚РёС‡РЅРѕ СЏРєС‰Рѕ РЅРµ РІРєР°Р·Р°РЅРѕ)
    
    Returns:
        Р†РЅС„РѕСЂРјР°С†С–СЏ РїСЂРѕ РІСЃС‚Р°РЅРѕРІР»РµРЅРёР№ С†РµРЅС‚СЂ
    """
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
    import math
    
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
        
        # РџРµСЂРµРІС–СЂРєР° РІР°Р»С–РґРЅРѕСЃС‚С– РєРѕРѕСЂРґРёРЅР°С‚
        if request.north <= request.south or request.east <= request.west:
            raise ValueError(f"РќРµРІС–СЂРЅС– РєРѕРѕСЂРґРёРЅР°С‚Рё: north={request.north} <= south={request.south} Р°Р±Рѕ east={request.east} <= west={request.west}")
        
        # РљРѕРЅРІРµСЂС‚СѓС”РјРѕ lat/lon bbox РІ UTM РґР»СЏ РіРµРЅРµСЂР°С†С–С— СЃС–С‚РєРё
        from services.crs_utils import bbox_latlon_to_utm
        bbox_utm = bbox_latlon_to_utm(
            request.north, request.south, request.east, request.west
        )
        bbox_meters = bbox_utm[:4]  # (minx, miny, maxx, maxy)
        to_wgs84 = bbox_utm[6]  # Р¤СѓРЅРєС†С–СЏ РґР»СЏ РєРѕРЅРІРµСЂС‚Р°С†С–С— UTM -> WGS84 (С–РЅРґРµРєСЃ 6)
        minx, miny, maxx, maxy = bbox_meters

        max_grid_cells = 1500
        if grid_type == 'square':
            estimated_cells = (math.ceil((maxx - minx) / request.hex_size_m) + 1) * (
                math.ceil((maxy - miny) / request.hex_size_m) + 1
            )
        elif grid_type == 'circle':
            diameter_m = request.hex_size_m
            estimated_cells = (math.ceil((maxx - minx) / diameter_m) + 1) * (
                math.ceil((maxy - miny) / diameter_m) + 1
            )
        else:
            hex_width = math.sqrt(3) * request.hex_size_m
            hex_height = 1.5 * request.hex_size_m
            estimated_cells = (math.ceil((maxx - minx) / hex_width) + 2) * (
                math.ceil((maxy - miny) / hex_height) + 2
            )

        if estimated_cells > max_grid_cells:
            suggested_size = math.ceil(
                request.hex_size_m * math.sqrt(estimated_cells / max_grid_cells) / 100
            ) * 100
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Сітка занадто щільна для цієї області: приблизно {estimated_cells} клітинок. "
                    f"Збільште розмір клітинки до ~{suggested_size} м або виберіть меншу область."
                ),
            )
        
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
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] РџРѕРјРёР»РєР° РіРµРЅРµСЂР°С†С–С— СЃС–С‚РєРё: {e}\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"РџРѕРјРёР»РєР° РіРµРЅРµСЂР°С†С–С— СЃС–С‚РєРё: {str(e)}")


class ZoneGenerationRequest(BaseModel):
    """Р—Р°РїРёС‚ РґР»СЏ РіРµРЅРµСЂР°С†С–С— РјРѕРґРµР»РµР№ РґР»СЏ РІРёР±СЂР°РЅРёС… Р·РѕРЅ"""
    model_config = ConfigDict(protected_namespaces=())
    
    zones: List[dict]  # РЎРїРёСЃРѕРє Р·РѕРЅ (GeoJSON features)
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
    export_format: str = Field(default="3mf", pattern="^(stl|3mf)$")
    context_padding_m: float = Field(default=400.0, ge=0.0, le=5000.0)
    # Fast mode for stitching diagnostics: generate only terrain (optionally with water depression)
    terrain_only: bool = False
    include_parks: bool = True
    parks_height_mm: float = Field(default=0.6, ge=0.1, le=5.0)
    parks_embed_mm: float = Field(default=1.0, ge=0.0, le=2.0)
    include_pois: bool = False # POI removed but keep field for compatibility
    is_ams_mode: bool = False
    canonical_mask_bundle_dir: Optional[str] = None
    auto_canonicalize_masks: bool = True


@app.post("/api/generate-zones", response_model=GenerationResponse)
async def generate_zones_endpoint(request: ZoneGenerationRequest, background_tasks: BackgroundTasks):

    if not request.zones or len(request.zones) == 0:
        raise HTTPException(status_code=400, detail="РќРµ РІРёР±СЂР°РЅРѕ Р¶РѕРґРЅРѕС— Р·РѕРЅРё")
    
    # РљР РРўРР§РќРћ: Р’РёР·РЅР°С‡Р°С”РјРѕ РіР»РѕР±Р°Р»СЊРЅРёР№ С†РµРЅС‚СЂ РґР»СЏ Р’РЎР†Р„Р‡ СЃС–С‚РєРё.
    # If client provides city bbox, use it for a stable reference; otherwise fallback to selected zones bbox.
    # Р¦Рµ Р·Р°Р±РµР·РїРµС‡СѓС”, С‰Рѕ РІСЃС– Р·РѕРЅРё РІРёРєРѕСЂРёСЃС‚РѕРІСѓСЋС‚СЊ РѕРґРЅСѓ С‚РѕС‡РєСѓ РІС–РґР»С–РєСѓ (0,0)
    # С– С–РґРµР°Р»СЊРЅРѕ РїС–РґС…РѕРґСЏС‚СЊ РѕРґРЅР° РґРѕ РѕРґРЅРѕС—
    print(f"[INFO] Р’РёР·РЅР°С‡РµРЅРЅСЏ РіР»РѕР±Р°Р»СЊРЅРѕРіРѕ С†РµРЅС‚СЂСѓ РґР»СЏ РІСЃС–С”С— СЃС–С‚РєРё ({len(request.zones)} Р·РѕРЅ)...")
    
    selected_grid_bbox = None
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
    selected_grid_bbox = {
        'north': max(all_lats),
        'south': min(all_lats),
        'east': max(all_lons),
        'west': min(all_lons)
    }

    grid_bbox = None
    # 1) Use explicit city bbox only for larger batches. For small interactive
    # selections, a full-city DEM scan blocks the request before a task id is returned.
    try:
        if request.north is not None and request.south is not None and request.east is not None and request.west is not None:
            if len(request.zones) > 3 and float(request.north) > float(request.south) and float(request.east) > float(request.west):
                grid_bbox = {
                    "north": float(request.north),
                    "south": float(request.south),
                    "east": float(request.east),
                    "west": float(request.west),
                }
    except Exception:
        grid_bbox = None

    # 2) Fallback: compute bbox from selected zones (fast interactive behavior)
    if grid_bbox is None:
        grid_bbox = selected_grid_bbox
    
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
    # cache version bump: elevation baseline logic changed (needs refresh)
    city_key = f"v4_{grid_bbox_latlon[0]:.6f}_{grid_bbox_latlon[1]:.6f}_{grid_bbox_latlon[2]:.6f}_{grid_bbox_latlon[3]:.6f}_z{int(request.terrarium_zoom)}_zs{float(request.terrain_z_scale):.3f}_ms{float(request.model_size_mm):.1f}"
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

    if cached_elev is not None:
        global_elevation_ref_m = float(cached.get("elevation_ref_m"))
        global_baseline_offset_m = float(cached.get("baseline_offset_m") or 0.0)
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

        global_elevation_ref_m, global_baseline_offset_m = calculate_global_elevation_reference(
            zones=request.zones,
            source_crs=source_crs,
            terrarium_zoom=request.terrarium_zoom if hasattr(request, 'terrarium_zoom') else 15,
            z_scale=float(request.terrain_z_scale),
            sample_points_per_zone=25,  # РљС–Р»СЊРєС–СЃС‚СЊ С‚РѕС‡РѕРє РґР»СЏ СЃРµРјРїР»С–РЅРіСѓ РІ РєРѕР¶РЅС–Р№ Р·РѕРЅС–
            global_center=global_center,  # Р’РђР–Р›РР’Рћ: РїРµСЂРµРґР°С”РјРѕ РіР»РѕР±Р°Р»СЊРЅРёР№ С†РµРЅС‚СЂ РґР»СЏ РєРѕРЅРІРµСЂС‚Р°С†С–С— РєРѕРѕСЂРґРёРЅР°С‚
            explicit_bbox=explicit_grid_bbox_tuple  # РљР РРўРР§РќРћ: Р’РёРєРѕСЂРёСЃС‚РѕРІСѓС”РјРѕ СЃС‚Р°Р±С–Р»СЊРЅРёР№ BBOX РјС–СЃС‚Р°
        )
    
    if global_elevation_ref_m is not None:
        print(f"[INFO] Р“Р»РѕР±Р°Р»СЊРЅРёР№ elevation_ref_m: {global_elevation_ref_m:.2f}Рј (РІРёСЃРѕС‚Р° РЅР°Рґ СЂС–РІРЅРµРј РјРѕСЂСЏ)")
        print(f"[INFO] Р“Р»РѕР±Р°Р»СЊРЅРёР№ baseline_offset_m: {global_baseline_offset_m:.3f}Рј")
    else:
        print(f"[WARN] РќРµ РІРґР°Р»РѕСЃСЏ РѕР±С‡РёСЃР»РёС‚Рё РіР»РѕР±Р°Р»СЊРЅРёР№ elevation_ref_m, РєРѕР¶РЅР° Р·РѕРЅР° РІРёРєРѕСЂРёСЃС‚РѕРІСѓРІР°С‚РёРјРµ Р»РѕРєР°Р»СЊРЅСѓ РЅРѕСЂРјР°Р»С–Р·Р°С†С–СЋ")
    
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
            context_padding_m=request.context_padding_m,
            terrain_only=bool(getattr(request, "terrain_only", False)),
            include_parks=request.include_parks,
            parks_height_mm=request.parks_height_mm,
            parks_embed_mm=request.parks_embed_mm,
            # include_pois is not in GenerationRequest anymore or hidden
            # РљР РРўРР§РќРћ: РџРµСЂРµРґР°С”РјРѕ РіР»РѕР±Р°Р»СЊРЅС– РїР°СЂР°РјРµС‚СЂРё РґР»СЏ СЃРёРЅС…СЂРѕРЅС–Р·Р°С†С–С— РІРёСЃРѕС‚
            elevation_ref_m=global_elevation_ref_m,  # Р“Р»РѕР±Р°Р»СЊРЅР° Р±Р°Р·РѕРІР° РІРёСЃРѕС‚Р° РґР»СЏ РІСЃС–С… Р·РѕРЅ
            baseline_offset_m=global_baseline_offset_m,  # Р“Р»РѕР±Р°Р»СЊРЅРµ Р·РјС–С‰РµРЅРЅСЏ baseline
            preserve_global_xy=True,  # IMPORTANT: export in a shared coordinate frame for stitching
            grid_step_m=global_grid_step_m,  # GLOBAL GRID FIX
            is_ams_mode=request.is_ams_mode,
        )
        
        # Р“РµРЅРµСЂСѓС”РјРѕ РјРѕРґРµР»СЊ РґР»СЏ Р·РѕРЅРё
        task_id = str(uuid.uuid4())
        zone_id_str = zone.get('id', f'zone_{zone_idx}')
        props = zone.get("properties") or {}
        zone_row = props.get("row")
        zone_col = props.get("col")
        task = GenerationTask(task_id=task_id, request=zone_request)
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

    _apply_default_canonical_bundle_if_needed(
        request,
        zone_id=zone_id,
        zone_row=zone_row,
        zone_col=zone_col,
        zone_polygon_coords=zone_polygon_coords,
    )
    _normalize_request_base_thickness(request, zone_prefix=zone_prefix)
    
    print(f"[DEBUG] {zone_prefix} AMS Mode: {'ENABLED' if request.is_ams_mode else 'DISABLED'}")
    
    try:
        runtime_context = prepare_generation_runtime_context(
            request=request,
            zone_prefix=zone_prefix,
        )
        latlon_bbox = runtime_context.latlon_bbox
        global_center = runtime_context.global_center

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
        )
        if workflow_result.terrain_only_result is not None:
            return
        print(f"[OK] Model generation completed. Task ID: {task_id}, Zone ID: {zone_id}, File: {workflow_result.output_file_abs}")
        
        
    except Exception as e:
        print(f"[ERROR] === РџРћРњРР›РљРђ Р“Р•РќР•Р РђР¦Р†Р‡ РњРћР”Р•Р›Р† === Task ID: {task_id}, Zone ID: {zone_id}, Error: {e}")
        import traceback
        traceback.print_exc()
        task.fail(str(e))
        # IMPORTANT: don't re-raise from background task, otherwise Starlette logs it as ASGI error
        # and it can interrupt other tasks. The failure is already recorded in task state.
        return


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
