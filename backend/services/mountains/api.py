# -*- coding: utf-8 -*-
"""HTTP-шар режиму «Гори» (APIRouter). main.py: `from services.mountains.api import router, bind; bind(tasks, GenerationTask, OUTPUT_DIR, rate_limit); app.include_router(router)`.

Ендпоїнти:
  GET  /api/mountains/presets?locale=uk     — відомі вершини (з фото-превʼю у /mountains/presets/*.jpg на фронті)
  GET  /api/mountains/figures               — бібліотека фігурок (з ліцензіями)
  POST /api/mountains/agent                 — {text, locale, base?} → {spec, understood[], warnings[], questions[], confidence}
  POST /api/mountains/preview               — {lat, lon, area_km, size_mm, height_mm?} → hillshade PNG (data-URL) + цифри
  POST /api/mountains/generate              — {spec} → {task_id}; статус — /api/status/{task_id} (як у решти режимів)
  POST /api/worlds/agent                    — {text, size_mm, shape?} → пояснення spec для «Світів»
"""
from __future__ import annotations

import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api", tags=["mountains"])
MAX_ACTIVE = 1
_ctx: dict[str, Any] = {}


def bind(tasks: dict, task_cls, output_dir: Path, rate_limit, release=None):
    _ctx.update(tasks=tasks, task_cls=task_cls, output_dir=Path(output_dir), rate_limit=rate_limit, release=release)


class AgentRequest(BaseModel):
    text: str = Field(default="", max_length=2000)
    locale: str = Field(default="uk", max_length=5)
    base: Optional[dict] = None


class PreviewRequest(BaseModel):
    lat: float = Field(ge=-85, le=85)
    lon: float = Field(ge=-180, le=180)
    area_km: float = Field(default=4.0, ge=1.0, le=60.0)
    size_mm: float = Field(default=200.0, ge=60, le=400)
    height_mm: Optional[float] = Field(default=None, ge=20, le=300)
    satellite: bool = True


class GenerateRequest(BaseModel):
    spec: dict


class WorldAgentRequest(BaseModel):
    text: str = Field(default="", max_length=2000)
    size_mm: float = Field(default=120.0, ge=40, le=220)
    shape: Optional[str] = None


@router.get("/mountains/presets")
async def presets(locale: str = "uk"):
    from .presets import public
    return {"presets": public(locale)}


@router.get("/mountains/figures")
async def figures(locale: str = "uk"):
    from .figures import library
    out = []
    for f in library():
        out.append({"id": f["id"], "name": f["name"].get(locale) or f["name"]["en"], "kind": f["kind"], "default_height_mm": f["default_height_mm"],
                    "min_height_mm": f["min_height_mm"], "max_height_mm": f["max_height_mm"], "license": f.get("license"), "source": f.get("source"),
                    "thumb": f"/mountains/figures/{f['id']}.jpg"})
    return {"figures": out}


@router.post("/mountains/agent")
async def agent(req: AgentRequest):
    from .agent import understand
    try:
        return understand(req.text, req.locale, req.base)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(); raise HTTPException(500, f"Агент не зміг розібрати опис: {exc}")


@router.post("/worlds/agent")
async def worlds_agent(req: WorldAgentRequest):
    from .agent import understand_world
    try:
        return understand_world(req.text, req.size_mm, req.shape)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(); raise HTTPException(500, f"Агент не зміг розібрати опис: {exc}")


_preview_cache: dict[str, tuple[float, dict]] = {}


@router.post("/mountains/preview")
async def preview(req: PreviewRequest):
    from .pipeline import quick_preview
    key = f"{req.lat:.4f},{req.lon:.4f},{req.area_km:.2f},{req.size_mm:.0f},{req.height_mm},{req.satellite}"
    hit = _preview_cache.get(key)
    if hit and time.time() - hit[0] < 3600:
        return hit[1]
    try:
        import anyio
        res = await anyio.to_thread.run_sync(lambda: quick_preview(req.lat, req.lon, req.area_km, req.size_mm, req.height_mm, req.satellite))
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(); raise HTTPException(502, f"Не вдалося отримати висоти для цієї ділянки: {exc}")
    if len(_preview_cache) > 200:
        _preview_cache.clear()
    _preview_cache[key] = (time.time(), res)
    return res


def _run_task(task_id: str, spec: dict):
    from .pipeline import run
    tasks = _ctx["tasks"]; task = tasks.get(task_id)
    if task is None:
        return
    try:
        task.update_status("processing", 5, "Перевіряю параметри…")
        out = run(spec, _ctx["output_dir"], f"mountain_{task_id[:8]}",
                  progress=lambda p, m: task.update_status("processing", int(p), m), log=lambda s: print(s, flush=True))
        task.set_output("3mf", out["3mf"]); task.set_output("glb", out["glb"])
        if out.get("tiles_zip"):
            task.set_output("tiles_zip", out["tiles_zip"])
        if out.get("preview_png"):
            task.set_output("preview_png", out["preview_png"])
        if out.get("paint_jpg"):
            task.set_output("paint_jpg", out["paint_jpg"])
        task.complete(out["3mf"])
        m = out["meta"]
        task.message = f"Готово · {m['place']['name']} · 1:{m['scale']} · {m['height_mm']:.0f} мм"
        # фронт читає це поле як у «Світах»: що саме збудовано + посилання на додаткові файли
        task.world_spec = {"mode": "mountain", "place": m["place"]["name"], "scale": m["scale"], "zexag": m["zexag"], "height_mm": m["height_mm"],
                           "size_mm": m["size_mm"], "sources": m["sources"], "tiles": m["tiles"], "figures": m["figures"],
                           "tiles_zip": f"/files/{Path(out['tiles_zip']).name}" if out.get("tiles_zip") else None,
                           "preview_png": f"/files/{Path(out['preview_png']).name}" if out.get("preview_png") else None,
                           "paint_jpg": f"/files/{Path(out['paint_jpg']).name}" if out.get("paint_jpg") else None,
                           "seconds": m["seconds"], "watertight": m["watertight"]}
        print(f"[MNT] {task_id} done in {m['seconds']} s", flush=True)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        task.fail(f"Помилка генерації гори: {exc}")
    finally:
        rel = _ctx.get("release")
        if rel:
            try:
                rel(task_id)               # gc + malloc_trim — як після решти важких задач (інцидент 06.09)
            except Exception:
                pass


@router.post("/mountains/generate")
async def generate(req: GenerateRequest, background: BackgroundTasks):
    from .agent import normalize_spec
    if not _ctx:
        raise HTTPException(503, "Режим «Гори» не ініціалізовано")
    spec, warn = normalize_spec(req.spec or {})
    if not spec.get("place"):
        raise HTTPException(422, "Оберіть місце (точку на мапі або вершину зі списку)")
    # VM має 4 ГБ: одночасно рахуємо не більше MAX_ACTIVE гір (решта режимів мають свою чергу)
    active = sum(1 for t in _ctx["tasks"].values() if getattr(t, "status", "") == "processing" and (getattr(t, "request", {}) or {}).get("mode") == "mountain")
    if active >= MAX_ACTIVE:
        raise HTTPException(429, "Зараз рахуються інші гори — спробуйте за хвилину")
    task_id = str(uuid.uuid4())
    _ctx["tasks"][task_id] = _ctx["task_cls"](task_id=task_id, request={"mode": "mountain", "place": spec["place"]["name"], "size_mm": spec["size_mm"]})
    background.add_task(_run_task, task_id, spec)
    print(f"[MNT] task {task_id} {spec['place']['name']} {spec['size_mm']} мм", flush=True)
    return {"task_id": task_id, "status": "processing", "message": "Задача створена", "warnings": warn, "spec": spec}
