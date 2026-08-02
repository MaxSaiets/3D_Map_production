"""Оркестратор сервісу «макет квартири»: файл → план → друкована модель.

Дві публічні операції, і між ними ЗАВЖДИ стоїть людина:

    analyze(bytes)          → PlanVector у пікселях + гіпотези масштабу + превʼю
    build(plan, m_per_px)   → водонепроникний меш + 3MF/STL/GLB

Розділення принципове. Автоматика вгадує стіни й масштаб добре, але не
ідеально, а помилка масштабу непомітна на екрані й виявляється лише після
друку й доставки. Тому між analyze і build обов'язково стоїть підтвердження
користувача — це і є головний елемент продукту, а не «поліш».
"""
from __future__ import annotations

import base64
import io
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .builder import BuildOptions, BuildResult, PlanBuildError, build_plan_mesh
from .plan_model import PlanVector
from .preprocess import PreparedImage, PreprocessError, prepare
from .scale import ScaleResult, interior_area_px2, resolve_scale
from .vectorize import VectorizeConfig, masks_to_plan

MAX_UPLOAD_BYTES = 25 * 1024 * 1024      # Caddy пропускає 50 МБ; лишаємо запас
PREVIEW_MAX_SIDE = 1400
PREVIEW_JPEG_QUALITY = 82

# Правдоподібність результату — межі, за якими продовжувати небезпечно.
MIN_AREA_M2 = 8.0
MAX_AREA_M2 = 400.0
PARTITION_RANGE_M = (0.05, 0.20)
BEARING_RANGE_M = (0.20, 0.60)


class FloorplanError(ValueError):
    """Помилка, яку можна показати користувачу як є."""


@dataclass
class AnalyzeResult:
    plan: PlanVector                      # координати В ПІКСЕЛЯХ превʼю
    scale: ScaleResult
    preview_data_url: str
    image_size_px: Tuple[int, int]
    detector: str                         # nn | cv
    confidence: float
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timings_ms: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        m_per_px = self.scale.chosen.m_per_px
        width_m = self.image_size_px[0] * m_per_px
        plan_w, plan_h = self.plan.size_m()
        return {
            "plan": self.plan.to_dict(),
            "scale": self.scale.to_dict(),
            "preview": self.preview_data_url,
            "image_size_px": list(self.image_size_px),
            "detector": self.detector,
            "confidence": round(self.confidence, 3),
            "notes": self.notes,
            "warnings": self.warnings,
            "timings_ms": self.timings_ms,
            "estimate": {
                "plan_width_m": round(plan_w * m_per_px, 2),
                "plan_height_m": round(plan_h * m_per_px, 2),
                "area_m2": round(plan_w * m_per_px * plan_h * m_per_px, 1),
                "sheet_width_m": round(width_m, 2),
                # Площа приміщень у пікселях² — з неї фронтенд рахує масштаб,
                # коли користувач вписує загальну площу квартири з договору.
                "interior_px2": round(interior_area_px2(self.plan), 1),
            },
        }


def _to_preview_data_url(rgb: np.ndarray) -> Tuple[str, Tuple[int, int]]:
    """Вирівняне зображення → data-URL для фронтенда.

    Навмисно НЕ кладемо план користувача у /files: це креслення чужої квартири,
    і в цьому проєкті вже одного разу протікали персональні дані через
    статичну роздачу. Дані живуть у відповіді й далі — лише в браузері."""
    from PIL import Image

    h, w = rgb.shape[:2]
    scale = min(1.0, PREVIEW_MAX_SIDE / max(h, w))
    img = Image.fromarray(rgb)
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=PREVIEW_JPEG_QUALITY, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}", (img.width, img.height)


def decode_data_url(value: str) -> bytes:
    """'data:image/png;base64,...' або чистий base64 → байти."""
    if not value:
        raise FloorplanError("Файл не передано.")
    raw = value.strip()
    if raw.startswith("data:"):
        _header, _, payload = raw.partition(",")
        if not payload:
            raise FloorplanError("Некоректний data-URL.")
        raw = payload
    try:
        data = base64.b64decode(raw, validate=False)
    except Exception as exc:
        raise FloorplanError("Не вдалось розкодувати файл.") from exc
    if len(data) > MAX_UPLOAD_BYTES:
        raise FloorplanError(
            f"Файл завеликий ({len(data) / 1e6:.1f} МБ). Максимум "
            f"{MAX_UPLOAD_BYTES / 1e6:.0f} МБ — стисніть або зменшіть роздільність."
        )
    if len(data) < 512:
        raise FloorplanError("Файл порожній або пошкоджений.")
    return data


def _downscale_masks(wall: np.ndarray, door: Optional[np.ndarray],
                     window: Optional[np.ndarray], max_side: int):
    """Зменшує маски до робочої роздільності (INTER_NEAREST — класи не змішуємо)."""
    import cv2

    height, width = wall.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return wall, door, window
    scale = max_side / float(longest)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))

    def _shrink(mask):
        if mask is None:
            return None
        return cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)

    return _shrink(wall), _shrink(door), _shrink(window)


def _scale_sanity(plan_px: PlanVector, m_per_px: float) -> List[str]:
    """Фізична перевірка масштабу: товщини стін і площа мусять бути з нашого світу.

    Це остання сітка безпеки перед друком. Масштаб, за якого стіна виходить
    завтовшки 4 м, — точно помилка, навіть якщо OCR у ньому «впевнений»."""
    problems: List[str] = []
    width_px, height_px = plan_px.size_m()
    width_m, height_m = width_px * m_per_px, height_px * m_per_px
    area = width_m * height_m
    if area < MIN_AREA_M2 or area > MAX_AREA_M2:
        problems.append(
            f"За цим масштабом квартира виходить {width_m:.1f}×{height_m:.1f} м "
            f"({area:.0f} м²) — це поза межами правдоподібного. Уточніть масштаб лінійкою."
        )
    thicknesses = sorted({round(w.thickness_m * m_per_px, 3) for w in plan_px.walls})
    if thicknesses:
        thinnest, thickest = thicknesses[0], thicknesses[-1]
        if thinnest < PARTITION_RANGE_M[0] * 0.6 or thinnest > PARTITION_RANGE_M[1] * 2.5:
            problems.append(
                f"Найтонша стіна виходить {thinnest * 100:.0f} см — для перегородки "
                f"очікуємо 5-20 см. Перевірте масштаб."
            )
        if thickest > BEARING_RANGE_M[1] * 2.0:
            problems.append(
                f"Найтовща стіна виходить {thickest * 100:.0f} см — це забагато навіть "
                f"для несучої. Перевірте масштаб."
            )
    return problems


def analyze(data: bytes, *, reference_px: Optional[float] = None,
            reference_m: Optional[float] = None, use_ocr: bool = True,
            use_nn: bool = True) -> AnalyzeResult:
    """Файл плану → векторний план у пікселях + гіпотези масштабу."""
    timings: Dict[str, int] = {}
    notes: List[str] = []
    warnings: List[str] = []

    t0 = time.time()
    try:
        prepared: PreparedImage = prepare(data)
    except PreprocessError as exc:
        raise FloorplanError(str(exc)) from exc
    timings["preprocess"] = int((time.time() - t0) * 1000)
    notes.extend(prepared.notes)

    # ── Детекція: нейромережа, якщо є; інакше класичний CV ────────────────────
    t0 = time.time()
    wall_mask = door_mask = window_mask = None
    detector, confidence = "cv", 0.0
    if use_nn:
        try:
            from . import detect_nn

            nn_result = detect_nn.detect(prepared.rgb)
        except Exception:
            nn_result = None
        if nn_result is not None and nn_result.wall_mask.sum() > 0:
            wall_mask = nn_result.wall_mask
            door_mask = nn_result.door_mask
            window_mask = nn_result.window_mask
            detector, confidence = "nn", nn_result.confidence
            notes.extend(nn_result.notes)

    if wall_mask is None:
        from .detect_cv import detect_walls

        cv_result = detect_walls(prepared.rgb)
        wall_mask = cv_result.wall_mask
        confidence = cv_result.confidence
        notes.extend(cv_result.notes)
        if cv_result.wall_mask.sum() == 0:
            raise FloorplanError(
                "Не вдалось знайти стіни. Переконайтесь, що на фото видно весь план, "
                "він не розмитий і зроблений згори."
            )
    timings["detect"] = int((time.time() - t0) * 1000)

    # ── Векторизація ─────────────────────────────────────────────────────────
    # ЄДИНА РОБОЧА РОЗДІЛЬНІСТЬ. Векторизація має етапи, квадратичні за кількістю
    # відрізків, а їх кількість росте з роздільністю: на аркуші 2400 px це 17.5 с
    # проти 2 с на 1400 px — при однаковому результаті, бо мережа все одно
    # працює на 640, а геометрія стін повністю визначена вже на 1200-1400 px.
    # Заодно маска, план і превʼю опиняються в одному масштабі, і зникає окремий
    # перерахунок координат нижче.
    t0 = time.time()
    wall_mask, door_mask, window_mask = _downscale_masks(
        wall_mask, door_mask, window_mask, PREVIEW_MAX_SIDE,
    )
    plan_px = masks_to_plan(wall_mask, door_mask, window_mask,
                            cfg=VectorizeConfig(), confidence=confidence)
    # ТУТ БУЛО вимірювання товщини по самому кресленню — прибрано після заміру.
    # Гіпотеза «маска завищує товщину» виявилась хибною: distance transform у
    # векторизаторі вже дає 3-5% похибки (5.6 px → 7.0, 26.6 → 27.5), а профіль
    # поперек стіни ловив сусідню штриховку й давав 11 px замість 7. Розклад
    # похибки підошви показав, що товщина взагалі не винна: при однаковій
    # товщині на обох планах IoU 0.751 проти 0.768 — тобто вся різниця в
    # ПОЛОЖЕННІ стін, а не в їхній ширині.
    timings["vectorize"] = int((time.time() - t0) * 1000)
    notes.extend(plan_px.notes)
    if not plan_px.walls:
        raise FloorplanError(
            "Стіни знайдено, але їх не вдалось перетворити на лінії. "
            "Спробуйте чіткіше фото або більший файл."
        )

    # ── Масштаб ──────────────────────────────────────────────────────────────
    t0 = time.time()
    pdf_pts = None
    if prepared.vector_paths:
        try:
            pdf_pts = prepared.vector_paths[0].get("page_pts")
        except Exception:
            pdf_pts = None
    scale_result = resolve_scale(
        plan_px, rgb=prepared.rgb, reference_px=reference_px, reference_m=reference_m,
        pdf_page_pts=pdf_pts, image_px=list(prepared.size), use_ocr=use_ocr,
    )
    timings["scale"] = int((time.time() - t0) * 1000)

    # ── Превʼю (і перерахунок координат у його масштаб) ───────────────────────
    # Коефіцієнт рахуємо від РОЗМІРУ МАСКИ, а не від вихідного зображення:
    # маска вже зменшена до робочої роздільності, і план разом із нею. Ділити на
    # ширину оригіналу означало б застосувати те саме зменшення вдруге — на
    # аркушах понад 1400 px план стиснувся б удвічі проти креслення під ним.
    preview_url, preview_size = _to_preview_data_url(prepared.rgb)
    shrink = preview_size[0] / max(1, wall_mask.shape[1])
    if abs(shrink - 1.0) > 1e-6:
        for wall in plan_px.walls:
            wall.x1 *= shrink
            wall.y1 *= shrink
            wall.x2 *= shrink
            wall.y2 *= shrink
            wall.thickness_m *= shrink
        for opening in plan_px.openings:
            opening.width_m *= shrink
        scale_result.chosen.m_per_px /= shrink
        for candidate in scale_result.candidates:
            candidate.m_per_px /= shrink
    plan_px.image_size_px = preview_size
    plan_px.m_per_px = scale_result.chosen.m_per_px

    warnings.extend(_scale_sanity(plan_px, scale_result.chosen.m_per_px))
    if scale_result.chosen.source in ("assumed", "door"):
        warnings.append(
            "Масштаб визначено приблизно — обовʼязково перевірте його лінійкою "
            "по відомому розміру."
        )

    return AnalyzeResult(
        plan=plan_px, scale=scale_result, preview_data_url=preview_url,
        image_size_px=preview_size, detector=detector, confidence=confidence,
        notes=notes, warnings=warnings, timings_ms=timings,
    )


def build(plan_dict: Dict[str, Any], m_per_px: float, options: Optional[BuildOptions] = None,
          progress: Optional[Callable[[int, str], None]] = None) -> BuildResult:
    """Відредагований користувачем план (у пікселях) → друкований меш."""
    if m_per_px <= 0:
        raise FloorplanError("Масштаб не заданий.")
    plan = PlanVector.from_pixel_dict(plan_dict, m_per_px)
    if not plan.walls:
        raise FloorplanError("У плані не лишилось жодної стіни.")

    width_m, height_m = plan.size_m()
    area = width_m * height_m
    if area < MIN_AREA_M2 or area > MAX_AREA_M2:
        raise FloorplanError(
            f"Розміри плану ({width_m:.1f}×{height_m:.1f} м = {area:.0f} м²) виглядають "
            f"помилковими. Перевірте масштаб перед генерацією."
        )
    try:
        return build_plan_mesh(plan, options or BuildOptions(), progress=progress)
    except PlanBuildError as exc:
        raise FloorplanError(str(exc)) from exc


def export_outputs(result: BuildResult, out_dir: str, basename: str) -> Dict[str, str]:
    """Меш → 3MF/STL/GLB поруч із рештою моделей проєкту.

    Свідомо експортуємо напряму, а НЕ через export_generation_outputs: та
    функція заново вписує геометрію в model_size_mm, а builder уже видав
    міліметри. Подвійне масштабування — саме та помилка розміру, заради якої
    весь цей сервіс і городиться. (Так само робить generate_custom_task.)"""
    os.makedirs(out_dir, exist_ok=True)
    mesh = result.mesh
    outputs: Dict[str, str] = {}
    for fmt in ("3mf", "stl", "glb"):
        path = os.path.join(out_dir, f"{basename}.{fmt}")
        try:
            mesh.export(path)
            outputs[fmt] = path
        except Exception as exc:  # noqa: BLE001
            print(f"[floorplan] export {fmt} failed: {exc}", flush=True)
    if not outputs:
        raise FloorplanError("Не вдалось зберегти модель у жодному форматі.")
    return outputs
