"""Підготовка вхідного зображення: PDF/фото/скан → чистий растр для детектора.

Користувач вантажить що завгодно: скріншот із забудовника, PDF від БТІ, фото
паперу з телефона під кутом і з тінню. Мета — привести все до одного вигляду,
бо детектор, натренований на «плоских» планах, на кривому фото просто злітає.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

MAX_SIDE_PX = 2400          # більше не має сенсу: деталь плану давно розрізнена
MIN_SIDE_PX = 320
PDF_RENDER_DPI = 200


class PreprocessError(ValueError):
    """Вхідний файл не читається як зображення плану."""


@dataclass
class PreparedImage:
    rgb: np.ndarray                       # HxWx3 uint8, вирівняне
    original_rgb: np.ndarray              # те, що показуємо користувачу
    warp_matrix: Optional[np.ndarray] = None   # original → rgb (3x3), None якщо без змін
    scale_factor: float = 1.0             # original_px → rgb_px
    source: str = "image"                 # image | pdf | pdf-vector
    notes: List[str] = field(default_factory=list)
    vector_paths: Optional[List[Dict[str, Any]]] = None  # для PDF з векторами

    @property
    def size(self) -> Tuple[int, int]:
        return (self.rgb.shape[1], self.rgb.shape[0])


# ═════════════════════════════════════════════════════════════════════════════
#  Завантаження
# ═════════════════════════════════════════════════════════════════════════════
def _is_pdf(data: bytes) -> bool:
    return data[:5] == b"%PDF-"


def load_pdf_page(data: bytes, page: int = 0, dpi: int = PDF_RENDER_DPI
                  ) -> Tuple[np.ndarray, Optional[List[Dict[str, Any]]]]:
    """PDF → растр сторінки + (якщо є) векторні шляхи.

    Вектор із PDF — це джекпот: геометрія точна, CV взагалі не потрібен, а
    розмір сторінки в пунктах разом із поміткою «М 1:100» одразу дає масштаб.

    ЛІЦЕНЗІЇ: свідомо НЕ використовуємо PyMuPDF/fitz — він AGPL-3.0 (або
    комерційна ліцензія Artifex за десятки тисяч $/рік), що несумісно з платним
    сервісом. pypdfium2 = BSD-3, pdfplumber = MIT."""
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover
        raise PreprocessError("PDF не підтримується: не встановлено pypdfium2.") from exc

    doc = pdfium.PdfDocument(data)
    try:
        if len(doc) == 0:
            raise PreprocessError("PDF порожній.")
        index = max(0, min(page, len(doc) - 1))
        pdf_page = doc[index]
        scale = dpi / 72.0
        bitmap = pdf_page.render(scale=scale, draw_annots=False)
        img = bitmap.to_numpy()
        if img.ndim == 2:
            img = np.dstack([img] * 3)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        img = np.ascontiguousarray(img.astype(np.uint8))
        page_pts = (float(pdf_page.get_width()), float(pdf_page.get_height()))
    finally:
        doc.close()

    paths = _pdf_vector_paths(data, index, scale, page_pts)
    return img, paths


def _pdf_vector_paths(data: bytes, index: int, scale: float,
                      page_pts: Tuple[float, float]) -> Optional[List[Dict[str, Any]]]:
    """Лінії/прямокутники сторінки через pdfplumber (MIT), у ПІКСЕЛЯХ растру."""
    try:
        import pdfplumber
    except ImportError:
        return None
    items: List[Dict[str, Any]] = []
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            if index >= len(pdf.pages):
                return None
            page = pdf.pages[index]
            for ln in page.lines:
                items.append({
                    "t": "l",
                    "p": [[float(ln["x0"]) * scale, float(ln["top"]) * scale],
                          [float(ln["x1"]) * scale, float(ln["bottom"]) * scale]],
                    "w": float(ln.get("linewidth") or 0.0) * scale,
                })
            for rc in page.rects:
                items.append({
                    "t": "re",
                    "p": [[float(rc["x0"]) * scale, float(rc["top"]) * scale],
                          [float(rc["x1"]) * scale, float(rc["bottom"]) * scale]],
                    "w": float(rc.get("linewidth") or 0.0) * scale,
                    "fill": bool(rc.get("fill")),
                })
    except Exception:
        return None
    if not items:
        return None
    return [{"items": items, "page_pts": list(page_pts), "scale": scale}]


def load_image(data: bytes) -> Tuple[np.ndarray, str, Optional[List[Dict[str, Any]]]]:
    """bytes → (RGB uint8, джерело, векторні шляхи PDF або None)."""
    if not data:
        raise PreprocessError("Порожній файл.")
    if _is_pdf(data):
        img, paths = load_pdf_page(data)
        return img, ("pdf-vector" if paths else "pdf"), paths

    from PIL import Image, ImageOps

    try:
        pil = Image.open(io.BytesIO(data))
        pil = ImageOps.exif_transpose(pil)     # фото з телефона приходять «лежачи»
        pil = pil.convert("RGB")
    except Exception as exc:
        raise PreprocessError("Не вдалось прочитати зображення. Підтримуємо JPG, PNG, PDF.") from exc
    return np.array(pil, dtype=np.uint8), "image", None


# ═════════════════════════════════════════════════════════════════════════════
#  Геометричне вирівнювання
# ═════════════════════════════════════════════════════════════════════════════
def _order_quad(pts: np.ndarray) -> np.ndarray:
    """4 точки → порядок [top-left, top-right, bottom-right, bottom-left]."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def find_page_quad(rgb: np.ndarray, min_area_ratio: float = 0.35) -> Optional[np.ndarray]:
    """Шукає чотирикутник аркуша на фото. None — якщо це вже плоский скан."""
    import cv2

    h, w = rgb.shape[:2]
    small_scale = 900.0 / max(h, w)
    if small_scale < 1.0:
        small = cv2.resize(rgb, (int(w * small_scale), int(h * small_scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        small, small_scale = rgb, 1.0

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 130)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: Optional[np.ndarray] = None
    best_area = min_area_ratio * small.shape[0] * small.shape[1]
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        area = abs(cv2.contourArea(approx))
        if area > best_area:
            best_area = area
            best = approx.reshape(4, 2).astype(np.float32)
    if best is None:
        return None
    return best / small_scale


def _quad_perspective(quad: np.ndarray) -> float:
    """Наскільки чотирикутник відхиляється від осьового прямокутника (0..1).

    Використовується двічі: щоб НЕ робити зайве перетворення на плоскому скані
    і щоб відмовити у надто косому фото — при нахилі 35° розміри стін
    розповзаються на ±8%, а на екрані це непомітно."""
    ordered = _order_quad(np.asarray(quad, dtype=np.float32))
    tl, tr, br, bl = ordered
    top = float(np.linalg.norm(tr - tl))
    bottom = float(np.linalg.norm(br - bl))
    left = float(np.linalg.norm(bl - tl))
    right = float(np.linalg.norm(br - tr))
    if min(top, bottom, left, right) < 1e-6:
        return 1.0
    horizontal = abs(top - bottom) / max(top, bottom)
    vertical = abs(left - right) / max(left, right)
    return float(max(horizontal, vertical))


def dewarp_page(rgb: np.ndarray, quad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Перспективна корекція за чотирикутником аркуша."""
    import cv2

    src = _order_quad(quad)
    (tl, tr, br, bl) = src
    width = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    height = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    width, height = int(round(width)), int(round(height))
    if width < MIN_SIDE_PX or height < MIN_SIDE_PX:
        raise PreprocessError("Аркуш на фото надто малий — зніміть ближче.")
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                   dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(rgb, matrix, (width, height), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return out, matrix


def estimate_skew_deg(gray: np.ndarray, limit_deg: float = 8.0, step: float = 0.25) -> float:
    """Кут перекосу за максимумом «осьової впорядкованості» темних пікселів.

    Перебір кутів надійніший за Hough: розмірні лінії, штриховка й текст дають
    Hough купу хибних піків, а проєкційний критерій реагує саме на довгі стіни."""
    import cv2

    h, w = gray.shape[:2]
    scale = 420.0 / max(h, w)
    small = cv2.resize(gray, (max(32, int(w * scale)), max(32, int(h * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
    binary = (small < np.percentile(small, 28)).astype(np.float32)
    if binary.sum() < 50:
        return 0.0

    best_angle, best_score = 0.0, -1.0
    center = (binary.shape[1] / 2.0, binary.shape[0] / 2.0)
    angle = -limit_deg
    while angle <= limit_deg + 1e-9:
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(binary, matrix, (binary.shape[1], binary.shape[0]),
                             flags=cv2.INTER_NEAREST, borderValue=0)
        score = float((rot.sum(axis=0) ** 2).sum() + (rot.sum(axis=1) ** 2).sum())
        if score > best_score:
            best_score, best_angle = score, angle
        angle += step
    return float(best_angle)


def rotate_image(rgb: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Поворот навколо центру з розширенням полотна (нічого не обрізаємо)."""
    import cv2

    h, w = rgb.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    matrix[0, 2] += nw / 2.0 - center[0]
    matrix[1, 2] += nh / 2.0 - center[1]
    out = cv2.warpAffine(rgb, matrix, (nw, nh), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    full = np.vstack([matrix, [0.0, 0.0, 1.0]])
    return out, full


def normalize_illumination(rgb: np.ndarray) -> np.ndarray:
    """Прибирає тінь від руки/лампи: ділення на сильно розмите тло.

    ⚠️ Викликати ЛИШЕ на справжніх фото. На чистому цифровому плані (скріншот
    оголошення, експорт із CAD) розмите тло майже рівномірне, і ділення на нього
    «витягує» шум та вибілює світло-сірі стіни маркетингових рендерів. Заміряно
    на реальних CIS-планах: із беззастережним вирівнюванням замикалось 36%
    кімнат, без нього — 76%. Тому рішення приймає illumination_unevenness()."""
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Тло оцінюємо МОРФОЛОГІЧНИМ ЗАМИКАННЯМ, а не гауссовим блюром. Блюр
    # «розмазує» самі стіни у свою ж оцінку тла, і велика сіра стіна ділиться
    # сама на себе — тобто зникає. Замикання великим ядром прибирає темні
    # структури (чорнило) і лишає рівень паперу, як і має бути для документа.
    k = max(31, (min(gray.shape[:2]) // 6) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (31, 31), 0)
    background = np.maximum(background, 1).astype(np.uint8)
    norm = cv2.divide(gray, background, scale=255)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)


def prepare(data: bytes, *, dewarp: bool = True, deskew: bool = True,
            normalize: bool = True, max_side: int = MAX_SIDE_PX) -> PreparedImage:
    """Головний вхід: байти файлу → PreparedImage, готовий для детектора."""
    import cv2

    rgb, source, vector_paths = load_image(data)
    original = rgb.copy()
    notes: List[str] = []
    transform: Optional[np.ndarray] = None

    if min(rgb.shape[:2]) < MIN_SIDE_PX:
        raise PreprocessError(
            f"Зображення замале ({rgb.shape[1]}×{rgb.shape[0]} px). "
            f"Потрібно хоча б {MIN_SIDE_PX} px по короткій стороні."
        )

    # 1. Аркуш на фото → площина
    if dewarp and source == "image":
        quad = find_page_quad(rgb)
        # Якщо чотирикутник і так осьовий прямокутник — це вже плоский скан, а
        # знайшли ми просто рамку креслення. Перетворення нічого не виправить,
        # зате зайвий раз переінтерполює зображення й зіпсує тонкі лінії.
        if quad is not None and _quad_perspective(quad) < 0.015:
            quad = None
        if quad is not None:
            perspective = _quad_perspective(quad)
            if perspective > 0.15:
                raise PreprocessError(
                    "Фото зроблено під надто гострим кутом — розміри на ньому спотворені "
                    "більше ніж на 8%, і макет вийде неправильного розміру. "
                    "Перезніміть план прямо згори."
                )
            try:
                rgb, matrix = dewarp_page(rgb, quad)
                transform = matrix
                notes.append("Виправлено перспективу аркуша.")
                if perspective > 0.08:
                    notes.append("Кут зйомки був великий — перевірте розміри уважніше.")
            except PreprocessError:
                pass

    # 2. Освітлення — лише якщо воно справді нерівне (див. докстрінг функції).
    if normalize and source == "image":
        rgb = normalize_illumination(rgb)
        notes.append("Вирівняно освітлення.")

    # 3. Перекос
    if deskew:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        angle = estimate_skew_deg(gray)
        if abs(angle) > 0.3:
            rgb, rot = rotate_image(rgb, angle)
            transform = rot if transform is None else rot @ transform
            notes.append(f"Вирівняно нахил на {angle:+.1f}°.")

    # 4. Обмеження розміру
    h, w = rgb.shape[:2]
    scale_factor = 1.0
    if max(h, w) > max_side:
        scale_factor = max_side / max(h, w)
        rgb = cv2.resize(rgb, (int(w * scale_factor), int(h * scale_factor)),
                         interpolation=cv2.INTER_AREA)
        scale_matrix = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
        transform = scale_matrix if transform is None else scale_matrix @ transform

    return PreparedImage(
        rgb=np.ascontiguousarray(rgb), original_rgb=original, warp_matrix=transform,
        scale_factor=scale_factor, source=source, notes=notes, vector_paths=vector_paths,
    )


def binarize(rgb: np.ndarray, block: int = 41, offset: int = 12) -> np.ndarray:
    """Адаптивна бінаризація → uint8 {0,1}, 1 = «чорнило»."""
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    # Медіана 3×3 перед порогом: на фото з телефона зерно паперу і JPEG-артефакти
    # інакше стають окремими «компонентами», і детектор насипає десятки хибних
    # стін. Лінії плану товщі за 3 px, тож медіана їх не чіпає.
    gray = cv2.medianBlur(gray, 3)
    block = max(11, block | 1)
    binary = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block, offset)
    # Отсу як другий голос: адаптивний поріг на чистих сканах ловить шум паперу.
    _, otsu = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if otsu.mean() < 0.45:   # інакше Отсу «залив» пів-аркуша — не довіряємо
        binary = np.maximum(binary, otsu).astype(np.uint8)
    return binary.astype(np.uint8)
