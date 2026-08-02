"""Інференс сегментатора планів на onnxruntime (CPU).

Прод-обмеження, зашиті сюди:
  • НІКОЛИ не імпортує torch — на 4 ГБ VPS його немає й не буде;
  • 1-2 потоки (більше — і генерація мап поруч починає голодувати);
  • сесія створюється лениво й кешується: старт ~0.3 с, тримати варто;
  • відсутність файлу ваг НЕ помилка — пайплайн просто піде класичним шляхом.

ПЕРЕВІРЕНО І ВІДКИНУТО: дорощування маски мережі всередину морфологічно
замкненого чорнила (щоб добрати ширину «порожніх» стін). Виміряно на 24 планах
різних стилів: замкнених кімнат 88/126 із дорощуванням проти 92/126 без нього.
Справжня причина низького recall на порожніх стінах була в СИНТЕТИЦІ — вона
малювала контуром навіть 8-см перегородки, чого в реальних кресленнях не буває.

ТАКОЖ ВІДКИНУТО: уточнення меж стіни перетином маски з бінаризованим чорнилом
(refine_with_ink). Задум був підтягнути товщину, а на ділі воно РОЗРИВАЛО стіни
там, де лінія бліда (світло-сірі несучі — звичайна річ на друкованих планах):
замкнених кімнат 92/126 із ним проти 121/126 без нього. Товщину й так добре дає
distance transform у векторизаторі.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

CLASS_BG, CLASS_WALL, CLASS_DOOR, CLASS_WINDOW = 0, 1, 2, 3
DEFAULT_INPUT_SIZE = 640

_LOCK = threading.Lock()
_SESSION: Dict[str, object] = {}


def model_path() -> Optional[str]:
    """Шлях до ваг: env FLOORPLAN_ONNX або backend/models/floorplan_seg.onnx."""
    env = os.getenv("FLOORPLAN_ONNX", "").strip()
    if env:
        return env if os.path.exists(env) else None
    here = os.path.dirname(os.path.abspath(__file__))
    default = os.path.normpath(os.path.join(here, "..", "..", "models", "floorplan_seg.onnx"))
    return default if os.path.exists(default) else None


def is_available() -> bool:
    if model_path() is None:
        return False
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def _session():
    path = model_path()
    if path is None:
        return None
    with _LOCK:
        cached = _SESSION.get(path)
        if cached is not None:
            return cached
        try:
            import onnxruntime as ort
        except ImportError:
            return None
        opts = ort.SessionOptions()
        threads = max(1, int(os.getenv("FLOORPLAN_ONNX_THREADS", "2")))
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        except Exception:
            return None
        _SESSION[path] = session
        return session


@dataclass
class NnDetectResult:
    wall_mask: np.ndarray
    door_mask: np.ndarray
    window_mask: np.ndarray
    confidence: float
    notes: List[str]


def _letterbox(rgb: np.ndarray, size: int) -> Tuple[np.ndarray, float, int, int]:
    """Вписує зображення у квадрат size×size БЕЗ спотворення пропорцій.

    Пропорції тут не косметика: розтягнута картинка зсуває товщини стін по
    одній осі, а з товщини рахується фізичний розмір виробу."""
    import cv2

    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, scale, left, top


def choose_input_size(rgb: np.ndarray, floor: int = 512, ceiling: int = 640) -> int:
    """Роздільність інференсу під конкретне зображення.

    Стеля саме 640, а не «якомога більше»: модель навчена на 512, і на 960 px
    стіни для неї стають надто товстими — recall падає з 0.98 до 0.86, тобто
    частина перегородок просто зникає. Нижня межа 512 тримає деталь на дрібних
    сканах. Заміряно на синтетичному наборі."""
    longest = int(max(rgb.shape[0], rgb.shape[1]))
    size = min(ceiling, max(floor, longest))
    return max(floor, (size // 32) * 32)


def detect(rgb: np.ndarray, input_size: Optional[int] = None) -> Optional[NnDetectResult]:
    """RGB-план → маски стін/дверей/вікон у РОЗМІРІ ВХІДНОГО зображення.

    None означає «нейромережі немає» — це нормальний робочий стан, викликач
    має піти класичним шляхом."""
    import cv2

    session = _session()
    if session is None:
        return None

    size = max(256, int(input_size or choose_input_size(rgb)) // 32 * 32)
    canvas, scale, left, top = _letterbox(rgb, size)
    tensor = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    try:
        logits = session.run(["logits"], {"image": tensor})[0]
    except Exception:
        return None

    logits = logits[0]                                   # (C, size, size)
    exp = np.exp(logits - logits.max(axis=0, keepdims=True))
    probs = exp / np.clip(exp.sum(axis=0, keepdims=True), 1e-9, None)

    # ГІСТЕРЕЗИС замість голого argmax. На межі стіни ймовірність гуляє навколо
    # 0.5, і argmax дає «пилку» — а кожна зазубрина породжує відросток скелета й
    # зайвий відрізок у редакторі. Беремо впевнене ядро (p>0.70) і нарощуємо
    # його лише туди, де модель бодай схиляється до стіни (p>0.35). Той самий
    # прийом, що в детекторі країв Кенні, і з тієї самої причини.
    # Нижній поріг тримаємо на рівні argmax (0.5), а НЕ нижче: з 0.35 нарощування
    # затягувало розмиту облямівку навколо кожної стіни, і товщина зростала вдвічі
    # (precision падала до 0.43 при recall 0.91) — тобто виріб виходив із
    # неправильними пропорціями стін. Гістерезис тут прибирає слабкі плями, а не
    # розширює впевнені.
    structural_p = 1.0 - probs[CLASS_BG]
    core = (structural_p >= 0.60).astype(np.uint8)
    loose = (structural_p >= 0.50).astype(np.uint8)
    if core.sum() >= 30:
        grow = np.ones((3, 3), np.uint8)
        current = core
        for _ in range(60):
            grown = cv2.dilate(current, grow, iterations=1) & loose
            if np.array_equal(grown, current):
                break
            current = grown
        structural_mask = current.astype(bool)
    else:
        structural_mask = (probs.argmax(axis=0) != CLASS_BG)

    labels = probs.argmax(axis=0).astype(np.uint8)
    labels[~structural_mask] = CLASS_BG
    # Всередині структури клас беремо як argmax серед стіна/двері/вікно, щоб
    # «майже фон» не пробивав дірку посеред стіни.
    inner = probs[1:].argmax(axis=0).astype(np.uint8) + 1
    labels = np.where(structural_mask, inner, CLASS_BG).astype(np.uint8)

    h, w = rgb.shape[:2]
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    labels = labels[top:top + nh, left:left + nw]
    conf_map = probs.max(axis=0)[top:top + nh, left:left + nw]
    labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)

    structural = labels != CLASS_BG
    confidence = float(conf_map[conf_map > 0].mean()) if conf_map.size else 0.0
    if structural.sum() < 50:
        return NnDetectResult(
            wall_mask=np.zeros((h, w), np.uint8), door_mask=np.zeros((h, w), np.uint8),
            window_mask=np.zeros((h, w), np.uint8), confidence=0.0,
            notes=["Нейромережа не побачила стін на цьому зображенні."],
        )
    return NnDetectResult(
        wall_mask=(labels == CLASS_WALL).astype(np.uint8),
        door_mask=(labels == CLASS_DOOR).astype(np.uint8),
        window_mask=(labels == CLASS_WINDOW).astype(np.uint8),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        notes=[],
    )
