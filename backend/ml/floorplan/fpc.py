"""Floor Plan CIS — РЕАЛЬНІ пострадянські плани з ручною розміткою стін.

Чому саме цей датасет вирішальний: усі попередні заміри робились на СИНТЕТИЦІ,
яку я сам і генерував, — тобто модель перевірялась на власному відображенні.
Тут 500 справжніх планів з оголошень нерухомості РФ/СНД, розмічених людиною.
Це єдина чесна відповідь на питання «а на реальних кресленнях воно працює?».

Ліцензія CC BY 4.0 — комерційне використання дозволене (перевірено на записі
Zenodo 17871080). Класи в COCO: 1=door, 2=wall, 3=window — збігаються з нашими
(1=стіна, 2=двері, 3=вікно) з точністю до перестановки.

Використання:
    python -m ml.floorplan.fpc eval --split valid
    python -m ml.floorplan.fpc export --split train --out _datasets/fpc_train
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FPC_ROOT_CANDIDATES = [
    r"D:\floorplan_datasets\fpc",
    os.path.join(HERE, "_datasets", "fpc"),
]

# COCO-категорія → наш клас маски
COCO_TO_CLASS = {2: 1, 1: 2, 3: 3}      # wall→1, door→2, window→3


def fpc_root(explicit: Optional[str] = None) -> Optional[str]:
    for path in ([explicit] if explicit else []) + FPC_ROOT_CANDIDATES:
        if path and os.path.isdir(path):
            return path
    return None


def iter_split(split: str = "valid", root: Optional[str] = None
               ) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
    """Віддає (шлях, RGB, маска класів) для кожного зображення сплiту."""
    import cv2

    base = fpc_root(root)
    if base is None:
        raise SystemExit(
            "Датасет Floor Plan CIS не знайдено. Завантажте "
            "https://zenodo.org/records/17871080/files/fpc.zip і розпакуйте."
        )
    folder = os.path.join(base, split)
    ann_path = os.path.join(folder, "_annotations.coco.json")
    if not os.path.exists(ann_path):
        raise SystemExit(f"немає {ann_path}")
    with open(ann_path, encoding="utf-8") as handle:
        coco = json.load(handle)

    by_image: Dict[int, List[dict]] = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)

    for image_info in coco["images"]:
        path = os.path.join(folder, image_info["file_name"])
        if not os.path.exists(path):
            continue
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        # Спершу стіни, потім отвори — щоб отвори лягли ПОВЕРХ смуги стіни,
        # рівно як у нашій синтетиці.
        for target in (2, 1, 3):
            for ann in by_image.get(image_info["id"], []):
                if ann["category_id"] != target:
                    continue
                cls = COCO_TO_CLASS.get(target)
                if cls is None:
                    continue
                for polygon in ann.get("segmentation") or []:
                    if len(polygon) < 6:
                        continue
                    points = np.array(polygon, np.float32).reshape(-1, 2).round().astype(np.int32)
                    cv2.fillPoly(mask, [points], int(cls))
        yield path, rgb, mask


def evaluate(split: str = "valid", root: Optional[str] = None,
             limit: int = 0) -> Dict[str, float]:
    """Метрики маски нашої мережі проти ручної розмітки на РЕАЛЬНИХ планах."""
    from services.floorplan import detect_nn

    if not detect_nn.is_available():
        raise SystemExit("немає файлу ваг models/floorplan_seg.onnx")

    inter_w = union_w = pred_w = true_w = 0
    inter_s = union_s = pred_s = true_s = 0
    count = 0
    for _path, rgb, mask in iter_split(split, root):
        result = detect_nn.detect(rgb)
        if result is None:
            continue
        count += 1
        # 1) лише стіни; 2) уся структура (стіна+двері+вікно) — саме її
        #    отримує векторизатор
        pred_wall = result.wall_mask > 0
        true_wall = mask == 1
        inter_w += int((pred_wall & true_wall).sum())
        union_w += int((pred_wall | true_wall).sum())
        pred_w += int(pred_wall.sum())
        true_w += int(true_wall.sum())

        pred_all = (result.wall_mask | result.door_mask | result.window_mask) > 0
        true_all = mask > 0
        inter_s += int((pred_all & true_all).sum())
        union_s += int((pred_all | true_all).sum())
        pred_s += int(pred_all.sum())
        true_s += int(true_all.sum())
        if limit and count >= limit:
            break

    safe = lambda a, b: (a / b) if b else 0.0
    return {
        "images": count,
        "wall_iou": safe(inter_w, union_w),
        "wall_precision": safe(inter_w, pred_w),
        "wall_recall": safe(inter_w, true_w),
        "struct_iou": safe(inter_s, union_s),
        "struct_precision": safe(inter_s, pred_s),
        "struct_recall": safe(inter_s, true_s),
    }


def export_pairs(split: str, out_dir: str, root: Optional[str] = None) -> int:
    """Зберігає (зображення, маска) у npz — щоб тренування не парсило COCO щоразу."""
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    for path, rgb, mask in iter_split(split, root):
        name = os.path.splitext(os.path.basename(path))[0][:60]
        np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), image=rgb, mask=mask)
        written += 1
    print(f"[fpc] {split}: збережено {written} пар у {out_dir}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["eval", "export", "stats"])
    parser.add_argument("--split", default="valid")
    parser.add_argument("--root", default=None)
    parser.add_argument("--out", default=os.path.join(HERE, "_datasets", "fpc_train"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.command == "eval":
        metrics = evaluate(args.split, args.root, args.limit)
        print(f"[fpc:{args.split}] зображень {metrics['images']}")
        print(f"  СТІНИ      IoU={metrics['wall_iou']:.3f} "
              f"precision={metrics['wall_precision']:.3f} recall={metrics['wall_recall']:.3f}")
        print(f"  СТРУКТУРА  IoU={metrics['struct_iou']:.3f} "
              f"precision={metrics['struct_precision']:.3f} recall={metrics['struct_recall']:.3f}")
    elif args.command == "export":
        export_pairs(args.split, args.out, args.root)
    else:
        for split in ("train", "valid", "test"):
            try:
                total = sum(1 for _ in iter_split(split, args.root))
                print(f"{split}: {total} зображень")
            except SystemExit as exc:
                print(split, exc)


if __name__ == "__main__":
    main()
