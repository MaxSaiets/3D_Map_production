"""Чекпоінт PyTorch → ONNX для CPU-інференсу в проді.

На сервері (4 ГБ RAM, без GPU) torch не встановлюється взагалі — там лише
onnxruntime. Тому експорт — обов'язковий крок, а не опція.

Запуск:
    python -m ml.floorplan.export_onnx --ckpt _checkpoints/floorplan_seg_best.pt
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from ml.floorplan.model import NUM_CLASSES, FloorPlanUNet

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "..", "models", "floorplan_seg.onnx"))


def export(ckpt: str, out_path: str = DEFAULT_OUT, size: int = 512,
           opset: int = 17, verify: bool = True) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model = FloorPlanUNet(NUM_CLASSES, pretrained=False)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    # ⚠️ БЕЗ NormalizedModel. train.py подає в мережу RGB у діапазоні 0..1 БЕЗ
    # ImageNet-нормалізації, тож обгортка додавала б на інференсі перетворення,
    # якого модель ніколи не бачила. Симптом був підступний: жодної помилки, а
    # маска — усе чорнило креслення підряд (текст, розмірні лінії, меблі як
    # «стіни»), при тому що на валідації IoU лишалась 0.91. Препроцесинг
    # тренування й інференсу мусить збігатись байт у байт.
    wrapped = model.eval()

    dummy = torch.rand(1, 3, size, size)
    torch.onnx.export(
        wrapped, dummy, out_path,
        input_names=["image"], output_names=["logits"],
        opset_version=opset,
        # Динамічні H/W дають змогу ганяти план у 768 px, коли треба більше
        # деталі на тонких перегородках, без повторного експорту.
        dynamic_axes={"image": {2: "height", 3: "width"},
                      "logits": {2: "height", 3: "width"}},
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[onnx] saved {out_path} ({size_mb:.1f} MB, opset {opset})")

    if verify:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        session = ort.InferenceSession(out_path, opts, providers=["CPUExecutionProvider"])
        with torch.no_grad():
            reference = wrapped(dummy).numpy()
        t0 = time.time()
        got = session.run(["logits"], {"image": dummy.numpy()})[0]
        dt = time.time() - t0
        diff = float(np.abs(reference - got).max())
        agree = float((reference.argmax(1) == got.argmax(1)).mean())
        print(f"[onnx] verify: max|Δlogit|={diff:.5f} argmax agreement={agree * 100:.3f}% "
              f"cpu {dt * 1000:.0f} ms @{size}px")
        if agree < 0.999:
            raise SystemExit("ONNX і PyTorch розходяться — експорт непридатний.")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    export(args.ckpt, args.out, args.size, args.opset)


if __name__ == "__main__":
    main()
