"""Тренування сегментатора планів на синтетичних даних.

Дані генеруються НА ЛЬОТУ (synth.make_sample), тому:
  • на диску не лежить жодного гігабайта (важливо: на всіх дисках ~10 ГБ),
  • кожен приклад унікальний — перенавчитись на конкретну картинку неможливо,
  • розподіл ми контролюємо самі й можемо додати саме ті стилі, які приносять
    наші користувачі (штриховані стіни, розмірні ланцюжки, кирилиця).

Запуск:
    python -m ml.floorplan.train --steps 8000 --batch 4 --size 512
    python -m ml.floorplan.train --steps 200 --batch 2 --size 384   # димовий
"""
from __future__ import annotations

import argparse
import math
import os
import time
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ml.floorplan import synth
from ml.floorplan.model import NUM_CLASSES, FloorPlanUNet, SegLoss, per_class_iou

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "_checkpoints")


class SynthDataset(Dataset):
    """Нескінченний потік планів. `index` слугує зерном — набір детермінований
    і відтворюваний, попри те що не існує на диску.

    Два джерела розкладки:
      • BSP-генератор (свій) — необмежена кількість, але кімнати завжди
        прямокутні й стіни строго осьові;
      • РЕАЛЬНІ квартири зі Swiss Dwellings (CC BY 4.0) — справжня топологія з
        еркерами, скосами й нішами, якої BSP не породжує.
    Малюються обидва однаково (наші конвенції креслення), інакше модель вивчить
    не планування, а те, з якого генератора прийшов приклад."""

    def __init__(self, size: int, length: int, seed_base: int, clean_ratio: float = 0.15,
                 real_ratio: float = 0.5, real_split: str = "train",
                 val_holdout: int = 500, fpc_ratio: float = 0.35,
                 fpc_dir: str = ""):
        self.size = size
        self.length = length
        self.seed_base = seed_base
        self.clean_ratio = clean_ratio
        self.real_ratio = real_ratio
        self.real_split = real_split
        self.val_holdout = val_holdout
        # Floor Plan CIS — СПРАВЖНІ пострадянські плани з ручною розміткою.
        # Найцінніше джерело: усе інше ми малюємо самі, а тут домен рівно той,
        # який принесуть користувачі. Заміряно: без них IoU стін на реальних
        # планах 0.508 при 0.95 на власній синтетиці.
        self.fpc_ratio = fpc_ratio
        self.fpc_dir = fpc_dir or os.path.join(HERE, "_datasets", "fpc_train")
        self._layouts: Optional[list] = None
        self._fpc_files: Optional[List[str]] = None

    def _fpc_pool(self) -> List[str]:
        if self._fpc_files is None:
            if os.path.isdir(self.fpc_dir):
                self._fpc_files = sorted(
                    os.path.join(self.fpc_dir, f) for f in os.listdir(self.fpc_dir)
                    if f.endswith(".npz")
                )
            else:
                self._fpc_files = []
        return self._fpc_files

    def _fpc_sample(self, rng: np.random.Generator) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Реальний план + ручна маска, з легкою аугментацією.

        Свою degrade() тут НЕ застосовуємо: ці зображення вже реальні (скріншоти
        оголошень, скани), і накладати згори синтетичні спотворення означало б
        вчити модель на подвійному артефакті."""
        import cv2

        pool = self._fpc_pool()
        if not pool:
            return None
        path = pool[int(rng.integers(len(pool)))]
        try:
            data = np.load(path)
            rgb, mask = data["image"], data["mask"]
        except Exception:
            return None

        # випадковий кроп 70-100% — модель має бачити й фрагменти плану
        h, w = mask.shape[:2]
        keep = float(rng.uniform(0.70, 1.0))
        ch, cw = max(32, int(h * keep)), max(32, int(w * keep))
        y0 = int(rng.integers(0, max(1, h - ch + 1)))
        x0 = int(rng.integers(0, max(1, w - cw + 1)))
        rgb, mask = rgb[y0:y0 + ch, x0:x0 + cw], mask[y0:y0 + ch, x0:x0 + cw]

        scale = self.size / max(rgb.shape[:2])
        nw, nh = max(1, int(rgb.shape[1] * scale)), max(1, int(rgb.shape[0] * scale))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        pad_v, pad_h = self.size - nh, self.size - nw
        rgb = cv2.copyMakeBorder(rgb, pad_v // 2, pad_v - pad_v // 2, pad_h // 2,
                                 pad_h - pad_h // 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        mask = cv2.copyMakeBorder(mask, pad_v // 2, pad_v - pad_v // 2, pad_h // 2,
                                  pad_h - pad_h // 2, cv2.BORDER_CONSTANT, value=0)

        if rng.random() < 0.5:                       # м'яка фотометрія
            alpha, beta = float(rng.uniform(0.85, 1.15)), float(rng.uniform(-18, 18))
            rgb = np.clip(rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return rgb, mask

    def _real_pool(self) -> list:
        """Ліниво (у кожному воркері окремо) — pickle на 60 МБ не варто
        серіалізувати між процесами."""
        if self._layouts is None:
            from ml.floorplan.datasets import load_layouts

            everything = load_layouts()
            if not everything:
                self._layouts = []
            elif self.real_split == "val":
                self._layouts = everything[-self.val_holdout:]
            else:
                self._layouts = everything[:-self.val_holdout] or everything
        return self._layouts

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seed = self.seed_base + index
        # Частину прикладів лишаємо «чистими» (скріншот/PDF без деградацій) —
        # такі входи в проді теж бувають, і на них модель має бути ідеальною.
        clean = (seed % 100) < int(self.clean_ratio * 100)
        rng = np.random.default_rng(seed)
        # Пріоритет джерел: справжні плани → справжня геометрія → BSP.
        fpc = self._fpc_sample(rng) if (self.fpc_ratio > 0 and rng.random() < self.fpc_ratio) else None
        pool = self._real_pool() if (fpc is None and self.real_ratio > 0) else []
        use_real = bool(pool) and rng.random() < self.real_ratio
        if fpc is not None:
            rgb, mask = fpc
        elif use_real:
            from ml.floorplan.geom_render import render_layout as render_real

            layout = pool[int(rng.integers(len(pool)))]
            style = synth._rand_style(rng)
            style.px_per_m = float(rng.uniform(28.0, 75.0))
            style.margin_px = int(rng.integers(60, 150))
            try:
                raw_img, raw_mask = render_real(layout, rng, style)
                rgb, mask = synth.finalize(raw_img, raw_mask, rng,
                                           out_size=self.size, clean=clean)
            except Exception:
                rgb, mask, _spec = synth.make_sample(seed, out_size=self.size, clean=clean)
        else:
            rgb, mask, _spec = synth.make_sample(seed, out_size=self.size, clean=clean)
        x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255.0
        y = torch.from_numpy(np.ascontiguousarray(mask)).long()
        if seed % 2 == 0:                      # горизонтальне дзеркало
            x, y = torch.flip(x, [2]), torch.flip(y, [1])
        k = (seed // 7) % 4                    # повороти на 90°
        if k:
            x, y = torch.rot90(x, k, [1, 2]), torch.rot90(y, k, [0, 1])
        return x.contiguous(), y.contiguous()


def _class_weights() -> List[float]:
    """Стіни ~10% пікселів, двері/вікна <1%. Без ваг мережа впевнено вчиться
    передбачати самий лише фон."""
    return [0.4, 1.0, 3.0, 3.0]


def amp_is_safe(model: torch.nn.Module, device: str, batch: int, size: int) -> bool:
    """Чи можна довіряти fp16 на ЦІЙ машині при цьому батчі.

    Реальна знахідка: на GTX 1650 (sm_75) з cuDNN 9.1 згортки у fp16 при batch≥3
    мовчки повертають NaN — вхід скінченний, ваги скінченні, вихід NaN. Помилки
    немає, є лише loss=nan до кінця навчання. fp32 і fp16 з вимкненим cuDNN
    рахують правильно. Тому не вгадуємо, а перевіряємо одним прогоном."""
    if device != "cuda":
        return False
    # ОБОВ'ЯЗКОВО в eval-режимі: у train-режимі forward оновлює running-статистики
    # BatchNorm, і якщо проба сама повернула NaN (а саме це ми й перевіряємо), ці
    # NaN назавжди осідають у running_mean/var. Модель тоді нормально ТРЕНУЄТЬСЯ,
    # але в eval видає самі NaN — тиждень можна шукати.
    was_training = model.training
    model.eval()
    try:
        probe = torch.rand(max(1, batch), 3, size, size, device=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(probe)
        ok = bool(torch.isfinite(out).all())
        del probe, out
        torch.cuda.empty_cache()
        return ok
    except Exception:
        return False
    finally:
        model.train(was_training)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: str,
             amp: bool) -> Tuple[float, List[float]]:
    model.eval()
    losses: List[float] = []
    ious: List[List[float]] = []
    criterion = SegLoss(_class_weights())
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(x)
                loss = criterion(logits.float(), y)
            losses.append(float(loss))
            ious.append(per_class_iou(logits.float(), y))
    model.train()
    mean_iou = [float(np.nanmean([r[c] for r in ious])) for c in range(NUM_CLASSES)] if ious else []
    return (float(np.mean(losses)) if losses else float("nan")), mean_iou


def train(steps: int = 8000, batch: int = 4, size: int = 512, lr: float = 3e-4,
          workers: int = 4, eval_every: int = 500, resume: str = "",
          out_name: str = "floorplan_seg") -> str:
    os.makedirs(CKPT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device} batch={batch} size={size} steps={steps}", flush=True)

    train_ds = SynthDataset(size, steps * batch, seed_base=10_000_000)
    # Валідація на ВІДКЛАДЕНИХ реальних квартирах: перевіряємо узагальнення
    # на топологію, якої в навчанні не було, а не запамʼятовування генератора.
    val_ds = SynthDataset(size, 64, seed_base=1_000, real_ratio=0.8, real_split="val",
                          fpc_ratio=0.0)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=False, num_workers=workers,
                              pin_memory=(device == "cuda"), drop_last=True,
                              persistent_workers=workers > 0)
    val_loader = DataLoader(val_ds, batch_size=max(1, batch // 2), shuffle=False,
                            num_workers=max(0, workers // 2))

    model = FloorPlanUNet(NUM_CLASSES, pretrained=True).to(device)
    amp = amp_is_safe(model, device, batch, size)
    if device == "cuda" and not amp:
        print("[train] fp16 на цій GPU дає NaN — вмикаю fp32 (повільніше, але правильно).",
              flush=True)
    if resume and os.path.exists(resume):
        model.load_state_dict(torch.load(resume, map_location=device))
        print(f"[train] resumed from {resume}", flush=True)
    model.train()

    criterion = SegLoss(_class_weights()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=max(1, steps), pct_start=0.15,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_wall_iou = -1.0
    best_path = os.path.join(CKPT_DIR, f"{out_name}_best.pt")
    last_path = os.path.join(CKPT_DIR, f"{out_name}_last.pt")
    t0 = time.time()
    running: List[float] = []
    skipped = 0

    for step, (x, y) in enumerate(train_loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits = model(x)
            loss = criterion(logits.float(), y)

        # fp16 у перших ітераціях цілком легально дає inf/NaN, поки GradScaler
        # добирає масштаб. Такий крок треба ПРОПУСТИТИ, а не пускати в backward:
        # інакше NaN розтікається по вагах і навчання мовчки вмирає, показуючи
        # loss=nan до самого кінця.
        if not torch.isfinite(loss):
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            if step <= steps:
                scheduler.step()
            if skipped in (25, 100, 400):
                print(f"[warn] пропущено {skipped} кроків із не-скінченною втратою", flush=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step <= steps:
            scheduler.step()
        running.append(float(loss))

        if step % 50 == 0:
            speed = step * batch / max(1e-6, time.time() - t0)
            window = [v for v in running[-50:] if math.isfinite(v)]
            shown = np.mean(window) if window else float("nan")
            print(f"[{step}/{steps}] loss={shown:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                  f"{speed:.1f} img/s" + (f" skipped={skipped}" if skipped else ""), flush=True)
        if step % eval_every == 0 or step == steps:
            vloss, ious = evaluate(model, val_loader, device, amp)
            names = ("bg", "wall", "door", "window")
            pretty = "  ".join(f"{n}={v:.3f}" for n, v in zip(names, ious))
            print(f"[eval @{step}] val_loss={vloss:.4f}  IoU: {pretty}", flush=True)
            torch.save(model.state_dict(), last_path)
            if len(ious) > 1 and ious[1] > best_wall_iou:
                best_wall_iou = ious[1]
                torch.save(model.state_dict(), best_path)
                print(f"[eval @{step}] new best wall IoU {best_wall_iou:.4f} -> {best_path}",
                      flush=True)
        if step >= steps:
            break

    print(f"[train] done in {(time.time() - t0) / 60:.1f} min, best wall IoU {best_wall_iou:.4f}",
          flush=True)
    return best_path if os.path.exists(best_path) else last_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--out-name", type=str, default="floorplan_seg")
    args = parser.parse_args()
    train(steps=args.steps, batch=args.batch, size=args.size, lr=args.lr,
          workers=args.workers, eval_every=args.eval_every, resume=args.resume,
          out_name=args.out_name)


if __name__ == "__main__":
    main()
