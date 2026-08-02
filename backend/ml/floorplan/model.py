"""U-Net із енкодером ResNet-18 для сегментації планів.

Чому саме так:
  • ResNet-18 (ImageNet) — енкодер, попередньо навчений на ФОТО. Наші входи —
    фотографії паперу з тінями й шумом, і предтренована згортка дає стійкість,
    якої мережа «з нуля» на синтетиці не набуває.
  • U-Net-декодер зі skip-зв'язками: межа стіни має бути піксельно точною, бо з
    неї далі рахується товщина, а з товщини — розмір виробу.
  • 4 класи (фон/стіна/двері/вікно) при 512×512 і батчі 4 у fp16 займають
    ~2.2 ГБ — влазить у GTX 1650 (4 ГБ), яка є на машині розробника.
  • Прод НЕ бачить torch узагалі: модель експортується в ONNX і крутиться на
    onnxruntime CPU у 4 ГБ VPS (~1.5 с на план у 1-2 потоки).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 4
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor]) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if skip is not None:
            # Розміри можуть розійтись на 1 px при непарних входах.
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class FloorPlanUNet(nn.Module):
    """ResNet-18 encoder + U-Net decoder → логіти (B, 4, H, W)."""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)   # /2,  64
        self.pool = net.maxpool                                    # /4
        self.layer1 = net.layer1                                   # /4,  64
        self.layer2 = net.layer2                                   # /8,  128
        self.layer3 = net.layer3                                   # /16, 256
        self.layer4 = net.layer4                                   # /32, 512

        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 48)
        self.up0 = UpBlock(48, 0, 32)
        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)          # /2
        s1 = self.layer1(self.pool(s0))   # /4
        s2 = self.layer2(s1)       # /8
        s3 = self.layer3(s2)       # /16
        s4 = self.layer4(s3)       # /32
        d = self.up4(s4, s3)       # /16
        d = self.up3(d, s2)        # /8
        d = self.up2(d, s1)        # /4
        d = self.up1(d, s0)        # /2
        d = self.up0(d, None)      # /1
        return self.head(d)


class NormalizedModel(nn.Module):
    """Обгортка для ONNX: приймає uint8-подібний float 0..1 і сама нормалізує.

    Так рантайм не мусить знати про статистики ImageNet — менше шансів, що
    прод і тренування розійдуться в препроцесингу (класичне джерело «на
    ноутбуці працює, на сервері ні»)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.model(x)


# ── Втрати ───────────────────────────────────────────────────────────────────
def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """М'який Dice по класах. Потрібен поряд із CE, бо стіни займають ~10%
    пікселів, а двері/вікна — менше 1%: чиста крос-ентропія вчиться передбачати
    фон і формально має гарний loss."""
    num_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=1)
    onehot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).to(probs.dtype)
    dims = (0, 2, 3)
    intersection = torch.sum(probs * onehot, dims)
    cardinality = torch.sum(probs + onehot, dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


class SegLoss(nn.Module):
    def __init__(self, class_weights: Optional[List[float]] = None, dice_weight: float = 0.5):
        super().__init__()
        weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ваги класів мусять жити на тому ж пристрої, що й логіти. Покладатись на
        # зовнішній .to(device) крихко: у оцінювальному циклі критерій легко
        # створити заново й забути перенести.
        if self.ce.weight is not None and self.ce.weight.device != logits.device:
            self.ce.weight = self.ce.weight.to(logits.device)
        return self.ce(logits, target.long()) + self.dice_weight * dice_loss(logits, target)


@torch.no_grad()
def per_class_iou(logits: torch.Tensor, target: torch.Tensor,
                  num_classes: int = NUM_CLASSES) -> List[float]:
    pred = logits.argmax(dim=1)
    out: List[float] = []
    for c in range(num_classes):
        p, t = (pred == c), (target == c)
        union = float((p | t).sum())
        out.append(float((p & t).sum()) / union if union > 0 else float("nan"))
    return out
