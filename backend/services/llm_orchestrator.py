"""
llm_orchestrator.py — промт користувача → структурований spec для procedural_generator.

Режим «опиши світ» (задача #5). ДВА шляхи:
  1) Якщо є ANTHROPIC_API_KEY + пакет anthropic → Claude перетворює вільний опис на
     валідний spec + додає друкарські обмеження (orchestration + покращення промту).
  2) Інакше (за замовчуванням, працює ОДРАЗУ без ключа) → детермінований rule-based
     парсер ключових слів (укр+англ). Пайплайн повний; AI — апгрейд, коли додати ключ.

prompt_to_spec(prompt, size_mm) -> (spec: dict, source: "llm"|"rules").
"""
from __future__ import annotations

import os
import zlib
from typing import Any

_SHAPES = ["mountain", "island", "valley", "plateau", "ridges", "crater", "rolling"]

# Ключові слова (укр + англ) → форма
_SHAPE_KW = {
    "mountain": ["гор", "вершин", "пік", "mountain", "peak", "alp", "хребет гір"],
    "island": ["остр", "island", "архіпелаг", "atoll", "море навколо", "берег"],
    "valley": ["долин", "каньйон", "ущелин", "valley", "canyon", "gorge", "ріка"],
    "plateau": ["плато", "меза", "plateau", "mesa", "столова гора", "tableland"],
    "ridges": ["хреб", "хвил", "дюн", "ridge", "wave", "dune", "ripple", "брижі"],
    "crater": ["кратер", "вулкан", "crater", "volcano", "caldera", "метеорит"],
    "rolling": ["пагорб", "горб", "rolling", "hill", "поле", "степ", "prairie"],
}


def _seed_from(prompt: str) -> int:
    return zlib.crc32((prompt or "world").encode("utf-8")) & 0x7FFFFFFF


def _rule_based_spec(prompt: str, size_mm: float) -> dict:
    p = (prompt or "").lower()
    shape = "rolling"
    best = 0
    for sh, kws in _SHAPE_KW.items():
        hits = sum(1 for k in kws if k in p)
        if hits > best:
            best, shape = hits, sh
    # висота
    max_h = 18.0
    if any(k in p for k in ["висок", "гострий", "крут", "tall", "high", "steep", "dramatic", "епічн", "epic"]):
        max_h = 30.0
    elif any(k in p for k in ["низьк", "плоск", "пологий", "flat", "low", "gentle", "м'як"]):
        max_h = 8.0
    # шорсткість/деталізація
    rough = 0.5
    if any(k in p for k in ["детал", "гострий", "скеляст", "rough", "jagged", "rocky", "rugged", "хаотичн"]):
        rough = 0.85
    elif any(k in p for k in ["гладк", "плавн", "smooth", "soft", "rolling", "м'як"]):
        rough = 0.25
    return {
        "shape": shape,
        "width_mm": float(size_mm),
        "max_height_mm": max_h,
        "base_thickness_mm": 3.0,
        "roughness": rough,
        "seed": _seed_from(prompt),
        "label": (prompt or "")[:40],
    }


def _llm_spec(prompt: str, size_mm: float) -> dict | None:
    """Claude → spec. None якщо ключа/пакета нема або помилка (→ fallback на rules)."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic  # type: ignore
    except Exception:
        return None
    try:
        client = anthropic.Anthropic()
        sys_prompt = (
            "Ти — інженер 3D-друку. Перетвори опис користувача на ДРУКОВАНИЙ рельєфний світ. "
            "Поверни ЛИШЕ JSON зі ключами: shape (one of: "
            + ", ".join(_SHAPES) + "), max_height_mm (2-40, друковано), roughness (0-1), "
            "base_thickness_mm (1-8). Без пояснень, лише JSON."
        )
        msg = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=300,
            system=sys_prompt,
            messages=[{"role": "user", "content": prompt[:2000]}],
        )
        import json
        txt = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        start, end = txt.find("{"), txt.rfind("}")
        if start < 0 or end < 0:
            return None
        spec = json.loads(txt[start:end + 1])
        spec["width_mm"] = float(size_mm)
        spec["seed"] = _seed_from(prompt)
        spec["label"] = (prompt or "")[:40]
        return spec
    except Exception as exc:  # noqa: BLE001
        print(f"[LLM] orchestrate failed (fallback to rules): {exc}", flush=True)
        return None


def prompt_to_spec(prompt: str, size_mm: float = 120.0) -> tuple[dict, str]:
    """Головний вхід: промт → (spec, джерело). Завжди повертає валідний spec."""
    spec = _llm_spec(prompt, size_mm)
    if spec is not None:
        return spec, "llm"
    return _rule_based_spec(prompt, size_mm), "rules"
