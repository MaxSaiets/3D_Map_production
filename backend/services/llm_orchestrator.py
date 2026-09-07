"""
llm_orchestrator.py — промт користувача → структурований spec для procedural_generator.

Режим «опиши світ». ДВА шляхи:
  1) Якщо є ANTHROPIC_API_KEY + пакет anthropic → Claude перетворює вільний опис на
     валідний spec + додає друкарські обмеження.
  2) Інакше (за замовчуванням, працює ОДРАЗУ без ключа) → детермінований rule-based
     парсер ключових слів (укр+англ). Пайплайн повний; AI — апгрейд.

⭐ПЕРЕПИСАНО 2026-09-08 разом із procedural_generator (власник: «створюється херня»).
Що було не так:
  * Форм було 7, слів — по 5–7 на форму, тож більшість описів падали у дефолт
    «rolling»/«mountain» і всі промти давали один і той самий результат.
  * Не було форм volcano/archipelago, хоча приклади на сторінці прямо їх називають
    («Острів-вулкан у морі»).
  * Не розбиралися МОДИФІКАТОРИ («засніжені», «глибокий», «гострі»), тож висота й
    шорсткість були майже завжди дефолтні.
  * Скоринг був «перше слово перемагає»: «острів-вулкан» ставав островом, бо
    island перевірявся раніше за volcano. Тепер ваги + пріоритет специфічніших форм.

prompt_to_spec(prompt, size_mm) -> (spec: dict, source: "llm"|"rules").
"""
from __future__ import annotations

import os
import zlib
from typing import Any

SHAPES = [
    "mountain", "island", "valley", "plateau",
    "ridges", "crater", "rolling", "volcano", "archipelago",
]

# Ключові слова (укр + англ + рос-суржик, який реально пишуть) → форма.
# Вага > 1 у специфічних слів, щоб «острів-вулкан» дав вулкан, а не просто острів.
_SHAPE_KW: dict[str, list[tuple[str, float]]] = {
    "mountain": [
        ("гор", 1.0), ("вершин", 1.2), ("пік", 1.2), ("хребет", 1.4), ("альп", 1.3),
        ("скел", 1.0), ("mountain", 1.2), ("peak", 1.2), ("alp", 1.2), ("ridgeline", 1.2),
        ("summit", 1.2), ("масив", 0.8), ("карпат", 1.3), ("гімала", 1.3), ("ельбрус", 1.3),
    ],
    "volcano": [
        ("вулкан", 2.5), ("volcano", 2.5), ("лав", 1.6), ("lava", 1.6), ("магм", 1.6),
        ("кальдер", 2.0), ("caldera", 2.0), ("виверж", 1.8), ("eruption", 1.8), ("етна", 2.0),
    ],
    "crater": [
        ("кратер", 2.2), ("crater", 2.2), ("метеорит", 2.0), ("meteor", 2.0),
        ("impact", 1.6), ("місяц", 1.4), ("moon", 1.2), ("lunar", 1.6), ("вирв", 1.6),
    ],
    "archipelago": [
        ("архіпелаг", 2.5), ("archipelago", 2.5), ("острови", 2.0), ("islands", 2.0),
        ("atoll", 1.6), ("атол", 1.6), ("розсип остров", 2.0),
    ],
    "island": [
        ("остр", 1.4), ("island", 1.4), ("берег", 1.0), ("узбереж", 1.2), ("coast", 1.0),
        ("море навколо", 1.5), ("лагун", 1.2), ("тропічн", 1.0), ("пляж", 1.0), ("beach", 1.0),
    ],
    "valley": [
        ("каньйон", 2.0), ("canyon", 2.0), ("ущелин", 1.8), ("gorge", 1.8), ("долин", 1.4),
        ("valley", 1.4), ("рік", 1.2), ("річк", 1.2), ("river", 1.2), ("русл", 1.4),
        ("фьорд", 1.6), ("fjord", 1.6), ("яр", 1.0),
    ],
    "plateau": [
        ("плато", 2.0), ("plateau", 2.0), ("меза", 1.8), ("mesa", 1.8),
        ("столова гора", 2.0), ("tableland", 1.6), ("каньйон-плато", 1.5),
    ],
    "ridges": [
        ("дюн", 2.0), ("dune", 2.0), ("пустел", 1.6), ("desert", 1.6), ("бархан", 2.0),
        ("хвил", 1.2), ("wave", 1.2), ("ripple", 1.2), ("брижі", 1.2), ("сахар", 1.6),
        ("піщан", 1.2), ("sand", 1.2),
    ],
    "rolling": [
        ("пагорб", 1.6), ("горб", 1.4), ("hill", 1.4), ("rolling", 1.4), ("поле", 1.0),
        ("степ", 1.2), ("prairie", 1.2), ("лук", 1.0), ("meadow", 1.2), ("плавн", 1.0),
        ("тоскан", 1.4),
    ],
}

_TALL = ["висок", "гострий", "крут", "епічн", "величн", "стрімк", "могутн",
         "tall", "high", "steep", "dramatic", "epic", "towering", "jagged", "засніжен", "snow"]
_LOW = ["низьк", "плоск", "пологий", "м'як", "мʼяк", "спокійн", "flat", "low", "gentle", "soft", "calm"]
_ROUGH = ["детал", "гострий", "скеляст", "хаотичн", "дик", "суворий", "rough", "jagged",
          "rocky", "rugged", "wild", "harsh", "eroded", "ерод"]
_SMOOTH = ["гладк", "плавн", "округл", "smooth", "soft", "gentle", "rounded", "мінімаліст", "minimal"]


def _seed_from(prompt: str) -> int:
    return zlib.crc32((prompt or "world").encode("utf-8")) & 0x7FFFFFFF


def _score_shapes(p: str) -> dict[str, float]:
    return {sh: sum(w for kw, w in kws if kw in p) for sh, kws in _SHAPE_KW.items()}


def _rule_based_spec(prompt: str, size_mm: float, seed_extra: int = 0) -> dict:
    p = (prompt or "").lower()
    scores = _score_shapes(p)
    shape, best = "rolling", 0.0
    for sh, sc in scores.items():
        if sc > best:
            best, shape = sc, sh

    # Висота: базово залежить від форми (вулкан/гори високі, дюни низькі), далі
    # коригується модифікаторами з опису.
    base_h = {"mountain": 26.0, "volcano": 28.0, "plateau": 20.0, "crater": 16.0,
              "island": 18.0, "archipelago": 15.0, "valley": 22.0, "ridges": 12.0,
              "rolling": 12.0}.get(shape, 18.0)
    if any(k in p for k in _TALL):
        base_h *= 1.35
    if any(k in p for k in _LOW):
        base_h *= 0.6
    max_h = max(4.0, min(38.0, base_h))

    rough = 0.5
    if any(k in p for k in _ROUGH):
        rough = 0.85
    elif any(k in p for k in _SMOOTH):
        rough = 0.22

    # Ерозія: «дикий/еродований» рельєф — сильніша; «мінімалістичний» — слабша.
    erosion = 0.6
    if any(k in p for k in ("ерод", "eroded", "вивітр", "старі гори", "ancient")):
        erosion = 0.9
    elif any(k in p for k in _SMOOTH):
        erosion = 0.35

    return {
        "shape": shape,
        "width_mm": float(size_mm),
        "max_height_mm": round(max_h, 1),
        "base_thickness_mm": 3.0,
        "roughness": rough,
        "erosion": erosion,
        "seed": (_seed_from(prompt) + int(seed_extra)) & 0x7FFFFFFF,
        "label": (prompt or "")[:40],
    }


def _llm_spec(prompt: str, size_mm: float, seed_extra: int = 0) -> dict | None:
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
            "Ти — інженер 3D-друку. Перетвори опис користувача на ДРУКОВАНИЙ рельєфний спек. "
            "Поверни ЛИШЕ JSON з ключами: shape (одне з: " + ", ".join(SHAPES) + "), "
            "max_height_mm (4-38), roughness (0-1), erosion (0-1), base_thickness_mm (1-8). "
            "Без пояснень, лише JSON."
        )
        msg = client.messages.create(
            model=os.getenv("WORLDS_LLM_MODEL", "claude-opus-4-8"),
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
        spec["seed"] = (_seed_from(prompt) + int(seed_extra)) & 0x7FFFFFFF
        spec["label"] = (prompt or "")[:40]
        return spec
    except Exception as exc:  # noqa: BLE001
        print(f"[LLM] orchestrate failed (fallback to rules): {exc}", flush=True)
        return None


def prompt_to_spec(
    prompt: str,
    size_mm: float = 120.0,
    shape_override: str | None = None,
    seed_extra: int = 0,
) -> tuple[dict, str]:
    """Головний вхід: промт → (spec, джерело). Завжди повертає валідний spec.

    shape_override — форма, ЯВНО обрана користувачем у UI (тоді парсер не вгадує).
    seed_extra — «інший варіант» для того самого опису (кнопка перегенерації).
    """
    if shape_override and shape_override in SHAPES:
        spec = _rule_based_spec(prompt, size_mm, seed_extra)
        spec["shape"] = shape_override
        return spec, "user"
    spec = _llm_spec(prompt, size_mm, seed_extra)
    if spec is not None:
        return spec, "llm"
    return _rule_based_spec(prompt, size_mm, seed_extra), "rules"
