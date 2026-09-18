# -*- coding: utf-8 -*-
"""Агент режиму «Гори»: вільний опис → перевірений spec + пояснення «як я зрозумів».

Принцип «без помилок»: агент НІКОЛИ не запускає генерацію сам. Він повертає
  { spec, understood: [рядки укр.], warnings: [...], questions: [...], confidence }
і фронт показує це людині для підтвердження; усі числа проходять normalize_spec(), тож
неможливо отримати нефізичну модель (занадто велика, без місця, фігурка поза плитою).

Два шляхи: Claude (ANTHROPIC_API_KEY) → JSON за схемою, далі та сама валідація; інакше —
детермінований парсер (укр/англ/рос): місце (пресети → Nominatim), розмір, висота, ободок,
боки, фігурки, місце фігурки (вершина/схил/стіна), текстура.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional

import numpy as np
import requests

from . import presets as P
from .figures import library as figure_library

LIMITS = {"size_mm": (60.0, 400.0), "height_mm": (20.0, 300.0), "area_km": (1.0, 60.0), "frame_w": (0.0, 30.0), "frame_h": (0.0, 60.0), "fig_h": (6.0, 50.0)}
FRAME_STYLES = ("none", "flat", "rounded")
SIDE_STYLES = ("vertical", "rock", "slope")
_nominatim_last = [0.0]


def default_spec() -> dict:
    return {"place": None, "area_km": None, "size_mm": 200.0, "height_mm": None, "frame": {"style": "rounded", "width_mm": 10.0, "height_mm": 25.0},
            "sides": "slope", "base_mm": 3.0, "figures": [], "texture": "satellite", "bed_mm": 256.0, "notes": ""}


def _clamp(v, lo, hi):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return float(min(max(v, lo), hi))


def normalize_spec(spec: dict) -> tuple[dict, list[str]]:
    """Приводить будь-який (навіть частковий/кривий) spec до безпечного. → (spec, попередження)."""
    warn = []
    s = default_spec(); s.update({k: v for k, v in (spec or {}).items() if v is not None})
    pl = s.get("place")
    if isinstance(pl, dict) and "lat" in pl and "lon" in pl:
        try:
            lat, lon = float(pl["lat"]), float(pl["lon"])
            if not (-85 <= lat <= 85 and -180 <= lon <= 180):
                raise ValueError
            s["place"] = {"name": str(pl.get("name") or f"{lat:.4f}, {lon:.4f}")[:80], "lat": lat, "lon": lon, "source": str(pl.get("source") or "map")}
        except (TypeError, ValueError):
            s["place"] = None; warn.append("Координати місця некоректні — оберіть точку на мапі.")
    else:
        s["place"] = None
    size = _clamp(s.get("size_mm"), *LIMITS["size_mm"])
    if size is None:
        size = 200.0
    if float(spec.get("size_mm") or size) != size:
        warn.append(f"Розмір обмежено до {size:.0f} мм (діапазон {LIMITS['size_mm'][0]:.0f}–{LIMITS['size_mm'][1]:.0f}).")
    s["size_mm"] = round(size, 1)
    h = s.get("height_mm")
    if h is not None:
        hc = _clamp(h, LIMITS["height_mm"][0], min(LIMITS["height_mm"][1], size * 1.0))
        if hc is not None and abs(hc - float(h)) > 0.5:
            warn.append(f"Висоту обмежено до {hc:.0f} мм (не більше за ширину плити і не вище 300 мм).")
        s["height_mm"] = None if hc is None else round(hc, 1)
    ak = s.get("area_km")
    s["area_km"] = None if ak is None else _clamp(ak, *LIMITS["area_km"])
    fr = s.get("frame") if isinstance(s.get("frame"), dict) else {}
    style = str(fr.get("style") or "rounded").lower()
    if style not in FRAME_STYLES:
        style = "rounded"
    fw = _clamp(fr.get("width_mm", 10.0), *LIMITS["frame_w"]) or 0.0; fh = _clamp(fr.get("height_mm", 25.0), *LIMITS["frame_h"]) or 0.0
    if style == "none":
        fw = 0.0
    if fw > 0 and fw > size * 0.12:
        fw = round(size * 0.12, 1); warn.append(f"Ободок звужено до {fw} мм, щоб лишилось місце для рельєфу.")
    if fw > 0 and fh < 3:
        fh = 3.0
    s["frame"] = {"style": style, "width_mm": round(fw, 1), "height_mm": round(fh, 1)}
    sides = str(s.get("sides") or "slope").lower()
    s["sides"] = sides if sides in SIDE_STYLES else "slope"
    s["base_mm"] = _clamp(s.get("base_mm", 3.0), 2.0, 10.0) or 3.0
    s["texture"] = "satellite" if str(s.get("texture") or "satellite") != "none" else "none"
    s["bed_mm"] = _clamp(s.get("bed_mm", 256.0), 100.0, 600.0) or 256.0
    lib = {f["id"]: f for f in figure_library()}
    figs = []
    for f in (s.get("figures") or [])[:6]:
        if not isinstance(f, dict):
            continue
        fid = str(f.get("id") or "")
        if fid not in lib:
            warn.append(f"Фігурки «{fid}» немає в бібліотеці — пропущено."); continue
        where = str(f.get("where") or ("steepest" if lib[fid]["kind"] == "climbing" else "summit"))
        if where not in ("summit", "steepest", "point", "slope"):
            where = "summit"
        # типова висота фігурки залежить від плити: ~8 % ширини (Матергорн 400 мм → 17 мм людина, 20 мм хатина)
        dflt = lib[fid]["default_height_mm"] * float(np.clip(size / 200.0, 0.5, 2.0))
        fh_ = _clamp(f.get("height_mm") or dflt, lib[fid]["min_height_mm"], lib[fid]["max_height_mm"])
        item = {"id": fid, "where": where, "height_mm": round(fh_, 1)}
        if where == "point":
            fx, fy = _clamp(f.get("fx", 0.5), 0.0, 1.0), _clamp(f.get("fy", 0.5), 0.0, 1.0)
            item["fx"], item["fy"] = (0.5 if fx is None else fx), (0.5 if fy is None else fy)
        figs.append(item)
    s["figures"] = figs
    s["notes"] = str(s.get("notes") or "")[:300]
    return s, warn


# ── геокодер ─────────────────────────────────────────────────────────────────
def geocode(name: str) -> Optional[dict]:
    """Пресети → Nominatim (1 запит/с, лише природні обʼєкти/вершини)."""
    p = P.find_by_alias(name)
    if p:
        return {"name": p["name"]["uk"], "lat": p["lat"], "lon": p["lon"], "source": "preset", "preset_id": p["id"], "area_km": p["area_km"], "elev": p["elev"]}
    q = name.strip()
    if len(q) < 3:
        return None
    dt = time.time() - _nominatim_last[0]
    if dt < 1.1:
        time.sleep(1.1 - dt)
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search", params={"q": q, "format": "jsonv2", "limit": 5, "accept-language": "uk,en"},
                         headers={"User-Agent": "monadruk-mountains/1.0 (hello@monadruk.com)"}, timeout=12)
        _nominatim_last[0] = time.time()
        if r.status_code != 200:
            return None
        rows = r.json()
    except Exception as exc:  # noqa: BLE001
        print(f"[AGENT] nominatim failed: {exc}", flush=True); return None
    pref = [x for x in rows if x.get("category") == "natural" and x.get("type") in ("peak", "volcano", "ridge", "mountain_range", "saddle", "hill")] or rows
    if not pref:
        return None
    x = pref[0]
    return {"name": str(x.get("name") or x.get("display_name", "")).split(",")[0][:80], "lat": float(x["lat"]), "lon": float(x["lon"]), "source": "geocode",
            "type": x.get("type"), "display": str(x.get("display_name", ""))[:120]}


# ── парсер правил ────────────────────────────────────────────────────────────
_NUM = r"(\d+(?:[.,]\d+)?)"
_CM = re.compile(_NUM + r"\s*(см|cm|сантиметр)", re.I)
_MM = re.compile(_NUM + r"\s*(мм|mm|міліметр)", re.I)
_KM = re.compile(_NUM + r"\s*(км|km|кілометр)", re.I)
_X = re.compile(_NUM + r"\s*[x×хна]\s*" + _NUM + r"\s*(см|cm|мм|mm)?", re.I)
_PLACE_RE = re.compile(r"(?:гор[аиу]|mount|mt\.?|peak|вершин[аи]|вулкан|volcano|масив|хребет|mountain)\s+([A-Za-zА-Яа-яЇїІіЄєҐґ'’\-\s]{3,40})", re.I)


def _num(m):
    return float(m.group(1).replace(",", "."))


def rule_parse(text: str) -> tuple[dict, list[str]]:
    t = text.strip(); tl = t.lower()
    s = default_spec(); notes = []
    # розмір / висота
    for m in _X.finditer(t):
        a, b = _num(m), float(m.group(2).replace(",", ".")); unit = (m.group(3) or "см").lower()
        k = 10.0 if unit.startswith(("см", "cm")) else 1.0
        s["size_mm"] = max(a, b) * k; break
    cms = [(_num(m), m.start(), m.end()) for m in _CM.finditer(t)]; mms = [(_num(m), m.start(), m.end()) for m in _MM.finditer(t)]
    vals = sorted([(v * 10, a_, b_) for v, a_, b_ in cms] + mms, key=lambda x: x[1])
    size_set = bool(_X.search(t))
    for v, a_, b_ in vals:
        # контекст = поточне речення/клауза (до коми/крапки/«і»/«та»), окремо ДО і ПІСЛЯ числа
        before = re.split(r"[,.;]| і | та | and |with| з ", tl[:a_])[-1]
        after = re.split(r"[,.;]| і | та | and |with| з ", tl[b_:])[0]
        KW = {"h": r"висот|заввишк|висок|height|tall|вгору", "f": r"ободок|обідок|рамк|бортик|frame|rim|border",
              "p": r"люд|фігур|скелелаз|альпініст|figure|person|climber|хатин|будин|hut|people", "s": r"розмір|плит|стор|size|plate|wide|width|широк"}
        # найближче ключове слово ПЕРЕД числом вирішує, що це за число
        last = max(((m.end(), k) for k, rx in KW.items() for m in re.finditer(rx, before)), default=(None, None))[1]
        if last == "h":
            s["height_mm"] = v
        elif last == "f":
            s["frame"]["width_mm"] = v if v <= 30 else s["frame"]["width_mm"]
        elif last == "p":
            s["_fig_h"] = v
        elif last == "s" or not size_set:
            s["size_mm"] = v; size_set = True
        elif re.search(KW["h"], after[:14]):
            s["height_mm"] = v
        elif re.search(KW["f"], after[:14]):
            s["frame"]["width_mm"] = v if v <= 30 else s["frame"]["width_mm"]
        elif re.search(KW["p"], after[:20]):
            s["_fig_h"] = v
        elif s.get("height_mm") is None and v < s["size_mm"]:
            s["height_mm"] = v
    km = _KM.search(t)
    if km:
        s["area_km"] = _num(km)
    # ободок
    if re.search(r"без (ободк|обідк|рамк|бортик|основ|низ)|no (frame|rim|border|base)|frameless", tl):
        s["frame"]["style"] = "none"
    elif re.search(r"заокругл|округл|rounded|скругл", tl):
        s["frame"]["style"] = "rounded"
    elif re.search(r"пласк|плоск|фаск|flat|chamfer|прям", tl) and re.search(r"ободок|обідок|рамк|бортик|frame|rim", tl):
        s["frame"]["style"] = "flat"
    m = re.search(r"(?:ободок|обідок|рамк\w*|бортик|frame|rim)\D{0,20}?" + _NUM + r"\s*(мм|mm|см|cm)?\s*(?:заввишки|висот|high|tall)", tl)
    if m:
        s["frame"]["height_mm"] = _num(m) * (10 if (m.group(2) or "").startswith(("см", "cm")) else 1)
    # боки
    if re.search(r"похил|нахил|скіс|slope|sloped|спуск до|осип", tl):
        s["sides"] = "slope"
    elif re.search(r"скел[ья]ст|скельн|rock(y)? (wall|side)|фактур|текстур\w* (бок|стін)", tl):
        s["sides"] = "rock"
    elif re.search(r"вертикальн|рівн\w* стін|прям\w* стін|vertical|зріз", tl):
        s["sides"] = "vertical"
    # текстура
    if re.search(r"без (текстур|фото|супутник)|plain|сір\w* пластик", tl):
        s["texture"] = "none"
    # фігурки
    figs = []
    fh = s.pop("_fig_h", None)
    if re.search(r"скелелаз|лізе|канат|climb|rope", tl):
        figs.append({"id": "climber_rope", "where": "steepest", "height_mm": fh})
    if re.search(r"маха|стоїть на вершин|на вершині|waving|standing|альпініст на|людин\w* на верш|hiker", tl):
        figs.append({"id": "hiker_wave", "where": "summit", "height_mm": fh})
    elif re.search(r"людин|фігур|person|figure|чоловік|man\b", tl) and not figs:
        figs.append({"id": "hiker_wave", "where": "summit", "height_mm": fh})
    if re.search(r"хатин|будин|хиж|притулок|hut|cabin|house|chalet", tl):
        figs.append({"id": "alpine_hut", "where": "slope", "height_mm": None})
    n = re.search(r"(\d+)\s*(людей|люди|людин|фігур\w*|осіб|people|persons?|figures?)", tl)
    if n and not [f for f in figs if f["id"] != "alpine_hut"]:
        figs.insert(0, {"id": "hiker_wave", "where": "summit", "height_mm": fh})
    if n and figs:
        k = int(n.group(1)); base = [f for f in figs if f["id"] != "alpine_hut"]
        extra = [dict(base[i % len(base)], where="point", fx=0.35 + 0.3 * ((i * 7) % 3) / 2, fy=0.35 + 0.3 * ((i * 5) % 3) / 2) for i in range(len(base), min(k, 6))]
        figs += extra
    s["figures"] = figs
    # місце
    place = None
    p = P.find_by_alias(tl)
    if p:
        place = {"name": p["name"]["uk"], "lat": p["lat"], "lon": p["lon"], "source": "preset", "preset_id": p["id"], "area_km": p["area_km"], "elev": p["elev"]}
    else:
        m = _PLACE_RE.search(t)
        cand = m.group(1).strip() if m else None
        if not cand:
            caps = re.findall(r"\b([A-ZА-ЯЇІЄҐ][a-zа-яїієґ'’\-]{2,}(?:\s+[A-ZА-ЯЇІЄҐ][a-zа-яїієґ'’\-]{2,})?)", t)
            cand = caps[0] if caps else None
        if cand:
            place = geocode(cand)
            if place is None:
                notes.append(f"Не знайшов місце «{cand}» — оберіть точку на мапі.")
    if place:
        s["place"] = place
        if s["area_km"] is None and place.get("area_km"):
            s["area_km"] = place["area_km"]
    return s, notes


# ── Claude ───────────────────────────────────────────────────────────────────
def _llm_parse(text: str, locale: str) -> Optional[dict]:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic  # type: ignore
    except Exception:
        return None
    figs = ", ".join(f"{f['id']} ({f['name']['uk']}, kind={f['kind']})" for f in figure_library())
    presets = ", ".join(f"{p['id']}={p['name']['uk']}" for p in P.PRESETS)
    sys_prompt = (
        "Ти — інженер 3D-друку рельєфних моделей гір. Перетвори опис на JSON (лише JSON, без тексту):\n"
        '{"place": {"name": str, "preset_id": str|null, "query": str|null}, "area_km": number|null, "size_mm": number, '
        '"height_mm": number|null, "frame": {"style": "none|flat|rounded", "width_mm": number, "height_mm": number}, '
        '"sides": "vertical|rock|slope", "texture": "satellite|none", '
        '"figures": [{"id": str, "where": "summit|steepest|slope|point", "height_mm": number|null, "fx": number|null, "fy": number|null}], '
        '"notes": str}\n'
        f"Пресети: {presets}. Якщо місце не з пресетів — дай query для геокодера (назва вершини латиницею або мовою оригіналу). "
        f"Фігурки: {figs}. where=steepest для скелелазів, summit для тих, хто стоїть/махає, slope для хатини. "
        "Одиниці: см→мм. Розмір плити 60–400 мм, висота 20–300 мм (якщо не сказано — null). "
        "Ободок за замовчуванням rounded 10×25 мм; «без ободка/основи» → none. Боки за замовчуванням slope. "
        "Нічого не вигадуй: якщо параметра немає в тексті — лиши дефолт/null."
    )
    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(model=os.getenv("MOUNTAINS_LLM_MODEL", "claude-sonnet-5"), max_tokens=600, system=sys_prompt,
                                     messages=[{"role": "user", "content": text[:2000]}])
        txt = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        a, b = txt.find("{"), txt.rfind("}")
        if a < 0:
            return None
        d = json.loads(txt[a:b + 1])
        pl = d.get("place") or {}
        place = None
        if pl.get("preset_id"):
            p = next((x for x in P.PRESETS if x["id"] == pl["preset_id"]), None)
            if p:
                place = {"name": p["name"]["uk"], "lat": p["lat"], "lon": p["lon"], "source": "preset", "preset_id": p["id"], "area_km": p["area_km"], "elev": p["elev"]}
        if place is None and (pl.get("query") or pl.get("name")):
            place = geocode(str(pl.get("query") or pl.get("name")))
        d["place"] = place
        if d.get("area_km") is None and place and place.get("area_km"):
            d["area_km"] = place["area_km"]
        return d
    except Exception as exc:  # noqa: BLE001
        print(f"[AGENT] llm failed → rules: {exc}", flush=True); return None


# ── пояснення ────────────────────────────────────────────────────────────────
def explain(spec: dict) -> list[str]:
    lib = {f["id"]: f for f in figure_library()}
    out = []
    pl = spec.get("place")
    if pl:
        src = {"preset": "з бібліотеки вершин", "geocode": "знайдено на карті (OpenStreetMap)", "map": "точка з мапи"}.get(pl.get("source"), "")
        out.append(f"Місце: {pl['name']} ({pl['lat']:.4f}, {pl['lon']:.4f}) — {src}.")
    else:
        out.append("Місце: не вказано — оберіть точку на мапі або назвіть гору.")
    ak = spec.get("area_km")
    out.append(f"Ділянка: {ak:.1f} × {ak:.1f} км." if ak else "Ділянка: підберу автоматично (для гори ≈ 3–6 км).")
    size = spec["size_mm"]; fr = spec["frame"]
    inner = size - 2 * fr["width_mm"]
    out.append(f"Плита {size:.0f} × {size:.0f} мм" + (f", рельєф усередині {inner:.0f} мм." if fr["width_mm"] else "."))
    out.append(f"Висота: {spec['height_mm']:.0f} мм (вертикаль підберу під цю висоту)." if spec.get("height_mm") else "Висота: у природному масштабі (без перебільшення), але не вище 60 % ширини.")
    out.append({"none": "Без ободка: рельєф до самого краю.", "flat": f"Ободок плаский з фаскою, {fr['width_mm']:.0f} × {fr['height_mm']:.0f} мм.",
                "rounded": f"Ободок із заокругленим верхом, {fr['width_mm']:.0f} × {fr['height_mm']:.0f} мм."}[fr["style"]])
    out.append({"vertical": "Боки: рівний вертикальний зріз.", "rock": "Боки: вертикальні скельні стінки з ребрами (до 6 мм углиб).",
                "slope": "Боки: похилі скельні схили з осипом (без нависань)."}[spec["sides"]])
    for f in spec.get("figures", []):
        nm = lib.get(f["id"], {}).get("name", {}).get("uk", f["id"])
        where = {"summit": "на вершині", "steepest": "на найкрутішій стіні поруч із вершиною", "slope": "на пологому схилі нижче вершини", "point": "у вказаній точці"}[f["where"]]
        out.append(f"Фігурка: {nm}, {f['height_mm']:.0f} мм, {where}.")
    if not spec.get("figures"):
        out.append("Фігурок немає (можна додати: скелелаз, альпініст, хатина).")
    out.append("Текстура: супутниковий знімок для превʼю та гайду розпису." if spec["texture"] == "satellite" else "Без текстури (сірий пластик).")
    if spec.get("notes"):
        out.append(f"Примітка: {spec['notes']}")
    return out


def understand(text: str, locale: str = "uk", base: Optional[dict] = None) -> dict:
    """Головний вхід агента. base — поточний spec із форми (агент доповнює, а не стирає)."""
    t = (text or "").strip()
    raw = _llm_parse(t, locale) if t else None
    source = "llm" if raw else "rules"
    if raw is None:
        raw, notes = rule_parse(t)
    else:
        notes = []
    merged = dict(base or {})
    for k, v in raw.items():
        if v is None or (k == "figures" and not v):
            continue
        if k == "frame" and isinstance(v, dict) and isinstance(merged.get("frame"), dict):
            merged["frame"] = {**merged["frame"], **{kk: vv for kk, vv in v.items() if vv is not None}}
        else:
            merged[k] = v
    spec, warn = normalize_spec(merged)
    questions = []
    if not spec.get("place"):
        questions.append("Яку гору робимо? Назвіть її або поставте точку на мапі.")
    conf = 0.9 if source == "llm" else 0.7
    if not spec.get("place"):
        conf -= 0.3
    return {"spec": spec, "understood": explain(spec), "warnings": warn + notes, "questions": questions, "confidence": round(max(conf, 0.1), 2), "source": source}


# ── агент для «Світів» (процедурні) ──────────────────────────────────────────
_WORLD_UK = {"mountain": "гірський хребет", "island": "острів", "valley": "каньйон / долина", "plateau": "плато", "ridges": "дюни",
             "crater": "кратер", "rolling": "пагорби", "volcano": "вулкан", "archipelago": "архіпелаг"}


def understand_world(text: str, size_mm: float = 120.0, shape: Optional[str] = None) -> dict:
    from services.llm_orchestrator import prompt_to_spec
    spec, src = prompt_to_spec(text or "", float(size_mm), shape_override=shape)
    shp = spec.get("shape"); hu = _WORLD_UK.get(shp, shp)
    understood = [f"Форма: {hu}" + (" (обрано вами)" if src == "user" else " (за описом)") + ".",
                  f"Розмір плити: {spec.get('width_mm', size_mm):.0f} мм, висота рельєфу до {spec.get('max_height_mm', 0):.0f} мм.",
                  f"Шорсткість {float(spec.get('roughness', 0)):.1f} / ерозія {float(spec.get('erosion', 0)):.1f} (0 — гладко, 1 — дуже порізано).",
                  f"Основа {float(spec.get('base_thickness_mm', 3)):.0f} мм. Seed {spec.get('seed')} — той самий опис дасть той самий світ; «інший варіант» змінює seed."]
    warnings = []
    if not (text or "").strip():
        warnings.append("Опис порожній — використано типові параметри.")
    if src == "rules":
        warnings.append("Розібрано за ключовими словами (AI-ключ не задано): якщо форма не та — оберіть її кнопкою.")
    return {"spec": {"shape": shp, "shapeUk": hu, "size_mm": float(spec.get("width_mm", size_mm)), "max_height_mm": spec.get("max_height_mm"),
                     "roughness": spec.get("roughness"), "erosion": spec.get("erosion"), "seed": spec.get("seed")},
            "understood": understood, "warnings": warnings, "questions": [], "confidence": 0.85 if src != "rules" else 0.65, "source": src}
