"""Nova Poshta API клієнт — пошук міста + відділення (warehouse) для форми замовлення.

Ключ ТІЛЬКИ у server-side env `NOVA_POSHTA_API_KEY` (безкоштовний у кабінеті НП).
Без ключа → is_configured()=False → фронт показує ручне введення (як було).
Легкий in-memory кеш (довідник міст/відділень змінюється рідко) щоб не бити по
ліміту НП-API і не гальмувати форму.
"""
from __future__ import annotations

import os
import time

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None  # type: ignore

NP_URL = "https://api.novaposhta.ua/v2.0/json/"
_CACHE: dict[str, tuple[float, list]] = {}
_TTL = 3600.0  # 1 година


def is_configured() -> bool:
    return bool((os.getenv("NOVA_POSHTA_API_KEY") or "").strip()) and httpx is not None


def _request(model: str, method: str, props: dict) -> list:
    key = (os.getenv("NOVA_POSHTA_API_KEY") or "").strip()
    if not key or httpx is None:
        return []
    ck = f"{model}|{method}|{sorted(props.items())}"
    now = time.time()
    hit = _CACHE.get(ck)
    if hit and now - hit[0] < _TTL:
        return hit[1]
    body = {
        "apiKey": key,
        "modelName": model,
        "calledMethod": method,
        "methodProperties": props,
    }
    try:
        r = httpx.post(NP_URL, json=body, timeout=12.0)
        data = r.json()
    except Exception as exc:  # мережа/таймаут — віддаємо кеш якщо є
        print(f"[NOVA POSHTA] request failed: {exc}", flush=True)
        return hit[1] if hit else []
    out = data.get("data", []) if isinstance(data, dict) and data.get("success") else []
    if not (isinstance(data, dict) and data.get("success")):
        # лог помилки НП (напр. невалідний ключ) — щоб було видно в pm2 logs
        errs = data.get("errors") if isinstance(data, dict) else None
        if errs:
            print(f"[NOVA POSHTA] API errors: {errs}", flush=True)
    _CACHE[ck] = (now, out)
    return out


def search_cities(q: str, limit: int = 20) -> list[dict]:
    """Пошук міст за рядком (укр). Повертає [{ref, name, area, region}]."""
    q = (q or "").strip()
    if len(q) < 2:
        return []
    raw = _request("Address", "getCities", {"FindByString": q, "Limit": str(int(limit))})
    items = []
    for c in raw:
        if not (c.get("Ref") and c.get("Description")):
            continue
        items.append({
            "ref": c.get("Ref"),
            "name": c.get("Description"),
            "area": c.get("AreaDescription", ""),
            "region": c.get("RegionsDescription", ""),
        })
    return items


def search_warehouses(city_ref: str, q: str = "", limit: int = 60) -> list[dict]:
    """Відділення/поштомати для міста (за CityRef). [{ref, number, name, short}]."""
    city_ref = (city_ref or "").strip()
    if not city_ref:
        return []
    props: dict = {"CityRef": city_ref, "Limit": str(int(limit))}
    if (q or "").strip():
        props["FindByString"] = q.strip()
    raw = _request("AddressGeneral", "getWarehouses", props)
    items = []
    for w in raw:
        if not w.get("Ref"):
            continue
        items.append({
            "ref": w.get("Ref"),
            "number": w.get("Number", ""),
            "name": w.get("Description", ""),
            "short": w.get("ShortAddress", ""),
        })
    return items
