"""Підписка Monadruk Pro — місячний безлім (файли мап і гір, без лімітів частоти
генерації, комерційна ліцензія на згенеровані моделі). Оплата — LiqPay регулярний
платіж (action=subscribe, periodicity=month). Потрібно, щоб у мерчанта LiqPay була
увімкнена послуга «Регулярні платежі».

Сховище — data/subscriptions.json (той самий підхід, що й user_store: JSON + лок).
Запис = одна підписка (order_id «sub_…»), містить і доказ згоди покупця (час,
версію умов, тексти позначок, хеш IP) — це потрібно, щоб за потреби довести, що
людина свідомо погодилась на автопродовження і на негайне надання цифрового
контенту (втрата права на відмову від договору).

Доступ активний, поки paid_until (+ GRACE на повторну спробу списання банком) у
майбутньому. Кожне успішне списання продовжує paid_until на календарний місяць від
моменту платежу; подія ідемпотентна (повтор того самого callback не продовжує
вдруге). Скасування = unsubscribe у LiqPay, доступ лишається до кінця оплаченого
місяця, неповний місяць не повертається (див. умови підписки на сайті)."""
from __future__ import annotations

import calendar
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_FILE = Path("data").resolve() / "subscriptions.json"
_LOCK = threading.Lock()

# Версія умов підписки (сторінка /pro-terms). Міняти разом із текстом умов —
# у записі згоди фіксується, яку саме редакцію людина прийняла.
TERMS_VERSION = "2026-10-06"
GRACE = timedelta(days=3)
PLANS = {
    "UAH": float(os.getenv("SUB_PRICE_UAH", "2100")),
    "USD": float(os.getenv("SUB_PRICE_USD", "50")),
}


def sales_open() -> bool:
    """07.10.2026: продаж підписки вимкнено, доки власник не підготує документи.
    Увімкнути: SUB_SALES_OPEN=1 у backend/.env + `pm2 restart 3dmap-backend --update-env`
    (фронт /pro бере прапорець з /api/subscription/plans — перезбирати сайт не треба)."""
    return os.getenv("SUB_SALES_OPEN", "0").strip().lower() in ("1", "true", "yes", "on")


_PAID = {"success", "subscribed", "sandbox"}
_REVOKE = {"reversed"}  # повернення коштів → доступ знімається


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:  # noqa: BLE001
        return None


def add_month(dt: datetime) -> datetime:
    """Той самий день наступного місяця (31.01 → 28/29.02)."""
    y, m = (dt.year + 1, 1) if dt.month == 12 else (dt.year, dt.month + 1)
    d = min(dt.day, calendar.monthrange(y, m)[1])
    return dt.replace(year=y, month=m, day=d)


def _load() -> Dict[str, Any]:
    try:
        return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"subs": {}}


def _save(data: Dict[str, Any]) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = DATA_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, DATA_FILE)


# ── кеш для verify_token (викликається на кожен запит) ──────────────────────────
_cache: Dict[str, Any] = {"mtime": None, "active": {}}


def _active_map() -> Dict[str, str]:
    """uid → paid_until (ISO) для підписок, що зараз дають доступ. Кеш за mtime файлу."""
    try:
        mt = DATA_FILE.stat().st_mtime
    except FileNotFoundError:
        return {}
    if _cache["mtime"] != mt:
        act: Dict[str, str] = {}
        for s in _load().get("subs", {}).values():
            pu = s.get("paid_until")
            if pu and s.get("status") in ("active", "cancelled"):
                if s["uid"] not in act or pu > act[s["uid"]]:
                    act[s["uid"]] = pu
        _cache.update(mtime=mt, active=act)
    return _cache["active"]


def is_active(uid: Optional[str]) -> bool:
    if not uid:
        return False
    pu = _parse(_active_map().get(uid))
    return bool(pu and pu + GRACE > _now())


# ── створення ───────────────────────────────────────────────────────────────────
def create_pending(*, uid: str, email: str, currency: str, consent: Dict[str, Any]) -> Dict[str, Any]:
    currency = currency if currency in PLANS else "UAH"
    order_id = f"sub_{uid[:10]}_{int(time.time())}"
    rec = {
        "order_id": order_id, "uid": uid, "email": email or "",
        "amount": PLANS[currency], "currency": currency,
        "status": "pending", "created_at": _iso(_now()),
        "paid_until": None, "payments": [], "events": [],
        "consent": {**consent, "terms_version": TERMS_VERSION, "ts": _iso(_now())},
    }
    with _LOCK:
        data = _load()
        data.setdefault("subs", {})[order_id] = rec
        _save(data)
    return rec


def consent_record(*, ip: str, user_agent: str, locale: str, country: str,
                   texts: List[str]) -> Dict[str, Any]:
    salt = os.getenv("SECRET_KEY", "monadruk")
    return {
        "ip_hash": hashlib.sha256(f"{ip}|{salt}".encode()).hexdigest()[:24],
        "user_agent": (user_agent or "")[:200],
        "locale": (locale or "")[:8],
        "country": (country or "")[:2],
        "accepted": [t[:400] for t in texts][:5],
    }


# ── події LiqPay ────────────────────────────────────────────────────────────────
def is_subscription_order(order_id: str) -> bool:
    return str(order_id or "").startswith("sub_")


def apply_liqpay_event(info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Обробляє перевірений (підписаний) payload LiqPay для order_id «sub_…».
    Повертає оновлений запис або None. Ідемпотентно."""
    order_id = str(info.get("order_id") or "")
    if not is_subscription_order(order_id):
        return None
    status = str(info.get("status") or "").lower()
    pay_id = str(info.get("payment_id") or info.get("transaction_id") or "")
    with _LOCK:
        data = _load()
        s = data.get("subs", {}).get(order_id)
        if not s:
            print(f"[SUB] unknown subscription order {order_id} status={status}")
            return None
        ev_key = f"{status}:{pay_id}"
        if ev_key in s.setdefault("events", []):
            return s
        s["events"] = (s["events"] + [ev_key])[-200:]
        newly_active = False
        if status in _PAID:
            try:
                amt = float(info.get("amount") or 0)
            except Exception:  # noqa: BLE001
                amt = 0.0
            ccy = str(info.get("currency") or "")
            if ccy != s["currency"] or amt + 0.01 < float(s["amount"]):
                print(f"[SUB] {order_id}: amount/currency mismatch {amt} {ccy} vs {s['amount']} {s['currency']} — ignored")
                return s
            ms = info.get("end_date") or info.get("create_date")
            try:
                paid_at = datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc) if ms else _now()
            except Exception:  # noqa: BLE001
                paid_at = _now()
            # те саме списання під іншим статусом (subscribed + success): той самий
            # payment_id АБО той самий платіжний цикл (списання раз на ≥28 днів)
            for p in s["payments"]:
                p_at = _parse(p.get("ts"))
                if (pay_id and p.get("payment_id") == pay_id) or (p_at and abs(p_at - paid_at) < timedelta(days=7)):
                    if pay_id and not p.get("payment_id"):
                        p["payment_id"] = pay_id
                        _save(data)
                    return s
            s["payments"].append({"payment_id": pay_id, "amount": amt, "currency": ccy,
                                  "status": status, "ts": _iso(paid_at)})
            cur = _parse(s.get("paid_until"))
            new_until = add_month(max(cur, paid_at) if cur else paid_at)
            s["paid_until"] = _iso(new_until)
            newly_active = s.get("status") == "pending"
            if s.get("status") in ("pending", "failed"):
                s["status"] = "active"
        elif status == "unsubscribed":
            if s.get("status") in ("active", "pending"):
                s["status"] = "cancelled"
                s["cancelled_at"] = s.get("cancelled_at") or _iso(_now())
        elif status in _REVOKE:
            s["status"] = "refunded"
            s["paid_until"] = _iso(_now() - GRACE)
        elif status in ("failure", "error") and s.get("status") == "pending":
            s["status"] = "failed"
        s["last_status"] = status
        _save(data)
    if newly_active:
        _notify(f"⭐ Нова підписка Pro: {s.get('email') or s['uid']} — {s['amount']:g} {s['currency']}/міс")
    return s


def _notify(text: str) -> None:
    try:
        from services.order_service import _tg_post, telegram_configured
        if telegram_configured():
            _tg_post("sendMessage", text=text)
    except Exception as exc:  # noqa: BLE001
        print(f"[SUB] notify failed: {exc}")


# ── кабінет ─────────────────────────────────────────────────────────────────────
def latest_for_uid(uid: str) -> Optional[Dict[str, Any]]:
    subs = [s for s in _load().get("subs", {}).values() if s.get("uid") == uid]
    if not subs:
        return None
    rank = {"active": 3, "cancelled": 2, "pending": 1}
    subs.sort(key=lambda s: (rank.get(s.get("status"), 0), s.get("paid_until") or "", s.get("created_at") or ""))
    return subs[-1]


def public_view(s: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not s:
        return {"status": "none"}
    pu = _parse(s.get("paid_until"))
    active = bool(pu and pu + GRACE > _now() and s.get("status") in ("active", "cancelled"))
    return {
        "status": s.get("status"), "active": active,
        "order_id": s.get("order_id"),
        "amount": s.get("amount"), "currency": s.get("currency"),
        "paid_until": s.get("paid_until"),
        # наступне списання лише для активної (не скасованої) підписки
        "renews_at": s.get("paid_until") if s.get("status") == "active" else None,
        "cancelled_at": s.get("cancelled_at"),
        "payments": [{"amount": p["amount"], "currency": p["currency"], "ts": p["ts"]}
                     for p in s.get("payments", [])][-12:],
    }


def mark_cancelled(order_id: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        data = _load()
        s = data.get("subs", {}).get(order_id)
        if not s:
            return None
        if s.get("status") in ("active", "pending"):
            s["status"] = "cancelled"
            s["cancelled_at"] = _iso(_now())
            _save(data)
    _notify(f"Підписку Pro скасовано: {s.get('email') or s['uid']} (доступ до {s.get('paid_until')})")
    return s


def list_all() -> List[Dict[str, Any]]:
    return sorted(_load().get("subs", {}).values(), key=lambda s: s.get("created_at") or "", reverse=True)
