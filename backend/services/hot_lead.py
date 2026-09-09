# -*- coding: utf-8 -*-
"""Миттєве сповіщення власника про «гарячий лід» — клік «замовити в месенджері».

⭐Навіщо (реальний випадок з аналітики 07.09.2026, відвідувач 73a3e4bb…):
    12:39:23  messenger_order  {channel: tg, product: map, priceUah: 770}
    12:40:00  why_not_order    {reason: self}
Людина прийшла з Google, дійшла до кінця, натиснула «Замовити в Telegram» —
і за 37 секунд відповіла в опитуванні «надрукую сам». Тобто в чат вона,
найпевніше, так нічого й не написала.

Кнопка месенджера працює так: текст замовлення копіюється в буфер і
відкривається `t.me/monadruk` (звичайні акаунти Telegram не вміють prefill).
Далі все залежить від людини — вставить і надішле чи передумає. Якщо не
надішле, власник про цього відвідувача не дізнається взагалі, хоча це
найтепліший контакт за тиждень.

Тому: про КЛІК повідомляємо одразу, з посиланням на саму модель. Далі це вже
рішення власника — писати першим чи ні.

Дедуплікація тут навмисно проста й у памʼяті процесу: подія рідкісна (одна за
тиждень), а після рестарту бекенду краще прислати зайве сповіщення, ніж
проковтнути лід.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

__all__ = ["should_notify", "format_lead", "notify", "reset"]

_PRODUCT_UK = {
    "map": "3D-мапа",
    "keychain": "брелок",
    "magnet": "магніт",
    "relief": "рельєф",
    "panel": "панно",
}
_CHANNEL_UK = {"tg": "Telegram", "ig": "Instagram"}

_lock = threading.Lock()
_seen: Dict[Tuple[str, str], float] = {}
_hour_bucket: list[float] = []


def _dedupe_window_s() -> float:
    try:
        return max(0.0, float(os.getenv("HOT_LEAD_DEDUPE_S", "600")))
    except Exception:
        return 600.0


def _hourly_cap() -> int:
    try:
        return max(1, int(os.getenv("HOT_LEAD_HOURLY_CAP", "12")))
    except Exception:
        return 12


def reset() -> None:
    with _lock:
        _seen.clear()
        _hour_bucket.clear()


def should_notify(event: str, visitor: str, now: Optional[float] = None) -> bool:
    """Один лід від одного відвідувача — одне сповіщення за вікно дедуплікації.

    Плюс запобіжник на випадок, якщо колись подія почне сипатись пачками:
    більше `HOT_LEAD_HOURLY_CAP` сповіщень за годину не шлемо, щоб не зробити
    з телеграму власника шумовий канал, який він перестане читати.
    """
    if event != "messenger_order":
        return False
    ts = time.time() if now is None else float(now)
    key = (str(event), str(visitor or "—"))
    window = _dedupe_window_s()
    with _lock:
        for stale_key, seen_at in list(_seen.items()):
            if ts - seen_at > window:
                del _seen[stale_key]
        if key in _seen:
            return False

        _hour_bucket[:] = [t for t in _hour_bucket if ts - t <= 3600.0]
        if len(_hour_bucket) >= _hourly_cap():
            return False

        _seen[key] = ts
        _hour_bucket.append(ts)
        return True


def _esc(value: Any) -> str:
    """Telegram parse_mode=HTML: екрануємо лише те, що ламає розмітку."""
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def format_lead(rec: Dict[str, Any], *, site_url: str = "") -> str:
    props = rec.get("props") or {}
    product = str(props.get("product") or "")
    channel = str(props.get("channel") or "")
    price = str(props.get("priceUah") or "").strip()
    summary = str(props.get("summary") or "").strip()
    task_id = str(props.get("taskId") or "").strip()

    lines = ["🔥 <b>Гарячий лід</b> — натиснули «замовити в месенджері»"]
    label = _PRODUCT_UK.get(product, product or "модель")
    # Рекап із фронта вже починається з назви товару («3D-мапа · M · Київ»),
    # тож дописувати її вдруге не треба — виходило «3D-мапа · 3D-мапа · M · Київ».
    if summary:
        what = summary if summary.casefold().startswith(label.casefold()) else f"{label} · {summary}"
    else:
        what = label
    lines.append(f"🧩 {_esc(what)}")
    if price:
        lines.append(f"💰 {_esc(price)} ₴")
    if channel:
        lines.append(f"💬 {_esc(_CHANNEL_UK.get(channel, channel))}")

    where = [p for p in (rec.get("path"), rec.get("locale"), rec.get("cc")) if p]
    if where:
        lines.append("📍 " + _esc(" · ".join(str(w) for w in where)))
    ref = str(rec.get("ref") or "").strip()
    if ref:
        lines.append(f"↪️ {_esc(ref[:120])}")
    if task_id and site_url:
        lines.append(f"🔗 {_esc(site_url.rstrip('/'))}/share/{_esc(task_id)}")

    lines.append("")
    lines.append(
        "Текст замовлення вже в буфері у людини — але надіслати його вона могла й "
        "не встигнути. Якщо повідомлення не прийшло, це привід написати першим."
    )
    return "\n".join(lines)


def notify(rec: Dict[str, Any]) -> bool:
    """Надсилає сповіщення. Ніколи не кидає — трекінг важливіший за сповіщення."""
    try:
        from services import order_service as _os

        if not _os.telegram_configured():
            return False
        text = format_lead(rec, site_url=os.getenv("SITE_URL", "https://monadruk.com"))
        return bool(_os._tg_post("sendMessage", chat_id=_os._chat(), parse_mode="HTML", text=text))
    except Exception as exc:  # noqa: BLE001
        print(f"[HOT_LEAD] сповіщення не надіслано: {exc}", flush=True)
        return False


def notify_in_background(rec: Dict[str, Any]) -> None:
    """Трекінг не має чекати на Telegram: HTTP-відповідь іде одразу."""
    threading.Thread(target=notify, args=(dict(rec),), daemon=True).start()
