"""Order intake + Telegram CRM delivery.

When a customer places an order we drop everything into the Telegram bot that
already powers ops alerts: a formatted order card, then the model file (named
<Name>_<date>_<id4>.3mf), then the screenshots of exactly what the customer
designed (so we can verify text / framing before printing).

Config (backend .env):
    TG_BOT_TOKEN   — bot token (same bot as health alerts)
    TG_CHAT_ID     — chat to receive orders (TG_ORDERS_CHAT_ID overrides if set)
"""
from __future__ import annotations

import base64
import json
import os
import random
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

OUTPUT_DIR = Path("output").resolve()
# БЕЗПЕКА: orders.jsonl містить ПДн (імʼя/телефон/адреса/оплата) — НЕ можна в
# OUTPUT_DIR (віддається на /files). Тримаємо у DATA_DIR (не монтується).
DATA_DIR = Path("data").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
ORDERS_LOG = DATA_DIR / "orders.jsonl"
_TG_API = "https://api.telegram.org/bot{token}/{method}"


def _token() -> str:
    return os.getenv("TG_BOT_TOKEN", "").strip()


def _chat() -> str:
    return (os.getenv("TG_ORDERS_CHAT_ID") or os.getenv("TG_CHAT_ID") or "").strip()


def telegram_configured() -> bool:
    return bool(_token() and _chat())


def _sanitize_filename(name: str) -> str:
    name = (name or "order").strip()
    name = re.sub(r"[^\wЀ-ӿ \-]", "", name)  # keep latin/cyrillic/word/space/dash
    name = re.sub(r"\s+", "_", name).strip("_")
    return name[:40] or "order"


def _existing_order_numbers() -> set[str]:
    nums: set[str] = set()
    try:
        if ORDERS_LOG.exists():
            for line in ORDERS_LOG.read_text(encoding="utf-8").splitlines():
                try:
                    nums.add(str(json.loads(line).get("order_number")))
                except Exception:  # noqa: BLE001
                    continue
    except Exception:  # noqa: BLE001
        pass
    return nums


# Allocation of order numbers must be atomic: two orders arriving at the same
# time both read the journal, both pick the same "free" number and both write it
# → duplicate #1234. We serialize allocation with a process lock AND keep an
# in-memory set of numbers handed out this run but not yet flushed to the journal
# (a record is appended a few lines later in create_order). 4 digits → up to ~9000
# numbers; if the space ever saturates we widen to 5 digits so we never loop forever.
_order_num_lock = threading.Lock()
_reserved_order_numbers: set[str] = set()


def _new_order_number() -> str:
    with _order_num_lock:
        used = _existing_order_numbers() | _reserved_order_numbers
        # 4-digit space first, then 5-digit fallback once it is crowded.
        for lo, hi in ((1000, 9999), (10000, 99999)):
            # bounded random attempts, then a deterministic scan so a nearly-full
            # space still returns the first free slot instead of failing.
            for _ in range(200):
                n = f"{random.randint(lo, hi)}"
                if n not in used:
                    _reserved_order_numbers.add(n)
                    return n
            for cand in range(lo, hi + 1):
                n = str(cand)
                if n not in used:
                    _reserved_order_numbers.add(n)
                    return n
        # Astronomically unlikely (≈99k orders): fall back to a timestamp tail.
        n = f"{int(time.time()) % 1000000}"
        _reserved_order_numbers.add(n)
        return n


def _tg_post(method: str, **kwargs) -> bool:
    try:
        url = _TG_API.format(token=_token(), method=method)
        files = kwargs.pop("files", None)
        r = requests.post(url, data=kwargs, files=files, timeout=30)
        ok = r.ok and r.json().get("ok", False)
        if not ok:
            print(f"[ORDER][TG] {method} failed: {r.status_code} {r.text[:200]}")
        return ok
    except Exception as e:  # noqa: BLE001
        print(f"[ORDER][TG] {method} error: {e}")
        return False


def _find_model_file(task_id: Optional[str], output_file: Optional[str]) -> Optional[Path]:
    # КРИТИЧНО: оператору надсилаємо ЛИШЕ друкарський .3mf/.stl, НІКОЛИ GLB-прев'ю —
    # бо файл називається .3mf (нижче), і GLB-вміст у .3mf не відкриється у слайсері
    # («битий файл»). Якщо output_file = .glb (preview-таск) — ігноруємо й шукаємо
    # справжній .3mf по task_id; якщо його ще нема — None (оператор отримає файл
    # окремо, коли догенерується, або повідомлення «не знайдено»).
    if output_file:
        p = Path(output_file)
        if p.exists() and p.suffix.lower() in (".3mf", ".stl"):
            return p
    if not task_id:
        return None
    # match by task-id prefix or full id anywhere in the filename
    short = task_id.replace("-", "")[:8]
    candidates: List[Path] = []
    for p in OUTPUT_DIR.glob("*.3mf"):
        n = p.name
        if "_layout" in n or "_print_acceptance" in n:
            continue
        if task_id in n or short in n:
            candidates.append(p)
    if candidates:
        # newest, prefer the main model_ file
        candidates.sort(key=lambda x: (("model_" not in x.name), -x.stat().st_mtime))
        return candidates[0]
    return None


def send_model_document(order_number: Any, name: str, model_path: Any) -> bool:
    """Догенерований друкарський 3MF → окремим повідомленням оператору (фоновий вотчер
    у main.py викликає це, коли order-now-генерація завершилась ПІСЛЯ відправки картки)."""
    try:
        p = Path(model_path)
        if not p.exists() or p.suffix.lower() not in (".3mf", ".stl"):
            return False
        fname = f"{_sanitize_filename(name or 'model')}_{order_number}{p.suffix.lower()}"
        with open(p, "rb") as fh:
            _tg_post("sendDocument", chat_id=_chat(),
                     caption=f"✅ Модель замовлення #{order_number} — друкарський файл готовий",
                     files={"document": (fname, fh, "model/3mf")})
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[ORDER] send_model_document error: {e}")
        return False


def list_orders_for_uid(uid: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Замовлення користувача з журналу (новіші перші), лише безпечні поля."""
    if not uid or not ORDERS_LOG.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        for line in ORDERS_LOG.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if r.get("uid") != uid:
                continue
            out.append({k: r.get(k) for k in (
                "order_number", "created_at", "status", "product_type",
                "est_price", "delivery_method", "delivery_country", "delivery_city", "summary",
            )})
    except Exception as e:  # noqa: BLE001
        print(f"[ORDER] list_orders_for_uid failed: {e}")
    out.reverse()
    return out[:limit]


def _delivery_text(o: Dict[str, Any]) -> str:
    method = (o.get("delivery_method") or "").lower()
    country = o.get("delivery_country") or ""
    city = o.get("delivery_city") or ""
    branch = o.get("delivery_branch") or ""
    addr = o.get("delivery_address") or ""
    if method in ("nova", "np", "nova_poshta"):
        return f"Нова Пошта — {city}, відділення/поштомат {branch}".strip(" ,")
    if method in ("ukr", "ukrposhta", "ukr_poshta"):
        return f"Укрпошта — індекс {branch}, {city} {addr}".strip(" ,")
    if method in ("novapost_eu", "nova_eu"):
        return f"Nova Post (EU) — {country}, {city}, відділення {branch}".strip(" ,")
    if method == "meest":
        return f"Meest — {country}, {city}, {addr or branch}".strip(" ,")
    if method in ("pickup", "self"):
        return "Самовивіз"
    return " ".join(x for x in (country, city, branch, addr) if x) or "—"


def send_contact(name: str, phone: str, message: str, source: str = "") -> bool:
    """Lightweight 'contact us / leave a request' → Telegram CRM."""
    if not telegram_configured():
        print("[CONTACT] Telegram not configured.")
        return False
    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    lines = [
        "📨 <b>НОВЕ ЗВЕРНЕННЯ</b>",
        f"🗓 {now}",
        "",
        f"👤 {name or '—'}",
        f"📞 {phone or '—'}",
    ]
    if message:
        lines.append(f"💬 {message}")
    if source:
        lines.append(f"\n<i>джерело: {source}</i>")
    return _tg_post("sendMessage", chat_id=_chat(), parse_mode="HTML", text="\n".join(lines))


def create_order(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process an order: notify Telegram CRM with data + file + screenshots."""
    order_number = _new_order_number()
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")

    name = (payload.get("name") or "").strip() or "Без імені"
    phone = (payload.get("phone") or "").strip()
    product = payload.get("product_type") or "map"
    product_label = "Брелок з мапою" if product == "keychain" else "3D-мапа"
    summary = payload.get("summary") or {}
    comment = (payload.get("comment") or "").strip()
    task_id = payload.get("task_id")
    output_file = payload.get("output_file")
    screenshots: List[str] = payload.get("screenshots") or []

    record = {
        "order_number": order_number,
        "created_at": now.isoformat(),
        "status": "new",
        **{k: payload.get(k) for k in (
            "name", "phone", "product_type", "task_id",
            "delivery_method", "delivery_country", "delivery_city", "delivery_branch",
            "delivery_address", "comment", "est_price", "uid", "user_email", "payment_url",
        )},
        "summary": summary,
    }
    # persist CRM record (best-effort)
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(ORDERS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001
        print(f"[ORDER] log write failed: {e}")

    if not telegram_configured():
        print("[ORDER] Telegram not configured — order logged only.")
        return {"order_number": order_number, "telegram": False}

    # 1) order card
    lines = [
        f"🧾 <b>НОВЕ ЗАМОВЛЕННЯ #{order_number}</b>",
        f"🗓 {now.strftime('%d.%m.%Y %H:%M')}",
        "",
        f"👤 <b>{name}</b>",
        f"📞 {phone or '—'}",
        f"📦 {product_label}",
    ]
    if summary.get("city") or summary.get("district"):
        lines.append(f"📍 {summary.get('city','')} {summary.get('district','')}".strip())
    if summary.get("label") or summary.get("text"):
        lines.append(f"✍️ Текст: <b>{summary.get('label') or summary.get('text')}</b>")
    if summary.get("size"):
        lines.append(f"📐 Розмір: {summary.get('size')}")
    lines.append(f"🚚 {_delivery_text(payload)}")
    est_price = (payload.get("est_price") or "").strip()
    if est_price:
        lines.append(f"💰 Орієнтовно з сайту: <b>{est_price}</b> (без доставки)")
    if comment:
        lines.append(f"💬 {comment}")
    if payload.get("user_email"):
        lines.append(f"👥 Акаунт: {payload['user_email']}")
    lines.append("")
    if payload.get("payment_url"):
        lines.append("💳 Клієнту показано кнопку «Оплатити зараз» — перевір надходження перед друком.")
    else:
        lines.append("⚠️ Оплата — узгодити з клієнтом (онлайн-оплата ще не підключена).")
    _tg_post("sendMessage", chat_id=_chat(), parse_mode="HTML", text="\n".join(lines))

    # 2) model file (ЛИШЕ друкарський .3mf — _find_model_file НЕ віддасть GLB)
    model_path = _find_model_file(task_id, output_file)
    if model_path and model_path.exists():
        fname = f"{_sanitize_filename(name)}_{date_str}_{order_number}.3mf"
        try:
            with open(model_path, "rb") as fh:
                _tg_post("sendDocument", chat_id=_chat(),
                         caption=f"Модель замовлення #{order_number}",
                         files={"document": (fname, fh, "model/3mf")})
        except Exception as e:  # noqa: BLE001
            print(f"[ORDER] sendDocument error: {e}")
    elif task_id:
        # 3MF ще генерується (order-now: повна якість стартує при відкритті форми).
        # Фоновий вотчер у main.py дошле файл, щойно він буде готовий.
        _tg_post("sendMessage", chat_id=_chat(), parse_mode="HTML",
                 text=f"⏳ Модель #{order_number} ще генерується — друкарський 3MF надійде окремо щойно буде готовий.")
    else:
        _tg_post("sendMessage", chat_id=_chat(), parse_mode="HTML",
                 text=f"⚠️ Файл моделі для #{order_number} не знайдено (task_id={task_id}).")

    # 3) screenshots of what the customer designed
    for idx, shot in enumerate(screenshots[:4], start=1):
        try:
            b64 = shot.split(",", 1)[1] if "," in shot else shot
            data = base64.b64decode(b64)
            if len(data) < 200:
                continue
            _tg_post("sendPhoto", chat_id=_chat(),
                     caption=f"#{order_number} — прев'ю {idx}",
                     files={"photo": (f"order_{order_number}_{idx}.png", data, "image/png")})
        except Exception as e:  # noqa: BLE001
            print(f"[ORDER] screenshot {idx} error: {e}")

    return {"order_number": order_number, "telegram": True}


ORDER_STATUSES = ("new", "paid", "printed", "shipped", "done")


def set_order_status(order_number: str, status: str) -> bool:
    """Rewrite the matching order record's status in orders.jsonl.

    The journal is append-only JSONL with one record per order (plus separate
    `type:payment` event lines). We rewrite the file in place, updating the
    status of the order record whose `order_number` matches. Returns True if a
    matching order record was found and updated.

    Statuses: new, paid, printed, shipped, done.
    """
    status = (status or "").strip().lower()
    if status not in ORDER_STATUSES:
        raise ValueError(f"invalid status: {status!r}")
    if not ORDERS_LOG.exists():
        return False
    try:
        lines = ORDERS_LOG.read_text(encoding="utf-8").splitlines()
    except Exception as e:  # noqa: BLE001
        print(f"[ORDER] set_status read failed: {e}")
        return False

    found = False
    out: List[str] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:  # noqa: BLE001
            out.append(line)  # keep unparseable lines verbatim
            continue
        # Only the order record (not the `type:payment` event lines) carries status.
        if rec.get("type") != "payment" and str(rec.get("order_number")) == str(order_number):
            rec["status"] = status
            rec["status_updated_at"] = datetime.now().isoformat()
            found = True
            out.append(json.dumps(rec, ensure_ascii=False))
        else:
            out.append(line)

    if not found:
        return False
    try:
        tmp = ORDERS_LOG.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
        tmp.replace(ORDERS_LOG)
    except Exception as e:  # noqa: BLE001
        print(f"[ORDER] set_status write failed: {e}")
        return False
    return True


def mark_order_paid(order_id: str, info: Dict[str, Any]) -> None:
    """LiqPay-callback підтвердив оплату → лог-подія у журнал + нотифікація оператора
    в Telegram (щоб бачив, що замовлення вже оплачене і можна друкувати/відправляти)."""
    import time
    status = str(info.get("status") or "")
    amount = info.get("amount")
    ccy = info.get("currency") or ""
    paid = status in ("success", "sandbox")
    pay_id = info.get("payment_id") or info.get("transaction_id") or ""
    try:
        ORDERS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with ORDERS_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "type": "payment", "order_number": str(order_id), "paid": paid,
                "amount": amount, "currency": ccy, "status": status,
                "payment_id": pay_id, "ts": int(time.time()),
            }, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001
        print(f"[ORDER] mark_paid log error: {e}")
    # Fold the payment event into the order record: a successful payment flips
    # the order's status to "paid" so the admin board reflects it (без окремого
    # ручного кліку). Best-effort — лог-подія вже записана вище.
    if paid:
        try:
            set_order_status(str(order_id), "paid")
        except Exception as e:  # noqa: BLE001
            print(f"[ORDER] mark_paid status fold error: {e}")
        # ВОРОНКА submit→paid: оплата приходить LiqPay-вебхуком (браузера НЕМА),
        # тож фронт-funnel («paid») не спрацює. Дописуємо серверну funnel-подію в
        # ANALYTICS-лог, щоб admin-воронка показувала оплачені конверсії, а не
        # зупинялась на order_submit. Best-effort.
        try:
            from datetime import datetime, timezone
            _alog = DATA_DIR / "analytics.jsonl"
            _now = datetime.now(timezone.utc)
            _alog.parent.mkdir(parents=True, exist_ok=True)
            with _alog.open("a", encoding="utf-8") as af:
                af.write(json.dumps({
                    "ts": _now.isoformat(timespec="seconds"), "day": _now.strftime("%Y-%m-%d"),
                    "event": "funnel", "path": "/", "locale": "", "ref": "", "cc": "",
                    "visitor": "server", "props": {"step": "paid", "order": str(order_id)},
                }, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001
            print(f"[ORDER] mark_paid funnel-event error: {e}")
    if telegram_configured():
        emoji = "💰" if paid else "⏳"
        _tg_post("sendMessage", chat_id=_chat(), parse_mode="HTML",
                 text=f"{emoji} <b>Оплата LiqPay</b> · замовлення <b>#{order_id}</b>: {status} — {amount} {ccy}")
