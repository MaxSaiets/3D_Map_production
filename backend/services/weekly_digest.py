"""S-4 (2026-09-07) · Тижневий дайджест у Telegram власнику.

Чому: адмінку власник відкриває рідко, а «нуль продажів» помітив лише за листами.
Раз на тиждень (понеділок, WEEKLY_DIGEST_HOUR_UTC, за замовчуванням 07:00 UTC) бот
шле ОДНЕ коротке повідомлення: відвідувачі, генерації, кліки «Замовити», надіслані
замовлення, завантаження, месенджер-звернення, відповіді «що заважає замовити»,
ліди (завантажили, не замовили) і топ-сторінки. Той самий бот/чат, що й замовлення
(TG_BOT_TOKEN / TG_CHAT_ID). Порожні секрети → дайджест вимкнено (лог, без винятків).

Чистий шар (тестується без HTTP/sleep): build_digest(), next_digest_delay().
main._weekly_digest_loop лише склеює: агрегація analytics → текст → _tg_post.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

DIGEST_WEEKDAY = 0  # понеділок
DIGEST_HOUR_UTC = int(os.getenv("WEEKLY_DIGEST_HOUR_UTC", "7") or 7)
DIGEST_DAYS = 7

_REASON_LABELS = {
    "price": "дорого",
    "self": "надрукують самі",
    "look": "просто дивились",
    "abroad": "не в Україні",
    "other": "інше",
}


def next_digest_delay(now: datetime, weekday: int = DIGEST_WEEKDAY, hour: int = DIGEST_HOUR_UTC) -> float:
    """Секунди до наступного понеділка hour:00 UTC (строго в майбутньому, ≥60 с)."""
    target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    days_ahead = (weekday - now.weekday()) % 7
    target = target + timedelta(days=days_ahead)
    if target <= now:
        target += timedelta(days=7)
    return max(60.0, (target - now).total_seconds())


def _pairs(rows: Optional[Sequence[Any]], limit: int = 5) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for row in rows or []:
        try:
            k, v = row[0], int(row[1])
        except Exception:  # noqa: BLE001
            continue
        out.append((str(k), v))
        if len(out) >= limit:
            break
    return out


def build_digest(agg: Dict[str, Any], orders_week: int, leads_total: int, days: int = DIGEST_DAYS,
                 pending: Tuple[int, int] = (0, 0)) -> str:
    """Текст дайджесту з результату main._aggregate_analytics(days=7) + к-сть
    замовлень за тиждень + к-сть лідів. Ніяких e-mail-ів у повідомленні."""
    totals = agg.get("totals") or {}
    guided = agg.get("guided") or {}
    choices = guided.get("choices") or {}
    steps = {str(s.get("step")): int(s.get("count") or 0) for s in (guided.get("steps") or []) if isinstance(s, dict)}
    gen_total = int((guided.get("generate") or {}).get("total") or 0) or steps.get("generate", 0)
    results = choices.get("results") or {}
    visitors = int(totals.get("uniqueVisitors") or 0)
    pageviews = int(totals.get("pageviews") or 0)
    order_clicks = int(choices.get("orderClicks") or 0)
    downloads = int(choices.get("downloads") or 0)
    messenger = sum(v for _, v in _pairs(choices.get("messenger")))
    reasons = _pairs(choices.get("whyNotOrder"))

    # ⭐09.09.2026, знайдено на перевірці ЖИВОГО дайджеста перед першою розсилкою:
    # рядок «Завантажили файл: 32» насправді означав 31 клік ОДНІЄЇ людини плюс
    # один чужий. Власник прочитав би це як «32 покупці» — саме так я сам
    # помилився 08.09. Тому поруч із подіями показуємо ЛЮДЕЙ, і лише тоді, коли
    # числа розходяться (інакше «(людей: 3)» біля «3» — зайвий шум).
    def _with_people(count: int, people_key: str) -> str:
        people = int(choices.get(people_key) or 0)
        if people and people != count:
            return f"{count} (людей: {people})"
        return str(count)

    lines = [f"📊 Monadruk за {days} дн."]
    lines.append(f"👥 Відвідувачі: {visitors} · перегляди: {pageviews}")
    _ok, _fail = int(results.get("ok") or 0), int(results.get("fail") or 0)
    # Показуємо ✓/✗ лише коли подія результату реально приходила: порожнє
    # «(✓0 / ✗0)» поруч із «Генерації: 6» читалось як «усе зламано».
    lines.append(f"🧩 Генерації: {_with_people(gen_total, 'generatePeople')}"
                 + (f" (✓{_ok} / ✗{_fail})" if (_ok or _fail) else ""))
    lines.append(f"🛒 Клік «Замовити»: {_with_people(order_clicks, 'orderClickPeople')}"
                 f" · надіслані замовлення: {orders_week}")
    # Зависли на оплаті — гроші, які вже майже прийшли. Посилання на оплату є в
    # картці замовлення в адмінці (кнопка «Скопіювати посилання»).
    _pend_n, _pend_days = (pending or (0, 0))
    if _pend_n:
        _age = f", найстарішому {_pend_days} дн." if _pend_days else ""
        lines.append(f"⏳ Чекають оплати: {_pend_n}{_age} — посилання в картці замовлення")
    lines.append(f"⬇️ Завантажили файл: {_with_people(downloads, 'downloadPeople')}"
                 f" · у месенджер: {_with_people(messenger, 'messengerPeople')}")
    # Повторні кліки «Завантажити» — сигнал, що людина не бачить реакції інтерфейсу
    # (прод 07.09: один відвідувач дав 31 клік → 35 генерацій друку).
    reclicks = int((guided.get("downloadReclicks") or 0))
    if reclicks:
        lines.append(f"🔁 Повторні кліки «Завантажити»: {reclicks} — інтерфейс не показує реакції")
    if reasons:
        lines.append("❓ Чому не замовляють: " + ", ".join(f"{_REASON_LABELS.get(k, k)} {v}" for k, v in reasons))
    if leads_total:
        lines.append(f"📇 Ліди (завантажили, не замовили): {leads_total} — список в адмінці")
    top = _pairs(agg.get("topPaths"), 4)
    if top:
        lines.append("🔝 " + " · ".join(f"{p} {n}" for p, n in top))
    countries = _pairs(agg.get("byCountry"), 4)
    if countries:
        lines.append("🌍 " + " · ".join(f"{c} {n}" for c, n in countries))
    if orders_week == 0:
        lines.append("⚠️ Замовлень за тиждень немає.")
    return "\n".join(lines)


def count_pending_payment(lines: Sequence[str]) -> Tuple[int, int]:
    """(скільки замовлень висять неоплаченими, вік найстарішого в днях).

    ⭐Навіщо: замовлення з онлайн-оплатою стартує як `pending_payment`, і якщо
    клієнт не завершив оплату — воно просто лежить у журналі. Оператор отримав
    Telegram при оформленні, але через тиждень про нього вже ніхто не памʼятає.
    Реальний випадок: замовлення #1141 (10.08) висіло МІСЯЦЬ, і власник вважав,
    що продажів немає взагалі. Тепер про такі нагадує тижневий дайджест."""
    import json
    from datetime import datetime as _dt
    latest: Dict[str, Dict[str, Any]] = {}
    paid_ids = set()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        onum = str(rec.get("order_number") or "")
        if not onum:
            continue
        if rec.get("type") == "payment":
            if rec.get("paid"):
                paid_ids.add(onum)
            continue
        cur = latest.get(onum) or {}
        cur.update(rec)
        latest[onum] = cur

    oldest_days = 0
    count = 0
    now = _dt.now()
    for onum, rec in latest.items():
        if onum in paid_ids or str(rec.get("status") or "") != "pending_payment":
            continue
        count += 1
        created = str(rec.get("created_at") or "")
        try:
            age = (now - _dt.fromisoformat(created)).days
        except Exception:  # noqa: BLE001
            age = 0
        oldest_days = max(oldest_days, age)
    return count, oldest_days


def count_orders_since(lines: Sequence[str], since_iso: str) -> int:
    """К-сть НОВИХ замовлень (записи з order_number, не payment-події) з created_at ≥ since."""
    import json
    seen = set()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if rec.get("type") == "payment":
            continue
        onum = str(rec.get("order_number") or "")
        if not onum or onum in seen:
            continue
        if str(rec.get("created_at") or "") >= since_iso:
            seen.add(onum)
    return len(seen)
