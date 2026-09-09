"""S-4: чистий шар тижневого дайджесту (без HTTP/sleep)."""
from __future__ import annotations

import json
from datetime import datetime

from services import weekly_digest as wd


def test_next_digest_delay_monday_future():
    # Середа 10:00 → наступний понеділок 07:00 = 4 дні 21 год
    now = datetime(2026, 9, 9, 10, 0, 0)
    d = wd.next_digest_delay(now, weekday=0, hour=7)
    assert abs(d - (4 * 86400 + 21 * 3600)) < 1


def test_next_digest_delay_same_day_before_and_after_hour():
    mon_early = datetime(2026, 9, 7, 6, 0, 0)  # понеділок 06:00 → за 1 год
    assert abs(wd.next_digest_delay(mon_early, 0, 7) - 3600) < 1
    mon_late = datetime(2026, 9, 7, 8, 0, 0)   # понеділок 08:00 → через 7 днів −1 год
    assert abs(wd.next_digest_delay(mon_late, 0, 7) - (7 * 86400 - 3600)) < 1


def test_build_digest_text_contains_key_numbers_and_warning():
    agg = {
        "totals": {"uniqueVisitors": 41, "pageviews": 120},
        "guided": {
            "steps": [{"step": "view", "count": 30}, {"step": "generate", "count": 9}],
            "generate": {"total": 9},
            "choices": {
                "orderClicks": 2, "downloads": 5,
                "results": {"ok": 8, "fail": 1},
                "whyNotOrder": [["price", 3], ["self", 2]],
                "messenger": [["tg", 1], ["ig", 1]],
            },
        },
        "topPaths": [["/create", 40], ["/keychains", 20]],
        "byCountry": [["UA", 25], ["DE", 6]],
    }
    text = wd.build_digest(agg, orders_week=0, leads_total=4)
    assert "Відвідувачі: 41" in text and "перегляди: 120" in text
    assert "Генерації: 9 (✓8 / ✗1)" in text
    assert "Клік «Замовити»: 2" in text and "надіслані замовлення: 0" in text
    assert "Завантажили файл: 5" in text and "у месенджер: 2" in text
    assert "дорого 3" in text and "надрукують самі 2" in text
    assert "Ліди" in text and "4" in text
    assert "/create 40" in text and "UA 25" in text
    assert "Замовлень за тиждень немає" in text
    assert "@" not in text  # жодних e-mail-ів у повідомленні


def test_build_digest_no_warning_when_orders():
    text = wd.build_digest({"totals": {}, "guided": {}}, orders_week=3, leads_total=0)
    assert "надіслані замовлення: 3" in text
    assert "немає" not in text


def test_count_orders_since_ignores_payments_and_duplicates():
    lines = [
        json.dumps({"order_number": "1", "created_at": "2026-09-01T10:00:00", "status": "new"}),
        json.dumps({"order_number": "1", "created_at": "2026-09-01T10:00:00", "status": "paid"}),
        json.dumps({"order_number": "2", "created_at": "2026-08-20T10:00:00"}),
        json.dumps({"type": "payment", "order_number": "3", "paid": True, "created_at": "2026-09-05T10:00:00"}),
        json.dumps({"order_number": "4", "created_at": "2026-09-06T10:00:00"}),
        "not json",
    ]
    assert wd.count_orders_since(lines, "2026-08-31T00:00:00") == 2


def test_digest_hides_empty_result_counters():
    """Було: «Генерації: 6 (✓0 / ✗0)» — читалось як поломка, хоча просто не
    приходила подія guided_result (виправлено окремо у ScenarioFlow)."""
    agg = {"totals": {}, "guided": {"generate": {"total": 6},
                                    "choices": {"results": {"ok": 0, "fail": 0}}}}
    text = wd.build_digest(agg, orders_week=0, leads_total=0)
    assert "Генерації: 6" in text
    assert "✓0" not in text

    agg["guided"]["choices"]["results"] = {"ok": 5, "fail": 1}
    text2 = wd.build_digest(agg, orders_week=0, leads_total=0)
    assert "Генерації: 6 (✓5 / ✗1)" in text2


def test_digest_reports_download_reclicks():
    agg = {"totals": {}, "guided": {"downloadReclicks": 7, "choices": {}}}
    text = wd.build_digest(agg, orders_week=0, leads_total=0)
    assert "Повторні кліки «Завантажити»: 7" in text
    # без повторів рядка немає
    assert "Повторні кліки" not in wd.build_digest({"totals": {}, "guided": {}}, 0, 0)


def test_pending_payment_orders_are_reported():
    """Реальний випадок: #1141 (10.08) висіло МІСЯЦЬ неоплаченим, і власник
    вважав, що продажів немає. Дайджест має про такі нагадувати."""
    import json
    from datetime import datetime, timedelta
    old = (datetime.now() - timedelta(days=29)).isoformat()
    recent = (datetime.now() - timedelta(days=2)).isoformat()
    lines = [
        json.dumps({"order_number": "1141", "status": "pending_payment", "created_at": old}),
        json.dumps({"order_number": "6960", "status": "pending_payment", "created_at": recent}),
        json.dumps({"order_number": "6799", "status": "pending_payment", "created_at": old}),
        json.dumps({"type": "payment", "order_number": "6799", "paid": True}),   # оплачене — не рахуємо
        json.dumps({"order_number": "8001", "status": "new", "created_at": recent}),
        "not json",
    ]
    n, days = wd.count_pending_payment(lines)
    assert n == 2, f"очікували 2 завислих, отримали {n}"
    assert 28 <= days <= 30

    text = wd.build_digest({"totals": {}, "guided": {}}, orders_week=0, leads_total=0, pending=(n, days))
    assert "Чекають оплати: 2" in text and "найстарішому" in text
    # немає зависших → рядка немає
    assert "Чекають оплати" not in wd.build_digest({"totals": {}, "guided": {}}, 0, 0, pending=(0, 0))

def test_events_and_people_are_told_apart():
    """⭐Знайдено на перевірці ЖИВОГО дайджеста 09.09, перед першою розсилкою:
    «Завантажили файл: 32» — це був 31 клік ОДНІЄЇ людини плюс один чужий.
    Власник прочитав би 32 покупці. Саме так я сам помилився 08.09."""
    agg = {
        "totals": {"uniqueVisitors": 31, "pageviews": 58},
        "guided": {
            "generate": {"total": 6},
            "choices": {
                "downloads": 32, "downloadPeople": 2,
                "orderClicks": 0, "orderClickPeople": 0,
                "generatePeople": 3,
                "messenger": [["tg", 1]], "messengerPeople": 1,
                "results": {"ok": 0, "fail": 0},
            },
        },
    }
    text = wd.build_digest(agg, orders_week=0, leads_total=0)
    assert "Завантажили файл: 32 (людей: 2)" in text, text
    assert "Генерації: 6 (людей: 3)" in text, text
    # там, де числа збігаються, зайвого «(людей: 1)» не додаємо
    assert "у месенджер: 1" in text and "у месенджер: 1 (людей" not in text, text


def test_no_people_data_keeps_the_old_plain_line():
    """Старий агрегат без *People полів не має ламати дайджест."""
    agg = {
        "totals": {"uniqueVisitors": 10, "pageviews": 20},
        "guided": {"generate": {"total": 4},
                   "choices": {"downloads": 9, "orderClicks": 1, "results": {}}},
    }
    text = wd.build_digest(agg, orders_week=0, leads_total=0)
    assert "Завантажили файл: 9 ·" in text, text
    assert "людей" not in text, text
