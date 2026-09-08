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
