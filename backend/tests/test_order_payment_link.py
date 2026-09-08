"""Збереження посилання на оплату в записі замовлення.

⭐Навіщо (знайдено 08.09.2026 на живих даних): чек LiqPay будувався ПІСЛЯ
`create_order` і потрапляв лише у HTTP-відповідь браузеру. Клієнт закривав
вкладку — доплатити було нічим, а оператор не мав що надіслати. При цьому
статус стартує як `pending_payment` (бо LiqPay налаштований), тож замовлення
просто зависало: на проді так висіли два серпневі замовлення (#6960, #1141).
"""
from __future__ import annotations

import json

import pytest

from services import order_service as os_mod


@pytest.fixture()
def orders_log(tmp_path, monkeypatch):
    log = tmp_path / "orders.jsonl"
    monkeypatch.setattr(os_mod, "ORDERS_LOG", log)
    log.write_text(
        json.dumps({"order_number": "7001", "status": "pending_payment", "product_type": "map"}, ensure_ascii=False)
        + "\n"
        + json.dumps({"type": "payment", "order_number": "7001", "paid": False}, ensure_ascii=False)
        + "\n"
        + json.dumps({"order_number": "7002", "status": "new"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return log


def _records(log):
    return [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_liqpay_form_becomes_shareable_link(orders_log):
    ok = os_mod.attach_payment("7001", {
        "provider": "liqpay",
        "action_url": "https://www.liqpay.ua/api/3/checkout",
        "data": "eyJhIjoxfQ==",
        "signature": "sig+/=abc",
        "amount": 490,
        "currency": "UAH",
    })
    assert ok is True
    rec = next(r for r in _records(orders_log) if r.get("order_number") == "7001" and r.get("type") != "payment")
    assert rec["payment_url"].startswith("https://www.liqpay.ua/api/3/checkout?")
    # спецсимволи підпису мають бути закодовані, інакше посилання ламається
    assert "data=eyJhIjoxfQ%3D%3D" in rec["payment_url"]
    assert "sig%2B%2F%3Dabc" in rec["payment_url"]
    assert rec["payment_amount"] == 490 and rec["payment_currency"] == "UAH"
    assert rec["status"] == "pending_payment"  # статус не чіпаємо


def test_plain_url_is_used_as_is(orders_log):
    assert os_mod.attach_payment("7002", {"url": "https://pay.example/x", "amount": 170}) is True
    rec = next(r for r in _records(orders_log) if r.get("order_number") == "7002")
    assert rec["payment_url"] == "https://pay.example/x"


def test_other_orders_and_payment_events_untouched(orders_log):
    before = _records(orders_log)
    os_mod.attach_payment("7001", {"url": "https://pay.example/y"})
    after = _records(orders_log)
    assert len(after) == len(before)
    # рядок-подія оплати лишився без змін
    ev_before = [r for r in before if r.get("type") == "payment"][0]
    ev_after = [r for r in after if r.get("type") == "payment"][0]
    assert ev_before == ev_after
    # чуже замовлення не змінилось
    assert next(r for r in after if r.get("order_number") == "7002") == \
           next(r for r in before if r.get("order_number") == "7002")


def test_no_link_no_write(orders_log):
    before = orders_log.read_text(encoding="utf-8")
    assert os_mod.attach_payment("7001", {"amount": 490}) is False       # немає ні url, ні форми
    assert os_mod.attach_payment("7001", {"action_url": "x", "data": ""}) is False
    assert os_mod.attach_payment("", {"url": "https://pay.example/z"}) is False
    assert os_mod.attach_payment("9999", {"url": "https://pay.example/z"}) is False  # немає такого замовлення
    assert orders_log.read_text(encoding="utf-8") == before
