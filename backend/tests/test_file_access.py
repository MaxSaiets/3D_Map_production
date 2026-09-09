"""Оплачений доступ до друк-файлу (рішення власника 09.09.2026: 149 ₴).

Навіщо це існує (заміряно за 30 днів): модель створили 23 людини, з них лише 8 з
України. Друк і доставка — тільки по Україні, тож дві третини тих, хто робить усю
роботу зі створення мапи, не мали що купити. Файл не має логістики й митниці.

Головне правило, яке тримають ці тести: **оплата не витрачає безкоштовну квоту**.
Покупець не має платити двічі — грошима і своїм безкоштовним завантаженням.
"""
from __future__ import annotations

import json

import pytest

from services import file_access as fa


@pytest.fixture(autouse=True)
def access_log(tmp_path, monkeypatch):
    monkeypatch.setattr(fa, "ACCESS_LOG", tmp_path / "file_access.jsonl")
    yield tmp_path / "file_access.jsonl"


# ── ціна ─────────────────────────────────────────────────────────────────────

def test_price_defaults_to_the_owners_decision(monkeypatch, tmp_path):
    monkeypatch.delenv("FILE_PRICE_UAH", raising=False)
    monkeypatch.setattr(fa, "PRICING_PATH", tmp_path / "нема.json")
    assert fa.file_price_uah() == 149


def test_price_comes_from_pricing_json(monkeypatch, tmp_path):
    monkeypatch.delenv("FILE_PRICE_UAH", raising=False)
    p = tmp_path / "pricing.json"
    p.write_text(json.dumps({"file": {"price": 199}}), encoding="utf-8")
    monkeypatch.setattr(fa, "PRICING_PATH", p)
    assert fa.file_price_uah() == 199, "ціна має правитись у pricing.json без релізу"


def test_env_overrides_pricing(monkeypatch, tmp_path):
    p = tmp_path / "pricing.json"
    p.write_text(json.dumps({"file": {"price": 199}}), encoding="utf-8")
    monkeypatch.setattr(fa, "PRICING_PATH", p)
    monkeypatch.setenv("FILE_PRICE_UAH", "99")
    assert fa.file_price_uah() == 99


def test_broken_pricing_file_does_not_break_checkout(monkeypatch, tmp_path):
    monkeypatch.delenv("FILE_PRICE_UAH", raising=False)
    p = tmp_path / "pricing.json"
    p.write_text("{ це не json", encoding="utf-8")
    monkeypatch.setattr(fa, "PRICING_PATH", p)
    assert fa.file_price_uah() == 149


# ── доступ ───────────────────────────────────────────────────────────────────

def test_grant_then_access():
    assert fa.has_access("t1") is False
    assert fa.grant("t1", order_number="7001", email="A@B.com", amount=149) is True
    assert fa.has_access("t1") is True
    assert fa.has_access("t2") is False, "доступ дається на КОНКРЕТНУ модель"


def test_grant_is_idempotent(access_log):
    assert fa.grant("t1", order_number="7001") is True
    assert fa.grant("t1", order_number="7001") is False, "повтор не має дублювати запис"
    lines = [l for l in access_log.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1


def test_no_task_id_grants_nothing():
    for bad in ("", "   ", None):
        assert fa.grant(bad) is False          # type: ignore[arg-type]
    assert fa.has_access("") is False
    assert fa.has_access(None) is False


def test_email_is_normalised_and_searchable():
    fa.grant("t1", email="  Owner@Example.COM ")
    fa.grant("t2", email="other@example.com")
    assert fa.list_for_email("owner@example.com") == ["t1"]
    assert fa.list_for_email("OWNER@EXAMPLE.COM") == ["t1"]
    assert fa.list_for_email("") == []


def test_broken_line_does_not_hide_other_grants(access_log):
    fa.grant("t1")
    with access_log.open("a", encoding="utf-8") as fh:
        fh.write("{побитий рядок\n")
    fa.grant("t2")
    assert fa.has_access("t1") is True
    assert fa.has_access("t2") is True


def test_missing_log_means_no_access(access_log):
    assert not access_log.exists()
    assert fa.has_access("t1") is False        # і не падає


# ── звʼязок з оплатою ────────────────────────────────────────────────────────

def test_paid_order_opens_access_to_its_model(tmp_path, monkeypatch):
    """`mark_order_paid` — ЄДИНА точка підтвердження оплати (її кличуть і
    webhook LiqPay, і опитування статусу зі сторінки-подяки), тож хук саме там."""
    from services import order_service as os_mod

    log = tmp_path / "orders.jsonl"
    log.write_text(
        json.dumps({"order_number": "7001", "status": "pending_payment",
                    "task_id": "task-abc", "email": "buyer@example.com"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(os_mod, "ORDERS_LOG", log)
    monkeypatch.setattr(os_mod, "telegram_configured", lambda: False)

    assert fa.has_access("task-abc") is False
    os_mod.mark_order_paid("7001", {"status": "success", "amount": 149, "currency": "UAH"})
    assert fa.has_access("task-abc") is True, "оплата має відкривати доступ до файлу"


def test_unpaid_order_opens_nothing(tmp_path, monkeypatch):
    from services import order_service as os_mod

    log = tmp_path / "orders.jsonl"
    log.write_text(
        json.dumps({"order_number": "7002", "status": "pending_payment",
                    "task_id": "task-xyz"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(os_mod, "ORDERS_LOG", log)
    monkeypatch.setattr(os_mod, "telegram_configured", lambda: False)

    os_mod.mark_order_paid("7002", {"status": "failure", "amount": 149})
    assert fa.has_access("task-xyz") is False


def test_find_order_record_takes_the_latest_version(tmp_path, monkeypatch):
    """Журнал дописуваний: `set_order_status` і `attach_payment` перезаписують
    запис новим рядком. Актуальний — останній."""
    from services import order_service as os_mod

    log = tmp_path / "orders.jsonl"
    log.write_text("\n".join([
        json.dumps({"order_number": "7003", "status": "new", "task_id": "старий"}, ensure_ascii=False),
        json.dumps({"type": "payment", "order_number": "7003", "paid": False}, ensure_ascii=False),
        json.dumps({"order_number": "7003", "status": "pending_payment", "task_id": "новий"}, ensure_ascii=False),
    ]) + "\n", encoding="utf-8")
    monkeypatch.setattr(os_mod, "ORDERS_LOG", log)

    rec = os_mod.find_order_record("7003")
    assert rec is not None and rec["task_id"] == "новий"
    assert os_mod.find_order_record("9999") is None

# ── ендпоінти ────────────────────────────────────────────────────────────────

def test_access_endpoint_reports_price_and_state(monkeypatch):
    from fastapi.testclient import TestClient
    import main

    monkeypatch.setattr(main, "tasks", getattr(main, "tasks", {}), raising=False)
    c = TestClient(main.app)

    r = c.get("/api/file/access/task-none")
    assert r.status_code == 200
    body = r.json()
    assert body["paid"] is False
    assert body["priceUah"] > 0 and body["currency"] == "UAH"

    fa.grant("task-paid", order_number="1")
    assert c.get("/api/file/access/task-paid").json()["paid"] is True


def test_checkout_rejects_a_broken_email():
    from fastapi.testclient import TestClient
    import main

    c = TestClient(main.app)
    # «@example.com» проходило першу версію перевірки й доходило до створення
    # замовлення — саме цей рядок і зловив діру.
    for bad in ("", "   ", "не пошта", "a@b", "@example.com", "a@.com",
                "a@example.", "a b@example.com", "a@@example.com"):
        r = c.post("/api/file/checkout", json={"task_id": "task-1", "email": bad})
        assert r.status_code == 422, (bad, r.status_code, r.text[:120])


def test_checkout_does_not_charge_twice():
    """Людина повернулась за посиланням на вже куплений файл — грошей не беремо."""
    from fastapi.testclient import TestClient
    import main

    fa.grant("task-paid-2", order_number="2")
    c = TestClient(main.app)
    r = c.post("/api/file/checkout", json={"task_id": "task-paid-2", "email": "a@example.com"})
    assert r.status_code == 200
    assert r.json() == {"alreadyPaid": True, "taskId": "task-paid-2"}
