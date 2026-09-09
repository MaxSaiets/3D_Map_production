"""Сповіщення про гарячий лід (клік «замовити в месенджері»).

⭐Реальний випадок 07.09.2026 (відвідувач 73a3e4bb…, прийшов з Google):
    12:39:23  messenger_order {channel: tg, product: map, priceUah: 770}
    12:40:00  why_not_order   {reason: self}
Клік був, повідомлення в чат — найпевніше ні. Власник дізнався про цю людину
лише тому, що я читав сирий analytics.jsonl через дві доби.
"""
from __future__ import annotations

import pytest

from services import hot_lead


@pytest.fixture(autouse=True)
def _clean():
    hot_lead.reset()
    yield
    hot_lead.reset()


def _rec(**over):
    rec = {
        "event": "messenger_order",
        "path": "/create",
        "locale": "uk",
        "ref": "https://www.google.com/",
        "visitor": "73a3e4bba6483715",
        "cc": "UA",
        "props": {"channel": "tg", "product": "map", "priceUah": "770",
                  "taskId": "abc123", "summary": "3D-мапа · M · Київ"},
    }
    rec.update(over)
    return rec


# ── коли шлемо ───────────────────────────────────────────────────────────────

def test_messenger_click_notifies_once_per_visitor():
    assert hot_lead.should_notify("messenger_order", "v1") is True
    assert hot_lead.should_notify("messenger_order", "v1") is False, "подвійний клік — одне сповіщення"
    assert hot_lead.should_notify("messenger_order", "v2") is True, "інша людина — інший лід"


def test_other_events_never_notify():
    for event in ("pageview", "click", "why_not_order", "guided_download", "conv_generate"):
        assert hot_lead.should_notify(event, "v1") is False


def test_same_visitor_notifies_again_after_the_window(monkeypatch):
    monkeypatch.setenv("HOT_LEAD_DEDUPE_S", "600")
    t = 1000.0
    assert hot_lead.should_notify("messenger_order", "v1", now=t) is True
    assert hot_lead.should_notify("messenger_order", "v1", now=t + 599) is False
    assert hot_lead.should_notify("messenger_order", "v1", now=t + 601) is True


def test_hourly_cap_protects_the_owner_inbox(monkeypatch):
    """Запобіжник: телеграм власника має лишитись каналом, який він читає."""
    monkeypatch.setenv("HOT_LEAD_HOURLY_CAP", "3")
    t = 1000.0
    sent = [hot_lead.should_notify("messenger_order", f"v{i}", now=t + i) for i in range(6)]
    assert sent == [True, True, True, False, False, False]
    # через годину ліміт відпускає
    assert hot_lead.should_notify("messenger_order", "v99", now=t + 3700) is True


# ── що саме бачить власник ───────────────────────────────────────────────────

def test_message_is_actionable():
    text = hot_lead.format_lead(_rec(), site_url="https://monadruk.com")
    for must in ("3D-мапа", "Київ", "770", "Telegram", "/create", "UA",
                 "https://monadruk.com/share/abc123", "google.com"):
        assert must in text, f"у сповіщенні немає «{must}» — власник не зможе діяти"


def test_product_name_is_not_repeated():
    """Рекап із фронта вже починається з назви товару. Перша прод-перевірка
    09.09 показала «3D-мапа · 3D-мапа · M · Київ»."""
    text = hot_lead.format_lead(_rec(), site_url="https://monadruk.com")
    assert text.count("3D-мапа") == 1, text
    assert "3D-мапа · M · Київ" in text


def test_product_name_is_added_when_recap_lacks_it():
    rec = _rec()
    rec["props"] = dict(rec["props"], summary="M · Київ")
    text = hot_lead.format_lead(rec)
    assert "3D-мапа · M · Київ" in text


def test_message_survives_missing_fields():
    """Стара версія фронта ще шле подію без taskId/summary — сповіщення має
    лишитись коректним, а не впасти чи показати «None»."""
    text = hot_lead.format_lead({"event": "messenger_order",
                                 "props": {"channel": "tg", "product": "map"}})
    assert "None" not in text
    assert "3D-мапа" in text
    assert "/share/" not in text, "без taskId посилання не вигадуємо"


def test_html_is_escaped():
    text = hot_lead.format_lead(_rec(ref="<script>alert(1)</script>"), site_url="https://monadruk.com")
    assert "<script>" not in text
    assert "&lt;script&gt;" in text


def test_notify_is_silent_without_telegram(monkeypatch):
    monkeypatch.delenv("TG_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TG_CHAT_ID", raising=False)
    monkeypatch.delenv("TG_ORDERS_CHAT_ID", raising=False)
    assert hot_lead.notify(_rec()) is False


def test_notify_never_raises(monkeypatch):
    """Аналітика важливіша за сповіщення: падіння Telegram не має ламати /api/track."""
    from services import order_service as os_mod

    monkeypatch.setattr(os_mod, "telegram_configured", lambda: True)
    monkeypatch.setattr(os_mod, "_chat", lambda: "1")

    def boom(*_a, **_k):
        raise RuntimeError("telegram down")

    monkeypatch.setattr(os_mod, "_tg_post", boom)
    assert hot_lead.notify(_rec()) is False


def test_notify_passes_html_message(monkeypatch):
    from services import order_service as os_mod

    seen = {}
    monkeypatch.setattr(os_mod, "telegram_configured", lambda: True)
    monkeypatch.setattr(os_mod, "_chat", lambda: "42")
    monkeypatch.setattr(os_mod, "_tg_post", lambda method, **kw: seen.update(kw, method=method) or True)

    assert hot_lead.notify(_rec()) is True
    assert seen["method"] == "sendMessage"
    assert seen["chat_id"] == "42"
    assert seen["parse_mode"] == "HTML"
    assert "Гарячий лід" in seen["text"]
