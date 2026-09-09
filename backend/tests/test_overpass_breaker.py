"""Запобіжник Overpass: не чекати вісім хвилин на мертвий хост.

⭐Звідки взялось (прод-лог 08.09.2026, реальний користувач, Мадрид):
    13:35:09 Завантаження будівель...
    13:38:17 [WARN] ... buildings ... Connection refused        (126 с)
    13:40:49 [WARN] ... building_parts ... Connection refused   (152 с)
    13:42:58 [WARN] ... water/roads/bridges ... Connection refused (129 с)
    [TIMING] fetch_source: 471.24s
І в кінці — ПОРОЖНЯ ПЛАСТИНА, бо пайплайн вважав зону «розрідженою».

Два правила, які тут закріплені:
1. помилка РІВНЯ ЗʼЄДНАННЯ вмикає запобіжник → решта шарів падає миттєво;
2. порожня відповідь запобіжник НЕ вмикає — у полі справді може не бути
   будівель, і карантинити через це живе джерело не можна.
"""
from __future__ import annotations

import pytest

from services import overpass_health as oh


@pytest.fixture(autouse=True)
def _clean_state():
    oh.reset()
    yield
    oh.reset()


def _refused(host: str = "overpass-api.de") -> Exception:
    """Точний текст, який приходив у прод-логах."""
    return OSError(
        f"HTTPSConnectionPool(host='{host}', port=443): Max retries exceeded with url: "
        "/api/interpreter (Caused by NewConnectionError(\"HTTPSConnection(host="
        f"'{host}', port=443): Failed to establish a new connection: [Errno 111] "
        "Connection refused\"))"
    )


# ── розпізнавання причини ────────────────────────────────────────────────────

@pytest.mark.parametrize("exc", [
    _refused(),
    ConnectionRefusedError("[Errno 111] Connection refused"),
    OSError("Name or service not known"),
    OSError("Temporary failure in name resolution"),
    OSError("Network is unreachable"),
])
def test_connection_level_errors_are_recognised(exc):
    assert oh.is_connection_error(exc) is True


@pytest.mark.parametrize("exc", [
    RuntimeError("buildings: empty result from https://overpass-api.de/api"),
    ValueError("query too heavy"),
    None,
])
def test_data_level_problems_are_not_outages(exc):
    assert oh.is_connection_error(exc) is False


def test_wrapped_cause_is_found_through_the_chain():
    try:
        try:
            raise _refused()
        except Exception as inner:
            raise RuntimeError("roads fetch failed") from inner
    except RuntimeError as outer:
        assert oh.is_connection_error(outer) is True


# ── поведінка запобіжника ────────────────────────────────────────────────────

def test_opens_after_threshold_and_closes_on_success(monkeypatch):
    monkeypatch.setenv("OVERPASS_BREAKER_FAILS", "2")
    assert oh.note_failure(_refused()) is True
    assert oh.outage_active() is False, "одна помилка ще не аварія"
    assert oh.note_failure(_refused()) is True
    assert oh.outage_active() is True, "друга помилка поспіль має ввімкнути запобіжник"

    oh.note_success()
    assert oh.outage_active() is False, "жива відповідь має погасити запобіжник"


def test_empty_responses_never_open_the_breaker(monkeypatch):
    monkeypatch.setenv("OVERPASS_BREAKER_FAILS", "2")
    for _ in range(10):
        assert oh.note_failure(RuntimeError("water: empty result")) is False
    assert oh.outage_active() is False


def test_breaker_expires_after_cooldown(monkeypatch):
    monkeypatch.setenv("OVERPASS_BREAKER_FAILS", "1")
    monkeypatch.setenv("OVERPASS_BREAKER_COOLDOWN_S", "5")   # нижня межа
    fake = {"t": 1000.0}
    monkeypatch.setattr(oh.time, "time", lambda: fake["t"])

    oh.note_failure(_refused())
    assert oh.outage_active() is True
    fake["t"] += 4.0
    assert oh.outage_active() is True, "у межах паузи джерело ще в карантині"
    fake["t"] += 2.0
    assert oh.outage_active() is False, "після паузи джерелу дають чесний шанс"


def test_user_message_has_no_python_details():
    err = oh.OverpassUnavailableError("HTTPSConnectionPool(host='overpass-api.de', port=443)")
    assert "OpenStreetMap" in err.user_message
    assert "HTTPSConnection" not in err.user_message, "покупцю не показуємо нутрощі"
    assert "HTTPSConnectionPool" in str(err), "у лог технічна причина має потрапити"


# ── інтеграція з обома завантажувачами ───────────────────────────────────────

@pytest.mark.parametrize("modname", ["data_loader", "extras_loader"])
def test_loader_fails_fast_while_breaker_is_open(modname, monkeypatch):
    """Головна економія: коли запобіжник увімкнено, наступні шари НЕ ходять у
    мережу взагалі. Саме це перетворює 471 с на ~одну спробу."""
    import importlib

    mod = importlib.import_module(f"services.{modname}")
    monkeypatch.setenv("OVERPASS_BREAKER_FAILS", "1")
    monkeypatch.setattr(mod, "_overpass_endpoints", lambda: ["https://a.example/api"])
    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)

    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        raise _refused()

    with pytest.raises(oh.OverpassUnavailableError):
        mod._run_overpass_with_retries("buildings", fetch)
    assert calls["n"] == 1

    # наступні шари тієї ж генерації — жодного мережевого виклику
    for label in ("water", "roads", "bridges"):
        with pytest.raises(oh.OverpassUnavailableError):
            mod._run_overpass_with_retries(label, fetch)
    assert calls["n"] == 1, f"після аварії було ще {calls['n'] - 1} марних походів у мережу"


@pytest.mark.parametrize("modname", ["data_loader", "extras_loader"])
def test_settings_restored_even_when_breaker_trips(modname, monkeypatch):
    """Ранній вихід не має лишити ox.settings перемкнутими на мертве дзеркало."""
    import importlib

    import osmnx as ox

    mod = importlib.import_module(f"services.{modname}")
    monkeypatch.setenv("OVERPASS_BREAKER_FAILS", "1")
    monkeypatch.setattr(mod, "_overpass_endpoints", lambda: ["https://a.example/api"])
    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)

    ep_attr = "overpass_url" if hasattr(ox.settings, "overpass_url") else "overpass_endpoint"
    to_attr = "requests_timeout" if hasattr(ox.settings, "requests_timeout") else "timeout"
    before = (getattr(ox.settings, ep_attr), getattr(ox.settings, to_attr))

    with pytest.raises(oh.OverpassUnavailableError):
        mod._run_overpass_with_retries("buildings", lambda: (_ for _ in ()).throw(_refused()))

    assert (getattr(ox.settings, ep_attr), getattr(ox.settings, to_attr)) == before


def test_successful_layer_keeps_the_source_open(monkeypatch):
    """Перевірка на протилежну помилку: успіх не має лишати слідів карантину."""
    from services import data_loader as mod

    monkeypatch.setenv("OVERPASS_BREAKER_FAILS", "2")
    monkeypatch.setattr(mod, "_overpass_endpoints", lambda: ["https://a.example/api"])

    class _Ok:
        empty = False

    oh.note_failure(_refused())          # одна помилка вже є
    assert mod._run_overpass_with_retries("water", lambda: _Ok()) is not None
    assert oh.outage_active() is False
    oh.note_failure(_refused())          # лічильник мав обнулитись
    assert oh.outage_active() is False, "успіх не обнулив лічильник помилок"
