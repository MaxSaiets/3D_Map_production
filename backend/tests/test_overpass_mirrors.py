"""Перемикання Overpass на запасні дзеркала.

⭐Знайдено 08.09.2026 у прод-логах:
    [WARN] ... failed for buildings via https://overpass-api.de/api: ... host='overpass-api.de'
    [WARN] ... failed for buildings via https://overpass.private.coffee/api: ... host='overpass-api.de'
Другий рядок каже «через дзеркало», а помилка — від ПЕРШОГО хоста. Причина: у
встановленій osmnx налаштування звуться `overpass_url` / `requests_timeout`, а
код писав у `overpass_endpoint` / `timeout` — неіснуючі атрибути. Тобто
перемикання на дзеркала не працювало ЖОДНОГО РАЗУ, і генерації поза Україною
(де немає локальної OSM-бази) падали, щойно лягав основний Overpass.
"""
from __future__ import annotations

import osmnx as ox
import pytest

from services import data_loader, extras_loader


@pytest.mark.parametrize("mod", [data_loader, extras_loader], ids=["data_loader", "extras_loader"])
def test_each_attempt_uses_its_own_mirror(mod, monkeypatch):
    endpoints = ["https://a.example/api", "https://b.example/api", "https://c.example/api"]
    monkeypatch.setattr(mod, "_overpass_endpoints", lambda: endpoints)
    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)

    attr = "overpass_url" if hasattr(ox.settings, "overpass_url") else "overpass_endpoint"
    seen: list[str] = []

    def fetch():
        seen.append(getattr(ox.settings, attr))
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        mod._run_overpass_with_retries("buildings", fetch)

    assert seen == endpoints, f"кожна спроба має йти на СВОЄ дзеркало, а пішла: {seen}"


@pytest.mark.parametrize("mod", [data_loader, extras_loader], ids=["data_loader", "extras_loader"])
def test_settings_restored_after_run(mod, monkeypatch):
    monkeypatch.setattr(mod, "_overpass_endpoints", lambda: ["https://a.example/api"])
    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)
    attr = "overpass_url" if hasattr(ox.settings, "overpass_url") else "overpass_endpoint"
    to_attr = "requests_timeout" if hasattr(ox.settings, "requests_timeout") else "timeout"
    before_ep = getattr(ox.settings, attr)
    before_to = getattr(ox.settings, to_attr)

    with pytest.raises(RuntimeError):
        mod._run_overpass_with_retries("roads", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    assert getattr(ox.settings, attr) == before_ep, "endpoint не відновлено після прогону"
    assert getattr(ox.settings, to_attr) == before_to, "timeout не відновлено після прогону"


@pytest.mark.parametrize("mod", [data_loader, extras_loader], ids=["data_loader", "extras_loader"])
def test_success_on_mirror_stops_further_attempts(mod, monkeypatch):
    monkeypatch.setattr(mod, "_overpass_endpoints", lambda: ["https://dead.example/api", "https://alive.example/api", "https://third.example/api"])
    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)
    attr = "overpass_url" if hasattr(ox.settings, "overpass_url") else "overpass_endpoint"
    calls: list[str] = []

    class _Ok:
        empty = False

    def fetch():
        ep = getattr(ox.settings, attr)
        calls.append(ep)
        if "dead" in ep:
            raise RuntimeError("down")
        return _Ok()

    assert isinstance(mod._run_overpass_with_retries("water", fetch), _Ok)
    assert calls == ["https://dead.example/api", "https://alive.example/api"], calls


def test_real_osmnx_setting_names_are_known():
    """Щит на майбутні оновлення osmnx: якщо імена знову зміняться, впаде саме
    цей тест, а не мовчазний фолбек на один-єдиний хост."""
    assert hasattr(ox.settings, "overpass_url") or hasattr(ox.settings, "overpass_endpoint")
    assert hasattr(ox.settings, "requests_timeout") or hasattr(ox.settings, "timeout")


def test_site_preview_uses_the_real_timeout_setting():
    """`site_preview` навмисно ставить КОРОТКИЙ таймаут (25 с) для швидкого
    превʼю, але писав у `ox.settings.timeout` — неіснуючий атрибут у поточній
    osmnx, тож фактично чекав дефолтні 180 с.

    Модуль читаємо ТЕКСТОМ, а не імпортом: `site_preview` наразі не імпортується
    взагалі (тягне `run_canonical_preview_pipeline`, якого немає у
    `full_generation_pipeline`) і ніде в застосунку не підключений — мертвий код.
    Тест тримає саме правило про ім'я налаштування."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "services" / "site_preview.py").read_text(encoding="utf-8")
    assert "requests_timeout" in src, "site_preview має знати реальне ім'я налаштування"
    assert "ox.settings.timeout =" not in src, "лишилось присвоєння у неіснуючий атрибут"
