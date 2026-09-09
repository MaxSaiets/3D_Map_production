"""Ціна магніта не залежить від переданого розміру.

⭐Знайдено 09.09.2026 контрольним проходом по живому API:

    /api/quote?product=magnet&size_mm=40  → 350 ₴
    /api/quote?product=magnet&size_mm=50  → 350 ₴
    /api/quote?product=magnet&size_mm=60  → 210 ₴
    /api/quote?product=magnet&size_mm=80  → 490 ₴
    /api/quote?product=magnet             → 350 ₴

Тобто 60-мм магніт коштував ДЕШЕВШЕ за 40-мм, а запит без розміру повертав 350 ₴,
хоча прайс-сторінка і десяток статей блогу обіцяють 210 ₴. Причина: `magnet` не
мав власної гілки й падав у мапну, де ціна береться за НАЙБЛИЖЧИМ розміром із
таблиці {55, 60, 80, 110, 150}.

Покупцеві це не показувалось (guided-флоу питає `product=map&size_mm=60`), але
API суперечив опублікованій ціні. Тест тримає узгодженість.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture(scope="module")
def client():
    return TestClient(main.app)


def _price(client, **params) -> int:
    r = client.get("/api/quote", params=params)
    assert r.status_code == 200, r.text
    return int(r.json()["price"])


def test_magnet_price_is_the_same_for_any_requested_size(client):
    prices = {s: _price(client, product="magnet", size_mm=s) for s in (30, 40, 50, 55, 60, 80, 150)}
    assert len(set(prices.values())) == 1, f"магніт має одну ціну, а вийшло: {prices}"


def test_magnet_without_size_matches_the_published_price(client):
    no_size = _price(client, product="magnet")
    with_size = _price(client, product="magnet", size_mm=60)
    assert no_size == with_size, "запит без розміру не має давати іншу ціну"
    # Прайс-сторінка й статті блогу називають 210 ₴ — звіряємось із таблицею.
    canonical = int((main._load_pricing().get("map", {}).get("sizes_mm", {}) or {}).get("60", 210))
    assert no_size == canonical


def test_magnet_is_cheaper_than_the_smallest_map(client):
    """Здоровий глузд: магніт — найдешевший виріб після брелка."""
    assert _price(client, product="magnet") < _price(client, product="map", size_mm=55)


def test_relief_addon_does_not_apply_to_a_flat_magnet(client):
    assert _price(client, product="magnet", relief=True) == _price(client, product="magnet")


def test_map_and_keychain_pricing_unchanged(client):
    """Щит: правка магніта не мала зачепити решту прайсу."""
    pricing = main._load_pricing()
    sizes = {k: int(v) for k, v in (pricing.get("map", {}).get("sizes_mm", {}) or {}).items()}
    for size, expected in sizes.items():
        assert _price(client, product="map", size_mm=float(size)) == expected
    assert _price(client, product="keychain") == int(pricing.get("keychain", {}).get("base", 170))
    addon = int(pricing.get("map", {}).get("relief_addon", 0))
    assert _price(client, product="map", size_mm=80, relief=True) == sizes["80"] + addon
