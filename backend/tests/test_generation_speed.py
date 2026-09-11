"""Два прискорення генерації (12.09.2026), заміряні на проді 11.09.2026.

1. Overpass лежить → osmnx перед /status спить 60 с НА КОЖЕН ШАР (14:59:59 green →
   15:00:57 building_parts → 15:01:58 bridges). Передполітна TCP-перевірка
   (`overpass_health.preflight`) має відмовляти за секунди, а запобіжник —
   вмикатись з тієї самої помилки.
2. `TerrariumTileProvider.sample_points` був Python-циклом по ~500 тис. точок
   (24.6 с «Отримання з API»). Векторна версія має давати ТІ САМІ числа, що й
   еталонна `_bilinear_sample`, включно з краями тайлів.
"""
from __future__ import annotations

import socket
import time

import numpy as np
import pytest

from services import overpass_health as oh
from services import terrarium_tiles as tt


@pytest.fixture(autouse=True)
def _reset_breaker():
    oh.reset()
    yield
    oh.reset()


# ── 1. передполітна перевірка ────────────────────────────────────────────────

def test_preflight_disabled_by_env_returns_none(monkeypatch):
    monkeypatch.setenv("OVERPASS_PREFLIGHT", "0")
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: (_ for _ in ()).throw(AssertionError("не мало викликатись")))
    assert oh.preflight("https://overpass-api.de/api") is None


def test_preflight_refused_is_a_connection_error(monkeypatch):
    monkeypatch.setenv("OVERPASS_PREFLIGHT", "1")

    def refused(addr, timeout=None):
        assert addr == ("overpass-api.de", 443)
        assert timeout and timeout <= 3.0, "таймаут має бути секунди, не хвилина"
        raise ConnectionRefusedError(111, "Connection refused")

    monkeypatch.setattr(socket, "create_connection", refused)
    exc = oh.preflight("https://overpass-api.de/api")
    assert exc is not None
    assert oh.is_connection_error(exc), "запобіжник має зарахувати це як аварію зʼєднання"


def test_preflight_ok_when_host_listens(monkeypatch):
    monkeypatch.setenv("OVERPASS_PREFLIGHT", "1")

    class _Sock:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: _Sock())
    assert oh.preflight("https://overpass.kumi.systems/api") is None


def test_preflight_skips_localhost(monkeypatch):
    monkeypatch.setenv("OVERPASS_PREFLIGHT", "1")
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: (_ for _ in ()).throw(AssertionError("localhost не перевіряємо")))
    assert oh.preflight("http://localhost:12345/api") is None


def test_dead_host_fails_in_seconds_not_minutes(monkeypatch):
    """Головна перевірка: із мертвим хостом цикл повторів НЕ доходить до osmnx
    (де сидить 60-секундний сон), а після порогу запобіжника решта шарів
    відмовляє миттєво з OverpassUnavailableError."""
    from services import data_loader as dl

    monkeypatch.setenv("OVERPASS_PREFLIGHT", "1")
    monkeypatch.setenv("OVERPASS_BREAKER_FAILS", "2")
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: (_ for _ in ()).throw(ConnectionRefusedError(111, "Connection refused")))
    monkeypatch.setattr(dl, "_overpass_endpoints", lambda: ["https://overpass-api.de/api"])
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)

    calls = []

    def fetch_fn():
        calls.append(1)
        raise AssertionError("osmnx не мав викликатись: хост мертвий")

    t0 = time.monotonic()
    with pytest.raises(Exception):
        dl._run_overpass_with_retries("buildings", fetch_fn)     # 1-ша помилка
    with pytest.raises(oh.OverpassUnavailableError):
        dl._run_overpass_with_retries("roads", fetch_fn)         # 2-га → запобіжник
    with pytest.raises(oh.OverpassUnavailableError):
        dl._run_overpass_with_retries("water", fetch_fn)         # миттєво, без спроби
    assert time.monotonic() - t0 < 2.0
    assert calls == [], "жодного виклику osmnx"
    assert oh.outage_active()


# ── 2. векторна вибірка висот ────────────────────────────────────────────────

def _fake_provider(monkeypatch, seed=0):
    rng = np.random.default_rng(seed)
    tiles = {}

    def get_tile(self, key):
        if key not in tiles:
            tiles[key] = (rng.random((256, 256)) * 1000.0 + key.x * 7 + key.y * 3).astype(np.float32)
        return tiles[key]

    monkeypatch.setattr(tt.TerrariumTileProvider, "get_tile", get_tile)
    monkeypatch.setattr(tt.TerrariumTileProvider, "prefetch", lambda self, keys, workers=8: None)
    return tt.TerrariumTileProvider(cache_dir=str(__import__("tempfile").mkdtemp())), tiles


def _reference(provider, lats, lons, z):
    """Стара поточкова реалізація — еталон."""
    out = np.full(len(lats), np.nan, dtype=np.float32)
    for i in range(len(lats)):
        gx, gy = tt._latlon_to_global_pixel(float(lons[i]), float(lats[i]), z)
        tx, ty, px, py = tt._global_pixel_to_tile(gx, gy)
        tile = provider.get_tile(tt.TileKey(z=z, x=tx, y=ty))
        out[i] = tt._bilinear_sample(tile, px, py)
    return out


def test_vectorized_matches_reference_including_tile_edges(monkeypatch):
    provider, _ = _fake_provider(monkeypatch)
    z = 12
    rng = np.random.default_rng(1)
    lats = rng.uniform(48.0, 48.6, 4000)
    lons = rng.uniform(24.0, 24.9, 4000)
    # точки рівно на межах тайлів — там px = 0 або 255.999…
    n = 256.0 * (2 ** z)
    edge_lon = (np.arange(3) + 2321) * 256.0 / n * 360.0 - 180.0
    lats = np.concatenate([lats, np.full(3, 48.3)]); lons = np.concatenate([lons, edge_lon])
    got = provider.sample_points(lats, lons, z)
    ref = _reference(provider, lats, lons, z)
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-3)


def test_vectorized_handles_2d_input_and_empty(monkeypatch):
    provider, _ = _fake_provider(monkeypatch)
    lat2d = np.full((5, 7), 48.2); lon2d = np.linspace(24.0, 24.1, 35).reshape(5, 7)
    got = provider.sample_points(lat2d, lon2d, 12)
    assert got.shape == (35,) and np.isfinite(got).all()
    assert provider.sample_points(np.array([]), np.array([]), 12).size == 0


def test_missing_tile_leaves_nan_but_others_sampled(monkeypatch):
    provider, tiles = _fake_provider(monkeypatch)
    real = tt.TerrariumTileProvider.get_tile

    def flaky(self, key):
        return None if key.x % 2 == 0 else real(self, key)

    monkeypatch.setattr(tt.TerrariumTileProvider, "get_tile", flaky)
    lons = np.linspace(24.0, 24.9, 400); lats = np.full(400, 48.3)
    got = provider.sample_points(lats, lons, 12)
    assert np.isnan(got).any() and np.isfinite(got).any()


def test_vectorized_is_fast_on_a_relief_grid(monkeypatch):
    """700×700 = 490 тис. точок (сітка рельєфу друку) — секунди, не десятки."""
    provider, _ = _fake_provider(monkeypatch)
    lat, lon = np.meshgrid(np.linspace(48.0, 48.05, 700), np.linspace(24.0, 24.08, 700))
    t0 = time.monotonic()
    got = provider.sample_points(lat.ravel(), lon.ravel(), 15)
    assert time.monotonic() - t0 < 5.0
    assert np.isfinite(got).all()
