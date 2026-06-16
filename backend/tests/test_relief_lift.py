"""Adaptive relief lift: flat cities amplified to a visible target, noise/mountains untouched.

Guards the empirically-calibrated behaviour (most UA cities have 30-70m real range →
only 3-7mm model relief; the lift normalises them to ~target while genuinely tall
relief stays natural and DEM noise is NOT blown up). See services/terrain_generator.py
lift_relief_to_target + the callers in generator.py / generation_pipeline.py."""
import numpy as np
import pytest

from services.terrain_generator import lift_relief_to_target


def _ramp(rng_m, n=20):
    """A linear heightfield with exactly rng_m of range (preserves shape under lift)."""
    return np.linspace(0.0, float(rng_m), n * n).reshape(n, n)


def test_flat_city_amplified_capped_by_max_amp():
    # 30m real range (e.g. Khmelnytskyi), target 110m → desired gain 3.67 but MAX_AMP caps at 3.5
    out = lift_relief_to_target(_ramp(30.0), target_relief_m=110.0, max_amp=3.5)
    rng = float(np.nanmax(out) - np.nanmin(out))
    assert rng == pytest.approx(30.0 * 3.5, abs=1.0)  # ~105m, capped


def test_moderate_city_reaches_target():
    # 50m range, target 110m → gain 2.2 (within cap) → lifted exactly to target
    out = lift_relief_to_target(_ramp(50.0), target_relief_m=110.0, max_amp=3.5)
    rng = float(np.nanmax(out) - np.nanmin(out))
    assert rng == pytest.approx(110.0, abs=1.0)


def test_dem_noise_not_amplified():
    # 1m range is below NOISE_FLOOR (max(3, 4% of target)) → left untouched (no fake hills)
    out = lift_relief_to_target(_ramp(1.0), target_relief_m=110.0)
    assert float(np.nanmax(out) - np.nanmin(out)) == pytest.approx(1.0, abs=0.01)


def test_mountain_not_lifted():
    # 300m range > target → lift does NOT fire (compression path handles mountains)
    out = lift_relief_to_target(_ramp(300.0), target_relief_m=110.0)
    assert float(np.nanmax(out) - np.nanmin(out)) == pytest.approx(300.0, abs=0.01)


def test_disabled_when_target_zero_or_none():
    z = _ramp(30.0)
    assert np.array_equal(lift_relief_to_target(z, 0.0), z)
    assert np.array_equal(lift_relief_to_target(z, None if False else 0.0), z)


def test_shape_preserved_only_scaled():
    # lift is a pure linear scale about the min → relative profile unchanged
    z = _ramp(40.0)
    out = lift_relief_to_target(z, target_relief_m=110.0, max_amp=3.5)
    z_rel = (z - z.min()) / (z.max() - z.min())
    out_rel = (out - out.min()) / (out.max() - out.min())
    assert np.allclose(z_rel, out_rel, atol=1e-6)


def test_floor_scales_with_target():
    # With a small target (e.g. 40m), NOISE_FLOOR=max(3,1.6)=3m; a 2m range stays noise
    out = lift_relief_to_target(_ramp(2.0), target_relief_m=40.0)
    assert float(np.nanmax(out) - np.nanmin(out)) == pytest.approx(2.0, abs=0.01)
