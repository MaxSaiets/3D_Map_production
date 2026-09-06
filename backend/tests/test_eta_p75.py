"""P-1: ETA = 75-й перцентиль останніх 15 вимірів (не медіана) — недообіцяємо."""
from services import result_cache as rc


def _set(bucket, values):
    rc._load()
    rc._stats[bucket] = list(values)


def test_eta_uses_p75_not_median():
    _set("t:p75", [40, 58.7, 51.2, 39.8, 80])
    # відсортовано: 39.8, 40, 51.2, 58.7, 80 → індекс round(0.75*4)=3 → 58.7 → 59
    assert rc.eta_seconds("t:p75") == 59


def test_eta_falls_back_to_default_below_three_samples():
    _set("preview:t", [10, 20])
    assert rc.eta_seconds("preview:t") == int(round(rc._ETA_DEFAULTS.get("preview", 90)))


def test_eta_uses_only_last_15():
    _set("t:last15", [1000] * 10 + [30] * 15)
    assert rc.eta_seconds("t:last15") == 30
