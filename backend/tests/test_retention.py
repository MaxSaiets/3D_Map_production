"""Unit tests for services/retention.py — model-file cleanup by age, with a
longer grace period for files referenced by an order."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from services import retention


def _touch(path: Path, age_days: float, content: bytes = b"x") -> None:
    path.write_bytes(content)
    ts = time.time() - age_days * 86400.0
    os.utime(path, (ts, ts))


@pytest.fixture
def dirs(tmp_path):
    out = tmp_path / "output"
    data = tmp_path / "data"
    out.mkdir()
    data.mkdir()
    return out, data


def test_old_unreferenced_model_deleted(dirs, monkeypatch):
    out, data = dirs
    monkeypatch.setattr(retention, "MODEL_RETENTION_DAYS", 30)
    monkeypatch.setattr(retention, "RETENTION_KEEP_ORDERED_DAYS", 1095)

    old_file = out / "model_100_deadbeef_cafebabe.3mf"
    _touch(old_file, age_days=40)

    res = retention.run_retention(out, data)
    assert not old_file.exists()
    assert res["deleted"] == 1
    assert res["freed_bytes"] > 0


def test_new_model_kept(dirs, monkeypatch):
    out, data = dirs
    monkeypatch.setattr(retention, "MODEL_RETENTION_DAYS", 30)
    monkeypatch.setattr(retention, "RETENTION_KEEP_ORDERED_DAYS", 1095)

    new_file = out / "model_100_deadbeef_cafebabe.3mf"
    _touch(new_file, age_days=5)

    res = retention.run_retention(out, data)
    assert new_file.exists()
    assert res["deleted"] == 0


def test_ordered_file_kept_past_default_retention(dirs, monkeypatch):
    out, data = dirs
    monkeypatch.setattr(retention, "MODEL_RETENTION_DAYS", 30)
    monkeypatch.setattr(retention, "RETENTION_KEEP_ORDERED_DAYS", 1095)

    task_id = "12345678-aaaa-bbbb-cccc-1234567890ab"
    short = task_id.replace("-", "")[:8]
    ordered_file = out / f"model_100_{short}_cafebabe.3mf"
    _touch(ordered_file, age_days=200)  # older than 30d, younger than 1095d

    orders_log = data / "orders.jsonl"
    orders_log.write_text(
        json.dumps({"order_number": "1001", "task_id": task_id}) + "\n",
        encoding="utf-8",
    )

    res = retention.run_retention(out, data)
    assert ordered_file.exists()
    assert res["kept_ordered"] == 1
    assert res["deleted"] == 0


def test_ordered_file_expires_after_long_retention(dirs, monkeypatch):
    out, data = dirs
    monkeypatch.setattr(retention, "MODEL_RETENTION_DAYS", 30)
    monkeypatch.setattr(retention, "RETENTION_KEEP_ORDERED_DAYS", 60)

    task_id = "12345678-aaaa-bbbb-cccc-1234567890ab"
    short = task_id.replace("-", "")[:8]
    ordered_file = out / f"model_100_{short}_cafebabe.3mf"
    _touch(ordered_file, age_days=100)

    orders_log = data / "orders.jsonl"
    orders_log.write_text(
        json.dumps({"order_number": "1001", "task_id": task_id}) + "\n",
        encoding="utf-8",
    )

    res = retention.run_retention(out, data)
    assert not ordered_file.exists()
    assert res["deleted"] == 1


def test_protected_names_never_deleted(dirs, monkeypatch):
    out, data = dirs
    monkeypatch.setattr(retention, "MODEL_RETENTION_DAYS", 1)
    monkeypatch.setattr(retention, "RETENTION_KEEP_ORDERED_DAYS", 1095)

    protected = [
        out / "users.json",
        out / "orders.jsonl",
        out / "analytics.jsonl",
        out / "panel_batches.json",
        out / "pricing_ua.json",
        out / "golden_baseline.json",
        out / ".gitkeep",
    ]
    for p in protected:
        _touch(p, age_days=500)

    res = retention.run_retention(out, data)
    for p in protected:
        assert p.exists(), f"{p.name} should never be deleted by retention"
    assert res["deleted"] == 0


def test_print_files_and_layout_dir_expire(dirs, monkeypatch):
    out, data = dirs
    monkeypatch.setattr(retention, "MODEL_RETENTION_DAYS", 30)
    monkeypatch.setattr(retention, "RETENTION_KEEP_ORDERED_DAYS", 1095)

    uuid_ = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    files = [
        out / f"{uuid_}_print_acceptance.json",
        out / f"{uuid_}_print_check_config.ini",
        out / f"{uuid_}_print_package.zip",
    ]
    for f in files:
        _touch(f, age_days=40)
    layout_dir = out / f"{uuid_}_print_layout_parts"
    layout_dir.mkdir()
    _touch(layout_dir / "part1.png", age_days=40)

    res = retention.run_retention(out, data)
    for f in files:
        assert not f.exists()
    assert not layout_dir.exists()
    assert res["deleted"] == 4  # 3 files + 1 dir


def test_share_preview_png_expires(dirs, monkeypatch):
    out, data = dirs
    monkeypatch.setattr(retention, "MODEL_RETENTION_DAYS", 30)
    monkeypatch.setattr(retention, "RETENTION_KEEP_ORDERED_DAYS", 1095)

    previews = out / "previews"
    previews.mkdir()
    old_png = previews / "abcd1234efgh5678.png"
    _touch(old_png, age_days=40)
    new_png = previews / "freshtaskid12345.png"
    _touch(new_png, age_days=1)

    res = retention.run_retention(out, data)
    assert not old_png.exists()
    assert new_png.exists()
    assert res["deleted"] == 1


def test_dry_run_deletes_nothing(dirs, monkeypatch):
    out, data = dirs
    monkeypatch.setattr(retention, "MODEL_RETENTION_DAYS", 30)
    monkeypatch.setattr(retention, "RETENTION_KEEP_ORDERED_DAYS", 1095)

    old_file = out / "model_100_deadbeef_cafebabe.3mf"
    _touch(old_file, age_days=40)

    res = retention.run_retention(out, data, dry_run=True)
    assert old_file.exists()
    assert res["deleted"] == 1
    assert res["freed_bytes"] > 0


def test_disabled_when_zero(dirs, monkeypatch):
    out, data = dirs
    monkeypatch.setattr(retention, "MODEL_RETENTION_DAYS", 0)

    old_file = out / "model_100_deadbeef_cafebabe.3mf"
    _touch(old_file, age_days=400)

    res = retention.run_retention(out, data)
    assert old_file.exists()
    assert res == {"deleted": 0, "freed_bytes": 0, "kept_ordered": 0, "errors": 0}


def test_unrelated_file_pattern_ignored(dirs, monkeypatch):
    out, data = dirs
    monkeypatch.setattr(retention, "MODEL_RETENTION_DAYS", 1)
    monkeypatch.setattr(retention, "RETENTION_KEEP_ORDERED_DAYS", 1095)

    weird = out / "some_random_notes.txt"
    _touch(weird, age_days=400)

    res = retention.run_retention(out, data)
    assert weird.exists()
    assert res["deleted"] == 0
