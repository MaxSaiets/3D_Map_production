"""Unit tests for services/user_store.py — model expiry flag + account deletion."""
from __future__ import annotations

import json

import pytest

from services import user_store


@pytest.fixture
def store(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    data_dir = tmp_path / "data"
    output_dir.mkdir()
    data_dir.mkdir()
    monkeypatch.setattr(user_store, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(user_store, "DATA_DIR", data_dir)
    monkeypatch.setattr(user_store, "USERS_FILE", data_dir / "users.json")
    return output_dir, data_dir


def test_list_models_flags_missing_file_as_expired(store):
    output_dir, _data_dir = store
    uid = "u1"
    user_store.add_model(uid, "a@b.com", {
        "task_id": "task-1", "download_url": "/files/model_100_deadbeef_cafebabe.3mf",
    })
    # File never created on disk → expired
    models = user_store.list_models(uid)
    assert len(models) == 1
    assert models[0]["expired"] is True

    # Now create the backing file → no longer expired
    (output_dir / "model_100_deadbeef_cafebabe.3mf").write_bytes(b"x")
    models2 = user_store.list_models(uid)
    assert "expired" not in models2[0]


def test_list_models_keeps_entries_when_deleted(store):
    """expired entries are flagged, never removed from history."""
    uid = "u1"
    user_store.add_model(uid, "a@b.com", {"task_id": "t1", "download_url": "/files/missing.3mf"})
    user_store.add_model(uid, "a@b.com", {"task_id": "t2", "download_url": "/files/missing2.3mf"})
    models = user_store.list_models(uid)
    assert len(models) == 2
    assert all(m["expired"] for m in models)


def test_delete_user_removes_files_and_record(store):
    output_dir, _data_dir = store
    uid = "u1"
    task_id = "12345678-aaaa-bbbb-cccc-1234567890ab"
    model_file = output_dir / f"model_100_{task_id.replace('-', '')[:8]}_cafebabe.3mf"
    model_file.write_bytes(b"x")
    user_store.add_model(uid, "a@b.com", {
        "task_id": task_id, "download_url": f"/files/{model_file.name}",
    })
    # A sibling print-package file referencing the same task id, not directly
    # linked via download_url — should still be swept via token matching.
    sibling = output_dir / f"{task_id}_print_package.zip"
    sibling.write_bytes(b"y")

    res = user_store.delete_user(uid)
    assert res["deleted_models"] == 1
    assert res["deleted_files"] == 2
    assert not model_file.exists()
    assert not sibling.exists()
    assert user_store.list_models(uid) == []
    assert user_store.get_quota(uid, "a@b.com", False)["downloads"] == 0  # fresh record after delete


def test_delete_user_removes_grids(store):
    uid = "u1"
    user_store.save_grid(uid, "a@b.com", {"name": "grid1"})
    assert len(user_store.list_grids(uid)) == 1
    user_store.delete_user(uid)
    assert user_store.list_grids(uid) == []


def test_delete_user_unknown_uid_is_noop(store):
    res = user_store.delete_user("no-such-user")
    assert res == {"deleted_models": 0, "deleted_files": 0}


def test_delete_user_does_not_touch_orders(store):
    """Orders are accounting records kept independent of the account — this
    module never touches orders.jsonl at all."""
    _output_dir, data_dir = store
    orders_log = data_dir / "orders.jsonl"
    orders_log.write_text(json.dumps({"order_number": "1", "uid": "u1"}) + "\n", encoding="utf-8")
    user_store.add_model("u1", "a@b.com", {"task_id": "t1"})
    user_store.delete_user("u1")
    assert orders_log.exists()
    assert "1" in orders_log.read_text(encoding="utf-8")
