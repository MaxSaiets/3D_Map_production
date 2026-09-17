"""Повторний клік «Створити» під час генерації не плодить однакових задач.

Прод 11.09.2026 18:50:51–58 UTC: шість POST /api/generate з ТИМИ САМИМИ
координатами за сім секунд (розширений режим, кнопка без гарду) → шість задач
у черзі на 2-ядерній VM. Тепер /api/generate повертає вже живу задачу.
"""
from __future__ import annotations

import uuid
from datetime import timedelta

import main
from services.generation_task import GenerationTask


def _task(key, status="processing", age_s=0.0, cancelled=False):
    t = GenerationTask(task_id=str(uuid.uuid4()), request=None)
    t.status = status
    t.cache_key = key
    t.cancelled = cancelled
    t.created_at = t.created_at - timedelta(seconds=age_s)
    main.tasks[t.task_id] = t
    return t


def test_finds_live_task_with_same_key(monkeypatch):
    monkeypatch.setattr(main, "tasks", {})
    live = _task("k1", "processing")
    assert main._find_inflight_task("k1") is live
    assert main._find_inflight_task("k1", exclude_task_id=live.task_id) is None
    assert main._find_inflight_task("other") is None


def test_ignores_done_cancelled_and_stale(monkeypatch):
    monkeypatch.setattr(main, "tasks", {})
    _task("k", "completed")
    _task("k", "failed")
    _task("k", "processing", cancelled=True)
    _task("k", "processing", age_s=3600)  # зависла після рестарту — не перехоплює
    assert main._find_inflight_task("k") is None
    q = _task("k", "queued")
    assert main._find_inflight_task("k") is q


def test_endpoint_returns_existing_task_for_repeat_click(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(main, "tasks", {})
    # фонова генерація не потрібна: задача просто лишається «processing»
    monkeypatch.setattr(main, "generate_model_task", lambda *a, **k: None)
    monkeypatch.setattr(main._rc, "lookup", lambda key: None)
    client = TestClient(main.app)
    body = {"north": 48.2100, "south": 48.2050, "east": 16.3750, "west": 16.3680, "model_size_mm": 80}
    first = client.post("/api/generate", json=body).json()
    second = client.post("/api/generate", json=body).json()
    third = client.post("/api/generate", json={**body, "model_size_mm": 100}).json()
    assert first["status"] == "processing" and not first["deduplicated"]
    assert second["task_id"] == first["task_id"] and second["deduplicated"] is True
    assert third["task_id"] != first["task_id"], "інші параметри — інша задача"
    assert len(main.tasks) == 2


# ── «Скасувати» для гостя ─────────────────────────────────────────────────
def test_guest_can_cancel_anonymous_task(monkeypatch):
    """Прод 13.09: DELETE /api/task без входу → 401 → генерація крутила VM далі,
    а повторний «Створити» став у чергу за нею. Гість має скасовувати свою задачу."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(main, "tasks", {})
    t = _task("k", "processing")
    client = TestClient(main.app)
    r = client.delete(f"/api/task/{t.task_id}")
    assert r.status_code == 200 and r.json()["cancelled"] is True
    assert t.cancelled and t.status == "cancelled"
    assert main._find_inflight_task("k") is None, "скасована задача більше не перехоплює нові запити"


def test_owned_task_still_needs_login(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(main, "tasks", {})
    t = _task("k", "processing")
    t.owner_uid = "uid-1"
    client = TestClient(main.app)
    assert client.delete(f"/api/task/{t.task_id}").status_code == 401
    assert not t.cancelled


def test_pipeline_stops_at_stage_boundary_after_cancel():
    """Межа етапу кидає GenerationCancelled, коли прапорець уже стоїть."""
    from services.generation_task import GenerationCancelled
    from services import full_generation_pipeline as fgp
    import inspect

    src = inspect.getsource(fgp.run_full_generation_pipeline)
    assert "raise GenerationCancelled(name)" in src and 'getattr(task, "cancelled", False)' in src
    # і головний раннер її ловить без позначки «помилка»
    main_src = inspect.getsource(main.generate_model_task)
    assert "except _GenerationCancelled" in main_src
    assert issubclass(GenerationCancelled, Exception)
