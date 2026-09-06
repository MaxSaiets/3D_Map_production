"""
Тести для нічного прогріву кешу шаблонів (services/template_warm.py).

Покриває:
  (a) persist_template_body → пише DATA_DIR/template_bodies.json, тіло
      санітизоване (заборонені ключі викинуто, лише JSON-примітиви).
  (b) next_warm_delay(now, hour) — чиста функція обчислення затримки.
  (c) run_template_warm — пропускає записи з попаданням у кеш
      (lookup_fn), для промахів викликає post_fn і опитує status_fn.
"""
from datetime import datetime

import pytest

from services import template_warm


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(template_warm, "DATA_DIR", tmp_path)
    monkeypatch.setattr(template_warm, "BODIES_PATH", tmp_path / "template_bodies.json")
    return tmp_path


class TestSanitizeAndPersist:
    def test_persist_writes_sanitized_body(self, tmp_data_dir):
        body = {
            "north": 50.45,
            "south": 50.40,
            "template_id": "kyiv-center",
            "screenshots": ["data:image/png;base64,AAA"],
            "auth_token": "secret-value",
            "nested": {"password": "x", "keep_me": 1.5},
            "some_list": [1, 2, {"token": "bad", "ok": True}],
        }
        template_warm.persist_template_body("kyiv-center", body, "cachekey123")

        assert template_warm.BODIES_PATH.exists()
        import json

        data = json.loads(template_warm.BODIES_PATH.read_text(encoding="utf-8"))
        assert "kyiv-center" in data
        entry = data["kyiv-center"]
        assert entry["cache_key"] == "cachekey123"
        assert "saved_at" in entry

        saved_body = entry["body"]
        assert saved_body["north"] == 50.45
        assert saved_body["south"] == 50.40
        # Заборонені ключі мають бути відсутні на всіх рівнях
        assert "screenshots" not in saved_body
        assert "auth_token" not in saved_body
        assert "password" not in saved_body["nested"]
        assert saved_body["nested"]["keep_me"] == 1.5
        assert saved_body["some_list"][2]["ok"] is True
        assert "token" not in saved_body["some_list"][2]

    def test_persist_merges_and_caps_entries(self, tmp_data_dir):
        for i in range(45):
            template_warm.persist_template_body(f"tpl-{i}", {"x": i}, f"key-{i}")
        data = template_warm.load_template_bodies()
        assert len(data) == template_warm.MAX_ENTRIES

    def test_persist_ignores_non_dict_body(self, tmp_data_dir):
        template_warm.persist_template_body("bad", "not-a-dict", "k")
        assert template_warm.load_template_bodies() == {}

    def test_load_missing_file_returns_empty(self, tmp_data_dir):
        assert template_warm.load_template_bodies() == {}


class TestNextWarmDelay:
    def test_target_later_today(self):
        now = datetime(2026, 9, 6, 1, 0, 0)
        delay = template_warm.next_warm_delay(now, 3)
        assert delay == pytest.approx(2 * 3600)

    def test_target_already_passed_rolls_to_tomorrow(self):
        now = datetime(2026, 9, 6, 5, 0, 0)
        delay = template_warm.next_warm_delay(now, 3)
        assert delay == pytest.approx(22 * 3600)

    def test_target_exact_hour_rolls_to_tomorrow(self):
        now = datetime(2026, 9, 6, 3, 0, 0)
        delay = template_warm.next_warm_delay(now, 3)
        assert delay == pytest.approx(24 * 3600)

    def test_clamps_out_of_range_hour(self):
        now = datetime(2026, 9, 6, 0, 0, 0)
        delay = template_warm.next_warm_delay(now, 30)  # clamps to 23
        assert delay == pytest.approx(23 * 3600)


class TestRunTemplateWarm:
    def test_skips_cache_hit_and_warms_miss(self):
        bodies = {
            "hot": {"body": {"a": 1}, "cache_key": "hot-key"},
            "cold": {"body": {"b": 2}, "cache_key": "cold-key"},
        }

        def lookup_fn(key):
            return {"output_file": "x"} if key == "hot-key" else None

        posted = []

        def post_fn(body):
            posted.append(body)
            return "task-cold-1"

        statuses = iter(["processing", "completed"])

        def status_fn(task_id):
            assert task_id == "task-cold-1"
            return next(statuses)

        sleeps = []
        results = template_warm.run_template_warm(
            post_fn, status_fn, lookup_fn, bodies,
            sleep_fn=lambda s: sleeps.append(s),
            poll_interval_s=1.0,
        )

        results_dict = dict(results)
        assert results_dict["hot"] == "skip_cached"
        assert results_dict["cold"] == "completed"
        assert posted == [{"b": 2}]
        # No poster call for the cached template.
        assert len(posted) == 1

    def test_timeout_when_status_never_completes(self):
        bodies = {"stuck": {"body": {"a": 1}, "cache_key": "k"}}

        def lookup_fn(_key):
            return None

        def post_fn(_body):
            return "task-stuck"

        def status_fn(_task_id):
            return "processing"

        results = template_warm.run_template_warm(
            post_fn, status_fn, lookup_fn, bodies,
            sleep_fn=lambda s: None,
            poll_interval_s=100.0,
            poll_timeout_s=250.0,
        )
        assert dict(results)["stuck"] == "timeout"

    def test_missing_body_reports_error_without_raising(self):
        bodies = {"broken": {"cache_key": "k"}}  # no "body" key

        results = template_warm.run_template_warm(
            lambda b: "tid", lambda t: "completed", lambda k: None, bodies,
            sleep_fn=lambda s: None,
        )
        assert dict(results)["broken"] == "error:no_body"

    def test_poster_exception_does_not_raise(self):
        bodies = {"x": {"body": {"a": 1}, "cache_key": "k"}}

        def post_fn(_body):
            raise RuntimeError("boom")

        results = template_warm.run_template_warm(
            post_fn, lambda t: "completed", lambda k: None, bodies,
            sleep_fn=lambda s: None,
        )
        outcome = dict(results)["x"]
        assert outcome.startswith("error:")

    def test_never_raises_out_of_loop(self):
        bodies = {"x": {"body": {"a": 1}, "cache_key": "k"}}

        def post_fn(_body):
            return None  # no task_id

        results = template_warm.run_template_warm(
            post_fn, lambda t: "completed", lambda k: None, bodies,
            sleep_fn=lambda s: None,
        )
        assert dict(results)["x"] == "error:no_task_id"
