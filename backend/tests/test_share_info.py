"""
Тести для GET /api/share/{task_id} (публічні метадані для сторінки шерингу).
"""
import pytest
from fastapi.testclient import TestClient

import main
from main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def tmp_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    main.tasks.clear()
    yield tmp_path
    main.tasks.clear()


TASK_ID = "abcd1234efgh5678"  # matches _SHARE_ID_RE (8-64 [A-Za-z0-9_-])


class TestShareInfo:
    def test_bad_id_returns_400(self, client, tmp_output_dir):
        resp = client.get("/api/share/x")  # too short for _SHARE_ID_RE
        assert resp.status_code == 400

    def test_nothing_found_returns_404(self, client, tmp_output_dir):
        resp = client.get(f"/api/share/{TASK_ID}")
        assert resp.status_code == 404

    def test_png_only(self, client, tmp_output_dir):
        previews = tmp_output_dir / "previews"
        previews.mkdir(parents=True, exist_ok=True)
        (previews / f"{TASK_ID}.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        resp = client.get(f"/api/share/{TASK_ID}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == TASK_ID
        assert data["png_url"] == f"/files/previews/{TASK_ID}.png"
        assert data["glb_url"] is None
        assert data["product"] is None

    def test_glb_only(self, client, tmp_output_dir):
        short = TASK_ID.replace("-", "")[:8]
        (tmp_output_dir / f"model_80_{short}_deadbeef.glb").write_bytes(b"glTF")

        resp = client.get(f"/api/share/{TASK_ID}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["glb_url"] == f"/files/model_80_{short}_deadbeef.glb"
        assert data["png_url"] is None

    def test_3mf_only_does_not_leak_as_glb(self, client, tmp_output_dir):
        """Коли на диску є лише .3mf (друкований формат без live-прев'ю),
        glb_url має лишитись null, а не хибно показати 3mf як glb."""
        short = TASK_ID.replace("-", "")[:8]
        (tmp_output_dir / f"model_80_{short}_deadbeef.3mf").write_bytes(b"PK\x03\x04")

        resp = client.get(f"/api/share/{TASK_ID}")
        assert resp.status_code == 404  # ні glb, ні png

    def test_product_from_in_memory_task(self, client, tmp_output_dir):
        previews = tmp_output_dir / "previews"
        previews.mkdir(parents=True, exist_ok=True)
        (previews / f"{TASK_ID}.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        class _Req:
            keychain_mode = True

        class _Task:
            request = _Req()

        main.tasks[TASK_ID] = _Task()

        resp = client.get(f"/api/share/{TASK_ID}")
        assert resp.status_code == 200
        assert resp.json()["product"] == "keychain"
