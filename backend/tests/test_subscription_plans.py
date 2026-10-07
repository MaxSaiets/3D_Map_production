"""07.10.2026: /api/subscription/plans віддає ціну ОДНОГО друк-файлу поруч із підпискою —
калькулятор на /pro («з якого файлу Pro вигідніший») рахує з тієї ж ціни, що й чек файлу."""
from fastapi.testclient import TestClient

from main import app
from services.file_access import file_price_uah


def test_plans_include_file_price_and_usd_hint():
    r = TestClient(app).get("/api/subscription/plans", headers={"cf-ipcountry": "UA"})
    assert r.status_code == 200
    j = r.json()
    assert j["suggested"] == "UAH"
    assert j["plans"]["UAH"] > 0 and j["plans"]["USD"] > 0
    assert j["file"]["UAH"] == file_price_uah()
    # підказка в доларах — з курсу pricing.json; якщо курсу нема, ключа просто немає
    if "USD" in j["file"]:
        assert 0 < j["file"]["USD"] < j["file"]["UAH"]


def test_plans_suggest_usd_outside_ukraine():
    j = TestClient(app).get("/api/subscription/plans", headers={"cf-ipcountry": "DE"}).json()
    assert j["suggested"] == "USD"
