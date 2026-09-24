from fastapi.testclient import TestClient

from backend.main import app


def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "device" in body
    assert "stages" in body


def test_list_videos_endpoint():
    client = TestClient(app)
    resp = client.get("/videos")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
