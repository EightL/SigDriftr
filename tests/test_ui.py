import pytest


pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient

from main import app


def test_ui_route_serves_demo_page() -> None:
    with TestClient(app) as client:
        response = client.get("/ui")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "SIGDRIFTR" in response.text
    assert "Cluster-Aware Output Layer" in response.text
    assert 'id="compareToggle"' in response.text
    assert 'id="scopeTopic2"' in response.text
    assert "Stage 7 UI" in response.text
    assert "Run Analysis" in response.text
