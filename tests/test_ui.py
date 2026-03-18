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
    assert "Drift Timeline" in response.text
    assert 'id="compareButton"' in response.text
    assert 'id="topicInput2"' in response.text
    assert "brief-status-banner" in response.text
    assert "insufficient data" in response.text
    assert "warming" in response.text
    assert "ready" in response.text
    assert "compare-grid" in response.text
    assert "getBriefStatusPresentation" in response.text
    assert "Interpret with care" in response.text
    assert "brief-headline is-muted" in response.text
