from fastapi.testclient import TestClient

from beekeeper_intel.api.app import create_app


def test_app_allows_local_frontend_cors_origin():
    client = TestClient(create_app())

    response = client.options(
        "/query",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"


def test_app_allows_onrender_preview_origin_via_regex():
    client = TestClient(create_app())

    response = client.options(
        "/query",
        headers={
            "Origin": "https://beekeeper-demo.onrender.com",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://beekeeper-demo.onrender.com"


def test_research_report_returns_structured_results_and_distributions():
    client = TestClient(create_app())

    response = client.post(
        "/research/report",
        json={
            "query": "Compare hobbyist vs commercial pain points in varroa monitoring and unmet needs.",
            "session_id": "api-report-1",
            "include_trace": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "research_synthesis"
    assert payload["results"]
    assert payload["results"][0]["statement"]
    assert payload["results"][0]["persona"]
    assert payload["results"][0]["source_titles"]
    assert payload["results"][0]["evidence_count"] >= 1
    assert payload["results"][0]["frequency_1_5"] >= 1
    assert "evidence_density" in payload["distributions"]
    assert len(payload["results"]) >= 7
    assert len(payload["citations"]) >= 7
