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
