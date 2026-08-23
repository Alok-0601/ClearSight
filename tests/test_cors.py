from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_github_pages_origin_is_allowed() -> None:
    response = client.options(
        "/verify",
        headers={
            "Origin": "https://alok-0601.github.io",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://alok-0601.github.io"


def test_streamlit_origin_is_allowed() -> None:
    response = client.options(
        "/verify",
        headers={
            "Origin": "https://clearsightt.streamlit.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://clearsightt.streamlit.app"
