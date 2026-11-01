import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.routes import verification


client = TestClient(app)
TEXT = "The city council approved funding for three new solar projects on Tuesday after a long debate."


@pytest.fixture(autouse=True)
def no_evidence_keys(monkeypatch):
    monkeypatch.delenv("GOOGLE_FACT_CHECK_API_KEY", raising=False)
    monkeypatch.delenv("GNEWS_API_KEY", raising=False)


@pytest.mark.parametrize(("prediction", "confidence", "verdict"), [
    ("FAKE", 0.58, "UNCERTAIN"),
    ("REAL", 0.649, "UNCERTAIN"),
    ("REAL", 0.65, "REAL"),
    ("FAKE", 0.93, "FAKE"),
])
def test_low_confidence_is_reported_as_uncertain(monkeypatch, prediction, confidence, verdict) -> None:
    monkeypatch.setattr(verification.ml_service, "predict", lambda title, body: (prediction, confidence))
    result = client.post("/verify", json={"text": TEXT}).json()
    assert result["prediction"] == prediction
    assert result["verdict"] == verdict
    history = client.get("/history").json()
    assert history[0]["id"] == result["id"]
    assert history[0]["verdict"] == verdict
