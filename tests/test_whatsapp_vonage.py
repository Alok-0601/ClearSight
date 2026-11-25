import asyncio
import base64
import hashlib
import hmac
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.routes import whatsapp_vonage
from backend.routes.verification import store
from backend.schemas import Article
from backend.services import vonage_api


SIGNATURE_SECRET = "test-signature-secret"
SANDBOX_NUMBER = "14157386102"
USER_NUMBER = "447700900001"
ARTICLE_TEXT = (
    "WASHINGTON (Reuters) - The Senate voted on Thursday to pass the bill, which "
    "lawmakers said would fund infrastructure projects through 2030."
)
client = TestClient(app)


class FakeVonageAPI:
    calls: list[tuple] = []
    urls: list[str] = []

    def __init__(self, api_key: str, api_secret: str, url: str) -> None:
        self.urls.append(url)

    async def send_text(self, to, from_, text):
        self.calls.append((to, from_, text))
        return "reply-uuid"


@pytest.fixture(autouse=True)
def bot_env(monkeypatch):
    monkeypatch.setenv("VONAGE_API_KEY", "test-key")
    monkeypatch.setenv("VONAGE_API_SECRET", "test-secret")
    monkeypatch.delenv("VONAGE_SIGNATURE_SECRET", raising=False)
    monkeypatch.delenv("VONAGE_MESSAGES_URL", raising=False)
    monkeypatch.delenv("GOOGLE_FACT_CHECK_API_KEY", raising=False)
    monkeypatch.delenv("GNEWS_API_KEY", raising=False)
    monkeypatch.setattr(whatsapp_vonage, "VonageMessagesAPI", FakeVonageAPI)
    FakeVonageAPI.calls = []
    FakeVonageAPI.urls = []
    whatsapp_vonage._recent_checks.clear()
    whatsapp_vonage._seen_messages.clear()


_next_uuid = iter(range(1, 10_000))


def inbound(text=None, message_type="text", **extra) -> dict:
    """A Messages API v1 inbound WhatsApp webhook body."""
    message = {
        "channel": "whatsapp",
        "message_uuid": f"aaaaaaaa-bbbb-4ccc-8ddd-{next(_next_uuid):012d}",
        "to": SANDBOX_NUMBER,
        "from": USER_NUMBER,
        "timestamp": "2026-11-25T12:00:00Z",
        "profile": {"name": "Test User"},
        "context_status": "none",
        "message_type": message_type,
        **extra,
    }
    if text is not None:
        message["text"] = text
    return message


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def signed_jwt(body: bytes, secret: str = SIGNATURE_SECRET, alg: str = "HS256", payload_hash: str | None = None) -> str:
    header = _b64url(json.dumps({"alg": alg, "typ": "JWT"}).encode())
    claims = _b64url(json.dumps({
        "iat": int(time.time()),
        "jti": "test-jti",
        "iss": "Vonage",
        "payload_hash": payload_hash or hashlib.sha256(body).hexdigest(),
        "api_key": "test-key",
    }).encode())
    signature = _b64url(hmac.new(secret.encode(), f"{header}.{claims}".encode(), hashlib.sha256).digest())
    return f"{header}.{claims}.{signature}"


def post(body: dict, path: str = "/whatsapp/vonage/inbound", token: str | None = None, raw: bytes | None = None):
    content = raw if raw is not None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return client.post(path, content=content, headers=headers)


def test_routes_are_disabled_without_credentials(monkeypatch) -> None:
    monkeypatch.delenv("VONAGE_API_SECRET")
    assert post(inbound("hi")).status_code == 404
    assert post({"status": "delivered"}, path="/whatsapp/vonage/status").status_code == 404
    assert FakeVonageAPI.calls == []


@pytest.mark.parametrize("greeting", ["hi", "Start", "help"])
def test_greeting_gets_help_from_the_sandbox_number(greeting) -> None:
    response = post(inbound(greeting))
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert FakeVonageAPI.calls == [(USER_NUMBER, SANDBOX_NUMBER, whatsapp_vonage.common.HELP_TEXT)]
    assert FakeVonageAPI.urls == [vonage_api.SANDBOX_MESSAGES_URL]


def test_article_text_is_checked_and_answered_once() -> None:
    history_before = len(store.history(100))
    assert post(inbound(ARTICLE_TEXT)).status_code == 200
    assert len(FakeVonageAPI.calls) == 1
    to, from_, reply = FakeVonageAPI.calls[0]
    assert (to, from_) == (USER_NUMBER, SANDBOX_NUMBER)
    assert "*Model verdict: REAL*" in reply
    assert "Evidence search is not configured" in reply
    assert len(reply) <= whatsapp_vonage.MAX_REPLY_LENGTH
    # Chat messages must not appear in the public /history.
    assert len(store.history(100)) == history_before


def test_link_is_extracted_and_checked(monkeypatch) -> None:
    fetched = []

    async def fake_extract(url):
        fetched.append(url)
        return Article(title="Senate passes bill", body=ARTICLE_TEXT, url=url, extraction_method="test")

    monkeypatch.setattr("backend.routes.verification.extract_article", fake_extract)
    post(inbound("look at this https://example.com/news/senate-bill"))
    assert fetched == ["https://example.com/news/senate-bill"]
    assert len(FakeVonageAPI.calls) == 1
    assert "Senate passes bill" in FakeVonageAPI.calls[0][2]


def test_non_text_messages_are_not_checked() -> None:
    image = inbound(message_type="image", image={"url": "https://example.com/a.jpg", "caption": ARTICLE_TEXT})
    assert post(image).status_code == 200
    assert FakeVonageAPI.calls == [(USER_NUMBER, SANDBOX_NUMBER, whatsapp_vonage.NON_TEXT_TEXT)]

    reaction = inbound(message_type="reaction", reaction={"action": "react", "emoji": "👍"})
    assert post(reaction).status_code == 200
    assert len(FakeVonageAPI.calls) == 1


def test_other_channels_and_repeats_are_acknowledged_without_reply() -> None:
    assert post(inbound("hi", channel="viber_service")).status_code == 200
    message = inbound("hi")
    post(message)
    post(message)
    assert len(FakeVonageAPI.calls) == 1


def test_rate_limit_per_sender() -> None:
    for _ in range(whatsapp_vonage.common.RATE_LIMIT + 1):
        post(inbound(ARTICLE_TEXT))
    assert "Too many checks" in FakeVonageAPI.calls[-1][2]


def test_messages_url_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("VONAGE_MESSAGES_URL", "https://api.nexmo.com/v1/messages")
    post(inbound("hi"))
    assert FakeVonageAPI.urls == ["https://api.nexmo.com/v1/messages"]


def test_valid_signature_is_accepted(monkeypatch) -> None:
    monkeypatch.setenv("VONAGE_SIGNATURE_SECRET", SIGNATURE_SECRET)
    body = json.dumps(inbound("hi")).encode()
    assert post({}, raw=body, token=signed_jwt(body)).status_code == 200
    assert len(FakeVonageAPI.calls) == 1


def test_missing_or_invalid_signature_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("VONAGE_SIGNATURE_SECRET", SIGNATURE_SECRET)
    body = json.dumps(inbound("hi")).encode()
    unsigned = signed_jwt(body).rsplit(".", 1)[0] + "."
    for token in [
        None,  # no Authorization header
        "not-a-jwt",
        signed_jwt(body, secret="wrong-secret"),
        signed_jwt(body, alg="none"),
        unsigned,
        signed_jwt(body, payload_hash=hashlib.sha256(b"other body").hexdigest()),
    ]:
        assert post({}, raw=body, token=token).status_code == 401
    # A valid token for one body cannot be reused with another body.
    other = json.dumps(inbound(ARTICLE_TEXT)).encode()
    assert post({}, raw=other, token=signed_jwt(body)).status_code == 401
    assert FakeVonageAPI.calls == []


def test_status_webhook_returns_200(monkeypatch) -> None:
    update = {
        "message_uuid": "aaaaaaaa-bbbb-4ccc-8ddd-0123456789ab",
        "to": USER_NUMBER,
        "from": SANDBOX_NUMBER,
        "timestamp": "2026-11-25T12:00:05Z",
        "status": "rejected",
        "channel": "whatsapp",
        "error": {"type": "https://developer.vonage.com/api-errors/messages#1340", "title": 1340},
    }
    assert post(update, path="/whatsapp/vonage/status").status_code == 200
    monkeypatch.setenv("VONAGE_SIGNATURE_SECRET", SIGNATURE_SECRET)
    body = json.dumps(update).encode()
    assert post({}, path="/whatsapp/vonage/status", raw=body, token=signed_jwt(body)).status_code == 200
    assert post({}, path="/whatsapp/vonage/status", raw=body).status_code == 401
    assert FakeVonageAPI.calls == []


def test_client_posts_text_to_sandbox_with_basic_auth(monkeypatch) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(202, json={"message_uuid": "aaaaaaaa-bbbb-4ccc-8ddd-0123456789ab"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        vonage_api.httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    api = vonage_api.VonageMessagesAPI("key", "secret")
    assert asyncio.run(api.send_text(USER_NUMBER, SANDBOX_NUMBER, "x" * 5000)) == "aaaaaaaa-bbbb-4ccc-8ddd-0123456789ab"

    (request,) = requests
    assert str(request.url) == "https://messages-sandbox.nexmo.com/v1/messages"
    assert request.headers["authorization"] == "Basic " + base64.b64encode(b"key:secret").decode()
    payload = json.loads(request.content)
    assert {k: payload[k] for k in ("message_type", "channel", "to", "from")} == {
        "message_type": "text", "channel": "whatsapp", "to": USER_NUMBER, "from": SANDBOX_NUMBER,
    }
    assert len(payload["text"]) == vonage_api.MAX_TEXT_LENGTH


def test_client_returns_none_when_vonage_rejects(monkeypatch) -> None:
    real_client = httpx.AsyncClient
    handler = lambda request: httpx.Response(401, json={"title": "Unauthorized", "detail": "Bad credentials."})  # noqa: E731
    monkeypatch.setattr(
        vonage_api.httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    assert asyncio.run(vonage_api.VonageMessagesAPI("key", "secret").send_text(USER_NUMBER, SANDBOX_NUMBER, "hi")) is None
