import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.routes import telegram
from backend.routes.verification import store
from backend.schemas import Article


SECRET = "test-secret"
ARTICLE_TEXT = (
    "WASHINGTON (Reuters) - The Senate voted on Thursday to pass the bill, which "
    "lawmakers said would fund infrastructure projects through 2030."
)
client = TestClient(app)


class FakeTelegramAPI:
    calls: list[tuple] = []

    def __init__(self, token: str) -> None:
        pass

    async def send_message(self, chat_id, text, reply_to=None):
        self.calls.append(("send", chat_id, text))
        return 100 + len(self.calls)

    async def edit_message(self, chat_id, message_id, text):
        self.calls.append(("edit", chat_id, text))


@pytest.fixture(autouse=True)
def bot_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", SECRET)
    monkeypatch.delenv("GOOGLE_FACT_CHECK_API_KEY", raising=False)
    monkeypatch.delenv("GNEWS_API_KEY", raising=False)
    monkeypatch.setattr(telegram, "TelegramAPI", FakeTelegramAPI)
    FakeTelegramAPI.calls = []
    telegram._recent_checks.clear()
    telegram._seen_updates.clear()


_next_update_id = iter(range(1, 10_000))


def post_message(text=None, chat_type="private", secret=SECRET, **extra):
    message = {"message_id": 7, "chat": {"id": 42, "type": chat_type}, **extra}
    if text is not None:
        message["text"] = text
    return client.post(
        "/telegram/webhook",
        json={"update_id": next(_next_update_id), "message": message},
        headers={"X-Telegram-Bot-Api-Secret-Token": secret},
    )


def test_webhook_is_disabled_without_configuration(monkeypatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN")
    assert post_message("/start").status_code == 404


def test_webhook_rejects_wrong_secret() -> None:
    assert post_message("/start", secret="wrong").status_code == 403
    assert FakeTelegramAPI.calls == []


def test_start_replies_with_help() -> None:
    assert post_message("/start").json() == {"ok": True}
    assert len(FakeTelegramAPI.calls) == 1
    assert "ClearSight news checker" in FakeTelegramAPI.calls[0][2]


def test_sticker_or_short_text_gets_help() -> None:
    post_message(sticker={"file_id": "x"})
    post_message("is this true?")
    assert [call[0] for call in FakeTelegramAPI.calls] == ["send", "send"]
    assert all("ClearSight news checker" in call[2] for call in FakeTelegramAPI.calls)


def test_article_text_is_checked_and_not_added_to_public_history() -> None:
    history_before = len(store.history(100))
    post_message(ARTICLE_TEXT)
    kinds = [call[0] for call in FakeTelegramAPI.calls]
    assert kinds == ["send", "edit"]
    assert FakeTelegramAPI.calls[0][2] == "🔎 Checking…"
    reply = FakeTelegramAPI.calls[1][2]
    assert "Model verdict: REAL" in reply
    assert "Evidence search is not configured" in reply
    assert len(store.history(100)) == history_before


def test_link_is_extracted_and_checked(monkeypatch) -> None:
    fetched = []

    async def fake_extract(url):
        fetched.append(url)
        return Article(title="Senate passes bill", body=ARTICLE_TEXT, url=url, extraction_method="test")

    monkeypatch.setattr("backend.routes.verification.extract_article", fake_extract)
    post_message("look at this https://example.com/news/senate-bill")
    assert fetched == ["https://example.com/news/senate-bill"]
    assert "Senate passes bill" in FakeTelegramAPI.calls[-1][2]


def test_failed_link_falls_back_to_message_text(monkeypatch) -> None:
    from backend.services.article_extractor import ArticleExtractionError

    async def failing_extract(url):
        raise ArticleExtractionError("The article page returned HTTP 403.")

    monkeypatch.setattr("backend.routes.verification.extract_article", failing_extract)
    post_message(f"{ARTICLE_TEXT} https://example.com/blocked")
    reply = FakeTelegramAPI.calls[-1][2]
    assert "checked the message text instead" in reply
    assert "Model verdict" in reply

    post_message("https://example.com/blocked")
    assert "HTTP 403" in FakeTelegramAPI.calls[-1][2]


def test_group_ignores_chatter_but_checks_replied_message() -> None:
    post_message(ARTICLE_TEXT, chat_type="group")
    assert FakeTelegramAPI.calls == []
    post_message("/check@ClearSightBot", chat_type="group", reply_to_message={"text": ARTICLE_TEXT})
    assert "Model verdict" in FakeTelegramAPI.calls[-1][2]


def test_rate_limit_per_chat() -> None:
    for _ in range(telegram.RATE_LIMIT + 1):
        post_message(ARTICLE_TEXT)
    assert "Too many checks" in FakeTelegramAPI.calls[-1][2]


def test_repeated_update_is_answered_once() -> None:
    body = {"update_id": 999_999, "message": {"message_id": 1, "chat": {"id": 1, "type": "private"}, "text": "/help"}}
    headers = {"X-Telegram-Bot-Api-Secret-Token": SECRET}
    client.post("/telegram/webhook", json=body, headers=headers)
    client.post("/telegram/webhook", json=body, headers=headers)
    assert len(FakeTelegramAPI.calls) == 1


def test_root_redirects_to_docs() -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_low_confidence_reply_says_uncertain(monkeypatch) -> None:
    from backend.routes import verification

    monkeypatch.setattr(verification.ml_service, "predict", lambda title, body: ("FAKE", 0.58))
    post_message(ARTICLE_TEXT)
    reply = FakeTelegramAPI.calls[-1][2]
    assert "Model verdict: UNCERTAIN" in reply
    assert "leaned FAKE" in reply
