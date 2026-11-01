import asyncio
from functools import partial

import httpx
import pytest

from backend.services import article_extractor
from backend.services.article_extractor import ArticleExtractionError, extract_article


# An IP literal resolves without DNS, so these tests never touch the network.
PUBLIC_URL = "http://8.8.8.8/news/story"
ARTICLE_HTML = (
    "<html><head><title>Council approves solar funding</title></head><body><article>"
    "<p>The city council approved funding for three new solar projects on Tuesday after a long debate.</p>"
    "</article></body></html>"
)


def use_transport(monkeypatch, handler) -> None:
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(article_extractor.httpx, "AsyncClient", partial(httpx.AsyncClient, transport=transport))


@pytest.mark.parametrize("host", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "100.64.0.1", "::1"])
def test_non_public_addresses_are_rejected(host: str) -> None:
    with pytest.raises(ArticleExtractionError):
        asyncio.run(article_extractor._ensure_public_host(host))


def test_public_address_is_allowed() -> None:
    asyncio.run(article_extractor._ensure_public_host("8.8.8.8"))


def test_identifies_itself_with_a_contact_url(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["user_agent"] = request.headers["user-agent"]
        return httpx.Response(200, html=ARTICLE_HTML)

    use_transport(monkeypatch, handler)
    article = asyncio.run(extract_article(PUBLIC_URL))
    assert seen["user_agent"].startswith("ClearSightBot/")
    assert "github.com/Alok-0601/ClearSight" in seen["user_agent"]
    assert article.title == "Council approves solar funding"


def test_site_blocking_automated_readers_gets_a_helpful_message(monkeypatch) -> None:
    use_transport(monkeypatch, lambda request: httpx.Response(403))
    with pytest.raises(ArticleExtractionError, match="Paste the article text instead"):
        asyncio.run(extract_article(PUBLIC_URL))


def test_redirect_to_private_address_is_blocked(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://127.0.0.1/admin"})

    use_transport(monkeypatch, handler)
    with pytest.raises(ArticleExtractionError, match="Private or local"):
        asyncio.run(extract_article(PUBLIC_URL))


def test_oversized_page_stops_downloading_at_the_cap(monkeypatch) -> None:
    chunk = b"<p>" + b"x" * 10_000 + b"</p>"
    sent = {"bytes": 0}

    async def endless_page():
        yield ARTICLE_HTML.replace("</article></body></html>", "").encode()
        for _ in range(1000):  # ~10 MB if read in full
            sent["bytes"] += len(chunk)
            yield chunk

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, content=endless_page())

    use_transport(monkeypatch, handler)
    article = asyncio.run(extract_article(PUBLIC_URL))
    assert article.body.startswith("The city council approved funding")
    # The page is ~10 MB; the download must stop near the 2 MB cap, not at the end.
    assert sent["bytes"] < 3_000_000
