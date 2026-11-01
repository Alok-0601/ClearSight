import asyncio
import ipaddress
import socket
from datetime import datetime
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from backend.schemas import Article


# Pages beyond this size are cut off; article text is near the top of the HTML.
MAX_PAGE_BYTES = 2_000_000
# The fetcher identifies itself honestly, with a contact URL as sites such as
# Wikipedia ask of automated clients. Sites that block automated readers are
# respected; the user is asked to paste the text instead.
_REQUEST_HEADERS = {
    "User-Agent": "ClearSightBot/1.0 (+https://github.com/Alok-0601/ClearSight; educational fact-check project)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
}
_BLOCKED_STATUSES = {401, 403, 429}


class ArticleExtractionError(ValueError):
    pass


async def _ensure_public_host(host: str) -> None:
    """Reject any destination that is not on the public internet before fetching a user URL."""
    if host.lower() == "localhost":
        raise ArticleExtractionError("Local URLs cannot be fetched.")
    try:
        addresses = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ArticleExtractionError("The article host could not be resolved.") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        # is_global also excludes ranges such as carrier-grade NAT (100.64.0.0/10)
        # that is_private does not cover.
        if not ip.is_global:
            raise ArticleExtractionError("Private or local network URLs cannot be fetched.")


async def _read_limited(response: httpx.Response) -> str:
    content = bytearray()
    async for chunk in response.aiter_bytes():
        content.extend(chunk)
        if len(content) >= MAX_PAGE_BYTES:
            break
    return bytes(content[:MAX_PAGE_BYTES]).decode(response.encoding or "utf-8", errors="replace")


def _meta(soup: BeautifulSoup, *names: str) -> str | None:
    for name in names:
        tag = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name})
        if tag and tag.get("content"):
            return tag["content"].strip()
    return None


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


async def extract_article(url: str) -> Article:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ArticleExtractionError("A valid HTTP or HTTPS article URL is required.")
    await _ensure_public_host(parsed.hostname)
    timeout = httpx.Timeout(12.0, connect=5.0)
    try:
        # Redirects are followed manually so every destination is screened
        # before a request is sent, rather than only after it was reached.
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False, headers=_REQUEST_HEADERS) as client:
            current_url = url
            for _ in range(4):
                current_host = urlparse(current_url).hostname
                if not current_host:
                    raise ArticleExtractionError("The article URL redirected to an invalid location.")
                await _ensure_public_host(current_host)
                async with client.stream("GET", current_url) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise ArticleExtractionError("The article page returned an invalid redirect.")
                        current_url = urljoin(current_url, location)
                        continue
                    response.raise_for_status()
                    html = await _read_limited(response)
                    final_url = str(response.url)
                    break
            else:
                raise ArticleExtractionError("The article page redirected too many times.")
    except httpx.TimeoutException as exc:
        raise ArticleExtractionError("The article page timed out.") from exc
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        if code in _BLOCKED_STATUSES:
            raise ArticleExtractionError(
                f"This site does not allow automated readers (HTTP {code}). Paste the article text instead."
            ) from exc
        raise ArticleExtractionError(f"The article page returned HTTP {code}.") from exc
    except httpx.HTTPError as exc:
        raise ArticleExtractionError("The article page could not be retrieved.") from exc

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "aside", "noscript"]):
        tag.decompose()
    title = _meta(soup, "og:title", "twitter:title") or (soup.title.string.strip() if soup.title and soup.title.string else None)
    publisher = _meta(soup, "article:publisher", "og:site_name", "publisher")
    published_at = _parse_date(_meta(soup, "article:published_time", "date", "publish-date"))
    article_root = soup.find("article") or soup.find("main") or soup.body
    paragraphs = article_root.find_all("p") if article_root else []
    body = "\n".join(paragraph.get_text(" ", strip=True) for paragraph in paragraphs)
    body = " ".join(body.split())
    if len(body) < 40:
        raise ArticleExtractionError("The page did not contain enough extractable article text.")
    return Article(
        title=title,
        body=body[:100_000],
        publisher=publisher,
        published_at=published_at,
        url=final_url,
        extraction_method="html_metadata_and_paragraphs",
    )
