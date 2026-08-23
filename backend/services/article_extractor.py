import asyncio
import ipaddress
import socket
from datetime import datetime
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from backend.schemas import Article


class ArticleExtractionError(ValueError):
    pass


async def _ensure_public_host(host: str) -> None:
    """Reject loopback/private destinations before fetching a user URL."""
    if host.lower() == "localhost":
        raise ArticleExtractionError("Local URLs cannot be fetched.")
    try:
        addresses = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ArticleExtractionError("The article host could not be resolved.") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_unspecified:
            raise ArticleExtractionError("Private or local network URLs cannot be fetched.")


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
    headers = {"User-Agent": "ClearSightBot/1.0 (+educational project)"}
    try:
        # Redirects are followed manually so every destination is screened
        # before a request is sent, rather than only after it was reached.
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False, headers=headers) as client:
            current_url = url
            for _ in range(4):
                current_host = urlparse(current_url).hostname
                if not current_host:
                    raise ArticleExtractionError("The article URL redirected to an invalid location.")
                await _ensure_public_host(current_host)
                response = await client.get(current_url)
                if not response.is_redirect:
                    break
                location = response.headers.get("location")
                if not location:
                    raise ArticleExtractionError("The article page returned an invalid redirect.")
                current_url = urljoin(current_url, location)
            else:
                raise ArticleExtractionError("The article page redirected too many times.")
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise ArticleExtractionError("The article page timed out.") from exc
    except httpx.HTTPStatusError as exc:
        raise ArticleExtractionError(f"The article page returned HTTP {exc.response.status_code}.") from exc
    except httpx.HTTPError as exc:
        raise ArticleExtractionError("The article page could not be retrieved.") from exc

    soup = BeautifulSoup(response.text, "html.parser")
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
        url=str(response.url),
        extraction_method="html_metadata_and_paragraphs",
    )
