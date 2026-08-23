import logging
import os
from datetime import datetime

import httpx

from backend.schemas import EvidenceItem, SourceStatus
from .cache import TTLCache


logger = logging.getLogger(__name__)
_cache: TTLCache[list[EvidenceItem]] = TTLCache()


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


async def search_news(query: str) -> tuple[list[EvidenceItem], SourceStatus]:
    api_key = os.getenv("GNEWS_API_KEY")
    if not api_key:
        return [], SourceStatus(source="gnews", enabled=False, status="not_configured", detail="GNEWS_API_KEY is not set.")
    cache_key = f"gnews:{query.lower()}"
    cached = await _cache.get(cache_key)
    if cached is not None:
        return cached, SourceStatus(source="gnews", enabled=True, status="ok", detail="cached")
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=4.0)) as client:
            response = await client.get(
                "https://gnews.io/api/v4/search",
                params={"q": query[:500], "lang": "en", "max": 8, "apikey": api_key},
            )
            response.raise_for_status()
        evidence = [
            EvidenceItem(
                source="gnews",
                type="news",
                title=item.get("title") or "Related coverage",
                url=item.get("url") or "",
                publisher=(item.get("source") or {}).get("name"),
                published_at=_parse_date(item.get("publishedAt")),
                summary=item.get("description"),
                relation_reason="Related news coverage does not itself establish whether the claim is true or false.",
            )
            for item in response.json().get("articles", [])
            if item.get("url")
        ]
        await _cache.set(cache_key, evidence)
        return evidence, SourceStatus(source="gnews", enabled=True, status="ok")
    except httpx.HTTPStatusError as exc:
        logger.warning("GNews request failed: %s", exc.response.status_code)
        return [], SourceStatus(source="gnews", enabled=True, status="error", detail=f"HTTP {exc.response.status_code}")
    except httpx.HTTPError:
        logger.exception("GNews request failed")
        return [], SourceStatus(source="gnews", enabled=True, status="error", detail="Request failed")
    except Exception:
        logger.exception("GNews response could not be normalized")
        return [], SourceStatus(source="gnews", enabled=True, status="error", detail="Invalid provider response")
