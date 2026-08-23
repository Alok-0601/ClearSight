import logging
import os

import httpx

from backend.schemas import EvidenceItem, SourceStatus
from .cache import TTLCache


logger = logging.getLogger(__name__)
_cache: TTLCache[list[EvidenceItem]] = TTLCache()


async def search_fact_checks(query: str) -> tuple[list[EvidenceItem], SourceStatus]:
    api_key = os.getenv("GOOGLE_FACT_CHECK_API_KEY")
    if not api_key:
        return [], SourceStatus(source="google_fact_check", enabled=False, status="not_configured", detail="GOOGLE_FACT_CHECK_API_KEY is not set.")
    cache_key = f"factcheck:{query.lower()}"
    cached = await _cache.get(cache_key)
    if cached is not None:
        return cached, SourceStatus(source="google_fact_check", enabled=True, status="ok", detail="cached")
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=4.0)) as client:
            response = await client.get(
                "https://factchecktools.googleapis.com/v1alpha1/claims:search",
                params={"query": query[:500], "key": api_key, "pageSize": 8},
            )
            response.raise_for_status()
        evidence: list[EvidenceItem] = []
        for claim in response.json().get("claims", []):
            for review in claim.get("claimReview", []):
                url = review.get("url")
                if not url:
                    continue
                evidence.append(EvidenceItem(
                    source="google_fact_check",
                    type="fact_check",
                    title=review.get("title") or claim.get("text") or "Fact check",
                    url=url,
                    publisher=review.get("publisher", {}).get("name"),
                    published_at=None,
                    summary=review.get("textualRating"),
                    claim=claim.get("text"),
                    rating=review.get("textualRating"),
                    relation_reason="Relation is assigned only after lexical claim matching and rating analysis.",
                ))
        await _cache.set(cache_key, evidence)
        return evidence, SourceStatus(source="google_fact_check", enabled=True, status="ok")
    except httpx.HTTPStatusError as exc:
        logger.warning("Google Fact Check request failed: %s", exc.response.status_code)
        return [], SourceStatus(source="google_fact_check", enabled=True, status="error", detail=f"HTTP {exc.response.status_code}")
    except httpx.HTTPError:
        logger.exception("Google Fact Check request failed")
        return [], SourceStatus(source="google_fact_check", enabled=True, status="error", detail="Request failed")
    except Exception:
        logger.exception("Google Fact Check response could not be normalized")
        return [], SourceStatus(source="google_fact_check", enabled=True, status="error", detail="Invalid provider response")
