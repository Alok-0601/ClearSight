import asyncio
import re

from backend.schemas import EvidenceItem, SourceStatus
from .factcheck_service import search_fact_checks
from .news_service import search_news


_STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "with"}
# Ratings are matched as whole words, mixed ratings first, so "Untrue",
# "Inaccurate" or "Half true" are never read as the word "true"/"accurate".
_MIXED_RATING = re.compile(
    r"\b(half|partly|partially|mixture|mixed|unproven|unverified|unsupported|"
    r"exaggerat\w*|missing context|needs context|lacks context|mostly true)\b"
)
_FALSE_RATING = re.compile(
    r"\b(false|fake|incorrect|inaccurate|misleading|pants on fire|scam|hoax|untrue|"
    r"fabricated|baseless|debunked|not (?:true|correct|accurate))\b"
)
_TRUE_RATING = re.compile(r"\b(true|correct|accurate|verified)\b")


def _keywords(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-zA-Z]{3,}", text.lower())
        if token not in _STOPWORDS
    }


def _annotate(item: EvidenceItem, article_text: str) -> EvidenceItem:
    article_terms = _keywords(article_text)
    evidence_terms = _keywords(" ".join(filter(None, [item.title, item.summary, item.claim])))
    overlap = len(article_terms & evidence_terms)
    # A lexical threshold prevents a rating for an unrelated claim from being
    # presented as support or contradiction for the submitted article.
    if overlap >= 5:
        relevance = "high"
    elif overlap >= 2:
        relevance = "medium"
    else:
        relevance = "low"
    relation = "related" if relevance != "low" else "not_determined"
    reason = "Related coverage found by keyword overlap; it is not a verdict on the submitted article."
    rating = (item.rating or "").lower()
    if item.type == "fact_check" and relevance == "high":
        if _MIXED_RATING.search(rating):
            relation, reason = "related", "A matching fact check gave a mixed or unproven rating, so it neither supports nor contradicts the claim."
        elif _FALSE_RATING.search(rating):
            relation, reason = "contradicts_claim", "A high-overlap fact check rated a matching claim as false or misleading."
        elif _TRUE_RATING.search(rating):
            relation, reason = "supports_claim", "A high-overlap fact check rated a matching claim as true or accurate."
        else:
            relation, reason = "related", "A matching fact check was found, but its rating could not be mapped safely."
    return item.model_copy(update={"relevance": relevance, "relation": relation, "relation_reason": reason})


async def retrieve_evidence(title: str | None, body: str) -> tuple[list[EvidenceItem], list[SourceStatus]]:
    query = (title or body[:400]).strip()
    fact_check_result, news_result = await asyncio.gather(
        search_fact_checks(query),
        search_news(query),
    )
    evidence = [*fact_check_result[0], *news_result[0]]
    annotated = [_annotate(item, f"{title or ''} {body}") for item in evidence]
    rank = {"high": 0, "medium": 1, "low": 2}
    annotated.sort(key=lambda item: (rank[item.relevance], item.type, item.title.lower()))
    return annotated, [fact_check_result[1], news_result[1]]
