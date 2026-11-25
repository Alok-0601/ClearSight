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


# Words that say nothing about the claim itself; left out of search queries.
_QUERY_SKIP = _STOPWORDS | {
    "breaking", "share", "before", "gets", "deleted", "watch", "video", "secretly", "said", "says", "this", "has", "all",
    "have", "had", "will", "would", "who", "which", "been", "were", "they", "their", "his", "her", "its", "you",
    "your", "through", "per", "cent", "time", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}
# "WASHINGTON (Reuters) - " or "NEW DELHI: " at the start of a report.
_DATELINE = re.compile(r"^\s*[A-Z][A-Za-z .,/\-]{0,40}(?:\([^)]*\))?\s*[-–—:]\s+")
# End of the first sentence; "U.S." and similar abbreviations do not count.
_SENTENCE_END = re.compile(r"(?<=[a-z0-9]{2}[.!?])\s+(?=[A-Z\"'])")


def search_query(body: str, max_words: int) -> str:
    """A short, headline-like keyword query built from the first sentence of an article.

    Messages from the bots have no headline. Searching with the raw text fails:
    GNews rejects long queries with punctuation (HTTP 400) and Google Fact Check
    finds no claim matching a whole paragraph.
    """
    first_sentence = _SENTENCE_END.split(_DATELINE.sub("", body[:600]).strip(), maxsplit=1)[0]
    words = [
        word for word in re.findall(r"[A-Za-z0-9']+", first_sentence)
        if len(word) > 1 and word.lower() not in _QUERY_SKIP
    ]
    return " ".join(words[:max_words])


async def retrieve_evidence(title: str | None, body: str) -> tuple[list[EvidenceItem], list[SourceStatus]]:
    has_title = bool(title and title.strip())
    if has_title:
        fact_check_query = news_query = title.strip()
    else:
        # News search treats every word as required, so it gets fewer words.
        fact_check_query = search_query(body, 8) or body[:200].strip()
        news_query = search_query(body, 6) or fact_check_query
    fact_check_result, news_result = await asyncio.gather(
        search_fact_checks(fact_check_query),
        search_news(news_query),
    )
    if not has_title and fact_check_result[1].status == "ok" and not fact_check_result[0]:
        # A shorter query often matches the wording of a published claim better.
        shorter = search_query(body, 5)
        if shorter and shorter != fact_check_query:
            fact_check_result = await search_fact_checks(shorter)
    evidence = [*fact_check_result[0], *news_result[0]]
    annotated = [_annotate(item, f"{title or ''} {body}") for item in evidence]
    rank = {"high": 0, "medium": 1, "low": 2}
    annotated.sort(key=lambda item: (rank[item.relevance], item.type, item.title.lower()))
    return annotated, [fact_check_result[1], news_result[1]]
