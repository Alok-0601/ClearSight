import pytest

from backend.schemas import EvidenceItem
from backend.services.evidence_aggregator import _annotate


def test_high_overlap_false_fact_check_contradicts_claim() -> None:
    item = EvidenceItem(
        source="google_fact_check",
        type="fact_check",
        title="Fact check: City council approves solar funding",
        url="https://example.com/fact-check",
        claim="The city council approved solar project funding",
        rating="False",
        relation_reason="pending",
    )
    result = _annotate(item, "The city council approved solar project funding in a vote yesterday.")
    assert result.relevance == "high"
    assert result.relation == "contradicts_claim"


def test_low_overlap_does_not_claim_a_verdict() -> None:
    item = EvidenceItem(
        source="gnews",
        type="news",
        title="Unrelated football results",
        url="https://example.com/news",
        relation_reason="pending",
    )
    result = _annotate(item, "The city council approved solar project funding in a vote yesterday.")
    assert result.relation == "not_determined"


@pytest.mark.parametrize(("rating", "expected"), [
    ("False", "contradicts_claim"),
    ("Mostly False", "contradicts_claim"),
    ("Untrue", "contradicts_claim"),
    ("Inaccurate", "contradicts_claim"),
    ("Not true", "contradicts_claim"),
    ("Pants on Fire!", "contradicts_claim"),
    ("True", "supports_claim"),
    ("Correct", "supports_claim"),
    ("Half true", "related"),
    ("Half-True", "related"),
    ("Mostly true", "related"),
    ("Unverified", "related"),
    ("Missing context", "related"),
    ("Satire", "related"),
])
def test_ratings_are_matched_as_whole_words(rating: str, expected: str) -> None:
    item = EvidenceItem(
        source="google_fact_check",
        type="fact_check",
        title="Fact check: City council approves solar funding",
        url="https://example.com/fact-check",
        claim="The city council approved solar project funding",
        rating=rating,
        relation_reason="pending",
    )
    result = _annotate(item, "The city council approved solar project funding in a vote yesterday.")
    assert result.relation == expected


def test_text_without_headline_is_searched_with_keywords() -> None:
    from backend.services.evidence_aggregator import search_query

    obama = ("BREAKING: Obama secretly signed an order to ban the Pledge of Allegiance in all schools! "
             "Share this before it gets deleted!!!")
    assert search_query(obama, 6) == "Obama signed order ban Pledge Allegiance"
    rbi = "NEW DELHI: The Reserve Bank of India on Friday kept the repo rate unchanged at 6.5 per cent."
    assert search_query(rbi, 6) == "Reserve Bank India kept repo rate"


def test_retrieve_evidence_without_title_sends_short_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from backend.schemas import SourceStatus
    from backend.services import evidence_aggregator

    seen: list[tuple[str, str]] = []

    async def fake_fact_checks(query: str):
        seen.append(("fact_check", query))
        return [], SourceStatus(source="google_fact_check", enabled=True, status="ok")

    async def fake_news(query: str):
        seen.append(("news", query))
        return [], SourceStatus(source="gnews", enabled=True, status="ok")

    monkeypatch.setattr(evidence_aggregator, "search_fact_checks", fake_fact_checks)
    monkeypatch.setattr(evidence_aggregator, "search_news", fake_news)
    body = ("WASHINGTON (Reuters) - The U.S. Senate on Tuesday passed a bill to fund the government through "
            "December, sending the measure to the House.")
    asyncio.run(evidence_aggregator.retrieve_evidence(None, body))
    assert ("fact_check", "Senate passed bill fund government December sending measure") in seen
    assert ("news", "Senate passed bill fund government December") in seen
    # no fact check found with 8 words, so a 5-word query is tried as well
    assert ("fact_check", "Senate passed bill fund government") in seen
    assert all(len(query) < 80 for _, query in seen)
