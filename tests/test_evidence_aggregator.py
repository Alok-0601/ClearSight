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
