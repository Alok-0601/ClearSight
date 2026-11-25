from datetime import datetime, timezone

from backend.routes import _whatsapp_common
from backend.routes.telegram import _format_result
from backend.schemas import Article, EvidenceItem, SourceStatus, VerificationResponse
from backend.services.factcheck_service import clean_text


def _result() -> VerificationResponse:
    snopes = EvidenceItem(
        source="google_fact_check",
        type="fact_check",
        title=clean_text("%%title%% Did Obama Sign Executive Order Banning the Pledge of ..."),
        url="https://www.snopes.com/fact-check/pledge-of-allegiance-ban/",
        publisher="Snopes.com",
        claim="President Obama has issued an executive order banning the Pledge of Allegiance in U.S. schools.",
        rating="False",
        relevance="high",
        relation="contradicts_claim",
        relation_reason="test",
    )
    return VerificationResponse(
        id="test", prediction="FAKE", verdict="FAKE", confidence=0.9, confidence_note="test",
        article=Article(title=None, body="text", extraction_method="provided_text"),
        evidence=[snopes], sources=[SourceStatus(source="google_fact_check", enabled=True, status="ok")],
        created_at=datetime.now(timezone.utc),
    )


def test_placeholders_are_removed_from_fact_check_titles() -> None:
    assert clean_text("%%title%% Did Obama Sign  an Order") == "Did Obama Sign an Order"


def test_bots_describe_the_fact_check_in_plain_language() -> None:
    telegram = _format_result(_result())
    whatsapp = _whatsapp_common.format_result(_result())
    for reply in (telegram, whatsapp):
        assert "%%" not in reply
        assert "rated a matching claim" in reply
        assert '"President Obama has issued an executive order banning the Pledge of Allegiance in U.S. schools."' in reply
    assert "❌ <b>Snopes.com</b> rated a matching claim <b>False</b>" in telegram
    assert "❌ *Snopes.com* rated a matching claim *False*" in whatsapp
    assert "Read the fact check: https://www.snopes.com/fact-check/pledge-of-allegiance-ban/" in whatsapp
