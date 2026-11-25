"""Plain-language wording of a fact check in the Telegram and WhatsApp replies."""

from backend.schemas import EvidenceItem

_ICONS = {"contradicts_claim": "❌", "supports_claim": "✅"}


def _shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def fact_check_parts(item: EvidenceItem) -> tuple[str, str, str, str]:
    """Icon, publisher, rating and checked claim of a fact check, ready to display."""
    icon = _ICONS.get(item.relation, "🔎")
    publisher = item.publisher or "A fact-checker"
    rating = _shorten(item.rating or "", 60)
    claim = _shorten(item.claim or item.title or "", 200)
    return icon, publisher, rating, claim


def claim_phrase(item: EvidenceItem) -> str:
    # Matching is by shared keywords, so the checked claim may differ from the message.
    return "a matching claim" if item.relevance == "high" else "a related claim"
