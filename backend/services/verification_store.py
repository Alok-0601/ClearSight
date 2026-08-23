from collections import OrderedDict

from backend.schemas import HistoryItem, VerificationResponse


class VerificationStore:
    """Bounded in-memory history; use a database before multi-instance deployment."""

    def __init__(self, max_items: int = 100) -> None:
        self.max_items = max_items
        self._items: OrderedDict[str, VerificationResponse] = OrderedDict()

    def add(self, verification: VerificationResponse) -> None:
        self._items[verification.id] = verification
        self._items.move_to_end(verification.id)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)

    def get(self, verification_id: str) -> VerificationResponse | None:
        return self._items.get(verification_id)

    def history(self, limit: int = 20) -> list[HistoryItem]:
        recent = list(self._items.values())[-limit:]
        return [
            HistoryItem(
                id=item.id,
                prediction=item.prediction,
                confidence=item.confidence,
                title=item.article.title,
                url=item.article.url,
                created_at=item.created_at,
            )
            for item in reversed(recent)
        ]
