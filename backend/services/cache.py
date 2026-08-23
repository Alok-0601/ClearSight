import asyncio
import time
from typing import Generic, TypeVar


T = TypeVar("T")


class TTLCache(Generic[T]):
    """Small in-process cache for repeat evidence searches.

    It deliberately has no persistence: cached API data is short-lived and a
    restart should always return the service to a clean state.
    """

    def __init__(self, ttl_seconds: int = 300, max_items: int = 128) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        self._items: dict[str, tuple[float, T]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> T | None:
        async with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            created, value = item
            if time.monotonic() - created > self.ttl_seconds:
                self._items.pop(key, None)
                return None
            return value

    async def set(self, key: str, value: T) -> None:
        async with self._lock:
            if len(self._items) >= self.max_items:
                oldest_key = min(self._items, key=lambda item_key: self._items[item_key][0])
                self._items.pop(oldest_key, None)
            self._items[key] = (time.monotonic(), value)
