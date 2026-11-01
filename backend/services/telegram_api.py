import logging

import httpx


logger = logging.getLogger(__name__)


class TelegramAPI:
    """Minimal Telegram Bot API client: the bot only sends and edits messages."""

    def __init__(self, token: str) -> None:
        self._base_url = f"https://api.telegram.org/bot{token}"

    async def _call(self, method: str, payload: dict) -> dict | None:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=4.0)) as client:
                response = await client.post(f"{self._base_url}/{method}", json=payload)
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            # The request URL contains the bot token, so log only the error type.
            logger.warning("Telegram %s failed: %s", method, type(exc).__name__)
            return None
        if not data.get("ok"):
            logger.warning("Telegram %s rejected: %s", method, data.get("description"))
            return None
        return data.get("result")

    async def send_message(self, chat_id: int, text: str, reply_to: int | None = None) -> int | None:
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": True},
        }
        if reply_to is not None:
            payload["reply_parameters"] = {"message_id": reply_to, "allow_sending_without_reply": True}
        result = await self._call("sendMessage", payload)
        return result.get("message_id") if result else None

    async def edit_message(self, chat_id: int, message_id: int, text: str) -> None:
        await self._call("editMessageText", {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
            "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": True},
        })
