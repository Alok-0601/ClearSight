import logging

import httpx


logger = logging.getLogger(__name__)

SANDBOX_MESSAGES_URL = "https://messages-sandbox.nexmo.com/v1/messages"
# The Messages API limit for WhatsApp text, "including unicode".
MAX_TEXT_LENGTH = 4096


class VonageMessagesAPI:
    """Minimal Vonage Messages API client: the bot only sends WhatsApp text messages."""

    def __init__(self, api_key: str, api_secret: str, url: str = SANDBOX_MESSAGES_URL) -> None:
        self._auth = (api_key, api_secret)
        self._url = url

    async def send_text(self, to: str, from_: str, text: str) -> str | None:
        """Send a text message and return its message_uuid, or None if Vonage did not accept it."""
        if len(text) > MAX_TEXT_LENGTH:
            text = text[: MAX_TEXT_LENGTH - 1] + "…"
        payload = {"message_type": "text", "channel": "whatsapp", "to": to, "from": from_, "text": text}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=4.0)) as client:
                response = await client.post(
                    self._url, json=payload, auth=self._auth, headers={"Accept": "application/json"},
                )
        except httpx.HTTPError as exc:
            # Log only the error type: the request carries the API key and secret.
            logger.warning("Vonage send failed: %s", type(exc).__name__)
            return None
        try:
            data = response.json()
        except ValueError:
            data = None
        if not isinstance(data, dict):
            data = {}
        if not response.is_success:
            logger.warning(
                "Vonage send rejected (HTTP %s): %s %s",
                response.status_code, data.get("title", ""), data.get("detail", ""),
            )
            return None
        return data.get("message_uuid")
