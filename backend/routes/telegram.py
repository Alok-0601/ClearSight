"""Telegram bot webhook.

Telegram POSTs each update to /telegram/webhook. The update is handled inside
the request (no background task) because serverless hosts such as Vercel may
freeze the function as soon as the response is returned.
"""

import hmac
import html
import logging
import os
import re
import time
from collections import OrderedDict, defaultdict, deque

from fastapi import APIRouter, Header, HTTPException, Request, status

from backend.schemas import Article, VerificationResponse, VerifyUrlRequest
from backend.services.telegram_api import TelegramAPI
from .verification import verify_article, verify_article_url


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/telegram", tags=["telegram"])

MIN_TEXT_LENGTH = 40
RATE_LIMIT = 5
RATE_WINDOW_SECONDS = 60
_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_COMMAND_RE = re.compile(r"^/(\w+)(?:@\w+)?\s*", re.DOTALL)

HELP_TEXT = (
    "<b>ClearSight news checker</b>\n\n"
    "Send me a news article link, or paste at least 40 characters of article text, "
    "and I will run it through the ClearSight model and look for related fact checks.\n\n"
    "In a group, reply to a message with /check.\n\n"
    "<i>This is a student project: a TF-IDF + Linear SVM model trained on the WELFake "
    "dataset. It is a second opinion, not a fact check.</i>"
)

# Both are per process. On serverless hosts each instance keeps its own copy,
# so they are best-effort protection, not a guarantee.
_recent_checks: defaultdict[int, deque[float]] = defaultdict(deque)
_seen_updates: OrderedDict[int, None] = OrderedDict()


def _rate_limited(chat_id: int) -> bool:
    now = time.monotonic()
    checks = _recent_checks[chat_id]
    while checks and now - checks[0] > RATE_WINDOW_SECONDS:
        checks.popleft()
    if len(checks) >= RATE_LIMIT:
        return True
    checks.append(now)
    return False


def _already_seen(update_id: int | None) -> bool:
    """Telegram re-sends an update if the webhook was slow; answer it only once."""
    if update_id is None:
        return False
    if update_id in _seen_updates:
        return True
    _seen_updates[update_id] = None
    while len(_seen_updates) > 500:
        _seen_updates.popitem(last=False)
    return False


def _message_text(message: dict) -> str:
    return (message.get("text") or message.get("caption") or "").strip()


def _target_text(message: dict) -> str | None:
    """Return the text to check, or None when the bot should stay silent."""
    text = _message_text(message)
    command = _COMMAND_RE.match(text)
    is_private = message.get("chat", {}).get("type") == "private"
    if command is None:
        # In groups the bot only acts on an explicit /check.
        return text if is_private else None
    name = command.group(1).lower()
    if name == "check":
        argument = text[command.end():].strip()
        return argument or _message_text(message.get("reply_to_message") or {})
    # An empty target makes the caller reply with the help text.
    return "" if is_private or name in {"start", "help"} else None


def _format_result(result: VerificationResponse) -> str:
    icon = "🔴" if result.prediction == "FAKE" else "🟢"
    lines = [f"{icon} <b>Model verdict: {result.prediction}</b> ({result.confidence:.0%} confidence)"]
    if result.article.title:
        lines.append(f"📰 {html.escape(result.article.title[:200])}")

    fact_checks = [item for item in result.evidence if item.type == "fact_check" and item.relevance != "low"][:3]
    news = [item for item in result.evidence if item.type == "news" and item.relevance != "low"][:3]
    if fact_checks:
        lines.append("\n<b>Related fact checks</b>")
        for item in fact_checks:
            rating = f" — rated <i>{html.escape(item.rating)}</i>" if item.rating else ""
            publisher = f" ({html.escape(item.publisher)})" if item.publisher else ""
            lines.append(f'• <a href="{html.escape(item.url)}">{html.escape(item.title[:120])}</a>{publisher}{rating}')
    if news:
        lines.append("\n<b>Related coverage</b>")
        for item in news:
            publisher = f" ({html.escape(item.publisher)})" if item.publisher else ""
            lines.append(f'• <a href="{html.escape(item.url)}">{html.escape(item.title[:120])}</a>{publisher}')
    if not fact_checks and not news:
        if all(source.status == "not_configured" for source in result.sources):
            lines.append("\nEvidence search is not configured on this server.")
        else:
            lines.append("\nNo closely related fact checks or coverage were found.")

    lines.append(
        "\n<i>Confidence comes from the model's decision margin and is not a calibrated "
        "probability. The model learned writing patterns from WELFake (mostly US news), "
        "so please confirm with a reliable source.</i>"
    )
    return "\n".join(lines)


async def _run_check(target: str) -> str:
    """Prefer a link in the message; fall back to the text if the link fails."""
    url_match = _URL_RE.search(target)
    remaining_text = _URL_RE.sub("", target).strip()
    url_error = None
    if url_match:
        try:
            # Validated like the /verify/url request body.
            url = str(VerifyUrlRequest(url=url_match.group(0)).url)
            return _format_result(await verify_article_url(url, record=False))
        except (HTTPException, ValueError) as exc:
            url_error = exc.detail if isinstance(exc, HTTPException) else "That link is not a valid URL."
            if len(remaining_text) < MIN_TEXT_LENGTH:
                return f"⚠️ I couldn't check that link: {html.escape(str(url_error))}"
    try:
        # record=False: chat messages must not appear in the public /history.
        result = await verify_article(Article(body=remaining_text if url_match else target), record=False)
    except HTTPException as exc:
        return f"⚠️ I couldn't check that text: {html.escape(str(exc.detail))}"
    reply = _format_result(result)
    if url_error:
        reply = f"(The link couldn't be opened, so I checked the message text instead.)\n\n{reply}"
    return reply


async def _handle_message(api: TelegramAPI, message: dict) -> None:
    chat_id = message["chat"]["id"]
    message_id = message.get("message_id")
    target = _target_text(message)
    if target is None:
        return
    if not _URL_RE.search(target) and len(target) < MIN_TEXT_LENGTH:
        await api.send_message(chat_id, HELP_TEXT, reply_to=message_id if target else None)
        return
    if _rate_limited(chat_id):
        await api.send_message(chat_id, "⏳ Too many checks in a minute. Please try again shortly.", reply_to=message_id)
        return
    status_id = await api.send_message(chat_id, "🔎 Checking…", reply_to=message_id)
    try:
        reply = await _run_check(target)
    except Exception:
        logger.exception("Telegram check failed")
        reply = "⚠️ Something went wrong while checking this. Please try again later."
    if status_id is None:
        await api.send_message(chat_id, reply, reply_to=message_id)
    else:
        await api.edit_message(chat_id, status_id, reply)


@router.post("/webhook", include_in_schema=False)
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
) -> dict[str, bool]:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    secret = os.getenv("TELEGRAM_WEBHOOK_SECRET")
    if not token or not secret:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Telegram bot is not configured.")
    if not hmac.compare_digest(x_telegram_bot_api_secret_token or "", secret):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid webhook secret.")

    update = await request.json()
    message = update.get("message")
    if not isinstance(message, dict) or "chat" not in message or _already_seen(update.get("update_id")):
        return {"ok": True}
    try:
        await _handle_message(TelegramAPI(token), message)
    except Exception:
        # Always acknowledge, otherwise Telegram keeps retrying the same update.
        logger.exception("Telegram update could not be handled")
    return {"ok": True}
