"""Message logic and WhatsApp formatting for the WhatsApp webhook(s).

The routes pass in the verification functions and their own rate-limit and
duplicate-tracking state, so each webhook stays independent.
"""

import asyncio
import logging
import re
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import Awaitable, Callable

from fastapi import HTTPException

from backend.schemas import Article, VerificationResponse, VerifyUrlRequest


logger = logging.getLogger(__name__)

MIN_TEXT_LENGTH = 40
RATE_LIMIT = 5
RATE_WINDOW_SECONDS = 60
# Answer before the host or the WhatsApp provider gives up on the request.
CHECK_TIMEOUT_SECONDS = 12
URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)

HELP_TEXT = (
    "*ClearSight news checker*\n\n"
    "Send me a news article link, or paste at least 40 characters of article text, "
    "and I will run it through the ClearSight model and look for related fact checks.\n\n"
    "_This is a student project: a TF-IDF + Linear SVM model trained on the WELFake "
    "dataset. It is a second opinion, not a fact check._"
)
RATE_LIMITED_TEXT = "⏳ Too many checks in a minute. Please try again shortly."
TIMEOUT_TEXT = "⌛ This is taking too long. Please try again, or paste the article text instead of the link."
ERROR_TEXT = "⚠️ Something went wrong while checking this. Please try again later."
DISCLAIMER = (
    "_Confidence comes from the model's decision margin and is not a calibrated "
    "probability. The model learned writing patterns from WELFake (mostly US news), "
    "so please confirm with a reliable source._"
)

VerifyText = Callable[..., Awaitable[VerificationResponse]]
VerifyUrl = Callable[..., Awaitable[VerificationResponse]]


def needs_help(target: str) -> bool:
    """Greetings such as "hi", "help" and "start", and anything else too short to check."""
    return not URL_RE.search(target) and len(target) < MIN_TEXT_LENGTH


def rate_limited(recent_checks: defaultdict[str, deque[float]], sender: str) -> bool:
    now = time.monotonic()
    checks = recent_checks[sender]
    while checks and now - checks[0] > RATE_WINDOW_SECONDS:
        checks.popleft()
    if len(checks) >= RATE_LIMIT:
        return True
    checks.append(now)
    return False


def already_seen(seen: OrderedDict[str, None], message_id: str | None) -> bool:
    """The provider may re-send a message if the webhook was slow; answer it only once."""
    if not message_id:
        return False
    if message_id in seen:
        return True
    seen[message_id] = None
    while len(seen) > 500:
        seen.popitem(last=False)
    return False


def utf16_length(text: str) -> int:
    # Emoji outside the Basic Multilingual Plane count twice towards message limits.
    return sum(2 if ord(char) > 0xFFFF else 1 for char in text)


def format_result(result: VerificationResponse, max_length: int | None = None, max_items: int = 3) -> str:
    # WhatsApp markup: *bold*, _italic_, and bare URLs on their own line.
    if result.verdict == "UNCERTAIN":
        lines = [
            f"🟡 *Model verdict: UNCERTAIN* (leaned {result.prediction}, only {result.confidence:.0%} confidence)"
        ]
    else:
        icon = "🔴" if result.verdict == "FAKE" else "🟢"
        lines = [f"{icon} *Model verdict: {result.verdict}* ({result.confidence:.0%} confidence)"]
    if result.article.title:
        lines.append(f"📰 {result.article.title[:200]}")

    fact_checks = [item for item in result.evidence if item.type == "fact_check" and item.relevance != "low"]
    news = [item for item in result.evidence if item.type == "news" and item.relevance != "low"]
    if fact_checks:
        lines.append("\n*Related fact checks*")
        for item in fact_checks[:max_items]:
            rating = f" — rated _{item.rating}_" if item.rating else ""
            publisher = f" ({item.publisher})" if item.publisher else ""
            lines.append(f"• {item.title[:120]}{publisher}{rating}\n{item.url}")
    if news:
        lines.append("\n*Related coverage*")
        for item in news[:max_items]:
            publisher = f" ({item.publisher})" if item.publisher else ""
            lines.append(f"• {item.title[:120]}{publisher}\n{item.url}")
    if not fact_checks and not news:
        if all(source.status == "not_configured" for source in result.sources):
            lines.append("\nEvidence search is not configured on this server.")
        else:
            lines.append("\nNo closely related fact checks or coverage were found.")

    lines.append(f"\n{DISCLAIMER}")
    reply = "\n".join(lines)
    # For short message limits, drop evidence items rather than the disclaimer.
    if max_length is not None and max_items > 1 and utf16_length(reply) > max_length:
        return format_result(result, max_length, max_items - 1)
    return reply


async def run_check(target: str, verify_text: VerifyText, verify_url: VerifyUrl, max_length: int | None = None) -> str:
    """Prefer a link in the message; fall back to the text if the link fails."""
    url_match = URL_RE.search(target)
    remaining_text = URL_RE.sub("", target).strip()
    url_error = None
    if url_match:
        try:
            # Validated like the /verify/url request body.
            url = str(VerifyUrlRequest(url=url_match.group(0)).url)
            return format_result(await verify_url(url, record=False), max_length)
        except (HTTPException, ValueError) as exc:
            url_error = exc.detail if isinstance(exc, HTTPException) else "That link is not a valid URL."
            if len(remaining_text) < MIN_TEXT_LENGTH:
                return f"⚠️ I couldn't check that link: {url_error}"
    try:
        # record=False: chat messages must not appear in the public /history.
        result = await verify_text(Article(body=remaining_text if url_match else target), record=False)
    except HTTPException as exc:
        return f"⚠️ I couldn't check that text: {exc.detail}"
    reply = format_result(result, max_length)
    if url_error:
        reply = f"(The link couldn't be opened, so I checked the message text instead.)\n\n{reply}"
    return reply


async def check_reply(
    target: str, verify_text: VerifyText, verify_url: VerifyUrl, timeout: float, max_length: int | None = None,
) -> str:
    """Run the check, turning a timeout or an unexpected error into a short reply."""
    try:
        return await asyncio.wait_for(run_check(target, verify_text, verify_url, max_length), timeout=timeout)
    except TimeoutError:
        logger.warning("WhatsApp check timed out")
        return TIMEOUT_TEXT
    except Exception:
        logger.exception("WhatsApp check failed")
        return ERROR_TEXT
