"""WhatsApp bot through the Vonage Messages API Sandbox.

Vonage POSTs each inbound message to /whatsapp/vonage/inbound and delivery
statuses to /whatsapp/vonage/status (Messages API v1 webhooks). As with the
Telegram bot, a message is checked and answered inside the request, because
serverless hosts such as Vercel may freeze the function as soon as the response
is returned. Vonage retries any non-2xx response, so valid requests are always
acknowledged with 200.
"""

import base64
import hashlib
import hmac
import json
import logging
import os
from collections import OrderedDict, defaultdict, deque

from fastapi import APIRouter, HTTPException, Request, status

from backend.services.vonage_api import SANDBOX_MESSAGES_URL, VonageMessagesAPI
from . import _whatsapp_common as common
from .verification import verify_article, verify_article_url


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/whatsapp/vonage", tags=["whatsapp"])

MAX_REPLY_LENGTH = 4096
NON_TEXT_TEXT = "📎 I can only check text. Send me a news article link, or paste at least 40 characters of article text."
# Reactions need no answer; other non-text messages get NON_TEXT_TEXT.
SILENT_MESSAGE_TYPES = {"reaction"}

# Both are per process. On serverless hosts each instance keeps its own copy,
# so they are best-effort protection, not a guarantee.
_recent_checks: defaultdict[str, deque[float]] = defaultdict(deque)
_seen_messages: OrderedDict[str, None] = OrderedDict()


def _credentials() -> tuple[str, str]:
    api_key = os.getenv("VONAGE_API_KEY")
    api_secret = os.getenv("VONAGE_API_SECRET")
    if not api_key or not api_secret:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="WhatsApp bot is not configured.")
    return api_key, api_secret


def _b64url_decode(part: str) -> bytes:
    return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))


def _verified_claims(token: str, secret: str) -> dict | None:
    """Return the claims of an HS256 JWT signed with `secret`, or None if it is not valid."""
    try:
        header_b64, claims_b64, signature_b64 = token.split(".")
        header = json.loads(_b64url_decode(header_b64))
        claims = json.loads(_b64url_decode(claims_b64))
        signature = _b64url_decode(signature_b64)
    except ValueError:
        return None
    # Accept HS256 only, so a token cannot pick "none" or another algorithm.
    if not isinstance(header, dict) or header.get("alg") != "HS256" or not isinstance(claims, dict):
        return None
    expected = hmac.new(secret.encode(), f"{header_b64}.{claims_b64}".encode(), hashlib.sha256).digest()
    return claims if hmac.compare_digest(signature, expected) else None


def _payload_hash_matches(claims: dict, body: bytes) -> bool:
    """payload_hash is the SHA-256 (hex) of the webhook body."""
    expected = claims.get("payload_hash")
    if not isinstance(expected, str):
        return False
    hashes = {hashlib.sha256(body).hexdigest()}
    try:
        # Vonage's own example hashes JSON.stringify(body), so accept that form too.
        compact = json.dumps(json.loads(body), separators=(",", ":"), ensure_ascii=False)
        hashes.add(hashlib.sha256(compact.encode()).hexdigest())
    except ValueError:
        pass
    return expected.lower() in hashes


def _check_signature(request: Request, body: bytes) -> None:
    """With VONAGE_SIGNATURE_SECRET set, require Vonage's signed-webhook JWT."""
    secret = os.getenv("VONAGE_SIGNATURE_SECRET")
    if not secret:
        return
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    claims = _verified_claims(token.strip(), secret) if scheme.lower() == "bearer" else None
    if claims is None or not _payload_hash_matches(claims, body):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid webhook signature.")


async def _read_json(request: Request) -> tuple[bytes, object]:
    body = await request.body()
    _check_signature(request, body)
    try:
        return body, json.loads(body)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The body must be JSON.") from None


async def _handle_message(api: VonageMessagesAPI, message: dict) -> None:
    # Reply from the number the user wrote to (the sandbox number).
    sender, our_number = str(message["from"]), str(message["to"])
    message_type = message.get("message_type")
    if message_type != "text":
        if message_type not in SILENT_MESSAGE_TYPES:
            await api.send_text(sender, our_number, NON_TEXT_TEXT)
        return
    target = str(message.get("text") or "").strip()
    if common.needs_help(target):
        # "hi", "start", "help" and anything else too short to check.
        await api.send_text(sender, our_number, common.HELP_TEXT)
        return
    if common.rate_limited(_recent_checks, sender):
        await api.send_text(sender, our_number, common.RATE_LIMITED_TEXT)
        return
    reply = await common.check_reply(
        target, verify_article, verify_article_url, common.CHECK_TIMEOUT_SECONDS, MAX_REPLY_LENGTH,
    )
    await api.send_text(sender, our_number, reply)


@router.post("/inbound", include_in_schema=False)
async def vonage_inbound(request: Request) -> dict[str, bool]:
    api_key, api_secret = _credentials()
    _, message = await _read_json(request)
    if not isinstance(message, dict) or message.get("channel") != "whatsapp":
        return {"ok": True}
    message_uuid = message.get("message_uuid")
    if (
        not message.get("from")
        or not message.get("to")
        or common.already_seen(_seen_messages, message_uuid if isinstance(message_uuid, str) else None)
    ):
        return {"ok": True}
    api = VonageMessagesAPI(api_key, api_secret, os.getenv("VONAGE_MESSAGES_URL") or SANDBOX_MESSAGES_URL)
    try:
        await _handle_message(api, message)
    except Exception:
        # Always acknowledge, otherwise Vonage keeps retrying the same message.
        logger.exception("Vonage WhatsApp message could not be handled")
    return {"ok": True}


@router.post("/status", include_in_schema=False)
async def vonage_status(request: Request) -> dict[str, bool]:
    _credentials()
    _, update = await _read_json(request)
    # Only logged: a failed reply (for example outside WhatsApp's 24-hour window) shows up here.
    if isinstance(update, dict) and update.get("status") in {"rejected", "undeliverable"}:
        error = update.get("error") if isinstance(update.get("error"), dict) else {}
        logger.warning(
            "Vonage message %s was %s: %s",
            update.get("message_uuid"), update.get("status"), error.get("title") or error.get("detail"),
        )
    return {"ok": True}
