"""Point the Telegram bot at the deployed API (run once after each domain change).

Usage:
    python scripts/set_telegram_webhook.py https://your-project.vercel.app

Reads TELEGRAM_BOT_TOKEN and TELEGRAM_WEBHOOK_SECRET from the environment or
.env; they must match the values set on the API host.
"""

import os
import sys

import httpx
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    secret = os.getenv("TELEGRAM_WEBHOOK_SECRET")
    if not token or not secret:
        sys.exit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_WEBHOOK_SECRET first.")
    webhook_url = f"{sys.argv[1].rstrip('/')}/telegram/webhook"
    api = f"https://api.telegram.org/bot{token}"

    response = httpx.post(f"{api}/setWebhook", json={
        "url": webhook_url,
        "secret_token": secret,
        "allowed_updates": ["message"],
        "drop_pending_updates": True,
    })
    print("setWebhook:", response.json())
    response = httpx.post(f"{api}/setMyCommands", json={"commands": [
        {"command": "check", "description": "Check an article link or text (reply to a message in groups)"},
        {"command": "help", "description": "How to use ClearSight"},
    ]})
    print("setMyCommands:", response.json())
    info = httpx.get(f"{api}/getWebhookInfo").json().get("result", {})
    print("Webhook URL:", info.get("url"), "| pending updates:", info.get("pending_update_count"))
    if info.get("last_error_message"):
        print("Last delivery error:", info["last_error_message"])


if __name__ == "__main__":
    main()
