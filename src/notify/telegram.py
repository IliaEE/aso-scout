"""
Daily digest to Telegram.

The point is not convenience. The queue's top rotates as scores shift, so a
niche that was interesting on Tuesday can be pushed off the visible list by
Thursday without anyone ever looking at it. A digest that arrives whether or
not you open the page makes the queue something you skim rather than something
you have to remember to visit.

Setup:
  1. Message @BotFather, /newbot, copy the token.
  2. Message your new bot once, then open
     https://api.telegram.org/bot<TOKEN>/getUpdates and read chat.id.
  3. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.

With either variable unset the digest is skipped silently — it is an optional
output, never a reason for a collection run to fail.
"""
from __future__ import annotations

import logging
import os
from typing import Protocol

import httpx

log = logging.getLogger(__name__)

API = "https://api.telegram.org/bot{token}/sendMessage"

# Telegram rejects messages over 4096 characters. The <pre> wrapper and a
# small margin come out of that budget.
MAX_CHARS = 3800


class Sender(Protocol):
    def send(self, text: str) -> bool: ...


def configured() -> bool:
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN")
                and os.environ.get("TELEGRAM_CHAT_ID"))


def split_message(text: str, limit: int = MAX_CHARS) -> list[str]:
    """
    Split on line boundaries so table rows are never cut mid-line.

    A row broken across two messages is unreadable in a monospace block, which
    defeats the whole reason for sending a table.
    """
    chunks: list[str] = []
    current: list[str] = []
    size = 0

    for line in text.split("\n"):
        # A single line longer than the limit cannot be preserved; hard-wrap it
        # rather than dropping it.
        while len(line) > limit:
            if current:
                chunks.append("\n".join(current))
                current, size = [], 0
            chunks.append(line[:limit])
            line = line[limit:]

        if size + len(line) + 1 > limit and current:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1

    if current:
        chunks.append("\n".join(current))
    return [c for c in chunks if c.strip()]


def _escape(text: str) -> str:
    """Minimal HTML escaping for Telegram's <pre> block."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class TelegramSender:
    def __init__(self, token: str | None = None, chat_id: str | None = None) -> None:
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")

    def send(self, text: str) -> bool:
        if not self.token or not self.chat_id:
            return False
        url = API.format(token=self.token)
        ok = True
        with httpx.Client(timeout=30.0) as client:
            for i, chunk in enumerate(split_message(text), start=1):
                try:
                    resp = client.post(url, json={
                        "chat_id": self.chat_id,
                        "text": f"<pre>{_escape(chunk)}</pre>",
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    })
                    if resp.status_code != 200:
                        log.error("telegram chunk %d failed: %s %s",
                                  i, resp.status_code, resp.text[:200])
                        ok = False
                except Exception:
                    log.exception("telegram chunk %d failed", i)
                    ok = False
        return ok


def send_digest(text: str, sender: Sender | None = None) -> bool:
    """
    Send the digest. Never raises: a failed notification must not take down a
    collection run that already did its real work.
    """
    if sender is None:
        if not configured():
            log.info("telegram not configured — skipping digest")
            return False
        sender = TelegramSender()
    try:
        return sender.send(text)
    except Exception:
        log.exception("digest send failed")
        return False
