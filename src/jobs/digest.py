"""
Send the digest right now, without waiting for a collection slot.

Two reasons this exists. The obvious one: after configuring Telegram you want
to see a message immediately rather than wait until 03:00 UTC. The less
obvious one: it is the only way to confirm the token and chat id actually work
before trusting a scheduled job to use them silently at night.

    python -m src.jobs.digest            # respects the once-a-day guard
    python -m src.jobs.digest --force    # sends regardless
    python -m src.jobs.digest --print    # prints instead of sending
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

from .. import db
from ..notify.telegram import configured, send_digest
from ..report.render import load_queue, render_clusters_table

log = logging.getLogger(__name__)

STATE_LAST_DIGEST = "last_digest_date"


def build() -> tuple[str, int]:
    """Render the digest. Returns (text, queue size)."""
    with db.connect() as conn:
        queue = load_queue(conn)
        return render_clusters_table(queue, conn), len(queue)


def main() -> int:
    ap = argparse.ArgumentParser(description="send the digest now")
    ap.add_argument("--force", action="store_true",
                    help="send even if one already went out today")
    ap.add_argument("--print", dest="show", action="store_true",
                    help="print to stdout instead of sending")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    db.init_db()

    text, size = build()
    if size == 0:
        print("очередь пуста — сначала нужен хотя бы один сбор")
        return 1

    if args.show:
        print(text)
        return 0

    if not configured():
        print(
            "TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID не заданы.\n"
            "Проверить содержимое без отправки: --print"
        )
        return 1

    today = datetime.now(timezone.utc).date().isoformat()
    if not args.force:
        with db.connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS run_state "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            row = conn.execute(
                "SELECT value FROM run_state WHERE key = ?", (STATE_LAST_DIGEST,)
            ).fetchone()
        if row and row["value"] == today:
            print("дайджест сегодня уже отправлялся. Повторить: --force")
            return 0

    if send_digest(text):
        with db.connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS run_state "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT INTO run_state (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (STATE_LAST_DIGEST, today),
            )
        print(f"отправлено ({size} запросов в очереди)")
        return 0

    print("отправить не удалось — причина в логе выше")
    return 1


if __name__ == "__main__":
    sys.exit(main())
