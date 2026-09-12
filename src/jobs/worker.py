"""
The autonomous worker: HTTP report + a daily collection run.

One process, one service, one volume. Railway cron would also work, but a
collection pass takes 15-40 minutes under the rate limit, and Railway skips a
cron run whose predecessor is still alive — an internal schedule avoids that
class of surprise entirely.

    python -m src.jobs.worker

Environment:
    COLLECT_HOUR   UTC hour to start the daily run (default 3)
    ROOTS_PER_RUN  probe roots per run (default 8)
    MAX_SNAPSHOTS  SERP snapshots per run (default 60)
    STORES         comma-separated storefronts (default us)
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from datetime import datetime, timezone

from .. import db
from ..config import settings
from ..seeds.rotation import pick_roots
from ..server import serve
from ..sources.http import AppleClient
from .collect import run

log = logging.getLogger(__name__)

STATE_LAST_RUN = "last_collect_date"


def _get_state(conn, key: str) -> str | None:
    row = conn.execute(
        "SELECT value FROM run_state WHERE key = ?", (key,)
    ).fetchone()
    return row["value"] if row else None


def _set_state(conn, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO run_state (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def ensure_state_table() -> None:
    with db.connect() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS run_state ("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )


async def collect_once() -> None:
    """One pass over every configured storefront."""
    roots_per_run = int(os.environ.get("ROOTS_PER_RUN", 8))
    max_snapshots = int(os.environ.get("MAX_SNAPSHOTS", 60))
    wanted = [s.strip().lower() for s in os.environ.get("STORES", "us").split(",")]
    stores = [s for s in settings.storefronts if s.key in wanted] or settings.storefronts[:1]

    async with AppleClient() as client:
        for store in stores:
            with db.connect() as conn:
                roots, reason = pick_roots(conn, store.key, roots_per_run)

            if not roots:
                log.warning("%s: no roots to probe", store.key)
                continue

            log.info("%s: probing %s (%s)", store.key, roots, reason)
            stats = await run(client, store, roots, deep=False,
                              max_snapshots=max_snapshots)
            log.info("%s: %s", store.key, stats)

        log.info("http: %s", client.stats)
        if client.stats.get("blocked"):
            log.error(
                "Requests were blocked (403). This egress IP cannot reach "
                "Apple — everything collected in this run is invalid."
            )


def scheduler_loop() -> None:
    hour = int(os.environ.get("COLLECT_HOUR", 3))
    log.info("scheduler armed for %02d:00 UTC daily", hour)

    while True:
        try:
            now = datetime.now(timezone.utc)
            today = now.date().isoformat()

            ensure_state_table()
            with db.connect() as conn:
                last = _get_state(conn, STATE_LAST_RUN)

            if now.hour == hour and last != today:
                log.info("starting daily collection for %s", today)
                # Claim the slot before running: a crash mid-run must not put
                # the worker into a restart loop that re-probes all day.
                with db.connect() as conn:
                    _set_state(conn, STATE_LAST_RUN, today)
                try:
                    asyncio.run(collect_once())
                    log.info("daily collection finished")
                except Exception:
                    log.exception("collection failed; will retry tomorrow")

            time.sleep(60)
        except Exception:
            log.exception("scheduler loop error")
            time.sleep(60)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )
    # Credentials first. Starting the scheduler and opening the database
    # before discovering that the server cannot bind wastes a restart cycle
    # and buries the real cause under unrelated log lines.
    from ..server import _credentials  # noqa: PLC0415

    _credentials()

    db.init_db()
    ensure_state_table()

    if os.environ.get("COLLECT_ON_BOOT", "").lower() in {"1", "true", "yes"}:
        log.info("COLLECT_ON_BOOT set — running one pass now")
        try:
            asyncio.run(collect_once())
        except Exception:
            log.exception("boot collection failed")

    threading.Thread(target=scheduler_loop, daemon=True, name="scheduler").start()

    httpd = serve()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
        httpd.shutdown()


if __name__ == "__main__":
    main()
