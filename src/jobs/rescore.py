"""
Recompute features and scores from stored snapshots. No network calls.

This is the payoff for storing raw API responses untouched: when a gate or a
weight turns out to be wrong, the fix applies retroactively to everything
already collected. No re-crawl, no waiting, no burned rate limit.

    python -m src.jobs.rescore
    python -m src.jobs.rescore --store us

Only the latest snapshot per keyword is used. Keywords already decided by hand
(go / no_go / narrow / in_astro) keep their status — a threshold change should
not silently overturn a human verdict.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter

from .. import db
from ..config import settings
from ..pipeline import features as feat
from ..pipeline import score as scoring

log = logging.getLogger(__name__)

# Statuses set by a person. Rescoring updates their numbers but not their fate.
HUMAN_DECIDED = {"in_astro", "go", "no_go", "narrow", "hold"}


def main() -> None:
    ap = argparse.ArgumentParser(description="rescore from stored snapshots")
    ap.add_argument("--store", help="limit to one storefront")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )

    db.init_db()
    before: Counter[str] = Counter()
    after: Counter[str] = Counter()
    moved: list[tuple[str, str, str, float]] = []

    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT k.id, k.term, k.storefront, k.status, k.suggest_pos,
                   (SELECT raw FROM serp_snapshots sp
                     WHERE sp.keyword_id = k.id
                     ORDER BY sp.taken_at DESC LIMIT 1) AS serp
            FROM keywords k
            WHERE (? IS NULL OR k.storefront = ?)
            """,
            (args.store, args.store),
        ).fetchall()

        if not rows:
            log.warning("no keywords in the database — run collect first")
            return

        for r in rows:
            before[r["status"]] += 1
            if not r["serp"]:
                after[r["status"]] += 1
                continue

            apps = json.loads(r["serp"])
            store = next(
                (s for s in settings.storefronts if s.key == r["storefront"]),
                settings.storefronts[0],
            )

            f = feat.extract(
                r["term"], r["storefront"], store.lang, apps, r["suggest_pos"]
            )
            if apps and apps[0].get("track_id"):
                f.top1_weekly_review_delta = db.review_delta(
                    conn, apps[0]["track_id"], r["storefront"]
                )

            s = scoring.score(f)

            if r["status"] in HUMAN_DECIDED:
                # Refresh the numbers, leave the verdict alone.
                conn.execute(
                    "INSERT INTO keyword_features (keyword_id, computed_at, payload) "
                    "VALUES (?, ?, ?)",
                    (r["id"], db.now_iso(), json.dumps(f.as_dict(), ensure_ascii=False)),
                )
                after[r["status"]] += 1
                continue

            db.store_score(conn, r["id"], f.as_dict(), s)
            after[s.verdict.value] += 1
            for g in s.gates_failed:
                after[f"gate_{g}"] += 1

            if r["status"] != s.verdict.value:
                moved.append((r["term"], r["status"], s.verdict.value, s.score))

    log.info("before: %s", dict(before))
    log.info("after:  %s", dict(after))

    promoted = [m for m in moved if m[2] == "queued"]
    if promoted:
        log.info("")
        log.info("%d keyword(s) promoted into the queue:", len(promoted))
        for term, old, new, sc in sorted(promoted, key=lambda m: -m[3]):
            log.info("  %-38s %s -> %s  score %.1f", term, old, new, sc)

    log.info("")
    log.info("Run `python -m src.jobs.report` to see the queue.")


if __name__ == "__main__":
    main()
