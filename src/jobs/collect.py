"""
The collection loop.

    verbs -> autocomplete probes -> suggestions -> objects -> new probes
                                         |
                                         v
                                    SERP snapshot
                                         |
                                    coherence check
                                         |
                                  features -> gates -> score

Ordering matters for cost: probe first (cheap, one request yields ~10
candidate keywords), then snapshot only what survived normalization. Snapshots
are the expensive part at ~1 request per keyword.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import Counter

from .. import db
from ..config import Storefront, settings
from ..pipeline import features as feat
from ..pipeline import score as scoring
from ..pipeline.coherence import dedup_terms
from ..seeds import expand
from ..seeds.verbs import SEED_VERBS
from ..sources import itunes, suggest
from ..sources.http import AppleClient, Fetcher

log = logging.getLogger(__name__)


async def harvest(
    client: Fetcher,
    store: Storefront,
    roots: list[str],
    conn,
    deep: bool,
) -> dict[str, int]:
    """Probe roots, persist the raw corpus, return {term: best_position}."""
    lang = f"{store.lang}_{store.country}"
    all_found: dict[str, int] = {}

    # Shallow mode probes only the bare root and the space suffix: 2 requests
    # per root instead of 28. Use it to scan wide, deep mode to mine a niche.
    suffixes = None if deep else ["", " "]

    for root in roots:
        found = await suggest.probe_prefix_set(
            client, root, store.country, lang,
            suffixes=suffixes, store_id=store.store_id,
        )
        if found:
            db.store_suggestions(conn, root, store.key, found)
        for term, pos in found.items():
            if term not in all_found or pos < all_found[term]:
                all_found[term] = pos
        log.info("probe %-18s -> %d suggestions", root, len(found))

    return all_found


async def refresh_tracked_apps(client: Fetcher, store: Storefront, conn) -> int:
    """
    Re-measure the apps behind every keyword we still care about.

    Without this the alert machinery is dead on arrival. Root rotation only
    moves forward, so an app snapshotted today is never revisited once its
    keyword's root is spent — and `find_alerts` needs three readings of the
    same app inside 30 days to notice a leader decaying. Most apps would sit
    at one reading forever.

    The lookup endpoint accepts up to 200 ids per call, so several hundred
    tracked apps cost one or two requests. Running this every pass is
    effectively free; extra passes on the same calendar day just overwrite
    the same row, which is harmless.
    """
    rows = conn.execute(
        """
        SELECT DISTINCT sp.raw
        FROM keywords k
        JOIN serp_snapshots sp ON sp.id = (
            SELECT id FROM serp_snapshots
             WHERE keyword_id = k.id ORDER BY taken_at DESC LIMIT 1
        )
        WHERE k.storefront = ? AND k.status IN ('queued', 'watchlist', 'in_astro')
        """,
        (store.key,),
    ).fetchall()

    track_ids: list[int] = []
    seen: set[int] = set()
    for r in rows:
        try:
            apps = json.loads(r["raw"])
        except (TypeError, ValueError):
            continue
        # Top-3 only. The tail of a SERP churns for reasons that have nothing
        # to do with the niche, and tracking it would multiply cost for noise.
        for a in apps[:3]:
            tid = a.get("track_id")
            if tid and tid not in seen:
                seen.add(tid)
                track_ids.append(tid)

    if not track_ids:
        return 0

    day = db.today_iso()
    updated = 0
    for i in range(0, len(track_ids), 200):
        batch = track_ids[i:i + 200]
        apps = await itunes.lookup(client, batch, store.country)
        for a in apps:
            if not a.get("track_id"):
                continue
            conn.execute(
                "INSERT INTO app_metrics_daily "
                "(track_id, storefront, day, rating, rating_count) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(track_id, storefront, day) DO UPDATE SET "
                "rating = excluded.rating, rating_count = excluded.rating_count",
                (a["track_id"], store.key, day, a.get("rating"), a.get("rating_count")),
            )
            updated += 1

    return updated


async def run(
    client: Fetcher,
    store: Storefront,
    roots: list[str],
    deep: bool = False,
    max_snapshots: int = 60,
) -> dict[str, int]:
    """One collection pass for one storefront."""
    stats = Counter()

    with db.connect() as conn:
        found = await harvest(client, store, roots, conn, deep)
        stats["suggestions"] = len(found)

        if not found:
            log.warning("no suggestions returned — endpoint may be blocked")
            return dict(stats)

        # Grow the lexicon for the next run.
        objects = expand.extract_objects(roots[0] if roots else "", list(found))
        probed = {
            r["prefix"] for r in conn.execute(
                "SELECT prefix FROM probed_prefixes WHERE storefront = ?", (store.key,)
            )
        }
        frontier = expand.next_probes(objects, probed)
        stats["new_probes"] = len(frontier)

        # 2 sightings, not 3: on a single pass over a few roots the corpus is
        # too small for 3 to ever trigger (last run discovered 0 verbs).
        new_verbs = expand.discover_verbs(list(found), set(SEED_VERBS), min_distinct=2)
        for verb in new_verbs:
            conn.execute(
                "INSERT INTO discovered_verbs (verb, found_at) VALUES (?, ?) "
                "ON CONFLICT(verb) DO UPDATE SET evidence = evidence + 1",
                (verb, db.now_iso()),
            )
        stats["new_verbs"] = len(new_verbs)

        # Dedup before spending SERP requests on spelling variants.
        canonical = dedup_terms(list(found))
        heads = sorted({canonical[t] for t in found}, key=lambda t: found.get(t, 99))
        stats["after_dedup"] = len(heads)

        # Snapshot and score.
        for term in heads[:max_snapshots]:
            if suggest.looks_like_brand(term):
                db.upsert_keyword(conn, term, store.key, "suggest",
                                  suggest_pos=found.get(term), kw_type="BRAND")
                stats["brand"] += 1
                continue

            kw_id = db.upsert_keyword(
                conn, term, store.key, "suggest", suggest_pos=found.get(term)
            )
            apps = await itunes.search(client, term, store.country, limit=10)
            if not apps:
                stats["empty_serp"] += 1
                continue

            db.store_serp(conn, kw_id, term, store.key, apps)

            f = feat.extract(term, store.key, store.lang, apps, found.get(term))
            if apps and apps[0].get("track_id"):
                f.top1_weekly_review_delta = db.review_delta(
                    conn, apps[0]["track_id"], store.key
                )

            s = scoring.score(f)
            db.store_score(conn, kw_id, f.as_dict(), s)
            stats[s.verdict.value] += 1
            # Per-gate counters. Without these, "rejected: 19" says nothing
            # about whether the thresholds are sane or the roots were bad.
            for g in s.gates_failed:
                stats[f"gate_{g}"] += 1

        # Keep the tracked set measured, so deltas and alerts can exist.
        refreshed = await refresh_tracked_apps(client, store, conn)
        if refreshed:
            stats["apps_refreshed"] = refreshed

    return dict(stats)


async def main() -> None:
    ap = argparse.ArgumentParser(description="ASO Scout collector")
    ap.add_argument("--store", default="us", help="storefront country code")
    ap.add_argument("--roots", nargs="*", help="probe roots (default: seed verbs)")
    ap.add_argument("--deep", action="store_true", help="probe a-z suffixes too")
    ap.add_argument("--limit-roots", type=int, default=8)
    ap.add_argument("--max-snapshots", type=int, default=60)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    db.init_db()
    store = next(
        (s for s in settings.storefronts if s.key == args.store.lower()),
        settings.storefronts[0],
    )
    roots = args.roots or SEED_VERBS[: args.limit_roots]

    async with AppleClient() as client:
        stats = await run(client, store, roots, args.deep, args.max_snapshots)
        log.info("done: %s", stats)
        log.info("http: %s", client.stats)
        if client.stats.get("blocked"):
            log.error(
                "requests were blocked (403). Check the egress IP before "
                "trusting any numbers from this run."
            )


if __name__ == "__main__":
    asyncio.run(main())
