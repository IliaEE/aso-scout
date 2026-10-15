"""
Turn measured demand into candidates by adding the competition side.

Apple tells us how big a search term is. It does not say who serves it, and
without that the trend reports are unusable: the first live run surfaced
"cinnabon", "mcdonalds rewards", "servus credit union" and "daves hot
chicken" as top movers. All real, all measured correctly, none of them a
niche anyone can build into.

That is not a flaw in the data. Movement belongs to brands, because brands
are what move — a launch, a campaign, a news cycle. Generic category demand
("pdf editor", "calorie tracker") is stable precisely because it is generic.
Which means the demand numbers only become useful once each term has been
through the same filters as everything else: who is in the top ten, is the
query simply an app's name, is the leader a fortress.

    python -m src.jobs.asa_enrich --top 200          # by popularity
    python -m src.jobs.asa_enrich --trending         # the movers instead
    python -m src.jobs.asa_enrich --min-pop 50 --top 500
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter

from .. import db
from ..asa.descriptors import tail_of
from ..asa.trends import movement, new_entrants, sustained
from ..config import settings
from ..db import connect as connect_db
from ..pipeline import features as feat
from ..pipeline import score as scoring
from ..sources import itunes
from ..sources.http import AppleClient, Fetcher
from .asa_pull import ensure_schema

log = logging.getLogger(__name__)


def candidates_by_popularity(
    conn, countries: list[str], min_pop: int, limit: int, max_pop: int = 100
) -> list[tuple[str, str, int]]:
    """
    Terms inside a popularity band, newest period only.

    A band, not a top-N. Sorting by popularity and taking the first 300 gave
    153 brands out of 300: the 70-90 range is where ChatGPT, Google and Uber
    Eats live, and nobody is taking those. Generic category demand sits lower,
    roughly 45-65, which is also the size that fits a $10k target — enough
    traffic to be worth serving, small enough that the leader is not a
    fortress.
    """
    ensure_schema(conn)
    ph = ",".join("?" * len(countries))
    newest = conn.execute(
        "SELECT MAX(period) p FROM asa_popularity WHERE granularity = 'WEEKLY_SUN_SAT'"
    ).fetchone()["p"]
    if not newest:
        return []
    rows = conn.execute(
        f"""
        SELECT term, country, MAX(pop_1to100) AS pop
        FROM asa_popularity
        WHERE granularity = 'WEEKLY_SUN_SAT' AND period = ?
          AND country IN ({ph}) AND pop_1to100 BETWEEN ? AND ?
        GROUP BY term, country
        ORDER BY pop DESC
        LIMIT ?
        """,
        [newest, *countries, min_pop, max_pop, limit],
    ).fetchall()
    return [(r["term"], r["country"], r["pop"]) for r in rows]


def candidates_from_descriptors(
    conn, countries: list[str], limit: int
) -> list[tuple[str, str, int]]:
    """
    Needs extracted from the descriptive half of branded terms.

    "be my eyes – helping the blind" is useless as a query and informative as
    a sentence: the tail names a need, written by the people searching for it.
    These cost no extra Apple calls to find — the terms are already stored,
    discarded as brands — and they still face every gate afterwards.
    """
    ensure_schema(conn)
    ph = ",".join("?" * len(countries))
    rows = conn.execute(
        f"""
        SELECT DISTINCT term, country FROM asa_popularity
        WHERE granularity = 'WEEKLY_SUN_SAT' AND country IN ({ph})
        """,
        countries,
    ).fetchall()

    known = {
        (r["term"], r["storefront"])
        for r in conn.execute("SELECT term, storefront FROM keywords")
    }

    out: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str]] = set()
    for r in rows:
        tail = tail_of(r["term"])
        if not tail:
            continue
        key = (tail, r["country"])
        if key in seen or key in known:
            continue
        seen.add(key)
        out.append((tail, r["country"], 0))   # popularity unknown for a tail
        if len(out) >= limit:
            break
    return out


def candidates_trending(
    conn, countries: list[str], limit: int
) -> list[tuple[str, str, int]]:
    """Movers and newcomers, deduplicated."""
    seen: dict[tuple[str, str], int] = {}
    for m in (new_entrants(conn, countries)
              + sustained(conn, countries)
              + movement(conn, countries)):
        key = (m.term, m.country)
        if key not in seen:
            seen[key] = m.popularity or 0
    return [(t, c, p) for (t, c), p in list(seen.items())[:limit]]


async def enrich(
    client: Fetcher, items: list[tuple[str, str, int]], skip_known: bool = True
) -> dict[str, int]:
    stats: Counter[str] = Counter()

    for term, country, pop in items:
        store = next(
            (s for s in settings.storefronts if s.key == country.lower()), None
        )
        if store is None:
            stats["unknown_store"] += 1
            continue

        if skip_known:
            with connect_db() as conn:
                known = conn.execute(
                    "SELECT 1 FROM serp_snapshots sp JOIN keywords k ON k.id = sp.keyword_id "
                    "WHERE k.term = ? AND k.storefront = ? LIMIT 1",
                    (term, store.key),
                ).fetchone()
            if known:
                stats["already_had_serp"] += 1
                continue

        apps = await itunes.search(client, term, store.country, limit=10)
        if not apps:
            stats["empty_serp"] += 1
            continue

        f = feat.extract(term, store.key, store.lang, apps, None)
        s = scoring.score(f)

        # The brand check is the whole point here: a term that IS an app's
        # name has demand that cannot be taken, however fast it is growing.
        is_brand = (f.top1_name_match or 0) >= 0.75

        def _write() -> None:
            with connect_db() as conn:
                kw_id = db.upsert_keyword(conn, term, store.key, "asa")
                db.store_serp(conn, kw_id, term, store.key, apps)
                db.store_score(conn, kw_id, f.as_dict(), s)
                conn.commit()

        db.write_with_retry(_write)

        if is_brand:
            stats["brand"] += 1
        else:
            stats[s.verdict.value] += 1
        log.info("  %-34s %-3s поп %-3s %-11s %s",
                 term[:34], country.upper(), pop,
                 "БРЕНД" if is_brand else s.verdict.value,
                 (apps[0].get("title") or "")[:34])

    return dict(stats)


def main() -> int:
    ap = argparse.ArgumentParser(description="add competition data to ASA terms")
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--min-pop", type=int, default=45)
    ap.add_argument("--max-pop", type=int, default=65,
                    help="upper end of the popularity band. Above ~65 is brand "
                         "territory (chatgpt, google, uber eats), not niches")
    ap.add_argument("--trending", action="store_true",
                    help="enrich the movers rather than the most popular")
    ap.add_argument("--descriptors", action="store_true",
                    help="enrich needs extracted from branded terms, e.g. "
                         "'be my eyes – helping the blind' -> 'helping the blind'")
    ap.add_argument("--countries", default="us,gb,ca,au")
    ap.add_argument("--all", action="store_true",
                    help="re-fetch terms that already have a SERP")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    db.init_db()
    countries = [c.strip().lower() for c in args.countries.split(",") if c.strip()]

    with db.connect() as conn:
        if args.descriptors:
            items = candidates_from_descriptors(conn, countries, args.top)
        elif args.trending:
            items = candidates_trending(conn, countries, args.top)
        else:
            items = candidates_by_popularity(
                conn, countries, args.min_pop, args.top, args.max_pop
            )

    if not items:
        print("нечего обогащать — сначала asa_pull")
        return 1

    log.info("%d терминов, ~%d запросов к App Store (≈%d мин при 18/мин)",
             len(items), len(items), max(len(items) // 18, 1))

    async def run() -> dict[str, int]:
        async with AppleClient() as client:
            out = await enrich(client, items, skip_known=not args.all)
            log.info("http: %s", client.stats)
            return out

    stats = asyncio.run(run())
    print(json.dumps(stats, ensure_ascii=False))
    print("\nДальше: python -m src.jobs.report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
