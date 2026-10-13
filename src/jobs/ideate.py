"""
Generate adjacent queries with the model, then make them earn their place.

    python -m src.jobs.ideate                # propose from the best niches
    python -m src.jobs.ideate --from 5       # seed from the top 5
    python -m src.jobs.ideate --stats        # how proposals have fared
    python -m src.jobs.ideate --dry-run      # propose, do not verify or store

Validation is the whole point. A proposal is a hypothesis: Apple's
autocomplete only returns strings people type, so a query that never appears
there has shown no demand at all, however sensible it reads. Each proposal
costs two Apple requests to check, which is why the prompt asks for a few good
ones rather than many.

Outcomes are recorded either way. A query rejected tonight must not be
proposed again tomorrow.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import Counter

from .. import db
from ..config import settings
from ..llm.ideate import (
    Proposal,
    already_proposed,
    ensure_table,
    generate,
    hit_rate,
    known_terms,
    mark,
    save_proposal,
)
from ..llm.provider import chat_for
from ..pipeline import features as feat
from ..pipeline import score as scoring
from ..report.render import group_into_clusters, load_queue
from ..sources import itunes, suggest
from ..sources.http import AppleClient, Fetcher

log = logging.getLogger(__name__)


async def verify(
    client: Fetcher, p: Proposal, store, conn
) -> tuple[str, str]:
    """
    Decide whether a proposed query is real. Returns (status, note).

    Two checks, in cost order:

      1. Does Apple suggest it? This is the demand test the rest of the
         pipeline gets for free. Accepting a query that never appears in
         autocomplete would mean importing the model's imagination as data.
      2. Does the SERP cohere and pass the gates? Same bar as any other
         keyword — no exemption for having been suggested by a model.
    """
    lang = f"{store.lang}_{store.country}"
    hints = await suggest.probe(client, p.query, store.country, lang, store.store_id)
    terms = {t for t, _ in hints}

    # Exact match, or Apple completing it into something longer. Both mean
    # people type this; a prefix that returns nothing means they do not.
    pos = None
    for term, position in hints:
        if term == p.query or term.startswith(p.query):
            pos = position
            break
    if not terms:
        return "no_demand", "автокомплит ничего не вернул"
    if pos is None:
        return "no_demand", f"нет в подсказках (вернулось: {list(terms)[:3]})"

    apps = await itunes.search(client, p.query, store.country, limit=10)
    if not apps:
        return "incoherent", "пустая выдача"

    f = feat.extract(p.query, store.key, store.lang, apps, pos)
    s = scoring.score(f)
    if s.verdict == scoring.Verdict.REJECTED:
        return "gated", ", ".join(s.gates_failed)

    kw_id = db.upsert_keyword(conn, p.query, store.key, "llm", suggest_pos=pos)
    db.store_serp(conn, kw_id, p.query, store.key, apps)
    db.store_score(conn, kw_id, f.as_dict(), s)
    conn.commit()
    return "accepted", f"score {s.score}, автокомплит #{pos}"


async def run(seed_count: int, dry_run: bool = False) -> dict[str, int]:
    chat = chat_for("ideate")
    if chat is None:
        log.warning("no API key set — skipping ideation")
        return {}
    if not chat.resolve_model():
        log.error("no usable model")
        return {}

    stats: Counter[str] = Counter()
    proposals: list[tuple[Proposal, object]] = []

    with db.connect() as conn:
        ensure_table(conn)
        clusters = group_into_clusters(load_queue(conn))
        # Seed from the strongest niches: adjacency to a good niche is more
        # likely to be good than adjacency to a weak one.
        seeds = sorted(clusters, key=lambda c: -c.rating.stars)[:seed_count]
        if not seeds:
            log.warning("queue is empty — collect first")
            return {}

        for cl in seeds:
            c = cl.head
            store = next(
                (s for s in settings.storefronts if s.key == c.storefront),
                settings.storefronts[0],
            )
            exclude = known_terms(conn, store.key) | already_proposed(conn, store.key)
            fresh = generate(chat, c.term, store.key, c.apps, exclude)
            stats["proposed"] += len(fresh)
            log.info("from %-26s -> %d ideas", c.term[:26],
                     len(fresh))
            for p in fresh:
                log.info("    %s  (%s)", p.query, p.why[:60])
                if not dry_run:
                    save_proposal(conn, p)
                proposals.append((p, store))
            if getattr(chat, "rate_limited", False):
                log.warning("rate limit reached — stopping generation")
                break
        if not dry_run:
            conn.commit()

    if dry_run or not proposals:
        return dict(stats)

    log.info("verifying %d proposals against Apple", len(proposals))
    async with AppleClient() as client:
        with db.connect() as conn:
            for p, store in proposals:
                status, note = await verify(client, p, store, conn)
                mark(conn, p.query, store.key, status, note)
                conn.commit()
                stats[status] += 1
                log.info("  %-10s %-30s %s", status, p.query[:30], note[:50])

    return dict(stats)


def main() -> int:
    ap = argparse.ArgumentParser(description="propose and verify new queries")
    ap.add_argument("--from", dest="seed_count", type=int,
                    default=int(os.environ.get("IDEATE_SEEDS", 5)))
    ap.add_argument("--dry-run", action="store_true",
                    help="show proposals without verifying or storing them")
    ap.add_argument("--stats", action="store_true",
                    help="how past proposals fared")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    db.init_db()

    if args.stats:
        with db.connect() as conn:
            rates = hit_rate(conn)
        if not rates:
            print("предложений ещё не было")
            return 0
        total = sum(rates.values())
        accepted = rates.get("accepted", 0)
        print(f"всего предложено: {total}")
        for status, n in sorted(rates.items(), key=lambda kv: -kv[1]):
            print(f"  {status:<12} {n:>4}  ({n / total:.0%})")
        print()
        if accepted == 0 and total >= 20:
            print("Ни одно предложение не прошло проверку. Модель выдумывает "
                  "запросы, которых не ищут — слой стоит отключить.")
        elif total:
            print(f"доля прошедших: {accepted / total:.0%}")
        return 0

    stats = asyncio.run(run(args.seed_count, args.dry_run))
    if not stats:
        print("ничего не сделано — смотрите лог")
        return 1
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
