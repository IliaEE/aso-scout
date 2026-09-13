"""
Choosing what to probe next.

An autonomous daily run cannot keep probing the same first six verbs. This
advances through the frontier on its own:

  1. seed verbs that have never been probed for this storefront
  2. then objects harvested from suggestions that have not been probed
  3. then verbs discovered from the corpus
  4. when everything is exhausted, re-probe the oldest prefixes so the
     suggestion corpus stays current

Everything is derived from tables the collector already fills, so there is no
extra state to keep in sync.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone

from ..seeds.expand import STOPWORDS, next_probes, singularize
from ..seeds.verbs import SEED_VERBS, TOO_BROAD

log = logging.getLogger(__name__)

# A probed prefix is not worth revisiting sooner than this.
REPROBE_AFTER_DAYS = int(os.environ.get("REPROBE_AFTER_DAYS", 7))


def probed(conn: sqlite3.Connection, storefront: str) -> set[str]:
    return {
        r["prefix"]
        for r in conn.execute(
            "SELECT prefix FROM probed_prefixes WHERE storefront = ?", (storefront,)
        )
    }


def harvested_objects(conn: sqlite3.Connection, storefront: str) -> Counter[str]:
    """Token frequency across every suggestion collected so far."""
    counter: Counter[str] = Counter()
    rows = conn.execute(
        "SELECT term FROM suggestions WHERE storefront = ?", (storefront,)
    ).fetchall()
    for r in rows:
        for tok in r["term"].split():
            tok = singularize(tok)
            if len(tok) < 3 or tok.isdigit() or tok in STOPWORDS:
                continue
            counter[tok] += 1
    return counter


def discovered_verbs(conn: sqlite3.Connection, min_evidence: int = 2) -> list[str]:
    return [
        r["verb"]
        for r in conn.execute(
            "SELECT verb FROM discovered_verbs WHERE evidence >= ? AND promoted = 0 "
            "ORDER BY evidence DESC",
            (min_evidence,),
        )
    ]


def stalest_prefixes(
    conn: sqlite3.Connection,
    storefront: str,
    limit: int,
    min_age_days: int | None = None,
) -> list[str]:
    """
    Oldest probes first — keeps the corpus fresh once the frontier is spent.

    `min_age_days` matters once collection runs several times a day: without
    it, a spent frontier means the same prefixes get re-probed every few
    hours, burning the rate limit on suggestions that barely change. Apple's
    hints move on a scale of weeks, not hours.
    """
    days = min_age_days if min_age_days is not None else REPROBE_AFTER_DAYS
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=days)
    ).isoformat()
    return [
        r["prefix"]
        for r in conn.execute(
            "SELECT prefix FROM probed_prefixes WHERE storefront = ? "
            "AND probed_at < ? ORDER BY probed_at ASC LIMIT ?",
            (storefront, cutoff, limit),
        )
    ]


def pick_roots(conn: sqlite3.Connection, storefront: str, n: int) -> tuple[list[str], str]:
    """
    Return (roots, reason). Reason is logged so a run explains its own choices.
    """
    done = probed(conn, storefront)

    fresh_seeds = [v for v in SEED_VERBS if v not in done]
    if fresh_seeds:
        picked = fresh_seeds[:n]
        if len(picked) == n:
            return picked, "новые глаголы из списка"
    else:
        picked = []

    remaining = n - len(picked)

    if remaining > 0:
        verbs = [v for v in discovered_verbs(conn) if v not in done]
        take = verbs[:remaining]
        picked += take
        remaining -= len(take)

    if remaining > 0:
        objects = harvested_objects(conn, storefront)
        frontier = next_probes(objects, done | set(picked), min_count=2, limit=remaining)
        picked += frontier
        remaining -= len(frontier)

    if remaining > 0:
        # Frontier exhausted. Re-probe the oldest prefixes: suggestions drift,
        # and a stale corpus is the one real fragility of this design.
        stale = [p for p in stalest_prefixes(conn, storefront, remaining * 3)
                 if p not in picked]
        picked += stale[:remaining]
        if not fresh_seeds:
            if not stale:
                # Everything is fresh and the frontier is empty. Doing nothing
                # is correct: re-probing unchanged prefixes would spend the
                # rate limit to learn nothing.
                return picked, (
                    f"фронтир исчерпан, все префиксы свежее "
                    f"{REPROBE_AFTER_DAYS} дн. — пропускаем прогон"
                )
            return picked, "фронтир исчерпан — обновляем самые старые префиксы"

    picked = [p for p in picked if p not in TOO_BROAD]
    return picked[:n], "смесь новых глаголов, найденных глаголов и объектов"
