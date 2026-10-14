"""
Pull weekly search-term popularity from Apple and store it.

    python -m src.jobs.asa_pull                  # last full week, priority genres
    python -m src.jobs.asa_pull --weeks 8        # backfill eight weeks
    python -m src.jobs.asa_pull --all-genres
    python -m src.jobs.asa_pull --stats          # what is stored

Why snapshots rather than a single current pull: a term's popularity on its
own says how big it is, not whether it is growing. Holding several weeks lets
`rankInGenre` be differenced, and a term climbing from 300th to 80th is the
only "becoming popular" signal this project has ever had access to.

Why the pull goes deep: the head of every genre is brands. The first live
check returned chatgpt, google, gmail, chrome, gemini, duckduckgo and outlook
in the top eight of Productivity. Useful categories start well below that, so
stopping at the first page would collect almost nothing worth having.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import date, timedelta

from .. import db
from ..asa.auth import AsaAuth, Credentials
from ..asa.popularity import GENRES, PRIORITY_GENRES, PopularityClient, PopularityRow

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS asa_popularity (
    term            TEXT NOT NULL,
    country         TEXT NOT NULL,
    genre           TEXT NOT NULL,
    period          TEXT NOT NULL,     -- week start or month
    granularity     TEXT NOT NULL,
    rank_in_genre   INTEGER,
    pop_in_genre    INTEGER,
    pop_1to100      INTEGER,
    pop_1to5        INTEGER,
    pulled_at       TEXT NOT NULL,
    PRIMARY KEY (term, country, genre, period, granularity)
);
CREATE INDEX IF NOT EXISTS idx_asa_term ON asa_popularity(term, country);
CREATE INDEX IF NOT EXISTS idx_asa_period ON asa_popularity(period, genre, country);
CREATE INDEX IF NOT EXISTS idx_asa_pop ON asa_popularity(pop_1to100 DESC);
"""


def ensure_schema(conn) -> None:
    conn.executescript(SCHEMA)


def weeks_back(n: int) -> list[tuple[str, str]]:
    """The last n completed Sunday-Saturday weeks, oldest first."""
    today = date.today()
    days_since_sunday = (today.weekday() + 1) % 7
    this_sunday = today - timedelta(days=days_since_sunday)
    out = []
    for i in range(n, 0, -1):
        start = this_sunday - timedelta(days=7 * i)
        out.append((start.isoformat(), (start + timedelta(days=6)).isoformat()))
    return out


def store(conn, rows: list[PopularityRow], granularity: str) -> int:
    ensure_schema(conn)
    now = db.now_iso()
    n = 0
    for r in rows:
        conn.execute(
            "INSERT INTO asa_popularity (term, country, genre, period, granularity, "
            "rank_in_genre, pop_in_genre, pop_1to100, pop_1to5, pulled_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(term, country, genre, period, granularity) DO UPDATE SET "
            "rank_in_genre=excluded.rank_in_genre, pop_in_genre=excluded.pop_in_genre, "
            "pop_1to100=excluded.pop_1to100, pop_1to5=excluded.pop_1to5, "
            "pulled_at=excluded.pulled_at",
            (r.search_term, r.country, r.genre, r.period, granularity,
             r.rank_in_genre, r.popularity_in_genre, r.popularity_1to100,
             r.popularity_1to5, now),
        )
        n += 1
    conn.commit()
    return n


def adopt_terms(conn, min_popularity: int, storefronts: list[str]) -> int:
    """
    Promote stored terms into the keyword queue.

    They arrive with a measured popularity rather than a guess, which is the
    whole point — but they still face the same gates as anything else. A
    popular term with an unbeatable top-3 is still unbeatable.
    """
    rows = conn.execute(
        """
        -- MAX, not the latest: a term that cleared the bar a few weeks ago
        -- and has since slipped is still a real search term worth looking at,
        -- and the slide is itself something the report can show.
        SELECT term, country, MAX(pop_1to100) AS pop
        FROM asa_popularity
        WHERE pop_1to100 >= ? AND country IN (%s)
        GROUP BY term, country
        """ % ",".join("?" * len(storefronts)),
        [min_popularity, *storefronts],
    ).fetchall()

    added = 0
    for r in rows:
        existing = conn.execute(
            "SELECT 1 FROM keywords WHERE term = ? AND storefront = ?",
            (r["term"], r["country"]),
        ).fetchone()
        if existing:
            continue
        db.upsert_keyword(conn, r["term"], r["country"], "asa")
        added += 1
    conn.commit()
    return added


def main() -> int:
    ap = argparse.ArgumentParser(description="pull Apple Ads popularity")
    ap.add_argument("--weeks", type=int, default=1,
                    help="how many completed weeks to pull, newest last")
    ap.add_argument("--all-genres", action="store_true")
    ap.add_argument("--countries", default="US,GB,CA,AU")
    ap.add_argument("--depth", type=int, default=500,
                    help="rows per genre; the head is all brands, so go deep")
    ap.add_argument("--adopt", type=int, default=0,
                    help="also queue terms with popularity >= this (0 = skip)")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    db.init_db()

    if args.stats:
        with db.connect() as conn:
            ensure_schema(conn)
            total = conn.execute("SELECT COUNT(*) c FROM asa_popularity").fetchone()["c"]
            if not total:
                print("данных ещё нет — запустите без --stats")
                return 0
            terms = conn.execute(
                "SELECT COUNT(DISTINCT term) c FROM asa_popularity").fetchone()["c"]
            periods = [r["period"] for r in conn.execute(
                "SELECT DISTINCT period FROM asa_popularity ORDER BY period")]
            print(f"строк: {total}, уникальных запросов: {terms}")
            print(f"недель: {len(periods)}  ({periods[0]} … {periods[-1]})")
            print()
            for r in conn.execute(
                "SELECT genre, country, COUNT(*) n FROM asa_popularity "
                "GROUP BY genre, country ORDER BY n DESC LIMIT 12"
            ):
                print(f"  {r['genre']:<26} {r['country'].upper()}  {r['n']}")
            if len(periods) < 2:
                print("\nДля трендов нужно минимум две недели: --weeks 4")
        return 0

    if Credentials.missing():
        print(f"не заданы: {', '.join(Credentials.missing())}")
        return 1

    countries = [c.strip().upper() for c in args.countries.split(",") if c.strip()]
    genres = GENRES if args.all_genres else PRIORITY_GENRES
    client = PopularityClient(AsaAuth())
    stats: Counter[str] = Counter()

    log.info("%d genres x %s, %d week(s), depth %d",
             len(genres), ",".join(countries), args.weeks, args.depth)

    with db.connect() as conn:
        for start, end in weeks_back(args.weeks):
            for genre in genres:
                rows = list(client.query(
                    countries, genre, start, end,
                    granularity="WEEKLY_SUN_SAT",
                    page_size=100, max_rows=args.depth,
                ))
                saved = store(conn, rows, "WEEKLY_SUN_SAT")
                stats["rows"] += saved
                log.info("  %s %-26s %4d rows", start, genre, saved)
                if not rows:
                    stats["empty"] += 1

        if args.adopt:
            added = adopt_terms(conn, args.adopt,
                                [c.lower() for c in countries])
            stats["queued"] = added
            log.info("queued %d new terms with popularity >= %d",
                     added, args.adopt)

    log.info("http: %s", client.stats)
    print(json.dumps(dict(stats), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
