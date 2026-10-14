"""
Reading movement out of the popularity snapshots.

A popularity number says how big a term is. It says nothing about whether the
term is going anywhere, and "going somewhere" is the only signal that lets you
reach a niche before it is crowded. That needs history, which is why a year of
weekly rows was worth pulling.

Four things this computes, in rising order of how strongly they should pull
your attention:

  movement        — rank change week over week. Noisy on its own: one TikTok
                    video moves a term 200 places.
  new entrants    — a term that was not in the top 500 at all and now is. The
                    purest form of an emerging need.
  sustained       — climbed in most of the last N weeks. One week is noise;
                    a month of climbing is a fact.
  seasonality     — whether this same climb happened a year ago. Without this
                    check, every August the tool would scream about
                    back-to-school terms as if they were discoveries.

The seasonality check is the reason the backfill went back twelve months
rather than eight weeks.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta

log = logging.getLogger(__name__)


@dataclass
class Movement:
    term: str
    country: str
    genre: str
    rank_now: int | None
    rank_before: int | None
    popularity: int | None
    delta: int                  # positive = climbing
    weeks_rising: int = 0
    weeks_tracked: int = 0
    is_new: bool = False
    seasonal: bool | None = None   # None = cannot tell, no data a year back

    @property
    def climbing(self) -> bool:
        return self.delta > 0

    @property
    def strength(self) -> str:
        if self.is_new:
            return "новичок"
        if self.weeks_tracked >= 4 and self.weeks_rising >= self.weeks_tracked - 1:
            return "устойчивый рост"
        if self.delta >= 50:
            return "рост"
        if self.delta <= -50:
            return "падение"
        return "стабильно"


def periods(conn: sqlite3.Connection, granularity: str = "WEEKLY_SUN_SAT") -> list[str]:
    return [
        r["period"] for r in conn.execute(
            "SELECT DISTINCT period FROM asa_popularity WHERE granularity = ? "
            "ORDER BY period",
            (granularity,),
        )
    ]


def movement(
    conn: sqlite3.Connection,
    countries: list[str],
    lookback_weeks: int = 1,
    min_delta: int = 30,
    granularity: str = "WEEKLY_SUN_SAT",
) -> list[Movement]:
    """
    Rank change between the newest period and `lookback_weeks` before it.

    Terms present in both are compared; terms present only in the newest are
    new entrants, which `new_entrants` reports separately with more care.
    """
    all_periods = periods(conn, granularity)
    if len(all_periods) < lookback_weeks + 1:
        return []

    now = all_periods[-1]
    before = all_periods[-(lookback_weeks + 1)]
    placeholders = ",".join("?" * len(countries))

    rows = conn.execute(
        f"""
        SELECT a.term, a.country, a.genre,
               b.rank_in_genre AS rank_now,
               a.rank_in_genre AS rank_before,
               b.pop_1to100    AS popularity
        FROM asa_popularity a
        JOIN asa_popularity b
          ON a.term = b.term AND a.country = b.country AND a.genre = b.genre
         AND b.granularity = a.granularity
        WHERE a.granularity = ? AND a.period = ? AND b.period = ?
          AND a.country IN ({placeholders})
          AND a.rank_in_genre IS NOT NULL AND b.rank_in_genre IS NOT NULL
        """,
        [granularity, before, now, *[c.lower() for c in countries]],
    ).fetchall()

    out = []
    for r in rows:
        delta = r["rank_before"] - r["rank_now"]   # climbing = positive
        if abs(delta) < min_delta:
            continue
        out.append(Movement(
            term=r["term"], country=r["country"], genre=r["genre"],
            rank_now=r["rank_now"], rank_before=r["rank_before"],
            popularity=r["popularity"], delta=delta,
        ))
    out.sort(key=lambda m: -m.delta)
    return out


def new_entrants(
    conn: sqlite3.Connection,
    countries: list[str],
    weeks_absent: int = 3,
    granularity: str = "WEEKLY_SUN_SAT",
) -> list[Movement]:
    """
    Terms in the newest period that were absent for the preceding weeks.

    This is the strongest early signal available: not a term that grew, but a
    need that did not visibly exist and now does. Requiring several weeks of
    absence filters out terms that merely oscillate around the 500 cutoff.
    """
    all_periods = periods(conn, granularity)
    if len(all_periods) < weeks_absent + 1:
        return []

    now = all_periods[-1]
    prior = all_periods[-(weeks_absent + 1):-1]
    ph_c = ",".join("?" * len(countries))
    ph_p = ",".join("?" * len(prior))

    rows = conn.execute(
        f"""
        SELECT term, country, genre, rank_in_genre, pop_1to100
        FROM asa_popularity
        WHERE granularity = ? AND period = ? AND country IN ({ph_c})
          AND term NOT IN (
              SELECT term FROM asa_popularity
              WHERE granularity = ? AND period IN ({ph_p})
          )
        ORDER BY rank_in_genre
        """,
        [granularity, now, *[c.lower() for c in countries], granularity, *prior],
    ).fetchall()

    return [
        Movement(
            term=r["term"], country=r["country"], genre=r["genre"],
            rank_now=r["rank_in_genre"], rank_before=None,
            popularity=r["pop_1to100"], delta=0, is_new=True,
        )
        for r in rows
    ]


def sustained(
    conn: sqlite3.Connection,
    countries: list[str],
    window: int = 4,
    min_total_gain: int = 60,
    granularity: str = "WEEKLY_SUN_SAT",
) -> list[Movement]:
    """
    Terms that climbed in most weeks of the window, not just the last one.

    One week of movement is noise. A month of it is a fact, and this is what
    feeds the monthly "look at this, no excuses" report.
    """
    all_periods = periods(conn, granularity)
    if len(all_periods) < window + 1:
        return []

    span = all_periods[-(window + 1):]
    ph_c = ",".join("?" * len(countries))
    ph_p = ",".join("?" * len(span))

    rows = conn.execute(
        f"""
        SELECT term, country, genre, period, rank_in_genre, pop_1to100
        FROM asa_popularity
        WHERE granularity = ? AND period IN ({ph_p}) AND country IN ({ph_c})
          AND rank_in_genre IS NOT NULL
        ORDER BY term, country, genre, period
        """,
        [granularity, *span, *[c.lower() for c in countries]],
    ).fetchall()

    series: dict[tuple[str, str, str], list[tuple[str, int, int]]] = {}
    for r in rows:
        key = (r["term"], r["country"], r["genre"])
        series.setdefault(key, []).append(
            (r["period"], r["rank_in_genre"], r["pop_1to100"] or 0)
        )

    out = []
    for (term, country, genre), points in series.items():
        # Only terms present across the whole window: a gap means we cannot
        # say it climbed steadily.
        if len(points) < len(span):
            continue
        ranks = [p[1] for p in points]
        rising = sum(1 for a, b in zip(ranks, ranks[1:]) if b < a)
        total = ranks[0] - ranks[-1]
        if total < min_total_gain or rising < len(ranks) - 2:
            continue
        out.append(Movement(
            term=term, country=country, genre=genre,
            rank_now=ranks[-1], rank_before=ranks[0],
            popularity=points[-1][2], delta=total,
            weeks_rising=rising, weeks_tracked=len(ranks) - 1,
        ))
    out.sort(key=lambda m: -m.delta)
    return out


def check_seasonality(
    conn: sqlite3.Connection,
    m: Movement,
    span_days: int = 28,
    granularity: str = "WEEKLY_SUN_SAT",
) -> bool | None:
    """
    Did this term move the same way at the same point last year?

    Compares the current move against the move over the same calendar span
    one year earlier. Same direction and comparable size means the calendar is
    doing the work, not a new need.

    Returns None when the needed weeks are not stored, which is the common
    case in the first year of collection: answering this needs history older
    than a year by the width of the comparison window, so with exactly twelve
    months almost everything is honestly unknown. It starts working as weeks
    accumulate — and "unknown" printed in the report is the correct thing to
    show until then, rather than a confident False.
    """
    all_periods = periods(conn, granularity)
    if len(all_periods) < 2 or m.delta == 0:
        return None
    try:
        now = date.fromisoformat(all_periods[-1])
    except ValueError:
        return None

    def nearest(target: date, tolerance: int = 14) -> str | None:
        best, best_gap = None, None
        for p in all_periods:
            try:
                gap = abs((date.fromisoformat(p) - target).days)
            except ValueError:
                continue
            if best_gap is None or gap < best_gap:
                best, best_gap = p, gap
        return best if best_gap is not None and best_gap <= tolerance else None

    end_last_year = nearest(now - timedelta(days=364))
    start_last_year = nearest(now - timedelta(days=364 + span_days))
    if not end_last_year or not start_last_year or end_last_year == start_last_year:
        return None

    rows = {
        r["period"]: r["rank_in_genre"]
        for r in conn.execute(
            "SELECT period, rank_in_genre FROM asa_popularity "
            "WHERE granularity = ? AND term = ? AND country = ? AND genre = ? "
            "AND period IN (?, ?)",
            (granularity, m.term, m.country, m.genre,
             start_last_year, end_last_year),
        )
    }
    a, b = rows.get(start_last_year), rows.get(end_last_year)
    if a is None or b is None:
        return None

    last_year_delta = a - b          # positive = climbed, same sign convention
    if last_year_delta == 0:
        return False
    same_direction = (last_year_delta > 0) == (m.delta > 0)
    comparable = abs(last_year_delta) >= max(abs(m.delta) * 0.5, 30)
    return same_direction and comparable


def annotate_seasonality(
    conn: sqlite3.Connection,
    items: list[Movement],
    granularity: str = "WEEKLY_SUN_SAT",
    span_days: int = 28,
) -> list[Movement]:
    """
    Mark each item seasonal / not / unknown.

    Keyword arguments on the call below are deliberate: an earlier version
    passed granularity positionally into the span_days slot, and every report
    died on `'int' + 'str'`. The unit tests missed it because they called
    check_seasonality directly and never went through this wrapper.
    """
    for m in items:
        m.seasonal = check_seasonality(
            conn, m, span_days=span_days, granularity=granularity
        )
    return items
