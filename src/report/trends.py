"""
Trends from stored history.

One thing this module will not do: report a trend in *query popularity*.
Nothing in the free data measures how many people search a term, so an arrow
claiming demand is rising would be invented. That number comes from Astro or
not at all.

What is real, and what each arrow actually means:

  * **traffic** — how fast the niche leader gains reviews, this week against
    the week before. Reviews track installs loosely, so acceleration means the
    niche is growing. Biased: apps that nag for ratings look busier than they
    are, so compare within a category, never across.

  * **demand** — the term's position in Apple's autocomplete over time. Apple
    orders suggestions by frequency, so climbing is a genuine demand signal.
    Sparse by nature: a prefix is re-probed weekly at most.

  * **competition** — our own score moving, which is the SERP getting weaker
    or stronger underneath the keyword.

Every function returns None when the history is too thin. A dash in the report
is information; a fabricated arrow is not.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

UP, DOWN, FLAT, NONE = "↑", "↓", "→", "·"

# Below this relative change, movement is noise rather than a trend.
FLAT_BAND = 0.15


@dataclass
class Trend:
    arrow: str
    label: str | None = None      # short human text, e.g. "+38%"
    detail: str | None = None     # longer line for the meta row

    @property
    def known(self) -> bool:
        return self.arrow != NONE


NO_TREND = Trend(NONE)


def _direction(old: float, new: float, band: float = FLAT_BAND) -> str:
    if old <= 0:
        return FLAT if new <= 0 else UP
    change = (new - old) / abs(old)
    if change > band:
        return UP
    if change < -band:
        return DOWN
    return FLAT


def traffic_trend(
    conn: sqlite3.Connection,
    track_id: int,
    storefront: str,
    window: int = 7,
) -> Trend:
    """
    Leader's review velocity this window vs the one before.

    Needs roughly two windows of readings. The refresh step in the collector
    exists to make sure these readings keep arriving after a keyword's root
    has been probed and left behind.
    """
    rows = conn.execute(
        "SELECT day, rating_count FROM app_metrics_daily "
        "WHERE track_id = ? AND storefront = ? ORDER BY day DESC LIMIT 40",
        (track_id, storefront),
    ).fetchall()
    if len(rows) < 4:
        return NO_TREND

    points = [
        (datetime.fromisoformat(r["day"]).date(), r["rating_count"] or 0)
        for r in rows
    ]
    newest = points[0][0]
    mid = newest - timedelta(days=window)
    oldest_wanted = newest - timedelta(days=window * 2)

    recent = [p for p in points if p[0] > mid]
    prior = [p for p in points if oldest_wanted <= p[0] <= mid]
    if len(recent) < 2 or len(prior) < 2:
        return NO_TREND

    def velocity(seq: list[tuple[object, int]]) -> float | None:
        seq = sorted(seq, key=lambda p: p[0])
        span = (seq[-1][0] - seq[0][0]).days
        if span <= 0:
            return None
        return (seq[-1][1] - seq[0][1]) / span

    v_new, v_old = velocity(recent), velocity(prior)
    if v_new is None or v_old is None:
        return NO_TREND

    arrow = _direction(v_old, v_new)
    if v_old > 0:
        pct = (v_new - v_old) / v_old * 100
        label = f"{pct:+.0f}%"
    else:
        label = None
    return Trend(arrow, label, f"отзывы у лидера: {v_old:.0f}/дн → {v_new:.0f}/дн")


def demand_trend(
    conn: sqlite3.Connection, term: str, storefront: str
) -> Trend:
    """
    Movement in Apple's autocomplete ordering. Lower position is better.

    This is the closest honest thing to a demand trend, and it is sparse:
    prefixes are re-probed no more than weekly, so most terms will have one
    observation for a long while.
    """
    rows = conn.execute(
        "SELECT position, seen_at FROM suggestions "
        "WHERE term = ? AND storefront = ? ORDER BY seen_at DESC LIMIT 20",
        (term, storefront),
    ).fetchall()
    if len(rows) < 2:
        return NO_TREND

    newest, oldest = rows[0], rows[-1]
    if newest["seen_at"] == oldest["seen_at"]:
        return NO_TREND

    new_pos, old_pos = newest["position"], oldest["position"]
    if new_pos == old_pos:
        return Trend(FLAT, None, f"автокомплит: стабильно #{new_pos}")
    # Inverted: position 2 beats position 8.
    arrow = UP if new_pos < old_pos else DOWN
    return Trend(arrow, f"#{new_pos}", f"автокомплит: #{old_pos} → #{new_pos}")


def score_trend(conn: sqlite3.Connection, keyword_id: int) -> Trend:
    """Our own score moving — the SERP weakening or hardening."""
    rows = conn.execute(
        "SELECT score, scored_at FROM keyword_scores "
        "WHERE keyword_id = ? ORDER BY scored_at DESC LIMIT 30",
        (keyword_id,),
    ).fetchall()
    if len(rows) < 2:
        return NO_TREND

    new, old = rows[0]["score"], rows[-1]["score"]
    # Rescoring after a threshold change moves every score at once; that is a
    # change in us, not in the market. Only report a sustained-looking move.
    if abs(new - old) < 3.0:
        return Trend(FLAT)
    arrow = UP if new > old else DOWN
    return Trend(arrow, f"{new - old:+.0f}", f"score: {old:.0f} → {new:.0f}")


def keyword_id_for(
    conn: sqlite3.Connection, term: str, storefront: str
) -> int | None:
    row = conn.execute(
        "SELECT id FROM keywords WHERE term = ? AND storefront = ?",
        (term, storefront),
    ).fetchone()
    return int(row["id"]) if row else None


def history_days(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT MIN(day) AS first FROM app_metrics_daily"
    ).fetchone()
    if not row or not row["first"]:
        return 0
    first = datetime.fromisoformat(row["first"]).replace(tzinfo=timezone.utc)
    return max((datetime.now(timezone.utc) - first).days, 0)
