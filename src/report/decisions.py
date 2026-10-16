"""
What you have already looked at.

The `decisions` table has been in the schema since the first version and was
never written to. That omission is why "live 360" came back six times in a
week: the queue is cumulative by design, nothing ever left it, and every
digest started from a blank slate.

A decision is per term, not per storefront. Dismissing "live 360" once should
silence it everywhere — a brand is a brand in all four English stores, and
having to dismiss the same thing four times is how a review habit dies.

Verdicts:
  skip   — not interesting, never show again
  astro  — taken into Astro for real validation; out of the idea list because
           it has moved to the next stage, not because it was rejected
  later  — interesting but not now; comes back after `LATER_DAYS`
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)

VERDICTS = {"skip", "astro", "later", "go", "no_go"}

# "later" is a snooze, not a dismissal. Two weeks is long enough that the
# queue has moved on, short enough that a good idea is not lost.
LATER_DAYS = 14


def record(
    conn: sqlite3.Connection,
    term: str,
    storefront: str,
    verdict: str,
    reason: str = "",
) -> bool:
    if verdict not in VERDICTS:
        return False
    conn.execute(
        "INSERT INTO decisions (term, storefront, decided_at, verdict, reason) "
        "VALUES (?, ?, ?, ?, ?)",
        (term, storefront, datetime.now(timezone.utc).isoformat(), verdict, reason),
    )
    conn.commit()
    return True


def hidden_terms(conn: sqlite3.Connection) -> dict[str, str]:
    """
    Terms that should not appear in the idea list, mapped to why.

    Keyed by term alone: a decision in one storefront hides the term in all of
    them. "later" expires; everything else is permanent until cleared.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=LATER_DAYS)).isoformat()
    out: dict[str, str] = {}
    try:
        rows = conn.execute(
            "SELECT term, verdict, decided_at FROM decisions ORDER BY decided_at"
        ).fetchall()
    except sqlite3.Error:
        return {}

    for r in rows:
        verdict = r["verdict"]
        if verdict == "later" and r["decided_at"] < cutoff:
            out.pop(r["term"], None)   # the snooze has run out
            continue
        out[r["term"]] = verdict
    return out


def clear(conn: sqlite3.Connection, term: str | None = None) -> int:
    """Undo decisions. Without a term, undo all of them."""
    if term:
        cur = conn.execute("DELETE FROM decisions WHERE term = ?", (term,))
    else:
        cur = conn.execute("DELETE FROM decisions")
    conn.commit()
    return cur.rowcount


def summary(conn: sqlite3.Connection) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for r in conn.execute(
            "SELECT verdict, COUNT(DISTINCT term) AS n FROM decisions GROUP BY verdict"
        ):
            out[r["verdict"]] = r["n"]
    except sqlite3.Error:
        pass
    return out
