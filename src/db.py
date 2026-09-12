"""
Storage.

SQLite locally, Postgres on Railway (set DATABASE_URL). The schema is
deliberately snapshot-first: raw API responses go in untouched and features
are derived. We do not yet know which fields will matter, and history cannot
be re-fetched after the fact.

`keywords.parent_id` carries lineage — which root probe produced a term, and
later which cluster head a term belongs to. Without it, clusters fall apart
and calibration has nothing to join on.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from .config import settings

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS keywords (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    term            TEXT NOT NULL,
    storefront      TEXT NOT NULL,
    source          TEXT NOT NULL,          -- suggest | metadata | matrix | llm_cluster
    parent_id       INTEGER,                -- lineage: probe root or cluster head
    canonical_term  TEXT,                   -- dedup target, NULL if canonical
    suggest_pos     INTEGER,                -- best autocomplete position seen
    cluster_head    TEXT,
    kw_type         TEXT,                   -- HEAD | BODY | TAIL | BRAND | ADJACENT
    status          TEXT NOT NULL DEFAULT 'new',
                    -- new|queued|watchlist|rejected|in_astro|go|no_go|narrow
    first_seen      TEXT NOT NULL,
    last_seen       TEXT NOT NULL,
    UNIQUE(term, storefront)
);
CREATE INDEX IF NOT EXISTS idx_kw_status ON keywords(status, storefront);
CREATE INDEX IF NOT EXISTS idx_kw_term   ON keywords(term);

-- Raw suggestion corpus. Survives the endpoint disappearing, which is the
-- single biggest fragility in this design.
CREATE TABLE IF NOT EXISTS suggestions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    prefix      TEXT NOT NULL,
    term        TEXT NOT NULL,
    position    INTEGER NOT NULL,
    storefront  TEXT NOT NULL,
    seen_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sugg_term   ON suggestions(term);
CREATE INDEX IF NOT EXISTS idx_sugg_prefix ON suggestions(prefix, storefront);

CREATE TABLE IF NOT EXISTS probed_prefixes (
    prefix      TEXT NOT NULL,
    storefront  TEXT NOT NULL,
    probed_at   TEXT NOT NULL,
    yield_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (prefix, storefront)
);

CREATE TABLE IF NOT EXISTS discovered_verbs (
    verb        TEXT PRIMARY KEY,
    found_at    TEXT NOT NULL,
    evidence    INTEGER NOT NULL DEFAULT 1,
    promoted    INTEGER NOT NULL DEFAULT 0
);

-- Raw SERP snapshots. Never edited, only appended.
CREATE TABLE IF NOT EXISTS serp_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword_id  INTEGER NOT NULL,
    term        TEXT NOT NULL,
    storefront  TEXT NOT NULL,
    taken_at    TEXT NOT NULL,
    raw         TEXT NOT NULL,              -- JSON: list of normalized apps
    FOREIGN KEY (keyword_id) REFERENCES keywords(id)
);
CREATE INDEX IF NOT EXISTS idx_serp_kw ON serp_snapshots(keyword_id, taken_at);

CREATE TABLE IF NOT EXISTS apps (
    track_id      INTEGER NOT NULL,
    storefront    TEXT NOT NULL,
    title         TEXT,
    primary_genre TEXT,
    last_raw      TEXT,
    updated_at    TEXT,
    PRIMARY KEY (track_id, storefront)
);

-- Daily rating counts. The delta is our only free install proxy, and it is
-- systematically biased toward apps that nag for reviews.
CREATE TABLE IF NOT EXISTS app_metrics_daily (
    track_id     INTEGER NOT NULL,
    storefront   TEXT NOT NULL,
    day          TEXT NOT NULL,
    rating       REAL,
    rating_count INTEGER,
    PRIMARY KEY (track_id, storefront, day)
);

CREATE TABLE IF NOT EXISTS keyword_features (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword_id  INTEGER NOT NULL,
    computed_at TEXT NOT NULL,
    payload     TEXT NOT NULL,
    FOREIGN KEY (keyword_id) REFERENCES keywords(id)
);
CREATE INDEX IF NOT EXISTS idx_feat_kw ON keyword_features(keyword_id, computed_at);

CREATE TABLE IF NOT EXISTS keyword_scores (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword_id   INTEGER NOT NULL,
    scored_at    TEXT NOT NULL,
    score        REAL NOT NULL,
    verdict      TEXT NOT NULL,
    gates_failed TEXT,
    factors      TEXT,
    notes        TEXT,
    FOREIGN KEY (keyword_id) REFERENCES keywords(id)
);
CREATE INDEX IF NOT EXISTS idx_score_kw ON keyword_scores(keyword_id, scored_at);

-- Ground truth from Astro, loaded manually. This is what calibration joins
-- the server's guesses against.
CREATE TABLE IF NOT EXISTS astro_facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    term        TEXT NOT NULL,
    storefront  TEXT NOT NULL,
    loaded_at   TEXT NOT NULL,
    popularity  REAL,
    difficulty  REAL,
    rank        INTEGER,
    raw         TEXT
);
CREATE INDEX IF NOT EXISTS idx_astro_term ON astro_facts(term, storefront);

CREATE TABLE IF NOT EXISTS decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    term        TEXT NOT NULL,
    storefront  TEXT NOT NULL,
    decided_at  TEXT NOT NULL,
    verdict     TEXT NOT NULL,   -- go | no_go | hold | narrow | other_store
    reason      TEXT
);
CREATE INDEX IF NOT EXISTS idx_dec_term ON decisions(term, storefront);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


@contextmanager
def connect(path: str | None = None) -> Iterator[sqlite3.Connection]:
    """
    SQLite connection with sane defaults.

    WAL is on so the report service can read while the worker writes. On
    Railway, swap this for psycopg — the SQL above is intentionally portable
    apart from AUTOINCREMENT.
    """
    target = path or settings.sqlite_path
    conn = sqlite3.connect(target, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(path: str | None = None) -> None:
    with connect(path) as conn:
        conn.executescript(SCHEMA)
    log.info("schema ready at %s", path or settings.sqlite_path)


# --- keyword helpers -----------------------------------------------------

def upsert_keyword(
    conn: sqlite3.Connection,
    term: str,
    storefront: str,
    source: str,
    parent_id: int | None = None,
    suggest_pos: int | None = None,
    kw_type: str | None = None,
) -> int:
    """Insert or refresh a keyword, returning its id. Keeps best suggest_pos."""
    ts = now_iso()
    cur = conn.execute(
        "SELECT id, suggest_pos FROM keywords WHERE term = ? AND storefront = ?",
        (term, storefront),
    )
    row = cur.fetchone()
    if row:
        best = row["suggest_pos"]
        if suggest_pos is not None and (best is None or suggest_pos < best):
            best = suggest_pos
        conn.execute(
            "UPDATE keywords SET last_seen = ?, suggest_pos = ? WHERE id = ?",
            (ts, best, row["id"]),
        )
        return int(row["id"])

    cur = conn.execute(
        """INSERT INTO keywords
           (term, storefront, source, parent_id, suggest_pos, kw_type,
            status, first_seen, last_seen)
           VALUES (?, ?, ?, ?, ?, ?, 'new', ?, ?)""",
        (term, storefront, source, parent_id, suggest_pos, kw_type, ts, ts),
    )
    return int(cur.lastrowid)


def store_suggestions(
    conn: sqlite3.Connection, prefix: str, storefront: str, found: dict[str, int]
) -> None:
    ts = now_iso()
    conn.executemany(
        "INSERT INTO suggestions (prefix, term, position, storefront, seen_at) "
        "VALUES (?, ?, ?, ?, ?)",
        [(prefix, term, pos, storefront, ts) for term, pos in found.items()],
    )
    conn.execute(
        "INSERT INTO probed_prefixes (prefix, storefront, probed_at, yield_count) "
        "VALUES (?, ?, ?, ?) ON CONFLICT(prefix, storefront) DO UPDATE SET "
        "probed_at = excluded.probed_at, yield_count = excluded.yield_count",
        (prefix, storefront, ts, len(found)),
    )


def store_serp(
    conn: sqlite3.Connection,
    keyword_id: int,
    term: str,
    storefront: str,
    apps: list[dict[str, Any]],
) -> None:
    ts = now_iso()
    conn.execute(
        "INSERT INTO serp_snapshots (keyword_id, term, storefront, taken_at, raw) "
        "VALUES (?, ?, ?, ?, ?)",
        (keyword_id, term, storefront, ts, json.dumps(apps, ensure_ascii=False)),
    )
    day = today_iso()
    for app in apps:
        if not app.get("track_id"):
            continue
        conn.execute(
            "INSERT INTO apps (track_id, storefront, title, primary_genre, last_raw, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(track_id, storefront) DO UPDATE SET "
            "title = excluded.title, last_raw = excluded.last_raw, updated_at = excluded.updated_at",
            (app["track_id"], storefront, app.get("title"), app.get("primary_genre"),
             json.dumps(app, ensure_ascii=False), ts),
        )
        conn.execute(
            "INSERT INTO app_metrics_daily (track_id, storefront, day, rating, rating_count) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(track_id, storefront, day) DO UPDATE SET "
            "rating = excluded.rating, rating_count = excluded.rating_count",
            (app["track_id"], storefront, day, app.get("rating"), app.get("rating_count")),
        )


def store_score(
    conn: sqlite3.Connection, keyword_id: int, features: dict[str, Any], scored: Any
) -> None:
    ts = now_iso()
    conn.execute(
        "INSERT INTO keyword_features (keyword_id, computed_at, payload) VALUES (?, ?, ?)",
        (keyword_id, ts, json.dumps(features, ensure_ascii=False)),
    )
    conn.execute(
        """INSERT INTO keyword_scores
           (keyword_id, scored_at, score, verdict, gates_failed, factors, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (keyword_id, ts, scored.score, scored.verdict.value,
         json.dumps(scored.gates_failed), json.dumps(scored.factors),
         json.dumps(scored.notes, ensure_ascii=False)),
    )
    conn.execute(
        "UPDATE keywords SET status = ? WHERE id = ?", (scored.verdict.value, keyword_id)
    )


def review_delta(
    conn: sqlite3.Connection, track_id: int, storefront: str, days: int = 7
) -> int | None:
    """
    Weekly review-count delta for an app. Returns None without two data
    points far enough apart — better no number than a fabricated one.
    """
    rows = conn.execute(
        "SELECT day, rating_count FROM app_metrics_daily "
        "WHERE track_id = ? AND storefront = ? ORDER BY day DESC LIMIT 30",
        (track_id, storefront),
    ).fetchall()
    if len(rows) < 2:
        return None
    newest, oldest = rows[0], rows[-1]
    span = (datetime.fromisoformat(newest["day"]) - datetime.fromisoformat(oldest["day"])).days
    if span < 2:
        return None
    delta = (newest["rating_count"] or 0) - (oldest["rating_count"] or 0)
    return int(delta / span * days)
