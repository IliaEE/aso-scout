"""
Astro -> ASO Scout bridge. Run locally on the Mac, whenever convenient.

    python -m src.astro.dump --out astro_facts.json
    python -m src.astro.dump --load astro_facts.json   # into the scout DB

Deliberately NOT a daemon. Demand and difficulty move slowly, and at 20-50
keywords a week the manual step costs minutes. A launchd agent would only be
worth it once a live app needs daily rank tracking.

Two things this does not do, on purpose:

1. It never writes to Astro's database. Core Data keeps its own primary-key
   bookkeeping in Z_PRIMARYKEY and Z_METADATA; inserting rows without
   updating those consistently corrupts the app. Keywords go in through
   Astro's UI, by hand.

2. It does not assume column names. Core Data mangles them (ZPOPULARITY,
   ZDIFFICULTY, ZKEYWORDNAME...) and they differ between app versions, so the
   schema is introspected at runtime and columns matched by pattern. Run
   --inspect first to see what your build actually has.

Read safety: Astro may hold the DB in WAL mode. Reading a hot database can
return a stale snapshot, so the file is copied together with its -wal and -shm
sidecars before reading. Closing Astro first is still the safest option.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB = (
    Path.home()
    / "Library/Containers/matteospada.it.ASO/Data/Library/Application Support/Astro/Model.sqlite"
)

# Core Data stores timestamps as seconds since 2001-01-01 UTC, not the Unix
# epoch. Forget this and every date lands in the 1970s.
APPLE_EPOCH_OFFSET = 978_307_200


def apple_ts_to_iso(value: Any) -> str | None:
    if value is None:
        return None
    try:
        ts = float(value) + APPLE_EPOCH_OFFSET
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def safe_copy(src: Path) -> Path:
    """Copy the DB plus WAL sidecars to temp so we read a consistent snapshot."""
    if not src.exists():
        raise SystemExit(
            f"Astro database not found at:\n  {src}\n\n"
            "Check that Astro is installed and has run at least once. "
            "If the path differs in your build, pass --db."
        )
    tmp = Path(tempfile.mkdtemp()) / src.name
    shutil.copy2(src, tmp)
    for suffix in ("-wal", "-shm"):
        side = src.with_name(src.name + suffix)
        if side.exists():
            shutil.copy2(side, tmp.with_name(tmp.name + suffix))
    return tmp


def columns_of(conn: sqlite3.Connection, table: str) -> list[str]:
    try:
        return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    except sqlite3.Error:
        return []


def find_column(cols: list[str], *patterns: str) -> str | None:
    """First column whose name contains any pattern, case-insensitive."""
    for pat in patterns:
        for col in cols:
            if pat.lower() in col.lower():
                return col
    return None


def inspect(db_path: Path) -> None:
    """Print the real schema. Run this first on a new Astro version."""
    conn = sqlite3.connect(f"file:{safe_copy(db_path)}?mode=ro", uri=True)
    try:
        tables = [
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        print(f"Tables ({len(tables)}):\n")
        for t in tables:
            cols = columns_of(conn, t)
            try:
                n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            except sqlite3.Error:
                n = "?"
            marker = " <--" if t.upper() in {"ZKEYWORD", "ZAPPLICATION", "ZDATAPOINT"} else ""
            print(f"  {t:<28} rows={n}{marker}")
            if marker:
                for c in cols:
                    print(f"      {c}")
    finally:
        conn.close()


def extract(db_path: Path, storefront: str = "us") -> list[dict[str, Any]]:
    """Pull keyword facts, mapping columns by pattern."""
    conn = sqlite3.connect(f"file:{safe_copy(db_path)}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cols = columns_of(conn, "ZKEYWORD")
        if not cols:
            raise SystemExit(
                "ZKEYWORD table not found. Run with --inspect and send me the "
                "output — column names differ between Astro versions."
            )

        col_term = find_column(cols, "keyword", "term", "name", "text")
        col_pop = find_column(cols, "popularity", "volume", "traffic")
        col_diff = find_column(cols, "difficulty", "competition")
        col_rank = find_column(cols, "currentrank", "rank", "position")
        col_store = find_column(cols, "store", "country")

        if not col_term:
            raise SystemExit(
                f"Could not find a keyword-name column in ZKEYWORD.\n"
                f"Columns present: {cols}\nRun --inspect and share the output."
            )

        select = [f"{col_term} AS term"]
        for alias, col in (
            ("popularity", col_pop), ("difficulty", col_diff),
            ("rank", col_rank), ("store", col_store),
        ):
            select.append(f"{col} AS {alias}" if col else f"NULL AS {alias}")

        rows = conn.execute(f"SELECT {', '.join(select)} FROM ZKEYWORD").fetchall()

        out = []
        for r in rows:
            term = (r["term"] or "").strip().lower()
            if not term:
                continue
            out.append({
                "term": term,
                "storefront": (r["store"] or storefront).strip().lower(),
                "popularity": r["popularity"],
                "difficulty": r["difficulty"],
                "rank": r["rank"],
            })

        missing = [n for n, c in
                   (("popularity", col_pop), ("difficulty", col_diff)) if not c]
        if missing:
            print(f"WARNING: no column matched {missing} — those fields will be "
                  f"null. Run --inspect and send the schema.", file=sys.stderr)

        return out
    finally:
        conn.close()


def load_into_scout(path: Path) -> int:
    """Load a dump into the scout DB as ground truth."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src import db  # noqa: PLC0415

    facts = json.loads(path.read_text())
    db.init_db()
    with db.connect() as conn:
        for f in facts:
            conn.execute(
                "INSERT INTO astro_facts (term, storefront, loaded_at, popularity, "
                "difficulty, rank, raw) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f["term"], f.get("storefront", "us"), db.now_iso(),
                 f.get("popularity"), f.get("difficulty"), f.get("rank"),
                 json.dumps(f, ensure_ascii=False)),
            )
            conn.execute(
                "UPDATE keywords SET status = 'in_astro' WHERE term = ? "
                "AND storefront = ? AND status IN ('queued', 'new')",
                (f["term"], f.get("storefront", "us")),
            )
    return len(facts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Astro bridge")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--inspect", action="store_true", help="print Astro's schema")
    ap.add_argument("--out", type=Path, help="write facts to JSON")
    ap.add_argument("--load", type=Path, help="load a JSON dump into the scout DB")
    ap.add_argument("--store", default="us")
    args = ap.parse_args()

    if args.inspect:
        inspect(args.db)
        return

    if args.load:
        print(f"loaded {load_into_scout(args.load)} facts")
        return

    facts = extract(args.db, args.store)
    payload = json.dumps(facts, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(payload)
        print(f"wrote {len(facts)} keyword facts to {args.out}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
