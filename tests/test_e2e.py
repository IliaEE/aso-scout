"""
End-to-end dry run against fixtures.

Exercises the full path: probe -> harvest -> dedup -> snapshot -> features ->
gates -> score -> DB -> report -> prompt. No network.
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import db
from src.config import Storefront, settings
from src.jobs.collect import run
from src.jobs.report import history_weeks, metadata_tokens, suggestions_for
from src.report.prompts import build_cluster_prompt, build_verdict_prompt
from src.report.render import find_alerts, load_queue, render_text
from tests import fixtures


async def main() -> int:
    tmp = tempfile.mkdtemp()
    settings.sqlite_path = str(Path(tmp) / "e2e.db")
    db.init_db()

    store = Storefront("us", "en", "United States", 143441)
    client = fixtures.build_client()

    print("=" * 62)
    print("COLLECTION PASS (fixtures, 2 roots)")
    print("=" * 62)
    stats = await run(client, store, roots=["scan", "noise"], deep=False)
    for k, v in sorted(stats.items()):
        print(f"  {k:<16} {v}")
    print(f"  {'http calls':<16} {len(client.calls)}")

    with db.connect() as conn:
        n_kw = conn.execute("SELECT COUNT(*) c FROM keywords").fetchone()["c"]
        n_sugg = conn.execute("SELECT COUNT(*) c FROM suggestions").fetchone()["c"]
        n_serp = conn.execute("SELECT COUNT(*) c FROM serp_snapshots").fetchone()["c"]
        n_verbs = conn.execute("SELECT COUNT(*) c FROM discovered_verbs").fetchone()["c"]
        print(f"\n  keywords={n_kw}  suggestions={n_sugg}  snapshots={n_serp} "
              f"discovered_verbs={n_verbs}")

        print("\n  status breakdown:")
        for r in conn.execute(
            "SELECT status, COUNT(*) c FROM keywords GROUP BY status ORDER BY c DESC"
        ):
            print(f"    {r['status']:<12} {r['c']}")

        queue = load_queue(conn)
        alerts = find_alerts(conn)
        weeks = history_weeks(conn)

        print("\n" + "=" * 62)
        print("WHAT YOU SEE ON MONDAY")
        print("=" * 62)
        print(render_text(queue, alerts, weeks))

        if queue:
            c = queue[0]
            print("\n" + "=" * 62)
            print(f'GENERATED CLUSTER PROMPT  ["{c.term}"]')
            print("=" * 62)
            prompt = build_cluster_prompt(
                term=c.term,
                storefront=c.storefront,
                apps=c.apps,
                suggestions=suggestions_for(conn, c.term, c.storefront),
                metadata_tokens=metadata_tokens(c.apps),
                pains=[("калибровка врёт", 41), ("подписка за экспорт", 33)],
            )
            print(prompt)

            print("=" * 62)
            print("GENERATED VERDICT PROMPT (after Astro data is loaded)")
            print("=" * 62)
            print(build_verdict_prompt(
                head=c.term,
                storefront=c.storefront,
                facts=[
                    {"term": "noise meter", "kw_type": "HEAD",
                     "popularity": 58, "difficulty": 34, "top1": "Decibel X"},
                    {"term": "decibel meter", "kw_type": "HEAD",
                     "popularity": 71, "difficulty": 52, "top1": "Decibel X"},
                    {"term": "spl meter", "kw_type": "TAIL",
                     "popularity": 19, "difficulty": 14, "top1": "Sonometer"},
                ],
                pains=[("калибровка врёт", 41)],
            ))

    return 0 if queue else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
