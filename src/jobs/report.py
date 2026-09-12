"""
Report job. Renders the weekly report; `--prompt N` emits the cluster prompt
for candidate N, ready to paste into Claude.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from datetime import datetime, timezone

from .. import db
from ..report.prompts import build_cluster_prompt
from ..report.render import find_alerts, load_queue, render_text

log = logging.getLogger(__name__)


def history_weeks(conn) -> int:
    row = conn.execute("SELECT MIN(taken_at) AS first FROM serp_snapshots").fetchone()
    if not row or not row["first"]:
        return 0
    first = datetime.fromisoformat(row["first"])
    if first.tzinfo is None:
        first = first.replace(tzinfo=timezone.utc)
    return max((datetime.now(timezone.utc) - first).days // 7, 0)


def metadata_tokens(apps: list[dict]) -> list[tuple[str, int]]:
    """Token frequency across competitor titles — free demand signal."""
    from ..seeds.expand import STOPWORDS, singularize

    counter: Counter[str] = Counter()
    for a in apps:
        text = f"{a.get('title','')} {a.get('subtitle_proxy','')}".lower()
        for tok in text.replace(":", " ").replace("-", " ").split():
            tok = singularize(tok.strip(".,!()&"))
            if len(tok) > 2 and tok not in STOPWORDS:
                counter[tok] += 1
    return counter.most_common()


def suggestions_for(conn, term: str, storefront: str) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT term FROM suggestions "
        "WHERE storefront = ? AND term LIKE ? ORDER BY position LIMIT 30",
        (storefront, f"%{term.split()[0]}%"),
    ).fetchall()
    return [r["term"] for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description="ASO Scout report")
    ap.add_argument("--prompt", type=int, help="emit cluster prompt for candidate N")
    ap.add_argument("--json", action="store_true", help="machine-readable queue")
    ap.add_argument("--clusters", action="store_true",
                    help="show every cluster and its members, to check merging")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    db.init_db()

    with db.connect() as conn:
        queue = load_queue(conn)
        alerts = find_alerts(conn)
        weeks = history_weeks(conn)

        if args.json:
            print(json.dumps(
                [{"term": c.term, "storefront": c.storefront, "score": c.score,
                  "notes": c.notes, "factors": c.factors} for c in queue],
                ensure_ascii=False, indent=2))
            return

        if args.clusters:
            # Over-merging is the dangerous failure: a good niche can hide
            # inside another cluster and never reach the report. This view
            # exists so merging can be checked by eye rather than trusted.
            from ..report.render import group_into_clusters  # noqa: PLC0415

            groups = group_into_clusters(queue)
            print(f"{len(groups)} ниш из {len(queue)} запросов\n")
            for i, cl in enumerate(groups, start=1):
                lead = cl.head.leader or {}
                print(f'{i:2}. {cl.head.term}  ·  score {cl.head.score}  ·  '
                      f'{cl.size} запр.')
                print(f'    топ-1: {lead.get("title", "?")[:44]}')
                for v in cl.variants:
                    vl = (v.leader or {}).get("title", "?")[:34]
                    mark = "" if vl == lead.get("title", "?")[:34] else f"   <- топ-1: {vl}"
                    print(f'      · {v.term:<38} {v.score:>5}{mark}')
                print()
            print("Разные топ-1 внутри кластера помечены — если их много,")
            print("склейка слишком агрессивна, поднимите cluster_overlap.")
            return

        if args.prompt is not None:
            if not 1 <= args.prompt <= len(queue):
                print(f"нет кандидата {args.prompt} (в очереди {len(queue)})")
                return
            c = queue[args.prompt - 1]
            print(build_cluster_prompt(
                term=c.term,
                storefront=c.storefront,
                apps=c.apps,
                suggestions=suggestions_for(conn, c.term, c.storefront),
                metadata_tokens=metadata_tokens(c.apps),
            ))
            return

        print(render_text(queue, alerts, weeks))


if __name__ == "__main__":
    main()
