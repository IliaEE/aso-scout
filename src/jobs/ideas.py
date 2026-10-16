"""
The idea list: every surviving candidate, with where its demand came from.

    python -m src.jobs.ideas              # everything, best first
    python -m src.jobs.ideas --measured   # only terms Apple sized for us
    python -m src.jobs.ideas --limit 30

Three provenances, and the difference matters more than the ranking:

  измерен   the term is in Apple's top-500 for its genre, so there is a real
            popularity number behind it.
  хвост     the term was extracted from the descriptive half of a branded
            search term ("be my eyes – helping the blind" -> "helping the
            blind"). A developer wrote those words to match what people
            search, which is why half of them survive the gates against
            12% for popularity-picked terms — but nobody measured the tail
            itself, so its demand is unverified.
  автокомплит  Apple suggested it, which proves people type it but says
            nothing about how many.

A term that is both measured and a tail is the strongest thing here: a
category description with a number attached.
"""
from __future__ import annotations

import argparse
import logging
import sys

from .. import db
from ..report.decisions import hidden_terms
from ..report.render import Candidate, group_into_clusters, load_queue
from .asa_pull import ensure_schema

log = logging.getLogger(__name__)


def popularity_map(conn, countries: list[str]) -> dict[tuple[str, str], int]:
    """{(term, country): popularity} from the newest weekly snapshot."""
    ensure_schema(conn)
    row = conn.execute(
        "SELECT MAX(period) p FROM asa_popularity WHERE granularity='WEEKLY_SUN_SAT'"
    ).fetchone()
    if not row or not row["p"]:
        return {}
    return {
        (r["term"], r["country"]): r["pop_1to100"]
        for r in conn.execute(
            "SELECT term, country, MAX(pop_1to100) AS pop_1to100 "
            "FROM asa_popularity WHERE granularity='WEEKLY_SUN_SAT' AND period = ? "
            "GROUP BY term, country",
            (row["p"],),
        )
    }


def source_map(conn) -> dict[tuple[str, str], str]:
    return {
        (r["term"], r["storefront"]): r["source"] or ""
        for r in conn.execute("SELECT term, storefront, source FROM keywords")
    }


def render(
    candidates: list[Candidate],
    conn,
    countries: list[str],
    limit: int = 40,
    measured_only: bool = False,
    only_source: str | None = None,
) -> str:
    pops = popularity_map(conn, countries)
    sources = source_map(conn)
    hidden = hidden_terms(conn)

    clusters = [
        c for c in group_into_clusters(candidates)
        if not c.is_brand and not c.is_junk and c.head.term not in hidden
    ]

    rows = []
    for cl in clusters:
        key = (cl.head.term, cl.head.storefront)
        pop = pops.get(key)
        src = sources.get(key, "")
        if measured_only and pop is None:
            continue
        if only_source and src != only_source:
            continue
        rows.append((cl, pop, src))

    # Measured demand first — a number beats an inference. Within that, the
    # rating decides.
    rows.sort(key=lambda r: (r[1] is None, -(r[1] or 0), -r[0].rating.stars))

    out: list[str] = []
    measured = sum(1 for _, p, _ in rows if p is not None)
    out.append(f"ИДЕИ · {', '.join(c.upper() for c in countries)} · "
               f"{len(rows)} ниш, из них {measured} с измеренным спросом")
    if hidden:
        out.append(f"скрыто вашими решениями: {len(hidden)}")
    out.append("")
    out.append(f"{'оцен':<6}{'мкт':<4}{'ниша':<26}{'поп':>5}  {'откуда':<12}лидер")
    out.append("─" * 76)

    for cl, pop, src in rows[:limit]:
        c, r = cl.head, cl.rating
        lead = (c.leader or {}).get("title", "")[:26]
        pop_s = str(pop) if pop is not None else "·"
        origin = {"asa": "ASA", "autocomplete": "автокомплит",
                  "suggest": "автокомплит", "llm": "llm"}.get(src, src or "?")
        # A term carrying both a measured size and a category description is
        # the one to look at first; say so rather than leaving it implicit.
        if pop is not None:
            origin = "ASA измер."
        # The actual search term, not the judge's Russian label: this is the
        # string you paste into Keyword Planner, and a translated name cannot
        # be cross-referenced with anything.
        out.append(f"{r.bar:<6}{c.storefront.upper():<4}{c.term[:24]:<26}"
                   f"{pop_s:>5}  {origin:<12}{lead}")
        if cl.display_name and cl.display_name.lower() != c.term.lower():
            out.append(f"      {cl.display_name}")
        notes = [s.key for s in r.signals if not s.met]
        if notes:
            out.append(f"      нет: {', '.join(notes)}")
        if cl.variants:
            out.append(f"      + {', '.join(v.term for v in cl.variants[:3])}")

    if len(rows) > limit:
        out.append("")
        out.append(f"...ещё {len(rows) - limit}")

    out.append("")
    out.append("поп — шкала Apple 1-100. Прочерк значит, что термин вне топ-500")
    out.append("своей категории: спрос не измерен, а не отсутствует.")
    out.append("")
    out.append("Стоимость привлечения здесь НЕ измеряется — ни один эндпоинт")
    out.append("Apple её не отдаёт. Проверяйте вручную в Keyword Planner по")
    out.append("короткому списку, прежде чем браться.")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="the idea list")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--measured", action="store_true",
                    help="only terms with a real popularity number")
    ap.add_argument("--countries", default="us,gb,ca,au")
    ap.add_argument("--source", choices=["asa", "suggest", "llm"],
                    help="only candidates from one origin")
    ap.add_argument("--pool", type=int, default=5000,
                    help="how deep into the queue to look. The default report "
                         "caps at 50 by score, which hid every ASA candidate "
                         "because they score lower than autocomplete terms")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(message)s")
    db.init_db()
    countries = [c.strip().lower() for c in args.countries.split(",") if c.strip()]

    with db.connect() as conn:
        # Load the whole queue, not the scored top 50. Score was tuned on
        # autocomplete terms, so ranking by it buries everything that came
        # from Apple's measured data — which is the opposite of useful.
        cands = load_queue(conn, limit=args.pool)
        if not cands:
            print("очередь пуста — запустите asa_enrich")
            return 1
        print(render(cands, conn, countries, args.limit, args.measured,
                     only_source=args.source))
    return 0


if __name__ == "__main__":
    sys.exit(main())
