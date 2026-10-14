"""
What is growing, what is new, what has been growing for a month.

    python -m src.jobs.asa_trends --weekly     # movement and new entrants
    python -m src.jobs.asa_trends --monthly    # sustained climbs only
    python -m src.jobs.asa_trends --term "calorie tracker"

The three views answer different questions and deserve different urgency:

  weekly   what moved. Noisy, worth a skim.
  monthly  what has been moving for a month. If it reached here unread, it
           is no longer optional.
  term     the full history of one term, for when you want to look closer.

Seasonality is marked rather than filtered out: a term that climbs every
September is still real demand, it just is not news.
"""
from __future__ import annotations

import argparse
import logging
import sys

from .. import db
from ..asa.trends import (
    Movement,
    annotate_seasonality,
    movement,
    new_entrants,
    periods,
    sustained,
)

log = logging.getLogger(__name__)


def fmt(m: Movement, show_delta: bool = True) -> str:
    arrow = "↑" if m.delta > 0 else ("↓" if m.delta < 0 else "·")
    rank = f"#{m.rank_now}" if m.rank_now else "—"
    pop = f"{m.popularity}" if m.popularity is not None else "·"
    season = ""
    if m.seasonal is True:
        season = "  ⟳ было и год назад"
    elif m.seasonal is None and not m.is_new:
        season = "  (год назад данных нет)"

    if m.is_new:
        body = f"НОВИЧОК в топ-500, сразу {rank}"
    elif show_delta:
        body = f"{arrow} {m.rank_before} → {m.rank_now}  ({m.delta:+d})"
    else:
        body = f"{arrow} {rank}"

    return (f"  {m.country.upper():<3} {m.term[:30]:<32} поп {pop:>3}  "
            f"{body}{season}")


def main() -> int:
    ap = argparse.ArgumentParser(description="popularity trends")
    ap.add_argument("--weekly", action="store_true")
    ap.add_argument("--monthly", action="store_true")
    ap.add_argument("--term")
    ap.add_argument("--countries", default="US,GB,CA,AU")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--months", action="store_true",
                    help="use monthly snapshots instead of weekly. Steadier, "
                         "and with 13 months stored the seasonality check "
                         "actually has something to compare against — the "
                         "weekly series has exactly 52 weeks, which is one "
                         "window short")
    args = ap.parse_args()
    gran = "MONTHLY" if args.months else "WEEKLY_SUN_SAT"
    unit = "месяцев" if args.months else "недель"

    logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(message)s")
    db.init_db()
    countries = [c.strip().lower() for c in args.countries.split(",") if c.strip()]

    with db.connect() as conn:
        weeks = periods(conn, gran)
        if not weeks:
            print("данных нет — запустите asa_pull")
            return 1

        if args.term:
            rows = conn.execute(
                "SELECT period, country, genre, rank_in_genre, pop_1to100 "
                "FROM asa_popularity WHERE term = ? AND granularity = ? "
                "ORDER BY country, genre, period",
                (args.term.strip().lower(), gran),
            ).fetchall()
            if not rows:
                print(f'"{args.term}" нет в данных Apple — значит он вне топ-500')
                print("своей категории. Это не доказательство отсутствия спроса,")
                print("но измерить его нечем.")
                return 0
            print(f'История "{args.term}"\n')
            for r in rows:
                print(f"  {r['period']}  {r['country'].upper():<3} "
                      f"{r['genre']:<24} #{r['rank_in_genre'] or '—':<5} "
                      f"поп {r['pop_1to100'] or '·'}")
            return 0

        print(f"данные: {len(weeks)} {unit}, {weeks[0]} … {weeks[-1]}\n")

        if args.monthly:
            items = annotate_seasonality(
                conn, sustained(conn, countries, granularity=gran),
                granularity=gran,
            )
            fresh = [m for m in items if m.seasonal is not True]
            label = "РОСЛИ НЕСКОЛЬКО МЕСЯЦЕВ" if args.months else "РОСЛИ ВЕСЬ МЕСЯЦ"
            print(f"{label} — {len(fresh)} шт (и ещё "
                  f"{len(items) - len(fresh)} сезонных)\n")
            print("Месяц роста это не шум. Если такое не было отсмотрено")
            print("раньше — смотреть сейчас.\n")
            for m in fresh[: args.limit]:
                print(fmt(m))
            if len(items) - len(fresh):
                print("\nсезонные (росли и год назад в это же время):")
                for m in [x for x in items if x.seasonal is True][:8]:
                    print(fmt(m))
            return 0

        # weekly is the default view
        entrants = new_entrants(conn, countries, granularity=gran)
        moves = annotate_seasonality(
            conn, movement(conn, countries, granularity=gran), granularity=gran
        )
        rising = [m for m in moves if m.climbing]
        falling = [m for m in moves if not m.climbing]

        print(f"НОВИЧКИ В ТОП-500 — {len(entrants)}\n")
        if entrants:
            print("Потребность, которой на прошлой неделе не было видно.\n")
            for m in entrants[: args.limit]:
                print(fmt(m))
        else:
            print("  нет\n")

        print(f"\nРАСТУТ — {len(rising)}\n")
        for m in rising[: args.limit]:
            print(fmt(m))

        print(f"\nПАДАЮТ — {len(falling)}\n")
        for m in falling[: min(args.limit, 8)]:
            print(fmt(m))

        print("\nПоп — шкала Apple 1-100. Ранг — место внутри категории.")
        print("Падение тоже сигнал: ниша сдувается, туда заходить поздно.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
