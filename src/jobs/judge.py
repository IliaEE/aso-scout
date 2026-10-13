"""
Run the LLM judge over candidates that have not been judged yet.

    python -m src.jobs.judge               # judge up to JUDGE_PER_RUN new ones
    python -m src.jobs.judge --limit 5     # small test
    python -m src.jobs.judge --models      # what can this account actually call
    python -m src.jobs.judge --requeue     # re-judge everything (rare)

Only unjudged keywords are sent. A verdict does not change unless the SERP
does, so re-judging nightly would spend the free tier re-deriving yesterday's
answer.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter

from .. import db
from ..llm.judge import ensure_table, judge_one, save_verdict
from ..llm.provider import GroqChat
from ..pipeline.features import name_match
from ..report.render import group_into_clusters, load_queue

log = logging.getLogger(__name__)


def run(limit: int, requeue: bool = False) -> dict[str, int]:
    chat = GroqChat()
    if not chat.configured:
        log.warning("GROQ_API_KEY not set — skipping judgement")
        return {}

    model = chat.resolve_model()
    if not model:
        log.error("no usable Groq model for this account; try --models")
        return {}
    log.info("judging with %s", model)

    stats: Counter[str] = Counter()

    with db.connect() as conn:
        ensure_table(conn)
        clusters = group_into_clusters(load_queue(conn))

        judged = set()
        if not requeue:
            judged = {
                (r["term"], r["storefront"])
                for r in conn.execute(
                    "SELECT term, storefront FROM llm_verdicts"
                )
            }

        # Best candidates first: if the rate limit runs out, it should run out
        # on the entries nobody was going to read anyway.
        clusters.sort(key=lambda c: -c.score)
        todo = [c for c in clusters if (c.head.term, c.head.storefront) not in judged]
        log.info("%d candidates, %d unjudged", len(clusters), len(todo))

        for cl in todo[:limit]:
            c = cl.head
            v = judge_one(
                chat, c.term, c.storefront, c.apps,
                [x.term for x in cl.variants],
                name_match=c.features.get("top1_name_match"),
            )
            if v is None:
                stats["failed"] += 1
                # The provider already retried and tried smaller models. If it
                # reports exhaustion there is nothing left to wait for, and
                # three more doomed calls only confirm it.
                if getattr(chat, "rate_limited", False):
                    log.warning("rate limit exhausted on every available model "
                                "— stopping. Already-judged verdicts are saved; "
                                "run again later to continue.")
                    break
                if chat.stats["failures"] >= 3:
                    log.warning("three failures in a row — stopping this run")
                    break
                continue
            save_verdict(conn, c.term, c.storefront, v, db.now_iso())
            stats[v.category] += 1
            log.info("  %-26s %-10s %.0f%%  %s",
                     c.term[:26], v.category, v.confidence * 100, v.niche_name)
            if v.note:
                log.info("  %-26s   %s", "", v.note[:90])

    stats["calls"] = chat.stats["calls"]
    if chat.stats.get("downgrades"):
        stats["model_downgrades"] = chat.stats["downgrades"]
    return dict(stats)


def selftest() -> int:
    """
    Score the judge against cases whose answer is already known.

    Exists because a run of twelve identical verdicts is unreadable: it could
    mean the sample was twelve niches, or it could mean the model answers
    "niche" to everything. Only a labelled set separates the two.
    """
    from ..llm.selftest import CASES, case_apps, summarize  # noqa: PLC0415

    chat = GroqChat()
    if not chat.configured:
        print("GROQ_API_KEY не задан")
        return 1
    model = chat.resolve_model()
    if not model:
        print("нет доступной модели")
        return 1

    print(f"проверка на {len(CASES)} размеченных случаях, модель {model}\n")
    results: list[tuple[str, str, str]] = []

    for term, expected, title, seller, rating, reviews in CASES:
        apps = case_apps(title, seller, rating, reviews)
        nm = name_match(term, title)
        v = judge_one(chat, term, "us", apps, [], name_match=nm)
        got = v.category if v else "ошибка"
        results.append((term, expected, got))
        mark = "ok" if got == expected else "MISS"
        print(f"  [{mark:>4}] {term:24} ожидалось {expected:10} получено {got}")
        if getattr(chat, "rate_limited", False):
            print("\n  лимит исчерпан — результат неполный")
            break

    print(summarize(results))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM judgement pass")
    ap.add_argument("--limit", type=int,
                    default=int(os.environ.get("JUDGE_PER_RUN", 30)))
    ap.add_argument("--requeue", action="store_true",
                    help="re-judge keywords that already have a verdict")
    ap.add_argument("--models", action="store_true",
                    help="list models this API key can call")
    ap.add_argument("--selftest", action="store_true",
                    help="judge a labelled set of known cases and score it")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    db.init_db()

    if args.models:
        chat = GroqChat()
        if not chat.configured:
            print("GROQ_API_KEY не задан")
            return 1
        models = chat.available_models()
        if not models:
            print("список моделей получить не удалось — проверьте ключ")
            return 1
        print(f"доступно {len(models)} моделей:")
        for m in sorted(models):
            print(f"  {m}")
        print(f"\nбудет использована: {chat.resolve_model()}")
        return 0

    if args.selftest:
        return selftest()

    stats = run(args.limit, args.requeue)
    if not stats:
        print("ничего не сделано — смотрите лог выше")
        return 1
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
