"""
Preflight check. Run this once after deploying, before trusting any numbers.

The blocking risk on a shared host like Railway is that the egress IP is
already burned by other scrapers. If that is the case, everything downstream
is silently empty or wrong, and you would rather find out in 30 seconds than
after a week of collection.

    python -m src.jobs.doctor
"""
from __future__ import annotations

import asyncio
import os
import sys

from .. import db
from ..config import on_railway, settings, volume_mounted
from ..sources import itunes, suggest
from ..sources.http import AppleClient

OK = "\033[32mOK\033[0m"
BAD = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"


async def main() -> int:
    print("ASO Scout preflight\n" + "=" * 46)
    problems = 0

    print(f"\nstorage: SQLite {settings.sqlite_path}")
    if on_railway() and not volume_mounted():
        print(f"  [{BAD}] running on Railway with NO volume mounted.")
        print("         The database is on the ephemeral filesystem and will be")
        print("         wiped on every deploy and restart. Snapshot history is")
        print("         the one thing that cannot be rebuilt — attach a volume")
        print("         before collecting anything.")
        problems += 1
    elif volume_mounted():
        print(f"  [{OK}] volume mounted at {os.environ['RAILWAY_VOLUME_MOUNT_PATH']}")
    try:
        db.init_db()
        with db.connect() as conn:
            conn.execute("SELECT 1").fetchone()
        print(f"  [{OK}] schema reachable and writable")
    except Exception as exc:
        print(f"  [{BAD}] storage: {exc}")
        problems += 1

    print(f"\nrate limit: {settings.requests_per_minute} req/min")
    if settings.requests_per_minute > 20:
        print(f"  [{WARN}] above the ~20/min ceiling Apple tolerates. Lower RPM.")

    if "example.com" in settings.user_agent:
        print(f"  [{WARN}] USER_AGENT still has the placeholder contact address.")
        print("         A traceable UA is what separates throttling from a ban.")

    async with AppleClient() as client:
        print("\nautocomplete endpoint (undocumented, the demand generator)")
        store = settings.storefronts[0]
        hints = await suggest.probe(
            client, "scan", store.country, f"{store.lang}_{store.country}",
            store.store_id,
        )
        if hints:
            print(f"  [{OK}] {len(hints)} suggestions, e.g. "
                  f"{', '.join(t for t, _ in hints[:4])}")
        else:
            print(f"  [{BAD}] no suggestions returned")
            print("         Either the IP is blocked or the endpoint changed shape.")
            print("         Run: python -m src.jobs.probe_matrix")
            print("         Without it the funnel has no source of validated queries.")
            problems += 1

        print("\nsearch endpoint (SERP snapshots)")
        apps = await itunes.search(client, "noise meter", "us", limit=5)
        if apps:
            print(f"  [{OK}] {len(apps)} results, #1 = {apps[0]['title'][:40]}")
        else:
            print(f"  [{BAD}] no search results returned")
            problems += 1

        st = client.stats
        print(f"\nhttp: {st['requests']} requests, {st['retries']} retries, "
              f"{st['blocked']} blocked, {st['failures']} failures")

        if st["blocked"]:
            print(f"\n  [{BAD}] Requests were blocked with 403.")
            print("  This egress IP cannot reach Apple. Nothing downstream will be")
            print("  valid. Move to a different host before collecting anything.")
            problems += 1

    # --- LLM layer -------------------------------------------------------
    from ..llm.provider import chat_for  # noqa: PLC0415

    print("\nLLM")
    for task, label in (("judge", "судья"), ("ideate", "генератор идей")):
        chat = chat_for(task)
        if chat is None:
            print(f"  [{WARN}] {label}: ключ не задан — работают только эвристики")
            continue

        models = chat.available_models()
        if not models:
            print(f"  [{BAD}] {label}: список моделей не получен — ключ неверный "
                  f"или отозван")
            problems += 1
            continue

        model = chat.resolve_model()
        if not model:
            print(f"  [{BAD}] {label}: нет пригодной модели среди {len(models)}")
            problems += 1
            continue

        # Which organization this key belongs to decides whether the two keys
        # share a rate limit. Same org means a second key bought nothing.
        reply = chat.complete(
            'Reply with exactly {"ok": true} and nothing else.',
            "ping",
            max_tokens=20,
        )
        if reply is None:
            print(f"  [{BAD}] {label}: {model} не отвечает")
            problems += 1
            continue

        shared = ""
        if task == "ideate":
            other = chat_for("judge")
            if other is not None and other.api_key == chat.api_key:
                shared = "  (тот же ключ, что у судьи)"
        print(f"  [{OK}] {label}: {model}, пауза {chat._min_interval:.1f}с{shared}")

    print("\n" + "=" * 46)
    if problems:
        print(f"{problems} blocking problem(s). Fix before collecting.")
        return 1
    print("Ready. Start collecting — history cannot be back-filled,")
    print("so the sooner snapshots begin, the sooner deltas become usable.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
