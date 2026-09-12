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
import sys

from .. import db
from ..config import settings
from ..sources import itunes, suggest
from ..sources.http import AppleClient

OK = "\033[32mOK\033[0m"
BAD = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"


async def main() -> int:
    print("ASO Scout preflight\n" + "=" * 46)
    problems = 0

    print(f"\nstorage: {'Postgres' if settings.use_postgres else 'SQLite ' + settings.sqlite_path}")
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

    print("\n" + "=" * 46)
    if problems:
        print(f"{problems} blocking problem(s). Fix before collecting.")
        return 1
    print("Ready. Start collecting — history cannot be back-filled,")
    print("so the sooner snapshots begin, the sooner deltas become usable.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
