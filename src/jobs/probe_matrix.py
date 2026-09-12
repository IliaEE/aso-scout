"""
Find a request shape that makes the hints endpoint return suggestions.

The endpoint parsed fine but returned an empty `hints` array, which means the
request reached Apple and Apple declined to answer. The usual cause is a
missing storefront identifier: these internal endpoints take the store from
the `X-Apple-Store-Front` header, not from a query parameter, so without it
there is no store to suggest for.

This tries a small matrix and reports which combination yields terms. Keep it
polite: one request every few seconds, roughly a dozen total.

    python -m src.jobs.probe_matrix
    python -m src.jobs.probe_matrix --q convert

A note on the user-agent row. The endpoint is public and unauthenticated, and
identifying the client type is how it is meant to be called — nothing is being
bypassed. It is still the undocumented, grey-area part of this system: fine for
internal research, not something to build a commercial product on. If it turns
out to be the deciding factor, treat that as a signal to budget for a licensed
ASO API sooner rather than later.
"""
from __future__ import annotations

import argparse
import asyncio
import plistlib

import httpx

from ..sources.suggest import HINTS_URL, _parse_hints

# Numeric App Store storefront IDs. The suffix after the comma is a client
# version marker that iTunes clients send; several values are accepted.
STOREFRONTS = {
    "us": 143441,
    "gb": 143444,
    "de": 143443,
    "ee": 143518,
    "ru": 143469,
}

ITUNES_UA = "iTunes/12.11 (Macintosh; OS X 10.15.7) AppleWebKit/605.1.15"
PLAIN_UA = "aso-scout/0.1 (keyword research)"


def variants(country: str) -> list[tuple[str, dict[str, str], dict[str, str]]]:
    """(label, extra_headers, extra_params)"""
    sf = STOREFRONTS.get(country, 143441)
    out: list[tuple[str, dict[str, str], dict[str, str]]] = []

    # Baseline: what we send today.
    out.append(("baseline (no storefront header)", {}, {"clientApplication": "Software"}))

    # Storefront header, a few accepted formats.
    for suffix in ("29", "32", "12"):
        out.append((
            f"X-Apple-Store-Front: {sf}-1,{suffix}",
            {"X-Apple-Store-Front": f"{sf}-1,{suffix}"},
            {"clientApplication": "Software"},
        ))
    out.append((
        f"X-Apple-Store-Front: {sf},29",
        {"X-Apple-Store-Front": f"{sf},29"},
        {"clientApplication": "Software"},
    ))

    # clientApplication alternatives, with the storefront header on.
    for app in ("MacSoftware", "iTunes"):
        out.append((
            f"clientApplication={app} + storefront",
            {"X-Apple-Store-Front": f"{sf}-1,29"},
            {"clientApplication": app},
        ))

    # No clientApplication at all.
    out.append((
        "no clientApplication + storefront",
        {"X-Apple-Store-Front": f"{sf}-1,29"},
        {},
    ))

    return out


async def try_one(
    client: httpx.AsyncClient,
    q: str,
    headers: dict[str, str],
    params: dict[str, str],
) -> tuple[int, int, list[str], str]:
    full = {"q": q, **params}
    try:
        resp = await client.get(HINTS_URL, params=full, headers=headers)
    except Exception as exc:
        return 0, 0, [], f"error: {type(exc).__name__}"

    if resp.status_code != 200:
        return resp.status_code, 0, [], ""

    try:
        parsed = plistlib.loads(resp.content)
    except Exception:
        try:
            parsed = resp.json()
        except Exception:
            return resp.status_code, len(resp.content), [], "unparseable"

    terms = _parse_hints(parsed)
    return resp.status_code, len(resp.content), terms, ""


async def main() -> None:
    ap = argparse.ArgumentParser(description="find a working hints request shape")
    ap.add_argument("--q", default="scan")
    ap.add_argument("--country", default="us")
    ap.add_argument("--delay", type=float, default=3.5)
    args = ap.parse_args()

    rows = variants(args.country)
    winners: list[str] = []

    print(f'probing hints for q="{args.q}" country={args.country}\n')
    print(f"{'variant':<42} {'ua':<8} {'code':>5} {'bytes':>6} {'terms':>6}")
    print("-" * 74)

    for ua_label, ua in (("plain", PLAIN_UA), ("itunes", ITUNES_UA)):
        async with httpx.AsyncClient(
            timeout=20.0, headers={"User-Agent": ua}, follow_redirects=True
        ) as client:
            for label, headers, params in rows:
                code, size, terms, note = await try_one(client, args.q, headers, params)
                flag = "  <--" if terms else ""
                print(
                    f"{label:<42} {ua_label:<8} {code:>5} {size:>6} "
                    f"{len(terms):>6}{flag}  {note}"
                )
                if terms:
                    winners.append(f"{label} | UA={ua_label}")
                    for t in terms[:6]:
                        print(f"      - {t}")
                await asyncio.sleep(args.delay)

        # No point running the second UA pass if the first already worked.
        if winners:
            break

    print("-" * 74)
    if winners:
        print("WORKING COMBINATIONS:")
        for w in winners:
            print(f"  {w}")
        print("\nSend me this list and I will lock the winner into suggest.py.")
    else:
        print("Nothing returned terms.")
        print()
        print("This does not block the project. Autocomplete is one of three")
        print("keyword sources; the other two use only the documented search")
        print("API and the public chart RSS, both of which already work for you.")
        print("Send me the table and I will switch the funnel to run on those.")


if __name__ == "__main__":
    asyncio.run(main())
