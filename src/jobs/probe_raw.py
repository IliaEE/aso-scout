"""
Raw-response diagnostic for the autocomplete endpoint.

The hints endpoint is undocumented, so when parsing fails the only way
forward is to look at what Apple actually sent. Run this and share the output.

    python -m src.jobs.probe_raw
    python -m src.jobs.probe_raw --q "convert" --country de
"""
from __future__ import annotations

import argparse
import asyncio
import json
import plistlib

import httpx

from ..config import settings
from ..sources.suggest import HINTS_URL, _parse_hints


async def main() -> None:
    ap = argparse.ArgumentParser(description="dump raw hints response")
    ap.add_argument("--q", default="scan")
    ap.add_argument("--country", default="us")
    ap.add_argument("--lang", default="en_us")
    ap.add_argument("--bytes", type=int, default=600)
    args = ap.parse_args()

    params = {
        "q": args.q,
        "clientApplication": "Software",
        "country": args.country,
        "l": args.lang,
    }

    async with httpx.AsyncClient(
        timeout=20.0,
        headers={"User-Agent": settings.user_agent},
        follow_redirects=True,
    ) as client:
        resp = await client.get(HINTS_URL, params=params)

    print(f"URL           {resp.url}")
    print(f"status        {resp.status_code}")
    print(f"content-type  {resp.headers.get('content-type', 'n/a')}")
    print(f"length        {len(resp.content)} bytes")
    print("-" * 60)
    print(f"first {args.bytes} bytes:\n")
    try:
        print(resp.content[: args.bytes].decode("utf-8", errors="replace"))
    except Exception:
        print(repr(resp.content[: args.bytes]))
    print("-" * 60)

    parsed = None
    for name, fn in (("JSON", lambda: json.loads(resp.content)),
                     ("plist", lambda: plistlib.loads(resp.content))):
        try:
            parsed = fn()
            print(f"parsed as {name}: OK")
            break
        except Exception as exc:
            print(f"parsed as {name}: failed ({type(exc).__name__})")

    if parsed is not None:
        if isinstance(parsed, dict):
            print(f"top-level keys: {list(parsed)[:12]}")
        terms = _parse_hints(parsed)
        print(f"terms extracted: {len(terms)}")
        for t in terms[:10]:
            print(f"  - {t}")
        if not terms:
            print("\nParsed fine but no terms found — the response nests them")
            print("somewhere our parser does not look. Share the bytes above.")
    else:
        print("\nNeither shape matched. Share the bytes above and I will")
        print("add a parser for this format.")


if __name__ == "__main__":
    asyncio.run(main())
