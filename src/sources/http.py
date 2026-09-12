"""
Polite async HTTP client for Apple's public endpoints.

Two non-negotiables baked in here:

1. A global token-bucket limiter shared by every source. Apple tolerates
   roughly 20 req/min per IP on the iTunes endpoints. Exceeding it gets the IP
   rate-limited, and on a shared host like Railway you may be sharing that
   budget with strangers already.

2. No proxy rotation. It violates Apple's terms and does not make the data
   any more accurate — it just hides the 429 that was telling you to slow down.

`fetch_json` is the only entry point. Pass a `transport` to run against
recorded fixtures instead of the live network.
"""
from __future__ import annotations

import asyncio
import json
import logging
import plistlib
import random
import time
from typing import Any, Protocol

import httpx

from ..config import settings

log = logging.getLogger(__name__)


class Fetcher(Protocol):
    """Anything that can turn a URL + params into parsed JSON."""

    async def fetch_json(
        self, url: str, params: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> Any | None: ...


class RateLimiter:
    """Token bucket. Shared process-wide so all sources respect one budget."""

    def __init__(self, per_minute: int) -> None:
        self.interval = 60.0 / max(per_minute, 1)
        self._lock = asyncio.Lock()
        self._next_slot = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            # Small jitter so concurrent workers don't align into bursts.
            self._next_slot = now + self.interval * random.uniform(0.9, 1.15)


class AppleClient:
    """Live client. Retries on 429/5xx with exponential backoff."""

    # Status codes worth retrying. 403 is NOT here: on Apple endpoints it
    # usually means blocked, and hammering it makes things worse.
    RETRY_ON = {429, 500, 502, 503, 504}

    def __init__(self, limiter: RateLimiter | None = None) -> None:
        self.limiter = limiter or RateLimiter(settings.requests_per_minute)
        self._client: httpx.AsyncClient | None = None
        self.stats = {"requests": 0, "retries": 0, "failures": 0, "blocked": 0}

    async def __aenter__(self) -> "AppleClient":
        self._client = httpx.AsyncClient(
            timeout=settings.request_timeout,
            headers={"User-Agent": settings.user_agent},
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client:
            await self._client.aclose()

    async def fetch_json(
        self, url: str, params: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> Any | None:
        if self._client is None:
            raise RuntimeError("AppleClient must be used as an async context manager")

        delay = 2.0
        for attempt in range(settings.max_retries):
            await self.limiter.acquire()
            self.stats["requests"] += 1
            try:
                resp = await self._client.get(url, params=params, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                log.warning("transport error on %s: %s", url, exc)
                self.stats["retries"] += 1
                await asyncio.sleep(delay)
                delay *= 2
                continue

            if resp.status_code == 200:
                try:
                    return resp.json()
                except json.JSONDecodeError:
                    pass
                # The hints endpoint returns an Apple plist, not JSON — XML on
                # some storefronts, binary on others. plistlib is stdlib and
                # handles both, so this costs no dependency.
                try:
                    return plistlib.loads(resp.content)
                except Exception:
                    pass
                # Neither shape matched. Log enough to identify what Apple
                # actually sent, because guessing from "parse failed" is
                # hopeless and this endpoint is undocumented.
                head = resp.content[:200]
                log.error(
                    "unparseable body from %s (content-type=%s, %d bytes). "
                    "First bytes: %r",
                    url,
                    resp.headers.get("content-type", "n/a"),
                    len(resp.content),
                    head,
                )
                self.stats["failures"] += 1
                return None

            if resp.status_code == 403:
                # Either the egress host is blocked or the IP is banned.
                # Loud, because it invalidates every downstream number.
                self.stats["blocked"] += 1
                log.error(
                    "403 from %s — IP may be blocked or host not allowed. "
                    "Deny reason: %s",
                    url,
                    resp.headers.get("x-deny-reason", "n/a"),
                )
                return None

            if resp.status_code in self.RETRY_ON:
                self.stats["retries"] += 1
                retry_after = resp.headers.get("retry-after")
                sleep_for = float(retry_after) if retry_after and retry_after.isdigit() else delay
                log.info("%s from %s, backing off %.1fs", resp.status_code, url, sleep_for)
                await asyncio.sleep(sleep_for)
                delay *= 2
                continue

            log.warning("unexpected %s from %s", resp.status_code, url)
            self.stats["failures"] += 1
            return None

        self.stats["failures"] += 1
        return None


class FixtureClient:
    """
    Offline client backed by recorded responses.

    Used by the test suite and by anyone wanting to exercise the pipeline
    logic without touching Apple. Keys are "<url>?<sorted params>".
    """

    def __init__(self, fixtures: dict[str, Any]) -> None:
        self.fixtures = fixtures
        self.calls: list[str] = []

    @staticmethod
    def make_key(url: str, params: dict[str, Any]) -> str:
        flat = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{url}?{flat}"

    async def __aenter__(self) -> "FixtureClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def fetch_json(
        self, url: str, params: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> Any | None:
        # Headers do not participate in the fixture key: they affect what
        # Apple returns, not which request we are identifying.
        key = self.make_key(url, params)
        self.calls.append(key)
        return self.fixtures.get(key)
