"""
Search term popularity from the Apple Ads Platform API.

This is the number the rest of the tool has been guessing at. Apple reports
it directly: `searchPopularity1to100` on the same scale the Search Ads UI
shows, plus `rankInGenre` — the term's position by volume inside its genre.

What it is NOT: a lookup service for arbitrary keywords. The dataset is the
top terms per genre, per country, per week. A long-tail phrase that never
enters that list has no popularity here, and no amount of querying produces
one. The honest shape of this source is "a ranked list we pull and keep",
not "ask it about any keyword you like".

Which is exactly why weekly snapshots matter: holding several weeks lets us
compute movement in `rankInGenre`, and a term climbing from 300th to 80th is
the "becoming popular" signal that nothing else here could ever see.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator

import httpx

from .auth import AsaAuth

log = logging.getLogger(__name__)

API_BASE = "https://api.ads.apple.com/v1"
POPULARITY_PATH = "/insights/apps/search-term-popularity/query"

# The genres the endpoint accepts. GAMES is included for completeness but the
# pipeline gates game keywords out anyway — we are not building games.
GENRES = [
    "PRODUCTIVITY_UTILITIES",
    "PHOTO_VIDEO",
    "HEALTH_FITNESS",
    "LIFESTYLE",
    "BUSINESS",
    "EDUCATION",
    "FINANCE",
    "FOOD_DRINK",
    "TRAVEL",
    "SHOPPING",
    "SOCIAL_NETWORKING",
    "SPORTS",
    "ENTERTAINMENT",
    "NEW_PUBLICATION",
    "GAMES",
]

# Genres worth pulling first: where a small team can actually ship something.
PRIORITY_GENRES = [
    "PRODUCTIVITY_UTILITIES",
    "PHOTO_VIDEO",
    "HEALTH_FITNESS",
    "BUSINESS",
    "FINANCE",
    "EDUCATION",
    "LIFESTYLE",
    "FOOD_DRINK",
]


@dataclass
class PopularityRow:
    search_term: str
    country: str
    genre: str
    period: str              # the week or month this row belongs to
    rank_in_genre: int | None
    popularity_in_genre: int | None
    popularity_1to100: int | None
    popularity_1to5: int | None

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> "PopularityRow | None":
        term = str(raw.get("searchTerm", "")).strip().lower()
        if not term:
            return None
        return cls(
            search_term=term,
            country=str(raw.get("countryOrRegion", "")).lower(),
            genre=str(raw.get("genre", "")),
            period=str(raw.get("week") or raw.get("month") or ""),
            rank_in_genre=raw.get("rankInGenre"),
            popularity_in_genre=raw.get("searchPopularityInGenre"),
            popularity_1to100=raw.get("searchPopularity1to100"),
            popularity_1to5=raw.get("searchPopularity1to5"),
        )


class PopularityClient:
    def __init__(self, auth: AsaAuth | None = None, timeout: float = 60.0) -> None:
        self.auth = auth or AsaAuth()
        self.timeout = timeout
        self.stats = {"requests": 0, "rows": 0, "failures": 0}

    def _post(self, body: dict[str, Any]) -> dict[str, Any] | None:
        self.stats["requests"] += 1
        try:
            with httpx.Client(timeout=self.timeout) as c:
                r = c.post(
                    f"{API_BASE}{POPULARITY_PATH}",
                    headers=self.auth.headers(),
                    json=body,
                )
        except Exception:
            self.stats["failures"] += 1
            log.exception("popularity request failed")
            return None

        if r.status_code != 200:
            self.stats["failures"] += 1
            log.error("popularity %s: %s", r.status_code, r.text[:300])
            return None
        try:
            return r.json()
        except Exception:
            self.stats["failures"] += 1
            log.error("unparseable popularity response")
            return None

    def query(
        self,
        countries: list[str],
        genre: str,
        start: str,
        end: str,
        granularity: str = "WEEKLY_SUN_SAT",
        page_size: int = 100,
        max_rows: int = 500,
    ) -> Iterator[PopularityRow]:
        """
        Pull one genre across countries for a period, newest-ranked first.

        Paginates by offset. `max_rows` is a guard rather than a target: the
        dataset tails off, and asking past the end costs requests for nothing.
        """
        offset = 0
        while offset < max_rows:
            body = {
                "filters": [
                    {"field": "countryOrRegion", "operator": "IN",
                     "value": [c.upper() for c in countries]},
                    {"field": "genre", "operator": "EQUALS", "value": genre},
                ],
                "timeRange": {
                    "start": start, "end": end, "granularity": granularity,
                },
                "sorting": [{"field": "rankInGenre", "order": "ASC"}],
                "pagination": {"offset": offset, "pageSize": page_size},
            }
            payload = self._post(body)
            if not payload:
                return

            rows = (payload.get("result") or {}).get("rows") or []
            if not rows:
                return

            for raw in rows:
                parsed = PopularityRow.parse(raw)
                if parsed:
                    self.stats["rows"] += 1
                    yield parsed

            if len(rows) < page_size:
                return
            offset += page_size

    def lookup(
        self, terms: list[str], countries: list[str], start: str, end: str,
        granularity: str = "WEEKLY_SUN_SAT", page_size: int = 100,
    ) -> Iterator[PopularityRow]:
        """
        Ask about specific terms.

        Only returns the ones present in the dataset. A term that comes back
        empty is not proof of zero demand — it means the term is outside the
        ranked list, which is most of the long tail.
        """
        for chunk_start in range(0, len(terms), 50):
            chunk = terms[chunk_start:chunk_start + 50]
            body = {
                "filters": [
                    {"field": "countryOrRegion", "operator": "IN",
                     "value": [c.upper() for c in countries]},
                    {"field": "searchTerm", "operator": "IN", "value": chunk},
                ],
                "timeRange": {
                    "start": start, "end": end, "granularity": granularity,
                },
                "pagination": {"offset": 0, "pageSize": page_size},
            }
            payload = self._post(body)
            if not payload:
                continue
            for raw in (payload.get("result") or {}).get("rows") or []:
                parsed = PopularityRow.parse(raw)
                if parsed:
                    self.stats["rows"] += 1
                    yield parsed
