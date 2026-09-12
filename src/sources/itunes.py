"""
iTunes Search API — SERP snapshots and app metadata.

Hard caveat, stated once and carried everywhere downstream: **this is not the
App Store ranking.** It is a different index. It is good enough to sift
thousands of keywords and spot obviously weak/stale competition, and it is
wrong often enough that the top-20 candidates must be verified by eye in the
Store or with real data from Astro before any decision.

Every raw response is stored untouched. We do not yet know which fields will
matter, and re-fetching history is impossible.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from .http import Fetcher

log = logging.getLogger(__name__)

SEARCH_URL = "https://itunes.apple.com/search"
LOOKUP_URL = "https://itunes.apple.com/lookup"


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def days_since(dt: datetime | None) -> int | None:
    if dt is None:
        return None
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max((now - dt).days, 0)


def normalize_app(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten one iTunes result into the fields the pipeline reasons about.

    On monetization: **the iTunes Search API does not expose in-app purchases
    for software.** `advisories` holds content ratings for music and film, not
    an IAP flag, so an earlier version of this code read it as one and marked
    every free app unmonetized — which rejected 39 of 39 keywords on a live
    run, including niches that are wall-to-wall subscriptions.

    `is_paid` (price > 0) is real and kept. `is_monetized` is retained only as
    a weak informational signal; nothing gates on it. Whether a niche makes
    money is verified by eye or in Astro, not here.
    """
    updated = _parse_date(raw.get("currentVersionReleaseDate"))
    released = _parse_date(raw.get("releaseDate"))
    price = raw.get("price")
    advisories = raw.get("advisories") or []

    title = (raw.get("trackName") or "").strip()
    # iTunes has no subtitle field; the closest proxy is the first sentence of
    # the description plus the seller name. Real subtitles need Astro.
    description = (raw.get("description") or "").strip()
    subtitle_proxy = description.split("\n")[0][:120] if description else ""

    return {
        "track_id": raw.get("trackId"),
        "bundle_id": raw.get("bundleId"),
        "title": title,
        "subtitle_proxy": subtitle_proxy,
        "seller": raw.get("sellerName"),
        "genres": raw.get("genres") or [],
        "primary_genre": raw.get("primaryGenreName"),
        "rating": raw.get("averageUserRating"),
        "rating_count": raw.get("userRatingCount") or 0,
        "price": price,
        "is_paid": bool(price and price > 0),
        # Always False in practice for software — kept so the field shape is
        # stable, but never trusted. See the docstring.
        "has_iap": "In-App Purchases" in advisories,
        "is_monetized": bool(price and price > 0),
        "updated_at": updated.isoformat() if updated else None,
        "released_at": released.isoformat() if released else None,
        "days_since_update": days_since(updated),
        "age_days": days_since(released),
        "languages": raw.get("languageCodesISO2A") or [],
        "content_rating": raw.get("contentAdvisoryRating"),
    }


async def search(
    client: Fetcher,
    term: str,
    country: str = "us",
    limit: int = 10,
    lang: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch the top-N for a keyword. Returns normalized app records in order."""
    params: dict[str, Any] = {
        "term": term,
        "country": country,
        "entity": "software",
        "limit": limit,
    }
    if lang:
        params["lang"] = lang

    payload = await client.fetch_json(SEARCH_URL, params)
    if not payload or "results" not in payload:
        return []
    return [normalize_app(r) for r in payload["results"]]


async def lookup(
    client: Fetcher, track_ids: list[int], country: str = "us"
) -> list[dict[str, Any]]:
    """
    Refresh metadata for known apps. Used by the daily metrics job to build
    the review-count delta that proxies installs.
    """
    if not track_ids:
        return []
    payload = await client.fetch_json(
        LOOKUP_URL,
        {"id": ",".join(str(i) for i in track_ids[:200]), "country": country},
    )
    if not payload or "results" not in payload:
        return []
    return [normalize_app(r) for r in payload["results"]]
