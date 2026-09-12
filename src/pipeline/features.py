"""
Features computed from a SERP snapshot.

Everything here is derived from stored snapshots, so features can be
recomputed and reweighted retroactively. That is why raw responses are kept
untouched in the DB.
"""
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import Any

from ..config import settings
from .coherence import coherence_score, top1_relevance


@dataclass
class Features:
    term: str
    storefront: str

    # Competition
    top3_rating_median: float | None
    top3_reviews_median: int
    top1_reviews: int
    stale_share: float           # share of top-10 not updated in > stale_days
    weak_share: float            # share of top-10 with rating < weak_rating
    thin_share: float            # share of top-10 with few reviews
    youngest_top5_age_days: int | None
    no_localization_share: float  # top-5 lacking the storefront's language

    # Money
    monetized_count: int
    paid_count: int
    games_share: float           # share of top-10 in the Games category
    top1_seller: str | None      # to detect platform giants

    # Relevance / demand proxies
    coherence: float
    top1_relevance: float
    suggest_position: int | None   # best autocomplete position seen
    in_metadata_share: float       # share of top-10 with term in title/subtitle

    # Install proxy (from review deltas, filled by the metrics job)
    top1_weekly_review_delta: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def est_top1_installs_monthly(self) -> int | None:
        """
        Crude install estimate for the leader.

        Systematically biased: apps that aggressively prompt for ratings look
        bigger than they are. Only ever compare within one category, and never
        present this without the error bar.
        """
        if self.top1_weekly_review_delta is None:
            return None
        weekly = self.top1_weekly_review_delta * settings.thresholds.review_to_install_multiplier
        return int(weekly * 4.33)


def _median_or_none(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.median(clean) if clean else None


def extract(
    term: str,
    storefront: str,
    lang: str,
    apps: list[dict[str, Any]],
    suggest_position: int | None = None,
) -> Features:
    """Compute all features for one keyword from its SERP snapshot."""
    t = settings.thresholds
    top10 = apps[:10]
    top5 = apps[:5]
    top3 = apps[:3]
    n = len(top10) or 1

    stale = sum(
        1 for a in top10
        if a.get("days_since_update") is not None and a["days_since_update"] > t.stale_days
    )
    weak = sum(1 for a in top10 if (a.get("rating") or 0) < t.weak_rating)
    thin = sum(1 for a in top10 if (a.get("rating_count") or 0) < t.thin_reviews)

    no_loc = 0
    if top5:
        for a in top5:
            langs = [str(x).lower() for x in (a.get("languages") or [])]
            if lang.lower() not in langs:
                no_loc += 1
        no_loc_share = no_loc / len(top5)
    else:
        no_loc_share = 0.0

    ages = [a["age_days"] for a in top5 if a.get("age_days") is not None]

    query_tokens = set(term.split())
    in_meta = sum(
        1 for a in top10
        if query_tokens & set(f"{a.get('title','')} {a.get('subtitle_proxy','')}".lower().split())
    )

    return Features(
        term=term,
        storefront=storefront,
        top3_rating_median=_median_or_none([a.get("rating") for a in top3]),
        top3_reviews_median=int(_median_or_none([a.get("rating_count", 0) for a in top3]) or 0),
        top1_reviews=int(top10[0].get("rating_count") or 0) if top10 else 0,
        stale_share=stale / n,
        weak_share=weak / n,
        thin_share=thin / n,
        youngest_top5_age_days=min(ages) if ages else None,
        no_localization_share=no_loc_share,
        games_share=sum(
            1 for a in top10
            if (a.get("primary_genre") or "") == "Games"
            or "Games" in (a.get("genres") or [])
        ) / n,
        top1_seller=(top10[0].get("seller") if top10 else None),
        monetized_count=sum(1 for a in top10 if a.get("is_monetized")),
        paid_count=sum(1 for a in top10 if a.get("is_paid")),
        coherence=coherence_score(term, top10),
        top1_relevance=top1_relevance(term, top10),
        suggest_position=suggest_position,
        in_metadata_share=in_meta / n,
    )
