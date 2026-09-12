"""
Gates and scoring.

Two-stage by design.

Gates are hard yes/no. A keyword that fails any gate is out, and the reason is
recorded so it never resurfaces in the report. One gate is special:
UNBEATABLE_TOP3 sends the keyword to a watchlist rather than the bin, because
a dominant leader can decay and the whole point of keeping snapshot history is
to notice when it does.

The score is a **product**, not a sum. Weakness x intent x demand-proxy. If
any factor is near zero the whole thing collapses. This is the guard against
the failure mode the free data makes likely: an empty niche looks fantastic on
competition metrics precisely because nobody wants it. A sum would let a weak
SERP alone carry a dead keyword to the top of the queue. A product will not.

Nothing here measures demand in absolute terms. We cannot, for free. The
ordinal proxies only rank candidates against each other; the real number comes
from Astro in the manual step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from ..config import GIANT_SELLERS, settings
from .features import Features


class Gate(str, Enum):
    NO_TRACTION = "no_traction"
    UNBEATABLE_TOP3 = "unbeatable_top3"
    INCOHERENT = "incoherent"
    EMPTY_SERP = "empty_serp"
    GAME_DOMINATED = "game_dominated"
    GIANT_DOMINATED = "giant_dominated"


class Verdict(str, Enum):
    QUEUED = "queued"        # passed everything, in the review queue
    WATCHLIST = "watchlist"  # strong leader now, monitor for decay
    REJECTED = "rejected"    # failed a terminal gate


@dataclass
class Scored:
    term: str
    storefront: str
    verdict: Verdict
    score: float
    gates_failed: list[str] = field(default_factory=list)
    factors: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def is_giant(seller: str | None) -> bool:
    """True when the #1 result belongs to a platform giant."""
    if not seller:
        return False
    low = seller.lower()
    return any(g in low for g in GIANT_SELLERS)


def check_gates(f: Features) -> list[Gate]:
    """Run the hard filters. Empty list means the keyword survives."""
    t = settings.thresholds
    failed: list[Gate] = []

    if f.top1_reviews == 0 and f.coherence == 0.0:
        failed.append(Gate.EMPTY_SERP)

    if f.coherence < t.min_coherence:
        failed.append(Gate.INCOHERENT)

    # The dead-niche gate. Nothing in the niche has traction, which almost
    # always means nobody is searching for it — a weak SERP is the symptom,
    # not the opportunity.
    if f.top1_reviews < t.min_top1_reviews:
        failed.append(Gate.NO_TRACTION)

    # Both conditions must hold. A 4.8-rated app with 200 reviews is beatable;
    # a 4.0-rated app with 500k reviews is beatable on relevance. Both at once
    # is not worth fighting with a new app.
    # Game queries pass every other gate: "block puzzle" really does return
    # apps with those words in the title, monetized, coherent. They are still
    # useless here — we are not building games, and casual-game keywords are
    # won with UA spend, not metadata.
    if f.games_share >= 0.5:
        failed.append(Gate.GAME_DOMINATED)

    if is_giant(f.top1_seller):
        failed.append(Gate.GIANT_DOMINATED)

    if (
        f.top3_rating_median is not None
        and f.top3_rating_median >= t.unbeatable_rating
        and f.top3_reviews_median >= t.unbeatable_reviews
    ):
        failed.append(Gate.UNBEATABLE_TOP3)

    return failed


def _weakness(f: Features) -> float:
    """How takeable the SERP is, 0..1."""
    parts = [
        f.stale_share,
        f.weak_share,
        f.thin_share,
        f.no_localization_share,
        1.0 - min(f.top1_relevance, 1.0),  # irrelevant #1 = open door
    ]
    weights = [0.30, 0.20, 0.20, 0.15, 0.15]
    return sum(p * w for p, w in zip(parts, weights))


def _market(f: Features) -> float:
    """
    How much traffic the niche demonstrably carries, 0..1.

    Log-scaled review count of the leader, mapped from the traction floor to
    the "fully proven" ceiling. This replaced a monetization factor that was
    computed from data the iTunes API never provides.

    It is a market-size proxy, not a revenue estimate, and it inherits the
    same bias as every review-based number here: apps that nag for ratings
    look bigger than they are.
    """
    t = settings.thresholds
    if f.top1_reviews <= 0:
        return 0.0
    lo = math.log10(max(t.min_top1_reviews, 1))
    hi = math.log10(max(t.proven_top1_reviews, t.min_top1_reviews + 1))
    val = (math.log10(f.top1_reviews) - lo) / (hi - lo)
    return max(0.0, min(val, 1.0))


def _demand_proxy(f: Features) -> float:
    """
    Ordinal demand proxy, 0..1. Not a volume estimate.

    Two independent signals: where Apple ranks the term in autocomplete, and
    how many competitors spent their limited metadata characters on it. The
    second is the underrated one — competitors have already done the research.
    """
    if f.suggest_position is None:
        pos_signal = 0.35  # unknown, not zero: absence of data is not absence of demand
    else:
        pos_signal = max(0.0, 1.0 - (f.suggest_position - 1) / 10.0)

    meta_signal = min(f.in_metadata_share * 1.5, 1.0)
    return 0.55 * pos_signal + 0.45 * meta_signal


def score(f: Features) -> Scored:
    """Gate, then score. Score is only meaningful for surviving keywords."""
    failed = check_gates(f)

    factors = {
        "weakness": round(_weakness(f), 3),
        "market": round(_market(f), 3),
        "demand_proxy": round(_demand_proxy(f), 3),
        "coherence": round(min(f.coherence * 1.5, 1.0), 3),
    }

    # Geometric-style product. Scaled to 0..100 for readability only.
    raw = (
        factors["weakness"]
        * factors["market"]
        * factors["demand_proxy"]
        * factors["coherence"]
    )
    value = round(raw ** 0.5 * 100, 1)  # sqrt softens the product's harshness

    notes: list[str] = []
    if f.stale_share >= 0.5:
        notes.append(f"{int(f.stale_share * 10)}/10 в топе не обновлялись больше года")
    if f.no_localization_share >= 0.6:
        notes.append(f"{int(f.no_localization_share * 5)}/5 из топа без локализации")
    if f.top1_relevance < 0.5:
        notes.append("топ-1 слабо релевантен запросу — Apple не нашёл ничего лучше")
    if f.top3_reviews_median < settings.thresholds.thin_reviews:
        notes.append("топ-3 тонкий по отзывам")
    if f.paid_count > 0:
        notes.append(f"{f.paid_count}/10 в топе платные — спрос платежеспособен")

    # If no weakness threshold was crossed, say which factor is actually
    # carrying the score rather than calling a mid-scoring candidate "weak".
    if not notes:
        top = max(factors, key=lambda k: factors[k])
        label = {
            "weakness": "топ уязвим по совокупности признаков",
            "market": "ниша с подтверждённым трафиком, но топ крепкий",
            "demand_proxy": "запрос высоко в автокомплите",
            "coherence": "запрос точно совпадает с нишей",
        }[top]
        notes.append(f"{label} (ведущий фактор: {top} {factors[top]})")

    if Gate.UNBEATABLE_TOP3 in failed and len(failed) == 1:
        return Scored(f.term, f.storefront, Verdict.WATCHLIST, value,
                      [g.value for g in failed], factors,
                      notes + ["лидер силён — следим за деградацией рейтинга"])

    if failed:
        return Scored(f.term, f.storefront, Verdict.REJECTED, value,
                      [g.value for g in failed], factors, notes)

    return Scored(f.term, f.storefront, Verdict.QUEUED, value, [], factors, notes)
