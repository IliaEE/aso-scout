"""
Coherence and deduplication.

Coherence answers "is this query a real niche?" without understanding
language. Method: ask Apple for the top-10 and measure how many of those apps
put the query's words in their own title. A real niche has apps named after
it — "scan pdf" surfaces "PDF Scanner", "Scan to PDF". A nonsense query
returns a grab bag with near-zero token overlap.

This is the filter that makes verb x object safe. We never have to judge
whether "scan blood pressure" makes sense: Apple returns junk for it, overlap
collapses, and it is dropped on a number.
"""
from __future__ import annotations

import logging
from typing import Any

from rapidfuzz import fuzz

from ..config import settings
from ..seeds.expand import content_tokens, singularize, tokens_of

log = logging.getLogger(__name__)


def coherence_score(term: str, apps: list[dict[str, Any]]) -> float:
    """
    Share of the top-10 whose title (plus subtitle proxy) contains the
    query's content tokens.

    Returns 0.0 for an empty SERP, which correctly fails the gate — a keyword
    Apple returns nothing for is not a keyword we can rank on.
    """
    if not apps:
        return 0.0

    query_tokens = set(content_tokens(term))
    if not query_tokens:
        return 0.0

    hits = 0
    for app in apps:
        haystack = f"{app.get('title', '')} {app.get('subtitle_proxy', '')}".lower()
        haystack_tokens = {singularize(t) for t in tokens_of(haystack)}
        # Partial credit is not useful here; we want "does this app claim the
        # niche", so require the majority of query tokens to be present.
        matched = len(query_tokens & haystack_tokens)
        if matched >= max(1, len(query_tokens) - 1):
            hits += 1

    return hits / len(apps)


def top1_relevance(term: str, apps: list[dict[str, Any]]) -> float:
    """
    How well the #1 result matches the query, 0..1.

    A low value on a high-demand term is a strong opportunity signal: it means
    Apple could not find anything genuinely relevant and is showing the best
    available approximation. That is a gap you can fill.
    """
    if not apps:
        return 0.0
    title = f"{apps[0].get('title', '')} {apps[0].get('subtitle_proxy', '')}"
    return fuzz.token_set_ratio(term, title.lower()) / 100.0


def dedup_terms(terms: list[str], ratio: int | None = None) -> dict[str, str]:
    """
    Collapse spelling variants onto a canonical form.

    "noise meter" / "noisemeter" / "noise metre" are one keyword. Returns
    {variant: canonical}. The canonical is the shortest surviving form, which
    is usually the one people actually type.
    """
    threshold = ratio if ratio is not None else settings.thresholds.fuzzy_dedup_ratio
    canonical: dict[str, str] = {}
    groups: list[list[str]] = []

    def squash(t: str) -> str:
        """Compare on singularized, space-free forms: 'noise metre' ~ 'noisemeter'."""
        return "".join(singularize(tok) for tok in tokens_of(t))

    # Shortest first so canonical forms are the compact ones.
    for term in sorted(set(terms), key=lambda t: (len(t), t)):
        placed = False
        for group in groups:
            # Compare against ANY member, not just the head. Comparing only to
            # the head loses transitive matches: a -> b and b -> c where a -> c
            # falls just under the threshold.
            if any(fuzz.ratio(squash(term), squash(member)) >= threshold
                   for member in group):
                group.append(term)
                placed = True
                break
        if not placed:
            groups.append([term])

    for group in groups:
        head = group[0]
        for member in group:
            canonical[member] = head

    return canonical
