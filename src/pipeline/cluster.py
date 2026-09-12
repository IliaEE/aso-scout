"""
Clustering by SERP overlap.

Two keywords belong to one niche when Apple shows the same apps for both.
That is it. No embeddings, no semantics, no LLM — we borrow Apple's own
judgement about what these queries mean.

    noise meter    n decibel meter    = 8/10  -> one cluster
    noise meter    n db meter         = 9/10  -> one cluster
    noise meter    n noise cancelling = 0/10  -> different niches

The failure mode is broad single-word terms: the top-10 for "fitness" is a
grab bag that overlaps with everything, so it would glue unrelated clusters
together. Hence `min_tokens_for_head` and the TOO_BROAD list — broad terms can
join a cluster but never lead one, and never seed a merge.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..config import settings
from ..seeds.verbs import TOO_BROAD

log = logging.getLogger(__name__)


def app_set(apps: list[dict[str, Any]], depth: int = 10) -> set[int]:
    """Track IDs of the top-N, used as the fingerprint of a SERP."""
    return {a["track_id"] for a in apps[:depth] if a.get("track_id")}


def overlap(a: set[int], b: set[int]) -> float:
    """Jaccard-style overlap, normalized by the smaller set."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def eligible_head(term: str) -> bool:
    """
    Can this term lead a cluster?

    Broad or single-token terms cannot: their SERPs are unfocused, and a
    cluster named after them would absorb unrelated keywords.
    """
    toks = term.split()
    if len(toks) < settings.thresholds.min_tokens_for_head:
        return False
    if term in TOO_BROAD:
        return False
    # A term made entirely of broad tokens is still broad.
    return not all(t in TOO_BROAD for t in toks)


@dataclass
class Cluster:
    head: str
    members: list[str] = field(default_factory=list)
    serp: set[int] = field(default_factory=set)

    @property
    def size(self) -> int:
        return len(self.members)


def top1_of(apps: list[dict[str, Any]]) -> int | None:
    """Track ID of the #1 result, or None."""
    return apps[0].get("track_id") if apps else None


class _Union:
    """Minimal union-find for single-linkage grouping."""

    def __init__(self, items: list[str]) -> None:
        self.parent = {i: i for i in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_clusters(
    serps: dict[str, list[dict[str, Any]]],
    threshold: float | None = None,
) -> list[Cluster]:
    """
    Group keywords into niches by SERP overlap, using single linkage.

    `serps` maps keyword -> ordered top-10 app records.

    Earlier this compared every term against the *head* only, which split
    obvious single niches: on a live run "compress image" and "compress
    photos" landed in separate clusters despite sharing the same #1 app,
    because their pairwise overlap sat just under the threshold while each
    overlapped a third variant. Single linkage fixes that — A-B and B-C become
    one cluster even when A-C falls short.

    Two terms are linked when either:
      * their top-10 overlap reaches the threshold, or
      * they share the same #1 result AND still show some overlap. The
        leader alone is not enough: one popular utility can top two unrelated
        queries, and merging on that would chain distinct niches together.

    Single linkage can chain: a long ladder of loosely-related terms may merge
    into one large cluster. Broad terms are the usual culprit, which is why
    TOO_BROAD terms are excluded from linking (they still get reported, just
    never glue other niches together).
    """
    thr = threshold if threshold is not None else settings.thresholds.cluster_overlap

    fingerprints = {term: app_set(apps) for term, apps in serps.items()}
    leaders = {term: top1_of(apps) for term, apps in serps.items()}
    terms = sorted(t for t in fingerprints if fingerprints[t])

    # Broad terms take part as members but never act as linking bridges.
    def links_allowed(t: str) -> bool:
        toks = t.split()
        return not all(tok in TOO_BROAD for tok in toks)

    uf = _Union(terms)
    for i, a in enumerate(terms):
        for b in terms[i + 1:]:
            if not (links_allowed(a) and links_allowed(b)):
                continue
            ov = overlap(fingerprints[a], fingerprints[b])
            same_leader = (
                leaders[a] is not None
                and leaders[a] == leaders[b]
                and ov >= settings.thresholds.same_leader_min_overlap
            )
            if same_leader or ov >= thr:
                uf.union(a, b)

    groups: dict[str, list[str]] = {}
    for t in terms:
        groups.setdefault(uf.find(t), []).append(t)

    clusters: list[Cluster] = []
    for members in groups.values():
        # Prefer an eligible head with the broadest SERP; fall back to the
        # first member so nothing is dropped.
        ranked = sorted(members, key=lambda t: (-len(fingerprints[t]), t))
        head = next((m for m in ranked if eligible_head(m)), ranked[0])
        serp: set[int] = set()
        for m in members:
            serp |= fingerprints[m]
        clusters.append(Cluster(head=head, members=sorted(members), serp=serp))

    # Terms with an empty SERP never entered linking; keep them visible.
    for term in sorted(fingerprints):
        if not fingerprints[term]:
            clusters.append(Cluster(head=term, members=[term], serp=set()))

    clusters.sort(key=lambda c: (-c.size, c.head))
    return clusters
