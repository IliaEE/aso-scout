"""
Pulling the need out of a branded search term.

Apple's search-term data is full of entries shaped like a brand plus a
description of what it does:

    be my eyes – helping the blind
    withu: workout & fitness app
    neateat - meal planner
    compass school manager

The brand half is useless to us — somebody owns it. The tail is not: it is a
plain-language statement of a need, written by the people searching for it.
"helping the blind", "meal planner", "workout app" are categories, and they
arrive already validated as things humans type.

This is a cheap source of candidates that costs no extra API calls: the terms
are already in the database, discarded as brands. The tails still have to
clear the same gates as anything else — a tail is a hypothesis, not a finding.
"""
from __future__ import annotations

import re

# Separators Apple's terms actually use between a name and its description.
SPLIT = re.compile(r"\s*[–—\-:|,]\s+|\s+[–—]\s*|:\s*")

# Words that make a tail a description of an app rather than of a need.
FILLER = {
    "app", "apps", "official", "mobile", "free", "pro", "plus", "premium",
    "lite", "hd", "online", "the", "a", "an", "и", "для",
}

# A tail of only these is a category label, not a need worth probing.
TOO_GENERIC = {
    "app", "apps", "game", "games", "tv", "music", "news", "shop", "store",
    "bank", "banking", "canada", "uk", "usa", "australia", "us",
}


def clean(part: str) -> str:
    out = re.sub(r"[^\w\s&+]", " ", part.lower())
    return re.sub(r"\s+", " ", out).strip()


def tail_of(term: str) -> str | None:
    """
    The descriptive half of a branded term, or None when there isn't one.

    Returns the part after the separator, stripped of filler. Rejects tails
    that are purely generic ("official app", "canada") because probing them
    would just rediscover the brand or a country.
    """
    parts = [clean(p) for p in SPLIT.split(term) if clean(p)]
    if len(parts) < 2:
        return None

    # The description is whatever follows the first separator; if a term has
    # several, the longest remaining chunk carries the most meaning.
    tail = max(parts[1:], key=lambda p: len(p.split()))

    # Strip filler from the edges only. Removing it everywhere turned
    # "helping the blind" into "helping blind" — the article carries no
    # meaning on its own but the phrase is what people actually type.
    words = tail.split()
    while words and words[0] in FILLER:
        words.pop(0)
    while words and words[-1] in FILLER:
        words.pop()
    if not words:
        return None
    if len(words) < 2:
        # One meaningful word only survives if it is not a bare category.
        if words[0] in TOO_GENERIC or len(words[0]) < 4:
            return None
    if all(w in TOO_GENERIC for w in words):
        return None

    out = " ".join(words)
    # A tail identical to the head is not a description of anything.
    head_words = parts[0].split()
    while head_words and head_words[0] in FILLER:
        head_words.pop(0)
    while head_words and head_words[-1] in FILLER:
        head_words.pop()
    head = " ".join(head_words)
    return None if out == head else out


def tails_from(terms: list[str]) -> dict[str, str]:
    """{tail: the term it came from} for every term that yields one."""
    out: dict[str, str] = {}
    for t in terms:
        tail = tail_of(t)
        if tail and tail not in out:
            out[tail] = t
    return out
