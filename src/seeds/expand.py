"""
Self-expanding lexicon.

The trick that makes this work without any language model: we already know the
parse, because we know what we probed. When the root "scan" returns
"scan pdf", the object is whatever follows the root. No POS tagger required.

Discovering new *verbs* is a frequency question, not a grammar question. A
token that shows up in first position across many distinct suggestions is
behaving like a verb in App Store search, whatever a dictionary says about it.
That is a deterministic rule over collected data — no LLM in the loop.

spacy is deliberately not a dependency. It would add ~500MB for a parse we get
free from lineage, and its notion of "verb" is about English, not about how
people search an app store.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict

import inflect

from .verbs import TOO_BROAD

log = logging.getLogger(__name__)

_inflect = inflect.engine()

# Tokens that carry no niche meaning and must never become probes.
STOPWORDS: set[str] = {
    "a", "an", "the", "and", "or", "for", "to", "of", "in", "on", "with",
    "my", "your", "me", "it", "is", "are", "how", "app", "apps", "free",
    "best", "pro", "plus", "premium", "new", "top", "online", "offline",
    "easy", "simple", "fast", "quick", "all", "any", "no", "not",
}


def singularize(token: str) -> str:
    """
    "documents" -> "document". Keeps the dedup layer from treating plural and
    singular forms as separate niches.

    inflect returns False when the word is already singular, which is why the
    guard is explicit rather than a truthiness check on a str.
    """
    # inflect raises on empty input, and stripping punctuation upstream can
    # legitimately produce an empty token.
    if not token or not token.isascii():
        return token
    try:
        result = _inflect.singular_noun(token)
    except Exception:
        return token
    return result if isinstance(result, str) and result else token


def tokens_of(term: str) -> list[str]:
    return [t for t in term.split() if t]


def content_tokens(term: str) -> list[str]:
    """Meaningful tokens: no stopwords, singularized."""
    return [singularize(t) for t in tokens_of(term) if t not in STOPWORDS]


def extract_objects(root: str, suggestions: list[str]) -> Counter[str]:
    """
    Pull the object side out of suggestions generated from a known root.

    Because `root` was the probe, everything that is not the root is the
    object. "scan" + "scan pdf to word" yields the phrase tokens pdf, word.
    """
    root_tokens = set(tokens_of(root))
    objects: Counter[str] = Counter()

    for sugg in suggestions:
        remainder = [
            singularize(t)
            for t in tokens_of(sugg)
            if t not in root_tokens and t not in STOPWORDS
        ]
        for tok in remainder:
            if len(tok) < 3 or tok.isdigit():
                continue
            objects[tok] += 1

    return objects


def discover_verbs(
    suggestions: list[str],
    known_verbs: set[str],
    min_distinct: int = 3,
) -> list[str]:
    """
    Find tokens acting as verbs: frequent in first position across distinct
    suggestions, and not already known.

    `min_distinct` is the guard against noise. One suggestion starting with a
    word proves nothing; three separate ones is a pattern.
    """
    leading: defaultdict[str, set[str]] = defaultdict(set)

    for sugg in suggestions:
        toks = tokens_of(sugg)
        if len(toks) < 2:
            continue
        head = singularize(toks[0])
        if head in STOPWORDS or head in known_verbs or len(head) < 3:
            continue
        leading[head].add(sugg)

    return sorted(
        head for head, seen in leading.items() if len(seen) >= min_distinct
    )


def next_probes(
    objects: Counter[str],
    already_probed: set[str],
    min_count: int = 2,
    limit: int = 50,
) -> list[str]:
    """
    Choose which harvested objects become the next round of probes.

    Objects seen only once are usually long-tail noise; requiring two
    sightings keeps the frontier from exploding. TOO_BROAD terms are excluded
    because they make useless cluster heads (see pipeline/cluster.py).
    """
    ranked = [
        obj
        for obj, count in objects.most_common()
        if count >= min_count
        and obj not in already_probed
        and obj not in TOO_BROAD
        and obj not in STOPWORDS
    ]
    return ranked[:limit]
