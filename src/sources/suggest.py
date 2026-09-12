"""
App Store autocomplete probing.

This is the engine of the whole system. We send prefixes and Apple hands back
the queries people actually type. A pairing that nobody searches for is never
returned, so validation is free and happens upstream of us.

Caveat worth repeating: this endpoint is undocumented. It can change shape or
disappear without notice. That is exactly why `store_suggestions` persists
every raw suggestion — if the endpoint dies, the accumulated corpus keeps the
pipeline running.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from ..config import STOREFRONT_IDS
from ..seeds.verbs import KNOWN_BRAND_HINTS, PROBE_SUFFIXES
from .http import Fetcher

log = logging.getLogger(__name__)

HINTS_URL = "https://search.itunes.apple.com/WebObjects/MZSearchHints.woa/wa/hints"

_CLEAN = re.compile(r"[^a-z0-9\s\-+&']")
_WS = re.compile(r"\s+")


def normalize(term: str) -> str:
    """Lowercase, strip punctuation noise, collapse whitespace."""
    out = _CLEAN.sub(" ", term.lower().strip())
    return _WS.sub(" ", out).strip()


def _parse_hints(payload: Any) -> list[str]:
    """
    The hints endpoint has returned a few different shapes over the years.
    Handle the ones seen in the wild rather than assuming one.
    """
    if payload is None:
        return []

    # Shape A: {"hints": [{"term": "scan pdf"}, ...]}
    if isinstance(payload, dict):
        hints = payload.get("hints")
        if isinstance(hints, list):
            out = []
            for h in hints:
                if isinstance(h, dict) and "term" in h:
                    out.append(str(h["term"]))
                elif isinstance(h, str):
                    out.append(h)
            return out
        # Shape B: {"suggestions": [...]}
        sugg = payload.get("suggestions")
        if isinstance(sugg, list):
            return [str(s.get("term", s)) if isinstance(s, dict) else str(s) for s in sugg]

    # Shape C: bare list of strings or dicts
    if isinstance(payload, list):
        out = []
        for item in payload:
            if isinstance(item, dict) and "term" in item:
                out.append(str(item["term"]))
            elif isinstance(item, str):
                out.append(item)
        return out

    return []


def root_matches(root: str, term: str) -> bool:
    """
    True when every token of the root appears as a complete token of the term.

    "block" matches "block puzzle" but not "blockchain".
    "blur"  matches "blur photo"   but not "blurams".
    """
    term_tokens = set(term.replace("-", " ").split())
    return all(tok in term_tokens for tok in root.split())


def looks_like_brand(term: str) -> bool:
    """Brand queries go to ASA bidding, not ASO. Cheap token check."""
    return any(tok in KNOWN_BRAND_HINTS for tok in term.split())


async def probe(
    client: Fetcher,
    prefix: str,
    country: str = "us",
    lang: str = "en_us",
    store_id: int | None = None,
) -> list[tuple[str, int]]:
    """
    Probe one prefix. Returns (suggestion, position) pairs.

    Two things are load-bearing here and both were found the hard way:

    1. `X-Apple-Store-Front` is required. Without it the endpoint returns an
       empty `hints` array — it has no store to suggest for, and the `country`
       and `l` query params do not substitute for the header.

    2. `clientApplication=Software` pins us to the App Store index. Drop it,
       or send `iTunes`/`MacSoftware`, and Apple answers from the music
       catalogue instead: probing "scan" returns "scandal", "scandroid",
       "kyla scanlon". Those parse fine and are completely useless, so the
       failure is silent rather than loud.

    Position matters: it is our only free ordinal proxy for demand. A term
    Apple puts second for its prefix is searched more than one it puts tenth.
    Ordinal, never cardinal — do not treat it as a volume number.
    """
    sid = store_id or STOREFRONT_IDS.get(country.lower())
    headers = {"X-Apple-Store-Front": f"{sid}-1,29"} if sid else None
    if not sid:
        log.warning(
            "no storefront ID known for country=%r — the hints endpoint will "
            "return an empty list. Add it to STOREFRONT_IDS in config.py.",
            country,
        )

    payload = await client.fetch_json(
        HINTS_URL,
        {"q": prefix, "clientApplication": "Software", "country": country, "l": lang},
        headers=headers,
    )
    raw = _parse_hints(payload)
    out: list[tuple[str, int]] = []
    for pos, term in enumerate(raw, start=1):
        norm = normalize(term)
        if norm and len(norm) > 2:
            out.append((norm, pos))
    return out


async def probe_prefix_set(
    client: Fetcher,
    root: str,
    country: str = "us",
    lang: str = "en_us",
    suffixes: list[str] | None = None,
    store_id: int | None = None,
) -> dict[str, int]:
    """
    Probe a root plus its a-z extensions, reaching into the tail.

    Returns {suggestion: best_position_seen}. Best position wins because a
    term surfacing at #1 for some prefix is a stronger demand signal than the
    same term at #8 for another.
    """
    suffixes = suffixes if suffixes is not None else PROBE_SUFFIXES
    found: dict[str, int] = {}
    for suffix in suffixes:
        prefix = f"{root}{suffix}"
        for term, pos in await probe(client, prefix, country, lang, store_id):
            # Keep only suggestions where the root appears as a whole word.
            # A 4-character prefix test is far too loose: probing "block" let
            # through "blockchain", and "blur" let through "blurams" (a camera
            # brand). Both parse fine and quietly poison the funnel.
            if not root_matches(root, term):
                continue
            if term not in found or pos < found[term]:
                found[term] = pos
    return found
