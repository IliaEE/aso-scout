"""
Letting the model propose queries, not just classify them.

Until now the model only sorted what autocomplete had already found, so it
raised precision and never recall. Generation changes that — but it also
removes the guarantee the whole pipeline rests on.

That guarantee is worth stating plainly: Apple's autocomplete only returns
strings people actually type, so every candidate from it has demonstrated
demand before we ever score it. A query invented by a model has demonstrated
nothing. "scan blood pressure" is a grammatical, plausible-sounding phrase
that nobody searches.

So proposals are hypotheses, and they go through exactly the same validation
as everything else: present in autocomplete, coherent SERP, past the gates.
Only survivors enter the queue, tagged with where they came from — because if
the survival rate turns out to be near zero, this feature should be deleted
rather than believed.

Proposals are stored whatever their fate. Re-proposing the same rejected
query every night is the same failure as re-showing the same niche.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from .provider import Chat, extract_json

log = logging.getLogger(__name__)

SYSTEM = """You are an App Store keyword researcher.

Given a niche that is known to exist, propose adjacent search queries a person
might type into the App Store — queries for NEARBY needs, not restatements of
the one you were given.

Rules:
- Propose what people type, not product descriptions. "remove photo
  background", not "background removal utility application".
- 2-4 words. Single words are too broad to be useful.
- No brand names and no app names.
- Adjacent, not identical: if given "compress image", do not return
  "compress images" or "image compressor". Return neighbouring needs like
  "convert heic to jpg" or "batch resize photos".
- Do not repeat anything in the exclusion list.
- Prefer concrete tasks with an obvious finished state.

Reply with a single JSON object:

{"ideas": [{"query": "...", "why": "<short Russian reason this need exists>"}]}

Propose at most 8. Fewer good ones beats more filler — every proposal costs a
real API call to verify, and a query nobody searches wastes that call."""


@dataclass
class Proposal:
    query: str
    why: str
    source_term: str
    storefront: str


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_proposals (
            query       TEXT NOT NULL,
            storefront  TEXT NOT NULL,
            source_term TEXT,
            why         TEXT,
            proposed_at TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'pending',
            -- pending | accepted | no_demand | incoherent | gated | duplicate
            checked_at  TEXT,
            note        TEXT,
            PRIMARY KEY (query, storefront)
        )
        """
    )


def already_proposed(conn: sqlite3.Connection, storefront: str) -> set[str]:
    """
    Every query ever proposed for this storefront, whatever became of it.

    Includes rejects on purpose: a query that failed validation last week will
    fail it again this week, and proposing it nightly is exactly the loop the
    decisions table was added to stop.
    """
    ensure_table(conn)
    return {
        r["query"]
        for r in conn.execute(
            "SELECT query FROM llm_proposals WHERE storefront = ?", (storefront,)
        )
    }


def known_terms(conn: sqlite3.Connection, storefront: str) -> set[str]:
    return {
        r["term"]
        for r in conn.execute(
            "SELECT term FROM keywords WHERE storefront = ?", (storefront,)
        )
    }


def save_proposal(
    conn: sqlite3.Connection, p: Proposal, status: str = "pending"
) -> None:
    ensure_table(conn)
    conn.execute(
        "INSERT INTO llm_proposals (query, storefront, source_term, why, "
        "proposed_at, status) VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(query, storefront) DO NOTHING",
        (p.query, p.storefront, p.source_term, p.why,
         datetime.now(timezone.utc).isoformat(), status),
    )


def mark(
    conn: sqlite3.Connection, query: str, storefront: str,
    status: str, note: str = "",
) -> None:
    conn.execute(
        "UPDATE llm_proposals SET status = ?, checked_at = ?, note = ? "
        "WHERE query = ? AND storefront = ?",
        (status, datetime.now(timezone.utc).isoformat(), note, query, storefront),
    )


def hit_rate(conn: sqlite3.Connection) -> dict[str, int]:
    """
    How proposals actually fared.

    The number that matters: if `accepted` stays near zero, the model is
    inventing queries nobody searches and this layer should be switched off
    rather than trusted because it feels clever.
    """
    ensure_table(conn)
    out: dict[str, int] = {}
    for r in conn.execute(
        "SELECT status, COUNT(*) AS n FROM llm_proposals GROUP BY status"
    ):
        out[r["status"]] = r["n"]
    return out


def build_prompt(
    source_term: str,
    storefront: str,
    apps: list[dict],
    exclude: set[str],
) -> str:
    lines = [
        f'Known niche: "{source_term}"',
        f"Storefront: {storefront.upper()}",
        "",
        "Apps currently serving it:",
    ]
    for a in apps[:5]:
        lines.append(f'  - "{str(a.get("title", "?"))[:60]}"')

    # Excluding what we already have is what keeps this from proposing the
    # same twenty queries every night.
    near = sorted(t for t in exclude if any(w in t for w in source_term.split()))
    if near:
        lines.append("")
        lines.append("Do NOT propose these, we already have them:")
        lines.append("  " + ", ".join(near[:40]))

    lines.append("")
    lines.append("Propose adjacent queries.")
    return "\n".join(lines)


def parse(raw: str | None, source_term: str, storefront: str) -> list[Proposal]:
    data = extract_json(raw or "")
    if not isinstance(data, dict):
        return []
    items = data.get("ideas")
    if not isinstance(items, list):
        return []

    out: list[Proposal] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        q = str(item.get("query", "")).strip().lower()
        # Reject what the rules already forbade rather than trusting the model
        # to have followed them.
        if not q or len(q.split()) < 2 or len(q) > 60 or q in seen:
            continue
        seen.add(q)
        out.append(Proposal(
            query=q,
            why=str(item.get("why", ""))[:160],
            source_term=source_term,
            storefront=storefront,
        ))
    return out[:8]


def generate(
    chat: Chat,
    source_term: str,
    storefront: str,
    apps: list[dict],
    exclude: set[str],
) -> list[Proposal]:
    raw = chat.complete(SYSTEM, build_prompt(source_term, storefront, apps, exclude))
    proposals = parse(raw, source_term, storefront)
    return [p for p in proposals if p.query not in exclude]
