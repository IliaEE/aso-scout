"""
The human-facing report.

Deliberately tiny: three candidates, one line of "why", one line of "what to
build". Everything else is available on tap but not pushed. A 50-row table is
an internal screen, not a report — it does not help anyone decide.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from ..config import settings
from ..pipeline.cluster import build_clusters, eligible_head


@dataclass
class ClusterCandidate:
    """
    One niche, not one keyword.

    The live run filled all three report slots with "compress image",
    "compress photos" and "compress photos & pictures" — three spellings of a
    single niche. cluster.py existed from the start but was never wired in,
    so the report was ranking variants against each other.
    """
    head: "Candidate"
    variants: list["Candidate"] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.head.score

    @property
    def size(self) -> int:
        return 1 + len(self.variants)


@dataclass
class Candidate:
    term: str
    storefront: str
    score: float
    notes: list[str]
    factors: dict[str, float]
    apps: list[dict[str, Any]]
    suggest_pos: int | None

    @property
    def why(self) -> str:
        if self.notes:
            return self.notes[0]
        # No weakness signal fired. Saying "разрыв между спросом и качеством
        # топа" here would assert something we never measured, so say what is
        # actually true: it survived the gates and nothing more.
        return "прошёл гейты, но явных признаков слабости топа нет — слабый кандидат"

    @property
    def band(self) -> str:
        if self.score >= 60:
            return "сильный"
        if self.score >= 35:
            return "средний"
        return "слабый"

    @property
    def leader(self) -> dict[str, Any] | None:
        return self.apps[0] if self.apps else None


def load_queue(conn: sqlite3.Connection, limit: int | None = None) -> list[Candidate]:
    """Latest score per keyword, queued only, best first."""
    n = limit or settings.queue_size
    rows = conn.execute(
        """
        SELECT k.term, k.storefront, k.suggest_pos,
               s.score, s.notes, s.factors,
               (SELECT raw FROM serp_snapshots sp
                 WHERE sp.keyword_id = k.id
                 ORDER BY sp.taken_at DESC LIMIT 1) AS serp
        FROM keywords k
        JOIN keyword_scores s ON s.id = (
            SELECT id FROM keyword_scores
             WHERE keyword_id = k.id ORDER BY scored_at DESC LIMIT 1
        )
        WHERE k.status = 'queued'
        ORDER BY s.score DESC
        LIMIT ?
        """,
        (n,),
    ).fetchall()

    out = []
    for r in rows:
        out.append(
            Candidate(
                term=r["term"],
                storefront=r["storefront"],
                score=r["score"],
                notes=json.loads(r["notes"] or "[]"),
                factors=json.loads(r["factors"] or "{}"),
                apps=json.loads(r["serp"] or "[]"),
                suggest_pos=r["suggest_pos"],
            )
        )
    return out


def group_into_clusters(candidates: list[Candidate]) -> list[ClusterCandidate]:
    """
    Collapse keyword candidates into niches by SERP overlap.

    The head of each cluster is its highest-scoring member, which is not
    necessarily the head cluster.py picked (that one optimises for SERP
    breadth; here we want the best candidate to lead the report).
    """
    by_term = {c.term: c for c in candidates}
    serps = {c.term: c.apps for c in candidates if c.apps}

    out: list[ClusterCandidate] = []
    seen: set[str] = set()

    for cl in build_clusters(serps):
        members = [by_term[t] for t in cl.members if t in by_term and t not in seen]
        if not members:
            continue
        members.sort(key=lambda c: -c.score)

        # A cluster led by a broad single word ("translate") is reported under
        # its best specific member instead.
        head = next((m for m in members if eligible_head(m.term)), members[0])
        rest = [m for m in members if m is not head]

        for m in members:
            seen.add(m.term)
        out.append(ClusterCandidate(head=head, variants=rest))

    # Candidates with no stored SERP never entered clustering.
    for c in candidates:
        if c.term not in seen:
            out.append(ClusterCandidate(head=c))

    out.sort(key=lambda c: -c.score)
    return out


def find_alerts(conn: sqlite3.Connection, days: int = 30) -> list[str]:
    """
    Decay signals on watchlist leaders.

    This is the only place absolute numbers would mislead and deltas tell the
    truth: a leader with 90k reviews looks unbeatable until you notice the
    rating has been sliding for a month.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    rows = conn.execute(
        """
        SELECT a.track_id, a.title, a.storefront,
               MIN(m.day) AS d0, MAX(m.day) AS d1
        FROM apps a JOIN app_metrics_daily m
          ON m.track_id = a.track_id AND m.storefront = a.storefront
        WHERE m.day >= ?
        GROUP BY a.track_id, a.storefront
        HAVING COUNT(*) >= 3
        """,
        (since,),
    ).fetchall()

    alerts: list[str] = []
    for r in rows:
        pair = conn.execute(
            "SELECT day, rating, rating_count FROM app_metrics_daily "
            "WHERE track_id = ? AND storefront = ? AND day IN (?, ?) ORDER BY day",
            (r["track_id"], r["storefront"], r["d0"], r["d1"]),
        ).fetchall()
        if len(pair) < 2:
            continue
        old, new = pair[0], pair[-1]
        if old["rating"] and new["rating"] and (old["rating"] - new["rating"]) >= 0.3:
            alerts.append(
                f'{r["title"]} ({r["storefront"].upper()}): рейтинг '
                f'{old["rating"]:.1f} → {new["rating"]:.1f} — окно на перехват'
            )
    return alerts


def render_text(
    candidates: list[Candidate], alerts: list[str], history_weeks: int
) -> str:
    """Plain-text weekly report. Same structure the dashboard renders."""
    week = datetime.now(timezone.utc).isocalendar().week
    date = datetime.now(timezone.utc).strftime("%d.%m.%Y")

    lines = [f"ASO SCOUT · отчёт за неделю {week} · {date}", "=" * 58, ""]

    if history_weeks < 4:
        lines += [
            f"⚠ истории накоплено {history_weeks} нед. — дельты и оценки",
            "  установок пока ненадёжны, абсолютные цифры уже можно читать",
            "",
        ]

    if alerts:
        lines.append("АЛЕРТЫ")
        for a in alerts[:5]:
            lines.append(f"  • {a}")
        lines.append("")

    if not candidates:
        lines += ["Кандидатов нет.", "",
                  "Это нормально на первых прогонах: либо история ещё не набрана,",
                  "либо гейты отсекли всё. Проверьте счётчики в логе сбора."]
        return "\n".join(lines)

    clusters = group_into_clusters(candidates)
    lines.append(f"НИШИ ({len(clusters)}) из {len(candidates)} запросов")
    lines.append("")

    for i, cl in enumerate(clusters[: settings.report_size], start=1):
        c = cl.head
        leader = c.leader
        lines.append(
            f'{i}. "{c.term}" · {c.storefront.upper()} · '
            f'score {c.score} ({c.band})'
        )
        lines.append(f"   {c.why}")
        if leader:
            lines.append(
                f'   Лидер: {leader.get("title", "?")} '
                f'{leader.get("rating") or 0:.1f}★ '
                f'{leader.get("rating_count") or 0} отзывов'
                + ("  ·  платный" if leader.get("is_paid") else "")
            )
        for note in c.notes[1:3]:
            lines.append(f"   {note}")
        if cl.variants:
            shown = ", ".join(v.term for v in cl.variants[:4])
            more = f" (+{len(cl.variants) - 4})" if len(cl.variants) > 4 else ""
            lines.append(f"   Кластер ({cl.size} запросов): {shown}{more}")
        lines.append("   [промпт кластера]  [в Astro]  [отклонить]")
        lines.append("")

    lines += ["-" * 58,
              "Спрос здесь косвенный. Точные цифры даст только Astro",
              "на шаге подтверждения."]
    return "\n".join(lines)
