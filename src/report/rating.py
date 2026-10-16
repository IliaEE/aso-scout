"""
Turning evidence into a decision.

`score 41.6` tells you nothing on its own, and invites false precision: 41.6
is not meaningfully better than 39.3. Worse, it came from a product of
weighted factors, so there was no way to check it. A formula you cannot audit
is indistinguishable from one that is wrong — and an earlier version of this
file put Life360 (three million reviews) at four stars.

Stars now count independent, checkable signals. Four stars means four of the
five below are true, and the report prints which. Every one can be verified
against the numbers in the same row.

The rating answers one question: **is this worth an Astro slot?** Not "will
this app succeed" — nothing here knows that, because none of the free data
measures search volume.
"""
from __future__ import annotations

from dataclasses import dataclass, field

@dataclass
class Signal:
    """One checkable reason a niche might be worth a slot."""
    key: str
    label: str
    met: bool
    evidence: str


BRAND_SUSPECT = 0.75


def signals(
    score: float,
    cluster_size: int,
    features: dict,
) -> list[Signal]:
    """
    Five independent checks, each readable off the row in the report.

    Stars used to come from a product of weighted factors — a number with no
    way to audit it. Counting signals instead means a four-star entry can be
    verified by looking: four of these five are true, and the report says
    which. A formula you cannot check is indistinguishable from a formula that
    is wrong.

    Deliberately independent: each one can be true while the others are false,
    so the count carries information. They are not weighted, because any
    weighting would be another unverifiable guess.
    """
    pos = features.get("suggest_pos")
    meta = features.get("in_metadata_share")
    stale = features.get("stale_share")
    relevance = features.get("top1_relevance")
    reviews = features.get("top1_reviews") or 0
    coherence = features.get("coherence")
    no_loc = features.get("no_localization_share")

    out: list[Signal] = []

    # 1. Demand. The only real signal available free: Apple orders its
    #    suggestions by how often people type them.
    demand_met = (pos is not None and pos <= 3) or (meta is not None and meta >= 0.5)
    out.append(Signal(
        "спрос", "запрос ищут",
        demand_met,
        f"автокомплит #{pos}" if pos else (
            f"в метаданных у {int(round((meta or 0) * 10))}/10" if meta is not None
            else "нет данных"),
    ))

    # 2. Room. Either the incumbents stopped shipping, or Apple has no exact
    #    match to show.
    room_met = (stale is not None and stale >= 0.4) or \
               (relevance is not None and relevance < 0.6)
    out.append(Signal(
        "место", "есть куда встать",
        room_met,
        f"{int(round((stale or 0) * 10))}/10 заброшены"
        + (", точного лидера нет" if relevance is not None and relevance < 0.6 else ""),
    ))

    # 3. Market size in the takeable band: proven traffic, not a fortress.
    market_met = 3_000 <= reviews <= 300_000
    out.append(Signal(
        "рынок", "трафик есть, лидер не крепость",
        market_met,
        f"{reviews // 1000}k отзывов у лидера" if reviews >= 1000 else f"{reviews} отзывов",
    ))

    # 4. A cluster, not a lone phrase. Several ways in means several places
    #    to put metadata, and weaker odds of it being a fluke.
    cluster_met = cluster_size >= 2
    out.append(Signal(
        "кластер", "несколько формулировок",
        cluster_met, f"{cluster_size} запросов",
    ))

    # 5. The SERP is actually about this, and nobody localized it.
    clean_met = (coherence is not None and coherence >= 0.5) or \
                (no_loc is not None and no_loc >= 0.6)
    out.append(Signal(
        "точность", "выдача по теме",
        clean_met,
        f"связность {coherence:.0%}" if coherence is not None else "нет данных",
    ))

    return out


# Legacy score bands, kept only so a stored score still orders rows within
# the same star count.
_BANDS = [(45.0, 5), (35.0, 4), (27.0, 3), (20.0, 2), (0.0, 1)]
CLUSTER_BONUS_AT = 3     # a cluster this size earns a star
CLUSTER_PENALTY_AT = 1   # a lone keyword loses one


@dataclass
class Rating:
    stars: int
    flags: list[str]
    reason: str
    signals: list["Signal"] = field(default_factory=list)

    @property
    def met_keys(self) -> list[str]:
        return [s.key for s in self.signals if s.met]

    @property
    def bar(self) -> str:
        return "★" * self.stars + "☆" * (5 - self.stars)


def _base_stars(score: float) -> int:
    for floor, stars in _BANDS:
        if score >= floor:
            return stars
    return 1


def rate(
    score: float = 0.0,
    cluster_size: int = 1,
    name_match: float = 0.0,
    stale_share: float = 0.0,
    top1_relevance: float = 1.0,
    features: dict | None = None,
) -> Rating:
    """Count the signals that hold. Stars are that count, floored at one."""
    feats = dict(features or {})
    feats.setdefault("stale_share", stale_share)
    feats.setdefault("top1_relevance", top1_relevance)

    sigs = signals(score, cluster_size, feats)
    met = [s for s in sigs if s.met]
    stars = max(1, min(5, len(met)))

    flags: list[str] = []
    if name_match >= BRAND_SUSPECT:
        flags.append("бренд?")
    if feats.get("stale_share", 0) >= 0.5:
        flags.append("топ устарел")

    missing = [s.key for s in sigs if not s.met]
    reason = ", ".join(f"{s.key}: {s.evidence}" for s in met) or "нет сильных признаков"
    if missing:
        reason += f" · не хватает: {', '.join(missing)}"

    return Rating(stars=stars, flags=flags, reason=reason, signals=sigs)


LEGEND = """★ = сколько из пяти признаков совпало. Каждый проверяется
по числам в той же строке:
  спрос     запрос в топ-3 автокомплита ИЛИ слово в метаданных у 5+/10
  место     4+/10 конкурентов заброшены ИЛИ точного лидера нет
  рынок     у лидера 3k-300k отзывов (трафик есть, но не крепость)
  кластер   2+ формулировки одной потребности
  точность  выдача действительно про этот запрос

«запр.» — сколько поисковых запросов в кластере: одна потребность,
разные формулировки. Больше запросов = шире охват метаданными.

Брендовые запросы вынесены отдельным списком: спрос на задачу
доказан, но саму формулировку занимает чужое приложение.

Флаги: «бренд?» формулировкой владеет приложение ·
«также в XX» ниша подтверждена в другом магазине ·
«топ устарел» конкуренты заброшены · «нет лидера» точного
совпадения в App Store нет"""
