"""
Turning numbers into a decision.

`score 41.6` tells you nothing on its own. Worse, it invites false precision:
41.6 is not meaningfully better than 39.3, and treating it as if it were
wastes attention. What a person actually needs is "look at this first" versus
"probably noise".

The rating answers one question: **how worthwhile is it to spend an Astro slot
on this?** Not "will this app succeed" — nothing here can know that, because
none of the free data measures demand.

Three inputs, because the raw score misses two things that matter:

  * **Cluster size.** One keyword is weak evidence; it can be a brand, a fluke
    or a dead end. Four keywords sharing a SERP means Apple sees a real
    category with several ways in — and several ways to place metadata.

  * **Brand suspicion.** A query that IS the top app's name passes every gate
    with a perfect coherence score, which is precisely what makes it
    dangerous. Those get pushed down hard.

Bands are absolute, not percentile. A relative scale would always produce
five-star entries even in a queue full of junk.
"""
from __future__ import annotations

from dataclasses import dataclass

# Score bands. Calibrated against what the formula can actually produce:
# score = sqrt(weakness * market * demand * coherence) * 100, and a genuinely
# strong combination (0.5, 0.8, 0.8, 1.0) lands near 57. Anything past 45 is
# already unusual, so the top band starts there rather than at a naive 80.
_BANDS = [(45.0, 5), (35.0, 4), (27.0, 3), (20.0, 2), (0.0, 1)]

BRAND_SUSPECT = 0.75     # name_match at or above this reads as a brand query
CLUSTER_BONUS_AT = 3     # a cluster this size earns a star
CLUSTER_PENALTY_AT = 1   # a lone keyword loses one


@dataclass
class Rating:
    stars: int
    flags: list[str]
    reason: str

    @property
    def bar(self) -> str:
        return "★" * self.stars + "☆" * (5 - self.stars)


def _base_stars(score: float) -> int:
    for floor, stars in _BANDS:
        if score >= floor:
            return stars
    return 1


def rate(
    score: float,
    cluster_size: int,
    name_match: float = 0.0,
    stale_share: float = 0.0,
    top1_relevance: float = 1.0,
) -> Rating:
    """Combine the signals into 1-5 stars plus human-readable flags."""
    stars = _base_stars(score)
    flags: list[str] = []
    reasons: list[str] = []

    if name_match >= BRAND_SUSPECT:
        # Flagged, not buried. The report routes brand queries into their own
        # section, so there is no need to crush the score as well — and doing
        # both would hide a real lead twice over.
        flags.append("бренд?")
        reasons.append("запрос совпадает с названием топ-1 — спрос на задачу есть, "
                       "но формулировку держит чужое приложение")
    elif cluster_size >= CLUSTER_BONUS_AT:
        stars += 1
        reasons.append(f"кластер из {cluster_size} запросов — ниша, а не одиночный ключ")
    elif cluster_size <= CLUSTER_PENALTY_AT:
        stars -= 1
        reasons.append("одиночный запрос — слабое свидетельство ниши")

    if stale_share >= 0.5:
        flags.append("топ устарел")
        reasons.append(f"{int(stale_share * 10)}/10 в топе не обновлялись больше года")

    if top1_relevance < 0.5 and name_match < BRAND_SUSPECT:
        flags.append("нет лидера")
        reasons.append("Apple не нашёл точного совпадения — место свободно")

    stars = max(1, min(5, stars))
    return Rating(
        stars=stars,
        flags=flags,
        reason="; ".join(reasons) if reasons else "прошёл гейты, особых сигналов нет",
    )


LEGEND = """★ = насколько стоит тратить слот Astro, не прогноз успеха
  ★★★★★  смотреть первым, скорее делать
  ★★★★☆  серьёзный кандидат, проверить в Astro
  ★★★☆☆  посмотреть конкурентов руками, потом решать
  ★★☆☆☆  низкий приоритет, вернуться если ничего лучше
  ★☆☆☆☆  скорее шум

«запр.» — сколько поисковых запросов в кластере: одна потребность,
разные формулировки. Больше запросов = шире охват метаданными.

Брендовые запросы вынесены отдельным списком: спрос на задачу
доказан, но саму формулировку занимает чужое приложение.

Флаги: «бренд?» формулировкой владеет приложение ·
«также в XX» ниша подтверждена в другом магазине ·
«топ устарел» конкуренты заброшены · «нет лидера» точного
совпадения в App Store нет"""
