"""
A labelled test set for the judge.

Twelve identical verdicts at 95% confidence prove nothing on their own: the
sample may simply have been twelve niches. The only way to tell a working
classifier from a rubber stamp is to hand it cases where the answer is already
known and count the misses.

Every case below is real, taken from live runs of this tool against the US,
GB, CA and AU stores. Review counts are approximate — what matters is the
relationship between the query and the name of the app Apple returns first.

Run it with:  python -m src.jobs.judge --selftest
"""
from __future__ import annotations

from typing import Any

# (query, expected_category, top-1 title, top-1 seller, rating, reviews)
CASES: list[tuple[str, str, str, str, float, int]] = [
    # --- brands: the query IS an app's name, however it is spelled ---------
    ("live 360", "brand", "Life360: Family Safety & GPS", "Life360", 4.8, 3_058_000),
    ("sign now", "brand", "SignNow: e-Signature app", "airSlate", 4.5, 17_949),
    ("pet desk", "brand", "PetDesk", "PetDesk", 4.8, 29_000),
    ("now", "brand", "NOW: TV, Movies, Live Sports", "Sky UK", 4.5, 125_000),
    ("saver", "brand", "Savers", "Savers Health & Beauty", 4.9, 5_000),
    ("rename", "brand", "Remini - AI Photo Enhancer", "Bending Spoons", 4.6, 352_000),
    ("dual mobile app", "brand", "Duo Mobile", "Cisco", 4.9, 2_288_000),
    ("smarter player lite", "brand", "Smarters Player Lite", "WHMCS SMARTERS", 4.8, 443_000),
    ("count the kicks", "brand", "Count the Kicks!", "Healthy Birth Day", 4.8, 2_000),
    ("translate now", "brand", "Translate Now - AI Translator", "Codeway", 4.6, 18_000),

    # --- too broad: one generic word, results are a grab bag ---------------
    ("split", "too_broad", "Splitwise", "Splitwise", 4.8, 120_000),
    ("sync", "too_broad", "Ford\u2122", "Ford Motor Company", 4.6, 40_000),
    ("mute", "too_broad", "Meitu AI Photo & Video Editor", "Meitu", 4.5, 60_000),

    # --- real niches: a function several apps compete to serve -------------
    ("compress image", "niche", "Compress Photos & Pictures", "Anuj Chaudhary", 4.7, 22_460),
    ("resize image", "niche", "Image Size", "Ruler Tech", 4.7, 29_773),
    ("scan pokemon cards", "niche", "Collectr - TCG Collector App", "Collectr", 4.9, 64_590),
    ("remove background", "niche", "Magic Eraser Background Editor", "Brandon Baars", 4.6, 90_000),
    ("blur face in video", "niche", "Blur-Video", "Shenzhen Weiyun", 4.5, 12_000),
    ("convert video to mp3", "niche", "Media Converter - video to mp3", "Fifth Lane", 4.6, 30_000),
    ("calculate work hours", "niche", "Hours Tracker: Time Calculator", "Carpe Diem", 4.7, 25_000),
]


def case_apps(title: str, seller: str, rating: float, reviews: int) -> list[dict[str, Any]]:
    """Minimal SERP shaped like a real one, enough for the prompt."""
    return [
        {"title": title, "seller": seller, "rating": rating,
         "rating_count": reviews, "price": 0.0},
        {"title": "Another App", "seller": "Someone Else", "rating": 4.2,
         "rating_count": max(reviews // 20, 100), "price": 0.0},
        {"title": "Third Option", "seller": "Third Dev", "rating": 4.0,
         "rating_count": max(reviews // 60, 50), "price": 0.0},
    ]


def summarize(results: list[tuple[str, str, str]]) -> str:
    """results: (query, expected, got)"""
    by_class: dict[str, list[tuple[str, str]]] = {}
    for query, expected, got in results:
        by_class.setdefault(expected, []).append((query, got))

    lines: list[str] = []
    total_hit = 0
    for expected in ("brand", "too_broad", "niche"):
        rows = by_class.get(expected, [])
        if not rows:
            continue
        hit = sum(1 for _, got in rows if got == expected)
        total_hit += hit
        lines.append(f"\n{expected}: {hit}/{len(rows)}")
        for query, got in rows:
            mark = "  ok  " if got == expected else " MISS "
            lines.append(f"  [{mark}] {query:24} -> {got}")

    total = len(results)
    lines.append(f"\n{'=' * 52}")
    lines.append(f"итого: {total_hit}/{total} верно")

    # A judge that answers "niche" to everything scores 7/20 here, which is
    # roughly what guessing the majority class gets. Anything near that means
    # the layer is not earning its tokens.
    niche_only = sum(1 for _, e, _ in results if e == "niche")
    lines.append(f"ответ 'niche' на всё дал бы: {niche_only}/{total}")
    if total_hit <= niche_only:
        lines.append("\nСудья не превосходит слепое угадывание. "
                     "Либо смените модель, либо отключите слой.")
    elif total_hit >= total * 0.8:
        lines.append("\nСудья различает классы — слой оправдан.")
    else:
        lines.append("\nЧастичная польза: проверьте промахи выше, "
                     "возможно поможет модель посильнее.")
    return "\n".join(lines)
