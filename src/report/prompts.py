"""
Prompt builders for the two manual steps.

These are string templates with substitution — no model call, no cost. The
server assembles evidence it already collected; the human pastes it into
Claude.

Prompt 1 (cluster expansion) hands over autocomplete suggestions, competitor
metadata tokens and review pains, so Claude groups and prioritizes real
material instead of inventing a cluster from one word.

Prompt 2 (verdict) carries the actual Astro numbers and asks for a decision.
"""
from __future__ import annotations

from typing import Any

from ..config import settings


def build_cluster_prompt(
    term: str,
    storefront: str,
    apps: list[dict[str, Any]],
    suggestions: list[str],
    metadata_tokens: list[tuple[str, int]],
    pains: list[tuple[str, int]] | None = None,
    slot_budget: int | None = None,
) -> str:
    """Prompt 1: expand a head keyword into a cluster ready to paste into Astro."""
    budget = slot_budget or min(settings.astro_slots // 2, 25)

    serp_lines = []
    for i, a in enumerate(apps[:10], start=1):
        upd = a.get("days_since_update")
        upd_s = f"upd {upd // 30}mo" if isinstance(upd, int) else "upd n/a"
        serp_lines.append(
            f'{i:2}. {a.get("title", "?")[:48]:<50} '
            f'{a.get("rating") or 0:.1f}* {a.get("rating_count") or 0:>7} {upd_s}'
        )

    meta_s = ", ".join(tok for tok, _ in metadata_tokens[:20])
    sugg_s = "\n ".join(suggestions[:25]) if suggestions else "(нет данных)"
    pains_s = (
        "\n ".join(f"{p} ({c})" for p, c in (pains or [])[:8])
        if pains else "(отзывы ещё не собраны)"
    )

    return f"""Задача: собрать поисковый кластер для оценки ниши в App Store.

КОНТЕКСТ (собран автоматически, не выдуман)

Головной запрос: {term}
Сторфронт: {storefront}

Топ-10 выдачи:
{chr(10).join(serp_lines)}

Подсказки автокомплита, где встречается ключ:
 {sugg_s}

Частые слова в метаданных топа:
 {meta_s}

Боли из отзывов 1-3*:
 {pains_s}

ЗАДАНИЕ

1. Собери кластер запросов вокруг головного. Оси расширения:
   - синонимы функции
   - единицы измерения и жаргон ниши
   - сценарии использования
   - формулировки с "app", "free", "online"
   - бытовой и профессиональный регистр отдельно

2. Помечай каждый запрос:
   HEAD     - голова кластера, основной трафик
   BODY     - та же потребность, другая формулировка
   TAIL     - узкий сценарий, низкий спрос но высокий интент
   BRAND    - бренд конкурента, для ASA а не для ASO
   ADJACENT - соседняя потребность, отдельное приложение, не смешивать

3. Отбрось запросы, означающие другую потребность.

4. Отдай максимум {budget} запросов, отсортированных по ожидаемой ценности.
   У меня ограничен бюджет слотов в трекере.

5. Отдельно предложи 2-3 варианта, как эту потребность могли бы называть,
   если бы стандарт формулировки ещё не сложился. Автокомплит такие не
   показывает, и это единственный способ их найти.

6. Отдельно: 3-5 гипотез позиционирования на основе болей выше.

ФОРМАТ ОТВЕТА

Сначала список для вставки в трекер - по одному запросу на строку,
без нумерации и комментариев, только HEAD/BODY/TAIL, не больше {budget}.

Затем таблица: запрос | тип | почему включён.

Затем блок BRAND. Затем блок ADJACENT. Затем гипотезы позиционирования.
"""


def build_verdict_prompt(
    head: str,
    storefront: str,
    facts: list[dict[str, Any]],
    pains: list[tuple[str, int]] | None = None,
) -> str:
    """Prompt 2: decide GO / NO-GO from real Astro numbers."""
    lines = []
    for f in facts:
        lines.append(
            f'{f.get("term", "?")[:28]:<30} '
            f'{(f.get("kw_type") or "?"):<6} '
            f'{f.get("popularity") if f.get("popularity") is not None else "-":>6} '
            f'{f.get("difficulty") if f.get("difficulty") is not None else "-":>6}   '
            f'{f.get("top1") or "-"}'
        )

    pains_s = (
        "\n ".join(f"{p} ({c})" for p, c in (pains or [])[:8])
        if pains else "(нет данных)"
    )

    return f"""Задача: решить, брать нишу в разработку или нет.

Кластер "{head}" / {storefront}. Данные Astro:

запрос                         тип     pop    diff   топ-1
{chr(10).join(lines)}

Боли из отзывов лидеров:
 {pains_s}

ВАЖНО: шкала popularity у Apple нелинейная. Суммировать значения нельзя.
Смотри на голову кластера и на то, сколько вариантов перешагнули порог.

Вопросы:
1. Достаточен ли спрос кластера? Или спрос только у одной формулировки,
   то есть это случайный запрос, а не ниша?
2. Не держит ли один лидер весь кластер? Если да, насколько он уязвим.
3. Какие 5-8 запросов брать в метаданные первыми.
4. Что приложение обязано уметь, чтобы забрать эти запросы, с опорой
   на боли из отзывов.
5. Вердикт, ровно один из:
   GO           - берём
   NO-GO        - не берём, с причиной
   СУЗИТЬ       - голова мёртвая, но живой TAIL, назови новую голову
   ДРУГОЙ СТОР  - спрос есть, но этот сторфронт занят, назови какой пробовать
   ADJACENT     - нашлась более интересная соседняя потребность, назови её

Если данных не хватает для вывода, скажи прямо, чего не хватает,
а не выдумывай цифры.
"""
