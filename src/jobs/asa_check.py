"""
Verify the Apple Ads credentials, one step at a time.

    python -m src.jobs.asa_check

Each step is checked separately because the failures look identical from the
outside: a wrong teamId, a key in the wrong format and a client created
against a different public key all surface as one terse rejection from Apple.
Knowing which step broke is most of the fix.
"""
from __future__ import annotations

import logging
import sys
from datetime import date, timedelta

from ..asa.auth import (
    MAX_SECRET_DAYS,
    WARN_BEFORE_DAYS,
    AsaAuth,
    AuthError,
    Credentials,
    describe_key_problem,
)
from ..asa.popularity import PRIORITY_GENRES, PopularityClient

OK = "\033[32mOK\033[0m"
BAD = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"


def last_full_week() -> tuple[str, str]:
    """The most recent completed Sunday-Saturday week."""
    today = date.today()
    # Sunday is weekday 6 in Python's Monday-based numbering.
    days_since_sunday = (today.weekday() + 1) % 7
    this_sunday = today - timedelta(days=days_since_sunday)
    start = this_sunday - timedelta(days=7)
    return start.isoformat(), (start + timedelta(days=6)).isoformat()


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(message)s")
    print("Apple Ads Platform API\n" + "=" * 52)

    missing = Credentials.missing()
    if missing:
        print(f"\n  [{BAD}] не заданы переменные: {', '.join(missing)}")
        print("\n  Их значения берутся из Настройки аккаунта → API")
        print("  у пользователя с ролью API. orgId — из интерфейса Apple Ads.")
        return 1

    creds = Credentials.from_env()
    assert creds is not None
    print(f"\n  clientId {creds.client_id[:12]}…  teamId {creds.team_id}  "
          f"keyId {creds.key_id}  orgId {creds.org_id}")

    # --- 1. the key itself ----------------------------------------------
    print("\n1. приватный ключ")
    problem = describe_key_problem(creds.private_key)
    if problem:
        print(f"  [{BAD}] {problem}")
        return 1
    print(f"  [{OK}] PKCS#8, {len(creds.private_key.splitlines())} строк")

    # --- 2. signing ------------------------------------------------------
    print("\n2. подпись client secret")
    auth = AsaAuth(creds)
    try:
        secret = auth.client_secret()
    except AuthError as exc:
        print(f"  [{BAD}] {exc}")
        return 1
    left = auth.secret_days_left()
    print(f"  [{OK}] подписан, {len(secret)} символов")
    if left is not None:
        line = f"  срок действия: {left} дн. (максимум {MAX_SECRET_DAYS}, продления нет)"
        if left <= WARN_BEFORE_DAYS:
            print(f"  [{WARN}] {line.strip()} — пора перевыпускать")
        else:
            print(line)

    # --- 3. token --------------------------------------------------------
    print("\n3. обмен на access token")
    try:
        token = auth.access_token()
    except AuthError as exc:
        print(f"  [{BAD}] {exc}")
        print("\n  Частые причины:")
        print("   · teamId или keyId не от этого клиента")
        print("   · клиент создан под другой публичный ключ")
        print("   · у пользователя отозвана роль API")
        return 1
    print(f"  [{OK}] получен, {len(token)} символов")

    # --- 4. the data we actually came for --------------------------------
    print("\n4. данные о популярности")
    start, end = last_full_week()
    print(f"  неделя {start} … {end}")
    client = PopularityClient(auth)
    rows = list(client.query(["US"], PRIORITY_GENRES[0], start, end, page_size=10,
                             max_rows=10))
    if not rows:
        print(f"  [{BAD}] строк не вернулось")
        print("  Данные за эту неделю могут быть ещё не опубликованы —")
        print("  попробуйте неделей раньше. Либо у роли нет доступа к Insights.")
        return 1

    print(f"  [{OK}] {len(rows)} строк, топ US / {PRIORITY_GENRES[0]}:\n")
    for r in rows[:8]:
        print(f"    #{str(r.rank_in_genre or '?'):<4} {r.search_term[:34]:<36} "
              f"популярность {r.popularity_1to100 or '?'}")

    print("\n" + "=" * 52)
    print("Авторизация работает. Можно собирать понедельные срезы.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
