"""
Authentication for the Apple Ads Platform API.

Two steps, and the first one is where everything goes wrong:

  1. A *client secret*: a JWT you sign yourself with your private key. Apple
     never issues it — you mint it locally from clientId, teamId and keyId.
  2. An *access token*: obtained by posting that secret to Apple. Lives an
     hour, so it is cached and re-derived as needed.

The private key must be **PKCS#8**. `openssl ecparam -genkey` alone does not
produce that, and the resulting file loads nowhere — the error it raises says
nothing useful about why, so `looks_like_pkcs8` checks the header explicitly
and says so in plain words instead.

The 180-day problem: Apple caps the client secret's lifetime at 180 days, with
no renewal, no warning and no grace period. For an unattended service that
means it simply stops working one day about six months from now, silently.
`days_until_expiry` exists so doctor and the digest can warn ahead of time
rather than leaving you to discover it from an empty report.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx
import jwt

log = logging.getLogger(__name__)

TOKEN_URL = "https://appleid.apple.com/auth/oauth2/token"
AUDIENCE = "https://appleid.apple.com"
SCOPE = "searchadsorg"

# Apple's hard cap. Not a default we chose — requesting longer is rejected.
MAX_SECRET_DAYS = 180

# Warn this far ahead, so there is time to rotate before anything breaks.
WARN_BEFORE_DAYS = 21


class AuthError(RuntimeError):
    pass


def looks_like_pkcs8(pem: str) -> bool:
    return "BEGIN PRIVATE KEY" in pem


def describe_key_problem(pem: str) -> str | None:
    """A readable reason the key will not work, or None if it looks fine."""
    if not pem or not pem.strip():
        return "ключ пустой — проверьте переменную ASA_PRIVATE_KEY"
    if "BEGIN EC PRIVATE KEY" in pem:
        return (
            "ключ в формате SEC1, а нужен PKCS#8. Преобразуйте:\n"
            "  openssl pkcs8 -topk8 -nocrypt -in ec-key.pem -out private-key.pem"
        )
    if "BEGIN PUBLIC KEY" in pem:
        return "это публичный ключ, а нужен приватный"
    if not looks_like_pkcs8(pem):
        return "не похоже на PEM с приватным ключом (нет строки BEGIN PRIVATE KEY)"
    if "\\n" in pem and "\n" not in pem:
        return (
            "переносы строк сохранились как литералы \\n. В Railway вставляйте "
            "ключ многострочным значением, а не одной строкой"
        )
    return None


@dataclass
class Credentials:
    client_id: str
    team_id: str
    key_id: str
    org_id: str
    private_key: str

    @classmethod
    def from_env(cls) -> "Credentials | None":
        fields = {
            "client_id": os.environ.get("ASA_CLIENT_ID", "").strip(),
            "team_id": os.environ.get("ASA_TEAM_ID", "").strip(),
            "key_id": os.environ.get("ASA_KEY_ID", "").strip(),
            "org_id": os.environ.get("ASA_ORG_ID", "").strip(),
            "private_key": os.environ.get("ASA_PRIVATE_KEY", ""),
        }
        if not all(fields.values()):
            return None
        return cls(**fields)

    @staticmethod
    def missing() -> list[str]:
        return [
            name for name in
            ("ASA_CLIENT_ID", "ASA_TEAM_ID", "ASA_KEY_ID",
             "ASA_ORG_ID", "ASA_PRIVATE_KEY")
            if not os.environ.get(name, "").strip()
        ]


def make_client_secret(creds: Credentials, valid_days: int = MAX_SECRET_DAYS) -> str:
    """
    Mint and sign the client secret.

    Claims follow Apple's spec: the team is the issuer, the client is both
    subject and the party the token is about, and the audience is Apple's
    identity service.
    """
    problem = describe_key_problem(creds.private_key)
    if problem:
        raise AuthError(problem)

    now = int(time.time())
    days = min(valid_days, MAX_SECRET_DAYS)
    payload = {
        "sub": creds.client_id,
        "aud": AUDIENCE,
        "iat": now,
        "exp": now + days * 24 * 3600,
        "iss": creds.team_id,
    }
    try:
        return jwt.encode(
            payload,
            creds.private_key,
            algorithm="ES256",
            headers={"alg": "ES256", "kid": creds.key_id},
        )
    except Exception as exc:
        raise AuthError(f"не удалось подписать client secret: {exc}") from exc


def days_until_expiry(client_secret: str) -> int | None:
    """Days left on a signed secret, without verifying it."""
    try:
        claims = jwt.decode(client_secret, options={"verify_signature": False})
    except Exception:
        return None
    exp = claims.get("exp")
    if not exp:
        return None
    left = datetime.fromtimestamp(exp, tz=timezone.utc) - datetime.now(timezone.utc)
    return max(int(left / timedelta(days=1)), 0)


class AsaAuth:
    """Holds credentials and hands out a live access token."""

    def __init__(self, creds: Credentials | None = None, timeout: float = 30.0) -> None:
        self.creds = creds or Credentials.from_env()
        self.timeout = timeout
        self._token: str | None = None
        self._token_expires = 0.0
        self._secret: str | None = None

    @property
    def configured(self) -> bool:
        return self.creds is not None

    def client_secret(self) -> str:
        if self._secret is None:
            if self.creds is None:
                raise AuthError("учётные данные Apple Ads не заданы")
            self._secret = make_client_secret(self.creds)
        return self._secret

    def secret_days_left(self) -> int | None:
        try:
            return days_until_expiry(self.client_secret())
        except AuthError:
            return None

    def access_token(self) -> str:
        # Refresh a minute early rather than racing the expiry.
        if self._token and time.time() < self._token_expires - 60:
            return self._token

        secret = self.client_secret()
        data = {
            "grant_type": "client_credentials",
            "client_id": self.creds.client_id,  # type: ignore[union-attr]
            "client_secret": secret,
            "scope": SCOPE,
        }
        try:
            with httpx.Client(timeout=self.timeout) as c:
                r = c.post(
                    TOKEN_URL,
                    data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded",
                             "Host": "appleid.apple.com"},
                )
        except Exception as exc:
            raise AuthError(f"сеть недоступна при запросе токена: {exc}") from exc

        if r.status_code != 200:
            # Apple's errors here are terse; include the body because the
            # difference between a wrong teamId and a wrong key is only
            # visible in it.
            raise AuthError(
                f"Apple отклонил запрос токена ({r.status_code}): {r.text[:300]}"
            )

        payload = r.json()
        token = payload.get("access_token")
        if not token:
            raise AuthError(f"в ответе нет access_token: {str(payload)[:200]}")

        self._token = token
        self._token_expires = time.time() + float(payload.get("expires_in", 3600))
        return token

    def headers(self) -> dict[str, str]:
        if self.creds is None:
            raise AuthError("учётные данные Apple Ads не заданы")
        return {
            "Authorization": f"Bearer {self.access_token()}",
            # Scopes the request to one organization. Without it the API
            # cannot tell which account is asking.
            "X-AP-Context": f"orgId={self.creds.org_id}",
            "Content-Type": "application/json",
        }
