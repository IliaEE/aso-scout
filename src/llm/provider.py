"""
LLM provider, Groq by default.

Pluggable on purpose: Groq is OpenAI-compatible, so swapping to another
provider is a base URL and a key, not a rewrite. The judge never imports a
provider directly — it takes a `Chat` and calls it.

Model selection is discovered, not hard-coded. Groq's catalogue moves: models
get deprecated, and some (llama-3.3-70b-versatile, llama-3.1-8b-instant among
them) have been moved behind enterprise access, so a fixed default silently
404s on a free account. The client asks the account what it can actually use
and picks the first match from a preference list.

The API key comes from the environment and is never written to a file or a
log. If it is missing the judge is skipped, not crashed: the heuristics still
work without it.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Protocol

import httpx

log = logging.getLogger(__name__)

GROQ_BASE = "https://api.groq.com/openai/v1"

# Preference order, best-for-this-task first. The job is short structured
# classification, so a mid-size model is plenty; the giants cost rate limit
# without judging brands any better.
PREFERRED_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

# Never used for judging, whatever the account offers: audio, safety filters
# and embeddings are not chat models.
NON_CHAT = ("whisper", "tts", "orpheus", "guard", "embed", "safeguard")


class Chat(Protocol):
    def _downgrade(self) -> bool:
        """Move to the next preferred model after persistent rate limiting."""
        if self._pinned or not self._model:
            return False
        available = [m for m in self.available_models()
                     if not any(x in m for x in NON_CHAT)]
        self._tried.add(self._model)

        # Preferred order first, then anything else the account has. A model
        # name missing from the list must not dead-end the run: this account
        # offers qwen3.8-27b, which no fixed list of mine predicted.
        ordered = [m for m in PREFERRED_MODELS if m in available]
        ordered += [m for m in sorted(available) if m not in ordered]

        for candidate in ordered:
            if candidate not in self._tried:
                log.warning("rate limited on %s — switching to %s",
                            self._model, candidate)
                self._model = candidate
                self.stats["downgrades"] += 1
                return True
        return False

    def complete(self, system: str, user: str, max_tokens: int = 700) -> str | None: ...


def extract_json(text: str) -> Any | None:
    """
    Pull a JSON object out of a model reply.

    Models wrap JSON in prose or fences no matter how firmly the prompt says
    otherwise, and a reasoning model may emit a <think> block first. Parsing
    defensively is cheaper than retrying.
    """
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost brace-balanced span.
    start = text.find("{")
    if start < 0:
        start = text.find("[")
    if start < 0:
        return None
    depth = 0
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    for i in range(start, len(text)):
        if text[i] == opener:
            depth += 1
        elif text[i] == closer:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


class GroqChat:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = GROQ_BASE,
        timeout: float = 45.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._model = model or os.environ.get("GROQ_MODEL") or None
        self._pinned = bool(model or os.environ.get("GROQ_MODEL"))
        self.stats = {"calls": 0, "failures": 0, "downgrades": 0}
        self.rate_limited = False
        self._retried_429 = False
        self._tried: set[str] = set()
        self._last_call = 0.0
        # The binding limit on Groq's free tier is tokens per minute, not
        # requests: the error body reads "tokens per minute (TPM): Limit 8000"
        # while each judgement costs roughly 950. Pacing by requests alone
        # produced a 429 on almost every other call.
        try:
            tpm = float(os.environ.get("GROQ_TPM", 8000))
        except ValueError:
            tpm = 8000.0
        try:
            est = float(os.environ.get("GROQ_TOKENS_PER_CALL", 1000))
        except ValueError:
            est = 1000.0
        token_interval = 60.0 / max(tpm / max(est, 1), 1) if tpm > 0 else 0.0

        try:
            rpm = float(os.environ.get("GROQ_RPM", 30))
        except ValueError:
            rpm = 30.0
        request_interval = 60.0 / rpm if rpm > 0 else 0.0

        # Whichever limit binds first.
        self._min_interval = max(token_interval, request_interval)

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def available_models(self) -> list[str]:
        try:
            with httpx.Client(timeout=self.timeout) as c:
                r = c.get(f"{self.base_url}/models", headers=self._headers())
            if r.status_code != 200:
                log.warning("model list failed: %s %s", r.status_code, r.text[:200])
                return []
            return [m["id"] for m in r.json().get("data", []) if m.get("id")]
        except Exception:
            log.exception("model list failed")
            return []

    def resolve_model(self) -> str | None:
        """Pick the best model this account can actually call."""
        if self._model:
            return self._model
        available = set(self.available_models())
        if not available:
            return None
        for candidate in PREFERRED_MODELS:
            if candidate in available:
                self._model = candidate
                log.info("using Groq model %s", candidate)
                return candidate
        # Nothing preferred: take any chat model the account offers. Groq's
        # catalogue changes names often enough that a fixed list goes stale.
        for m in sorted(available):
            if not any(x in m for x in NON_CHAT):
                self._model = m
                log.info("falling back to Groq model %s", m)
                return m
        return None

    def _downgrade(self) -> bool:
        """Move to the next preferred model after persistent rate limiting."""
        if self._pinned or not self._model:
            return False
        available = [m for m in self.available_models()
                     if not any(x in m for x in NON_CHAT)]
        self._tried.add(self._model)

        # Preferred order first, then anything else the account has. A model
        # name missing from the list must not dead-end the run: this account
        # offers qwen3.8-27b, which no fixed list of mine predicted.
        ordered = [m for m in PREFERRED_MODELS if m in available]
        ordered += [m for m in sorted(available) if m not in ordered]

        for candidate in ordered:
            if candidate not in self._tried:
                log.warning("rate limited on %s — switching to %s",
                            self._model, candidate)
                self._model = candidate
                self.stats["downgrades"] += 1
                return True
        return False

    def complete(self, system: str, user: str, max_tokens: int = 700) -> str | None:
        model = self.resolve_model()
        if not model or not self.configured:
            return None

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # Near-zero temperature: this is classification, and a creative
            # answer here is just an inconsistent one.
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        # Reasoning models burn output tokens on internal thinking, and those
        # count against the free tier's per-minute budget. Classifying a search
        # query needs none of it: the first run hit the limit after seven calls.
        if "gpt-oss" in model or "qwen3" in model:
            payload["reasoning_effort"] = "low"

        # Space the calls out. Free tiers cap requests per minute as well as
        # tokens, and arriving in a burst wastes the budget on 429s.
        if self._min_interval > 0:
            wait = self._last_call + self._min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)

        self.stats["calls"] += 1
        self._last_call = time.monotonic()
        try:
            with httpx.Client(timeout=self.timeout) as c:
                r = c.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )
        except Exception:
            self.stats["failures"] += 1
            log.exception("groq request failed")
            return None

        if r.status_code == 429:
            # The body names which limit was hit and for how long. Discarding
            # it, as the first version did, leaves nothing to act on.
            retry_after = r.headers.get("retry-after")
            log.warning("groq rate limit: %s", r.text[:300])

            if not self._retried_429:
                self._retried_429 = True
                try:
                    delay = float(retry_after) if retry_after else 20.0
                except ValueError:
                    delay = 20.0
                delay = min(delay, 60.0)
                log.info("waiting %.0fs and retrying once", delay)
                time.sleep(delay)
                self._last_call = time.monotonic()
                try:
                    with httpx.Client(timeout=self.timeout) as c:
                        r2 = c.post(f"{self.base_url}/chat/completions",
                                    headers=self._headers(), json=payload)
                    if r2.status_code == 200:
                        self._retried_429 = False
                        return r2.json()["choices"][0]["message"]["content"]
                except Exception:
                    log.exception("retry after rate limit failed")

            # Still limited. Drop to the next, smaller model rather than
            # abandoning the run: a lighter model judging the queue beats the
            # best model judging seven rows of it.
            if self._downgrade():
                self._retried_429 = False
                return self.complete(system, user, max_tokens)

            self.stats["failures"] += 1
            self.rate_limited = True
            return None
        if r.status_code != 200:
            self.stats["failures"] += 1
            # response_format is not supported by every model; retry plain.
            if "response_format" in r.text:
                payload.pop("response_format", None)
                try:
                    with httpx.Client(timeout=self.timeout) as c:
                        r = c.post(f"{self.base_url}/chat/completions",
                                   headers=self._headers(), json=payload)
                    if r.status_code == 200:
                        return r.json()["choices"][0]["message"]["content"]
                except Exception:
                    log.exception("groq retry failed")
            log.warning("groq %s: %s", r.status_code, r.text[:300])
            return None

        try:
            return r.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError):
            self.stats["failures"] += 1
            log.warning("unexpected groq response shape")
            return None


def chat_for(task: str) -> GroqChat | None:
    """
    A client configured for one task: "judge" or "ideate".

    Separate configuration per task, because the two jobs have different
    needs. Judging benefits from the strongest model available; proposing
    adjacent queries does not, and a smaller model there leaves more of the
    token budget for the work that needs it.

    One thing a second key does NOT buy: more throughput. Groq's limits are
    per organization — the 429 body says so outright, "in organization
    org_..." — so another key on the same account draws from the same 8000
    tokens a minute. Real extra capacity means either a paid tier or a
    genuinely different provider.

    Which is why base_url is configurable too: anything OpenAI-compatible
    works here, so IDEATE_BASE_URL can point at another provider with its own
    budget.

    Falls back to the shared GROQ_* settings when the task-specific ones are
    unset, so existing deployments keep working untouched.
    """
    prefix = task.upper()
    key = os.environ.get(f"{prefix}_API_KEY") or os.environ.get("GROQ_API_KEY", "")
    if not key:
        return None
    model = os.environ.get(f"{prefix}_MODEL") or os.environ.get("GROQ_MODEL") or None
    base = os.environ.get(f"{prefix}_BASE_URL") or os.environ.get("GROQ_BASE_URL") or GROQ_BASE
    chat = GroqChat(api_key=key, model=model, base_url=base)

    # Per-task pacing: an ideation pass is a handful of calls, a judging pass
    # is dozens, and they should not have to share one interval.
    tpm = os.environ.get(f"{prefix}_TPM")
    if tpm:
        try:
            est = float(os.environ.get(f"{prefix}_TOKENS_PER_CALL", 1000))
            chat._min_interval = 60.0 / max(float(tpm) / max(est, 1), 1)
        except ValueError:
            pass
    return chat


def default_chat() -> GroqChat | None:
    chat = GroqChat()
    return chat if chat.configured else None
