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
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]


class Chat(Protocol):
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
        self.stats = {"calls": 0, "failures": 0}

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
        # Nothing preferred: take any non-audio, non-guard chat model.
        for m in sorted(available):
            if not any(x in m for x in ("whisper", "tts", "guard", "embed")):
                self._model = m
                log.info("falling back to Groq model %s", m)
                return m
        return None

    def complete(self, system: str, user: str, max_tokens: int = 700) -> str | None:
        model = self.resolve_model()
        if not model or not self.configured:
            return None

        payload = {
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

        self.stats["calls"] += 1
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
            self.stats["failures"] += 1
            log.warning("groq rate limit hit — judging stops for this run")
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


def default_chat() -> GroqChat | None:
    chat = GroqChat()
    return chat if chat.configured else None
