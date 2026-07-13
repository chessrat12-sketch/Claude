"""Pluggable LLM backends behind one tiny interface.

The simulation only needs ``complete(prompt) -> str``. Concrete backends
(RunPod, an OpenAI-compatible endpoint, the Anthropic API, ...) live behind
that method so the engine never imports a vendor SDK directly. The default
``EchoBackend`` requires no network and lets the whole system run offline,
which keeps tests and CI hermetic.
"""

from __future__ import annotations

import json
import os
from typing import Protocol


class LLMBackend(Protocol):
    def complete(self, prompt: str) -> str: ...


class EchoBackend:
    """Offline stand-in that returns a valid but trivial action.

    It exists so the LLM code path is exercisable without a server. It always
    chooses to idle; real reasoning is delegated to a genuine backend.
    """

    def complete(self, _prompt: str) -> str:
        return json.dumps({"action": "idle", "reason": "offline echo backend"})


class OpenAICompatBackend:
    """Backend for any OpenAI-compatible /chat/completions endpoint.

    Covers RunPod vLLM deployments and NVIDIA API Catalog, which both speak the
    OpenAI schema. Network libraries are imported lazily so this module stays
    importable with no dependencies installed.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.8,
        max_tokens: int = 300,
    ) -> None:
        self.base_url = (base_url or os.environ.get("EC_LLM_BASE_URL", "")).rstrip("/")
        self.model = model or os.environ.get("EC_LLM_MODEL", "")
        self.api_key = api_key or os.environ.get("EC_LLM_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, prompt: str) -> str:
        import urllib.request  # stdlib, avoids a hard dependency

        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
        return body["choices"][0]["message"]["content"]
