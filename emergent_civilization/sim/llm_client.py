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

# Python's default urllib User-Agent ("Python-urllib/3.x") is a well-known
# automation signature that some providers' edge WAFs (e.g. Cloudflare) block
# outright — surfacing as an opaque "error code: 1010" with no mention of the
# real API at all. Sending a normal-looking one avoids that false positive.
_USER_AGENT = "EmergentCivilization/1.0 (+https://github.com; research-sim-client)"


class LLMBackend(Protocol):
    def complete(self, prompt: str) -> str: ...


def _sanitize_header_value(name: str, value: str) -> str:
    """Strip incidental whitespace and fail loudly on non-Latin-1 content.

    HTTP header values must be Latin-1. A stray character picked up while
    copy-pasting a key (a smart quote, a zero-width space, a BOM) produces a
    cryptic ``'latin-1' codec can't encode characters`` error deep inside
    urllib with no indication of which value or character caused it. Catching
    it here — at backend construction time, before any request is made —
    turns that into one clear, immediate error instead of the same confusing
    failure repeating every tick for every agent.
    """
    cleaned = value.strip()
    try:
        cleaned.encode("latin-1")
    except UnicodeEncodeError as e:
        raise ValueError(
            f"{name} contains a character that can't go in an HTTP header "
            f"(often an invisible character picked up while copy-pasting, or "
            f"leftover placeholder text). Re-copy the value fresh from the "
            f"source and paste it in on its own, with nothing else attached. "
            f"Value seen (repr, so hidden characters are visible): {cleaned!r}"
        ) from e
    return cleaned


class EchoBackend:
    """Offline stand-in that returns a valid but trivial action.

    It exists so the LLM code path is exercisable without a server. It always
    chooses to idle; real reasoning is delegated to a genuine backend.
    """

    def complete(self, _prompt: str) -> str:
        return json.dumps({"action": "idle", "reason": "offline echo backend"})


class MockReasoningBackend:
    """Offline stand-in that produces *varied* actions with plausible reasons.

    Unlike EchoBackend it actually looks at the observation embedded in the
    prompt and picks a survival-sensible action, so the LLM code path (build
    prompt -> complete -> parse -> act) can be demonstrated end to end without a
    real model or network. It is NOT the research subject — it only stands in
    for a genuine LLM so the reasoning-log demo runs anywhere. Point a real
    backend at it (env vars) to run the actual experiment.
    """

    def complete(self, prompt: str) -> str:
        obs = _extract_observation(prompt)
        me = obs.get("self", {})
        inv = me.get("inventory", {})
        hunger = me.get("hunger", 0)
        energy = me.get("energy", 100)
        food = inv.get("food", 0)

        if hunger >= 60 and food > 0:
            return _act("eat", reason="I'm getting hungry, so I'll eat now")
        if obs.get("is_night") and obs.get("nearby_threats") and not me.get("sheltered"):
            if obs.get("nearby_agents"):
                return _act("move", {"direction": _toward(me, obs["nearby_agents"][0])},
                            "wolves are out — safer to stick with the others")
            return _act("rest", reason="hunkering down for the night")
        if energy <= 25:
            return _act("rest", reason="low on energy, I need to recover")
        if inv.get("wood", 0) >= 3 and inv.get("stone", 0) >= 1 \
                and not me.get("sheltered") and not obs.get("nearby_shelters"):
            return _act("build", reason="I have the materials, I'll build a shelter")
        on = obs.get("on_tile_node")
        if on and (on["resource"] != "food" or food < 4):
            return _act("gather", reason=f"gathering {on['resource']} while I'm on it")
        nodes = obs.get("nearby_nodes", [])
        if nodes:
            return _act("move", {"direction": _toward(me, nodes[0])},
                        f"heading toward the {nodes[0]['resource']}")
        return _act("move", {"direction": "north"}, "exploring for resources")


class AnthropicBackend:
    """Backend for the Anthropic Messages API (recommended: a fast Claude model).

    Uses stdlib urllib so no SDK is required. Set EC_LLM_API_KEY (or
    ANTHROPIC_API_KEY) and optionally EC_LLM_MODEL.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 200,
        temperature: float = 0.8,
    ) -> None:
        self.model = model or os.environ.get("EC_LLM_MODEL", "claude-haiku-4-5-20251001")
        raw_key = api_key or os.environ.get("EC_LLM_API_KEY") \
            or os.environ.get("ANTHROPIC_API_KEY", "")
        self.api_key = _sanitize_header_value("EC_LLM_API_KEY/ANTHROPIC_API_KEY", raw_key)
        self.base_url = (base_url or os.environ.get("ANTHROPIC_BASE_URL",
                         "https://api.anthropic.com")).rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature

    def complete(self, prompt: str) -> str:
        import urllib.error
        import urllib.request

        payload = json.dumps(
            {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/v1/messages",
            data=payload,
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "User-Agent": _USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} from {self.base_url}: {detail}") from e
        return body["content"][0]["text"]


def pick_backend_from_env(*, allow_mock: bool = True) -> LLMBackend:
    """Choose a backend from environment variables, in this order:

    1. ``EC_LLM_BASE_URL`` set      -> OpenAICompatBackend (RunPod vLLM / NVIDIA API)
    2. ``EC_LLM_API_KEY`` or ``ANTHROPIC_API_KEY`` set -> AnthropicBackend
    3. otherwise, if ``allow_mock`` -> MockReasoningBackend (offline, no network)

    Raises ``RuntimeError`` if no credentials are found and ``allow_mock`` is
    False — used by callers (like the live viz server) where silently falling
    back to a fake model would be misleading.
    """
    if os.environ.get("EC_LLM_BASE_URL"):
        print(f"[llm] backend: OpenAICompatBackend ({os.environ['EC_LLM_BASE_URL']})")
        return OpenAICompatBackend()
    if os.environ.get("EC_LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
        model = os.environ.get("EC_LLM_MODEL", "claude-haiku-4-5-20251001")
        print(f"[llm] backend: AnthropicBackend (model={model})")
        return AnthropicBackend()
    if allow_mock:
        print("[llm] backend: MockReasoningBackend (offline — set EC_LLM_* for a real model)")
        return MockReasoningBackend()
    raise RuntimeError(
        "No LLM credentials found. Set EC_LLM_API_KEY (or ANTHROPIC_API_KEY) to use "
        "Anthropic, or EC_LLM_BASE_URL (+ EC_LLM_MODEL, EC_LLM_API_KEY) for an "
        "OpenAI-compatible endpoint — free options with no card required: Groq "
        "(https://api.groq.com/openai/v1), OpenRouter "
        "(https://openrouter.ai/api/v1, use a ':free'-tagged model), NVIDIA "
        "API Catalog (https://integrate.api.nvidia.com/v1); or Google Gemini "
        "(https://generativelanguage.googleapis.com/v1beta/openai), xAI Grok "
        "(https://api.x.ai/v1), a local Ollama server "
        "(http://localhost:11434/v1), or RunPod vLLM."
    )


# -- helpers for the mock backend -----------------------------------------
def _extract_observation(prompt: str) -> dict:
    """Pull the observation JSON block back out of a built prompt."""
    marker = "Current observation:\n"
    start = prompt.find(marker)
    if start == -1:
        return {}
    start += len(marker)
    end = prompt.find("\n\nYour action:", start)
    try:
        return json.loads(prompt[start:end if end != -1 else None])
    except json.JSONDecodeError:
        return {}


def _toward(me: dict, other: dict) -> str:
    (px, py), (tx, ty) = me.get("pos", [0, 0]), other.get("pos", [0, 0])
    if abs(tx - px) >= abs(ty - py):
        return "east" if tx > px else "west"
    return "north" if ty > py else "south"


def _act(action: str, args: dict | None = None, reason: str = "") -> str:
    return json.dumps({"action": action, "args": args or {}, "reason": reason})


class OpenAICompatBackend:
    """Backend for any OpenAI-compatible /chat/completions endpoint.

    Covers RunPod vLLM deployments, the NVIDIA API Catalog
    (https://integrate.api.nvidia.com/v1, free tier available), Google Gemini's
    OpenAI-compatible endpoint (https://generativelanguage.googleapis.com/v1beta/openai,
    free tier via a Google AI Studio key), xAI Grok (https://api.x.ai/v1,
    console.x.ai — needs billing/credits), Groq (https://api.groq.com/openai/v1,
    console.groq.com — free tier, no card required), OpenRouter
    (https://openrouter.ai/api/v1, openrouter.ai — ":free"-tagged models need
    no card), and fully local servers like Ollama (http://localhost:11434/v1)
    or LM Studio (http://localhost:1234/v1) — all speak the same OpenAI chat
    schema. Local servers generally ignore the Authorization header, so any
    non-empty EC_LLM_API_KEY placeholder (e.g. "ollama") works. Network
    libraries are imported lazily so this module stays importable with no
    dependencies installed.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.8,
        max_tokens: int | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("EC_LLM_BASE_URL", "")).rstrip("/")
        self.model = model or os.environ.get("EC_LLM_MODEL", "")
        raw_key = api_key or os.environ.get("EC_LLM_API_KEY", "")
        self.api_key = _sanitize_header_value("EC_LLM_API_KEY", raw_key)
        self.temperature = temperature
        # Default (150) is tuned for free-tier token-per-minute budgets. Some
        # models (e.g. Qwen3's "thinking" mode) spend a chunk of the output
        # budget on reasoning text before ever emitting the action JSON, and
        # can get truncated mid-thought at 150 (-> "unparseable model output").
        # Override with EC_LLM_MAX_TOKENS for those, since a private/paid
        # backend doesn't have the same TPM pressure a free tier does.
        self.max_tokens = (
            max_tokens if max_tokens is not None
            else int(os.environ.get("EC_LLM_MAX_TOKENS", "150"))
        )
        # Generous default: local CPU inference (Ollama/LM Studio) can be much
        # slower per call than a hosted API.
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        import urllib.error
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
                "User-Agent": _USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            # Surface the provider's actual error text (e.g. "invalid API key",
            # "model not found") instead of a bare "HTTP Error 400".
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} from {self.base_url}: {detail}") from e
        return body["choices"][0]["message"]["content"]
