"""Visualisation server: run the simulation live and stream world snapshots.

Advances the simulation on a background thread at a fixed tick rate and exposes
the latest render snapshot over HTTP. Three clients consume the same
``GET /state`` endpoint and always show identical worlds:

  * ``/``    a real 3D browser viewer (Three.js, orbit camera; vendored
             locally under viewer/vendor/ so it works with no CDN/network)
  * ``/iso`` a lightweight isometric-canvas viewer with zero dependencies
  * Unity    ``unity/Scripts/SceneBootstrap.cs`` for a full 3D engine client

    python -m server.viz_server              # then open http://localhost:8000
    python -m server.viz_server --port 9000 --tick-ms 300 --agents 12

    # Drive agents with a real LLM instead of the heuristic baseline.
    #
    # Easiest: copy .env.example to .env in this directory, fill in your real
    # values once, and every future run picks them up automatically — no more
    # retyping `set`/`export` each session. See .env.example for the format.
    #
    # --llm-gap-ms (default 8000) paces calls one-at-a-time instead of
    # bursting all agents then going idle — this is what actually controls
    # calls-per-minute against a provider's rate limit, independent of
    # --agents. The default is safe for Groq's free tier; raise it for
    # stricter limits, lower it for paid/high-limit APIs, or use 0 for local
    # Ollama (no rate limit at all).
    #
    # Or set them for just this session (any of these providers work). Pick one:

    # (a) Anthropic
    export EC_LLM_API_KEY=<anthropic-key>          # or ANTHROPIC_API_KEY
    python -m server.viz_server --llm --agents 4 --llm-gap-ms 1000

    # (b) NVIDIA API Catalog (free tier) — build.nvidia.com
    export EC_LLM_BASE_URL=https://integrate.api.nvidia.com/v1
    export EC_LLM_MODEL=meta/llama-3.1-8b-instruct   # or any model you enabled
    export EC_LLM_API_KEY=<nvidia-key>
    python -m server.viz_server --llm --agents 4

    # (c) Google Gemini (free tier, OpenAI-compatible endpoint) — aistudio.google.com/apikey
    export EC_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
    export EC_LLM_MODEL=gemini-2.0-flash
    export EC_LLM_API_KEY=<google-ai-studio-key>
    python -m server.viz_server --llm --agents 4

    # (d) xAI Grok (OpenAI-compatible endpoint) — console.x.ai (needs billing/credits)
    export EC_LLM_BASE_URL=https://api.x.ai/v1
    export EC_LLM_MODEL=grok-2-latest        # check console.x.ai for the current name
    export EC_LLM_API_KEY=<xai-key>
    python -m server.viz_server --llm --agents 4 --llm-gap-ms 1000

    # (e) Groq (free tier, no card required) — console.groq.com
    # Free-tier token-per-minute limits are tight (e.g. 6000 TPM on
    # llama-3.1-8b-instant) — the default --llm-gap-ms 8000 is tuned for this.
    export EC_LLM_BASE_URL=https://api.groq.com/openai/v1
    export EC_LLM_MODEL=llama-3.1-8b-instant  # see console.groq.com/docs/models
    export EC_LLM_API_KEY=<groq-key>
    python -m server.viz_server --llm --agents 4

    # (f) OpenRouter (":free"-tagged models, no card required) — openrouter.ai
    # Free models are also rate-limited; the default gap applies here too.
    export EC_LLM_BASE_URL=https://openrouter.ai/api/v1
    export EC_LLM_MODEL=meta-llama/llama-3.1-8b-instruct:free  # see openrouter.ai/models?max_price=0
    export EC_LLM_API_KEY=<openrouter-key>
    python -m server.viz_server --llm --agents 4

    # (g) Fully local (Ollama) — no cloud, no key needed beyond a placeholder,
    # no rate limit (just your CPU/GPU speed), so turn pacing off:
    export EC_LLM_BASE_URL=http://localhost:11434/v1
    export EC_LLM_MODEL=llama3.2
    export EC_LLM_API_KEY=ollama
    python -m server.viz_server --llm --agents 4 --llm-gap-ms 0

The server itself uses only the Python standard library (no framework, no
extra deps beyond an LLM API call when ``--llm`` is used).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sim import Agent, HeuristicPolicy, LLMPolicy, Personality, Simulation, make_scattered_world
from sim.envfile import load_env_file
from sim.llm_client import pick_backend_from_env
from sim.snapshot import world_snapshot

# Load .env (if present) before anything reads EC_LLM_* from the environment.
# Checks the current directory first, then this file's project root, so it
# works whether you run from emergent_civilization/ or elsewhere.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_loaded_env = load_env_file(os.getcwd(), _PROJECT_ROOT)

NAMES = [
    "Aria", "Boaz", "Cira", "Doran", "Esme", "Finn", "Gwen", "Hodr", "Ivo",
    "Juno", "Kai", "Lena", "Milo", "Nadia", "Oren", "Pia", "Quin", "Rhea",
]
VIEWER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "viewer")
VIEWER_3D = os.path.join(VIEWER_DIR, "village_3d.html")   # real 3D (Three.js), default
VIEWER_ISO = os.path.join(VIEWER_DIR, "village.html")     # isometric canvas, no-install
VENDOR_DIR = os.path.join(VIEWER_DIR, "vendor")            # vendored three.js (no CDN)


class LiveWorld:
    """Owns the simulation and the latest snapshot behind a lock."""

    def __init__(
        self, n_agents: int, size: int, tick_ms: int, seed: int,
        use_llm: bool = False, llm_gap_ms: int = 0,
    ) -> None:
        self.tick_ms = tick_ms
        rng = random.Random(seed)
        world = make_scattered_world(width=size, height=size, density=0.20, seed=seed)
        agents = []
        for i in range(n_agents):
            agents.append(
                Agent(
                    id=f"a{i}",
                    name=NAMES[i % len(NAMES)],
                    pos=(rng.randrange(size), rng.randrange(size)),
                    personality=Personality(
                        greed=rng.random(), sociability=rng.random(),
                        caution=rng.random(), curiosity=rng.random(),
                    ),
                )
            )
        if use_llm:
            # Fails loudly if no credentials are set — a live LLM run with no
            # key would otherwise be a confusing silent fallback.
            backend = pick_backend_from_env(allow_mock=False)
            policy_factory = lambda a: LLMPolicy(a, backend)  # noqa: E731
        else:
            policy_factory = lambda a: HeuristicPolicy(a.personality)  # noqa: E731
        self.sim = Simulation(
            world, agents, policy_factory=policy_factory, seed=seed,
            decision_gap=(llm_gap_ms / 1000.0) if use_llm else 0.0,
        )
        self._lock = threading.Lock()
        self._snapshot = self._build_snapshot()
        self._running = True

    def _build_snapshot(self) -> dict:
        m = self.sim.metrics
        agents = self.sim.agents
        stats = {
            "tick": self.sim.world.tick,
            "alive": sum(a.alive for a in agents.values()),
            "population": len(agents),
            "deaths": m.deaths,
            "trades": m.trades,
            "gifts": m.gifts,
            "shelters": m.structures_built,
            "tools": m.tools_crafted,
            "attacks": m.predator_attacks,
            "alliances": m.alliances(agents),
            "meanTrust": m.mean_trust(agents),
            "gini": round(m.wealth_gini(agents), 3),
            "priceWoodForFood": m.mean_exchange_ratio("wood", "food"),
        }
        return world_snapshot(
            self.sim.world, agents, self.sim.recent_events, stats,
            self.sim.ecology.predators,
        )

    def snapshot_json(self) -> bytes:
        with self._lock:
            return json.dumps(self._snapshot).encode()

    def run_loop(self) -> None:
        while self._running:
            if any(a.alive for a in self.sim.agents.values()):
                self.sim.step()
            snap = self._build_snapshot()
            with self._lock:
                self._snapshot = snap
            time.sleep(self.tick_ms / 1000.0)

    def stop(self) -> None:
        self._running = False


def make_handler(live: LiveWorld):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):  # silence per-request logging
            pass

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: str, content_type: str = "text/html; charset=utf-8") -> None:
            try:
                with open(path, "rb") as fh:
                    self._send(200, fh.read(), content_type)
            except FileNotFoundError:
                self._send(404, b"not found", "text/plain")

        def do_GET(self):  # noqa: N802
            if self.path.startswith("/state"):
                self._send(200, live.snapshot_json(), "application/json")
            elif self.path == "/health":
                self._send(200, b'{"ok":true}', "application/json")
            elif self.path in ("/", "/index.html", "/3d", "/village_3d.html"):
                self._send_file(VIEWER_3D)
            elif self.path in ("/iso", "/village.html"):
                self._send_file(VIEWER_ISO)
            elif self.path.startswith("/vendor/"):
                # Only serve the exact vendored filenames — no directory traversal.
                name = self.path[len("/vendor/"):]
                if name not in {"three.module.min.js", "OrbitControls.js"}:
                    self._send(404, b"not found", "text/plain")
                    return
                self._send_file(os.path.join(VENDOR_DIR, name), "application/javascript")
            else:
                self._send(404, b"not found", "text/plain")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Emergent Civilization viz server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--size", type=int, default=14)
    parser.add_argument("--tick-ms", type=int, default=400, help="ms between ticks")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--llm", action="store_true",
        help="drive agents with a real LLM (needs EC_LLM_API_KEY/ANTHROPIC_API_KEY "
             "or EC_LLM_BASE_URL) instead of the heuristic baseline",
    )
    parser.add_argument(
        "--llm-gap-ms", type=int, default=8000,
        help="milliseconds to wait between each agent's LLM call within a tick "
             "(only used with --llm). Calls are spaced out evenly instead of "
             "bursting all agents back-to-back then going idle — this is what "
             "actually controls calls-per-minute against a provider's rate "
             "limit, independent of --agents or --tick-ms. Default (8000ms, "
             "~7.5 calls/min) stays under Groq's free-tier ~6000 TPM in the "
             "worst case. Set lower for paid/high-limit APIs, or 0 for a "
             "local Ollama server with no rate limit.",
    )
    args = parser.parse_args()

    if _loaded_env:
        print(f"[env] loaded {_loaded_env}")

    if args.llm and args.llm_gap_ms > 0:
        calls_per_min = 60000 / args.llm_gap_ms
        print(f"[llm] pacing one call every {args.llm_gap_ms}ms "
              f"(~{calls_per_min:.1f} calls/min total, regardless of --agents). "
              f"Lower --llm-gap-ms for a faster/paid backend, or 0 for local Ollama.")
    elif args.llm:
        print("[llm] --llm-gap-ms 0: agents call back-to-back with no pacing — "
              "fine for local Ollama or a high-limit paid API, but likely to hit "
              "free-tier rate limits otherwise.")

    try:
        live = LiveWorld(
            args.agents, args.size, args.tick_ms, args.seed,
            use_llm=args.llm, llm_gap_ms=args.llm_gap_ms,
        )
    except (RuntimeError, ValueError) as e:
        raise SystemExit(f"error: {e}")

    threading.Thread(target=live.run_loop, daemon=True).start()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(live))
    print(f"Emergent Civilization village live at http://localhost:{args.port}")
    mode = "LLM agents" if args.llm else "heuristic baseline"
    print(f"  {args.agents} agents ({mode}) · {args.size}x{args.size} world · {args.tick_ms}ms/tick")
    print("  3D view: /  (Three.js, orbit camera)   |   Isometric: /iso")
    print("  GET /state for the raw snapshot (Unity/other clients). Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        live.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
