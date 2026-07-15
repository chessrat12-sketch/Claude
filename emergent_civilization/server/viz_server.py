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

    # Drive agents with a real LLM instead of the heuristic baseline. Pick one:

    # (a) Anthropic
    export EC_LLM_API_KEY=<anthropic-key>          # or ANTHROPIC_API_KEY
    python -m server.viz_server --llm --agents 4 --tick-ms 2000

    # (b) NVIDIA API Catalog (free tier) — build.nvidia.com
    export EC_LLM_BASE_URL=https://integrate.api.nvidia.com/v1
    export EC_LLM_MODEL=meta/llama-3.1-8b-instruct   # or any model you enabled
    export EC_LLM_API_KEY=<nvidia-key>
    python -m server.viz_server --llm --agents 4 --tick-ms 2000

    # (c) Google Gemini (free tier, OpenAI-compatible endpoint) — aistudio.google.com/apikey
    export EC_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
    export EC_LLM_MODEL=gemini-2.0-flash
    export EC_LLM_API_KEY=<google-ai-studio-key>
    python -m server.viz_server --llm --agents 4 --tick-ms 2000

    # (d) xAI Grok (OpenAI-compatible endpoint) — console.x.ai
    export EC_LLM_BASE_URL=https://api.x.ai/v1
    export EC_LLM_MODEL=grok-2-latest        # check console.x.ai for the current name
    export EC_LLM_API_KEY=<xai-key>
    python -m server.viz_server --llm --agents 4 --tick-ms 2000

    # (e) Fully local (Ollama) — no cloud, no key needed beyond a placeholder
    export EC_LLM_BASE_URL=http://localhost:11434/v1
    export EC_LLM_MODEL=llama3.1
    export EC_LLM_API_KEY=ollama
    python -m server.viz_server --llm --agents 4 --tick-ms 3000

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
from sim.llm_client import pick_backend_from_env
from sim.snapshot import world_snapshot

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
        self, n_agents: int, size: int, tick_ms: int, seed: int, use_llm: bool = False
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
        self.sim = Simulation(world, agents, policy_factory=policy_factory, seed=seed)
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
    args = parser.parse_args()

    if args.llm and args.tick_ms < 1500:
        print(f"[llm] note: {args.tick_ms}ms/tick is tight for API latency — each tick "
              f"calls the model once per agent, one after another. Consider --tick-ms 2000+ "
              f"and a small --agents count to keep it responsive and cheap.")

    try:
        live = LiveWorld(args.agents, args.size, args.tick_ms, args.seed, use_llm=args.llm)
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
