"""Visualisation server: run the simulation live and stream world snapshots.

Advances the simulation on a background thread at a fixed tick rate and exposes
the latest render snapshot over HTTP. Both the bundled browser viewer and a
Unity client consume the same ``GET /state`` endpoint, so the 3D village and the
zero-setup web view show identical worlds.

    python -m server.viz_server              # then open http://localhost:8000
    python -m server.viz_server --port 9000 --tick-ms 300 --agents 12

Uses only the Python standard library (no framework, no extra deps). Swap the
policy factory for ``LLMPolicy`` to visualise real LLM agents.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sim import Agent, HeuristicPolicy, Personality, Simulation, make_scattered_world
from sim.snapshot import world_snapshot

NAMES = [
    "Aria", "Boaz", "Cira", "Doran", "Esme", "Finn", "Gwen", "Hodr", "Ivo",
    "Juno", "Kai", "Lena", "Milo", "Nadia", "Oren", "Pia", "Quin", "Rhea",
]
VIEWER_HTML = os.path.join(os.path.dirname(os.path.dirname(__file__)), "viewer", "village.html")


class LiveWorld:
    """Owns the simulation and the latest snapshot behind a lock."""

    def __init__(self, n_agents: int, size: int, tick_ms: int, seed: int) -> None:
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
        self.sim = Simulation(
            world, agents,
            policy_factory=lambda a: HeuristicPolicy(a.personality),
            seed=seed,
        )
        self._lock = threading.Lock()
        self._snapshot = self._build_snapshot()
        self._running = True

    def _build_snapshot(self) -> dict:
        m = self.sim.metrics
        stats = {
            "tick": self.sim.world.tick,
            "alive": sum(a.alive for a in self.sim.agents.values()),
            "population": len(self.sim.agents),
            "deaths": m.deaths,
            "trades": m.trades,
            "messages": m.messages,
            "gini": round(m.wealth_gini(self.sim.agents), 3),
            "priceWoodForFood": m.mean_exchange_ratio("wood", "food"),
        }
        return world_snapshot(self.sim.world, self.sim.agents, self.sim.recent_events, stats)

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

        def do_GET(self):  # noqa: N802
            if self.path.startswith("/state"):
                self._send(200, live.snapshot_json(), "application/json")
            elif self.path == "/health":
                self._send(200, b'{"ok":true}', "application/json")
            elif self.path in ("/", "/index.html", "/village.html"):
                try:
                    with open(VIEWER_HTML, "rb") as fh:
                        self._send(200, fh.read(), "text/html; charset=utf-8")
                except FileNotFoundError:
                    self._send(404, b"viewer not found", "text/plain")
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
    args = parser.parse_args()

    live = LiveWorld(args.agents, args.size, args.tick_ms, args.seed)
    threading.Thread(target=live.run_loop, daemon=True).start()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(live))
    print(f"Emergent Civilization village live at http://localhost:{args.port}")
    print(f"  {args.agents} agents · {args.size}x{args.size} world · {args.tick_ms}ms/tick")
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
