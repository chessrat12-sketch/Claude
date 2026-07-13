"""v0.1 / v0.2 demo: ten agents try to survive in a scattered world.

Runs fully offline using the heuristic baseline policy, so it needs no LLM and
no network. It exists to prove the substrate is playable — that survival is
possible but not free — before wiring in real LLM agents. Swap
``policy_factory`` for one that returns ``LLMPolicy`` to run the real
experiment.

    python -m examples.run_v01_survival
"""

from __future__ import annotations

import random

from sim import Agent, HeuristicPolicy, Personality, Simulation, make_scattered_world

NAMES = ["Aria", "Boaz", "Cira", "Doran", "Esme", "Finn", "Gwen", "Hodr", "Ivo", "Juno"]


def build_agents(world, seed: int = 7) -> list[Agent]:
    rng = random.Random(seed)
    agents = []
    for i, name in enumerate(NAMES):
        pos = (rng.randrange(world.width), rng.randrange(world.height))
        personality = Personality(
            greed=rng.random(),
            sociability=rng.random(),
            caution=rng.random(),
            curiosity=rng.random(),
        )
        agents.append(Agent(id=f"a{i}", name=name, pos=pos, personality=personality))
    return agents


def main() -> None:
    world = make_scattered_world(width=14, height=14, density=0.20, seed=1)
    agents = build_agents(world)

    sim = Simulation(
        world,
        agents,
        policy_factory=lambda agent: HeuristicPolicy(agent.personality),
        seed=42,
    )

    def report(s: Simulation) -> None:
        if s.world.tick % 20 == 0:
            alive = sum(a.alive for a in s.agents.values())
            print(f"  tick {s.world.tick:3d} | alive {alive:2d}/{len(s.agents)}")

    print("Emergent Civilization — v0.1 survival baseline (offline)\n")
    sim.run(ticks=200, on_tick=report)

    print("\n=== summary ===")
    summary = sim.summary()
    for key, value in summary.items():
        print(f"{key:20s}: {value}")


if __name__ == "__main__":
    main()
