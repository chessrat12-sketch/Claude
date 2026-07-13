"""LLM-agent demo: every NPC is driven by an LLM through ``LLMPolicy``.

This runs the *experiment* wiring (partial observation -> prompt -> model ->
JSON action), and prints each agent's chosen action together with the free-form
``reason`` it gave — the qualitative trace of *why* it cooperated, traded, built
or fled. That reasoning log is the narrative evidence the research design cares
about (see ``docs/05_prompt_and_decision_loop.md``).

Backend selection (first that applies):
  * EC_LLM_BASE_URL set        -> OpenAICompatBackend  (RunPod vLLM / NVIDIA API)
  * EC_LLM_API_KEY / ANTHROPIC_API_KEY set -> AnthropicBackend
  * otherwise                  -> MockReasoningBackend (offline, no network)

    python -m examples.run_llm_agents
    EC_LLM_BASE_URL=... EC_LLM_MODEL=... EC_LLM_API_KEY=... python -m examples.run_llm_agents
"""

from __future__ import annotations

import os
import random

from sim import Agent, LLMPolicy, Personality, Simulation, make_scattered_world
from sim.llm_client import (
    AnthropicBackend,
    MockReasoningBackend,
    OpenAICompatBackend,
)

NAMES = ["Aria", "Boaz", "Cira", "Doran", "Esme", "Finn"]


def pick_backend():
    if os.environ.get("EC_LLM_BASE_URL"):
        print("backend: OpenAICompatBackend (", os.environ["EC_LLM_BASE_URL"], ")")
        return OpenAICompatBackend()
    if os.environ.get("EC_LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
        print("backend: AnthropicBackend")
        return AnthropicBackend()
    print("backend: MockReasoningBackend (offline — set EC_LLM_* for a real model)")
    return MockReasoningBackend()


def main() -> None:
    backend = pick_backend()
    rng = random.Random(11)
    world = make_scattered_world(width=10, height=10, density=0.22, seed=11)
    agents = [
        Agent(
            id=f"a{i}", name=NAMES[i], pos=(rng.randrange(10), rng.randrange(10)),
            personality=Personality(greed=rng.random(), sociability=rng.random(),
                                    caution=rng.random(), curiosity=rng.random()),
        )
        for i in range(len(NAMES))
    ]
    sim = Simulation(world, agents, policy_factory=lambda a: LLMPolicy(a, backend), seed=11)

    print("\nEmergent Civilization — LLM agents (reasoning trace)\n")
    for _ in range(24):
        sim.step()
        # Show what each agent just decided and why (from their memory + reason).
        night = "🌙" if sim.world.is_night else "☀️"
        line = []
        for a in sim.agents.values():
            if a.alive and a.memory:
                line.append(f"{a.name}:{a.memory[-1].kind}")
        print(f"tick {sim.world.tick:2d} {night}  " + "  ".join(line))

    print("\n=== summary ===")
    for k, v in sim.summary().items():
        print(f"{k:20s}: {v}")


if __name__ == "__main__":
    main()
