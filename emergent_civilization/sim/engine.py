"""The simulation loop that ties world, agents, policies and metrics together.

One tick = metabolism for every agent, then one decided-and-executed action per
living agent, then world processes (resource regrowth), then measurement. Agent
order is shuffled each tick so no agent has a permanent first-mover advantage.
"""

from __future__ import annotations

import random
from typing import Any, Callable

from .actions import ActionExecutor
from .agent import Agent
from .metrics import Metrics
from .observation import build_observation
from .policy import Policy
from .types import ActionType
from .world import World

# A policy factory receives an agent and returns the Policy that drives it, so
# each agent can carry its own LLM context / personality.
PolicyFactory = Callable[[Agent], Policy]


class Simulation:
    def __init__(
        self,
        world: World,
        agents: list[Agent],
        policy_factory: PolicyFactory,
        seed: int | None = None,
    ) -> None:
        self.world = world
        self.agents: dict[str, Agent] = {a.id: a for a in agents}
        self.policies: dict[str, Policy] = {a.id: policy_factory(a) for a in agents}
        self.executor = ActionExecutor(self.world, self.agents)
        self.metrics = Metrics()
        self._rng = random.Random(seed)

    def step(self) -> None:
        # 1. Metabolism (may kill agents before they act).
        for agent in self.agents.values():
            was_alive = agent.alive
            agent.metabolize(self.world.tick)
            if was_alive and not agent.alive:
                self.metrics.deaths += 1

        # 2. Decisions and actions, in randomised order.
        order = [a for a in self.agents.values() if a.alive]
        self._rng.shuffle(order)
        for agent in order:
            obs = build_observation(self.world, agent, self.agents)
            action = self.policies[agent.id].decide(obs)
            result = self.executor.execute(agent, action)
            self.metrics.record_action(agent.id, action.type.value)
            if action.type == ActionType.GATHER and result.ok:
                self.metrics.record_gather(result.data["resource"], result.data["amount"])

        # 3. World processes + measurement.
        self.world.step()
        self.metrics.record_population(self.world, self.agents)

    def run(self, ticks: int, on_tick: Callable[["Simulation"], None] | None = None) -> Metrics:
        for _ in range(ticks):
            if not any(a.alive for a in self.agents.values()):
                break
            self.step()
            if on_tick is not None:
                on_tick(self)
        return self.metrics

    def summary(self) -> dict[str, Any]:
        return self.metrics.summary(self.world, self.agents)
