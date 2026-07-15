"""The simulation loop that ties world, agents, policies and metrics together.

One tick = metabolism for every agent, then one decided-and-executed action per
living agent, then world processes (resource regrowth), then measurement. Agent
order is shuffled each tick so no agent has a permanent first-mover advantage.
"""

from __future__ import annotations

import random
import time
from typing import Any, Callable

from .actions import ActionExecutor
from .agent import Agent
from .ecology import Ecology
from .interactions import Interactions
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
        decision_gap: float = 0.0,
    ) -> None:
        self.world = world
        self.agents: dict[str, Agent] = {a.id: a for a in agents}
        self.policies: dict[str, Policy] = {a.id: policy_factory(a) for a in agents}
        self.interactions = Interactions()
        self.executor = ActionExecutor(self.world, self.agents, self.interactions)
        self.ecology = Ecology()
        self.metrics = Metrics()
        # Events (trades, speech, gifts, builds, predator attacks) produced
        # during the most recent tick, for metrics and the render snapshot.
        self.recent_events: list[dict] = []
        self._dead: set[str] = set()
        self._rng = random.Random(seed)
        # Seconds to sleep between each agent's decision within a tick. Zero
        # for the offline/heuristic path (default); a live LLM run sets this
        # so calls land one-at-a-time, evenly spaced, instead of bursting all
        # agents back-to-back and then going silent for the rest of the tick.
        # This is what actually controls calls-per-minute against a provider's
        # rate limit — independent of agent count or --tick-ms.
        self.decision_gap = decision_gap

    def step(self) -> None:
        # 1. Metabolism (may kill agents before they act).
        for agent in self.agents.values():
            agent.metabolize(self.world.tick)

        # 2. Decisions and actions, in randomised order.
        self.executor.events.clear()
        order = [a for a in self.agents.values() if a.alive]
        self._rng.shuffle(order)
        predators = self.ecology.predators
        for i, agent in enumerate(order):
            if i > 0 and self.decision_gap > 0:
                time.sleep(self.decision_gap)
            obs = build_observation(
                self.world, agent, self.agents, self.interactions, predators
            )
            action = self.policies[agent.id].decide(obs)
            result = self.executor.execute(agent, action)
            self.interactions.clear_inbox(agent.id)
            self.metrics.record_action(agent.id, action.type.value)
            if result.ok and action.type == ActionType.GATHER:
                self.metrics.record_gather(result.data["resource"], result.data["amount"])
            elif result.ok and action.type == ActionType.CRAFT:
                self.metrics.tools_crafted += 1

        # 3. Ecology: night, exposure, predators (can kill agents).
        eco_events = self.ecology.update(self.world, self.agents, self._rng)

        # 4. Drain events into metrics + record any new deaths with their cause.
        self.recent_events = list(self.executor.events) + eco_events
        for ev in self.recent_events:
            self.metrics.record_event(ev)
        for agent in self.agents.values():
            if not agent.alive and agent.id not in self._dead:
                self._dead.add(agent.id)
                self.metrics.record_death(agent.cause_of_death or "unknown")

        # 5. World processes, offer expiry + measurement.
        self.world.step()
        self.interactions.expire(self.world.tick)
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
