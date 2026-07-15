"""The simulation loop that ties world, agents, policies and metrics together.

One tick = metabolism for every agent, then one decided-and-executed action per
living agent, then world processes (resource regrowth), then measurement. Agent
order is shuffled each tick so no agent has a permanent first-mover advantage.
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from .actions import ActionExecutor
from .agent import Agent
from .ecology import Ecology
from .interactions import Interactions
from .metrics import Metrics
from .observation import build_observation
from .policy import Policy
from .types import Action, ActionType
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
        max_concurrency: int = 1,
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
        # Minimum seconds between the *start* of one agent's decision and the
        # next — across the whole run, not just within one tick. Zero for the
        # offline/heuristic path (default); a live LLM run sets this so calls
        # land one-at-a-time, evenly spaced, instead of bursting all agents
        # back-to-back and then going silent for the rest of the tick. This is
        # what actually controls calls-per-minute against a provider's rate
        # limit — independent of agent count or --tick-ms.
        #
        # Enforced by timestamp rather than a flat sleep-every-call: the wait
        # is `decision_gap` minus whatever time already elapsed (including the
        # previous call's own latency and the tick's world-processing time),
        # so a slow API call is credited toward the gap instead of the gap
        # being added on top of it. Tracking runs across tick boundaries too,
        # so the pacing holds even at the seam between one step() and the next.
        self.decision_gap = decision_gap
        self._last_decision_time: float | None = None
        # Number of agents whose decisions can be in flight at once. 1 (the
        # default) preserves the original one-at-a-time behaviour exactly —
        # every existing single-threaded test relies on that. Set this above 1
        # only when the backend can actually take concurrent load (e.g. your
        # own RunPod vLLM deployment, which is built for batched concurrent
        # requests) — it's what makes 100-agent scale practical, since a
        # shared free-tier API has no such headroom regardless of concurrency.
        # decision_gap is ignored when max_concurrency > 1: per-call pacing
        # and a concurrent worker pool are different throttling strategies for
        # different situations, and mixing them doesn't compose meaningfully.
        self.max_concurrency = max(1, max_concurrency)

    def step(self) -> None:
        # 1. Metabolism (may kill agents before they act).
        for agent in self.agents.values():
            agent.metabolize(self.world.tick)

        # 2. Decisions and actions, in randomised order.
        self.executor.events.clear()
        order = [a for a in self.agents.values() if a.alive]
        self._rng.shuffle(order)
        predators = self.ecology.predators
        # "thought" events — each agent's chosen action plus its own stated
        # reason, if any. Not used for anything mechanical; this is purely so
        # a human watching can see *why* an agent did what it did (the
        # research design's qualitative evidence trail — see
        # docs/05_prompt_and_decision_loop.md).
        thought_events: list[dict] = []

        if self.max_concurrency > 1:
            # Concurrent path: every agent decides against the same
            # start-of-decision-phase world state (not each other's
            # just-applied actions), then actions resolve afterward in the
            # shuffled order — a normal simultaneous-decide/sequential-resolve
            # pattern, and the only way to actually use a high-throughput
            # backend (e.g. your own RunPod vLLM deployment, built for batched
            # concurrent requests) concurrently instead of one call at a time.
            observations = {
                agent.id: build_observation(
                    self.world, agent, self.agents, self.interactions, predators
                )
                for agent in order
            }
            with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
                futures = {
                    agent.id: pool.submit(self.policies[agent.id].decide, observations[agent.id])
                    for agent in order
                }
                actions = {agent_id: fut.result() for agent_id, fut in futures.items()}
            for agent in order:
                self._resolve_action(agent, actions[agent.id], thought_events)
        else:
            # Sequential path (default, max_concurrency=1): each agent decides
            # against the latest state, immediately after the previous
            # agent's action already applied. Unchanged from the original
            # single-threaded behaviour — every existing test relies on this.
            for agent in order:
                if self.decision_gap > 0:
                    if self._last_decision_time is not None:
                        wait = self.decision_gap - (time.monotonic() - self._last_decision_time)
                        if wait > 0:
                            time.sleep(wait)
                    self._last_decision_time = time.monotonic()
                obs = build_observation(
                    self.world, agent, self.agents, self.interactions, predators
                )
                action = self.policies[agent.id].decide(obs)
                self._resolve_action(agent, action, thought_events)

        # 3. Ecology: night, exposure, predators (can kill agents).
        eco_events = self.ecology.update(self.world, self.agents, self._rng)

        # 4. Drain events into metrics + record any new deaths with their cause.
        self.recent_events = thought_events + list(self.executor.events) + eco_events
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

    def _resolve_action(self, agent: Agent, action: Action, thought_events: list[dict]) -> None:
        """Execute one already-decided action and record its effects."""
        result = self.executor.execute(agent, action)
        self.interactions.clear_inbox(agent.id)
        self.metrics.record_action(agent.id, action.type.value)
        if result.ok and action.type == ActionType.GATHER:
            self.metrics.record_gather(result.data["resource"], result.data["amount"])
        elif result.ok and action.type == ActionType.CRAFT:
            self.metrics.tools_crafted += 1
        text = f"{agent.name}: {action.type.value}"
        if action.reason:
            text += f" — {action.reason}"
        thought_events.append({"type": "thought", "a": agent.id, "b": "", "text": text})

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
