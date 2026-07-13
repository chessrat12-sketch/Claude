"""Collects the quantitative signals the research questions care about.

The proposal asks whether structure *emerges*: division of labour, trade, an
economy, trust, norms. This collector records the raw series from which those
higher-order phenomena are measured. Milestones past v0.2 (trade, trust, rules)
are wired here as counters so the plumbing exists before the behaviour does.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .agent import Agent
from .types import Resource
from .world import World


@dataclass
class Metrics:
    # Population.
    alive_over_time: list[int] = field(default_factory=list)
    deaths: int = 0
    deaths_by_cause: Counter = field(default_factory=Counter)

    # Activity (which action each agent takes — the raw signal for "did a
    # division of labour appear?").
    action_counts: Counter = field(default_factory=Counter)
    gather_by_resource: Counter = field(default_factory=Counter)
    per_agent_actions: dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )

    # Economy (populated from v0.3 onward).
    trades: int = 0
    price_log: list[dict] = field(default_factory=list)

    # Production & shelter (v0.4).
    tools_crafted: int = 0
    structures_built: int = 0
    gifts: int = 0

    # Threats (ecology).
    predator_attacks: int = 0

    # Society scaffolding (rule proposals populated from v0.7 onward).
    messages: int = 0
    rule_proposals: int = 0

    # -- recording --------------------------------------------------------
    def record_action(self, agent_id: str, action_type: str) -> None:
        self.action_counts[action_type] += 1
        self.per_agent_actions[agent_id][action_type] += 1

    def record_gather(self, resource: str, amount: int) -> None:
        self.gather_by_resource[resource] += amount

    def record_trade(self, give: dict, receive: dict) -> None:
        self.trades += 1
        self.price_log.append({"give": dict(give), "receive": dict(receive)})

    def record_message(self) -> None:
        self.messages += 1

    def record_event(self, ev: dict) -> None:
        """Fold a social/ecology event into the right counter."""
        kind = ev.get("type")
        if kind == "trade":
            self.record_trade(ev["give"], ev["receive"])
        elif kind == "speak":
            self.messages += 1
        elif kind == "gift":
            self.gifts += 1
        elif kind == "build":
            self.structures_built += 1
        elif kind == "attack":
            self.predator_attacks += 1

    def record_death(self, cause: str) -> None:
        self.deaths += 1
        self.deaths_by_cause[cause or "unknown"] += 1

    # -- relationships ----------------------------------------------------
    def alliances(self, agents: dict[str, Agent], threshold: float = 0.3) -> int:
        """Count mutually-trusting living pairs — the seed of alliances."""
        living = [a for a in agents.values() if a.alive]
        count = 0
        for i, a in enumerate(living):
            for b in living[i + 1:]:
                if a.relationships.get(b.id, 0) >= threshold and \
                        b.relationships.get(a.id, 0) >= threshold:
                    count += 1
        return count

    def mean_trust(self, agents: dict[str, Agent]) -> float:
        vals = [v for a in agents.values() for v in a.relationships.values()]
        return round(statistics.mean(vals), 3) if vals else 0.0

    def mean_exchange_ratio(self, give_res: str, recv_res: str) -> float | None:
        """Average units of ``recv_res`` paid per unit of ``give_res``.

        This is the emergent "price" — never set anywhere, only read back out of
        the trades that actually happened. Returns None if that pair never traded.
        """
        ratios = []
        for entry in self.price_log:
            g, r = entry["give"], entry["receive"]
            if list(g) == [give_res] and list(r) == [recv_res] and g[give_res]:
                ratios.append(r[recv_res] / g[give_res])
        return round(statistics.mean(ratios), 3) if ratios else None

    def record_population(self, world: World, agents: dict[str, Agent]) -> None:
        self.alive_over_time.append(sum(1 for a in agents.values() if a.alive))

    # -- analysis ---------------------------------------------------------
    def wealth_gini(self, agents: dict[str, Agent]) -> float:
        """Gini coefficient of inventory size — a wealth-concentration proxy."""
        wealth = sorted(a.wealth for a in agents.values() if a.alive)
        n = len(wealth)
        if n == 0 or sum(wealth) == 0:
            return 0.0
        cum = sum((i + 1) * w for i, w in enumerate(wealth))
        return (2 * cum) / (n * sum(wealth)) - (n + 1) / n

    def role_specialization(self) -> dict[str, str]:
        """Rough "job" label per agent = its most frequent action.

        A pure eyeball on whether agents differentiated. Not a claim of a real
        role — just the dominant behaviour, which is where roles would first
        show up.
        """
        roles = {}
        for agent_id, counts in self.per_agent_actions.items():
            if counts:
                roles[agent_id] = counts.most_common(1)[0][0]
        return roles

    def summary(self, world: World, agents: dict[str, Agent]) -> dict:
        survival_ticks = [
            (a.died_tick if a.died_tick is not None else world.tick) - a.born_tick
            for a in agents.values()
        ]
        return {
            "ticks": world.tick,
            "alive": self.alive_over_time[-1] if self.alive_over_time else 0,
            "deaths": self.deaths,
            "deaths_by_cause": dict(self.deaths_by_cause),
            "mean_survival_ticks": round(statistics.mean(survival_ticks), 1)
            if survival_ticks
            else 0,
            "action_counts": dict(self.action_counts),
            "gather_by_resource": dict(self.gather_by_resource),
            "tools_crafted": self.tools_crafted,
            "structures_built": self.structures_built,
            "gifts": self.gifts,
            "predator_attacks": self.predator_attacks,
            "trades": self.trades,
            "messages": self.messages,
            "price_wood_for_food": self.mean_exchange_ratio("wood", "food"),
            "alliances": self.alliances(agents),
            "mean_trust": self.mean_trust(agents),
            "wealth_gini": round(self.wealth_gini(agents), 3),
            "world_resources": {r.value: n for r, n in world.total_resources().items()},
            "roles": self.role_specialization(),
        }
