"""The agent: a persistent bundle of body, mind, and social ties.

Each agent owns its own memory, personality, goals, inventory and
relationships. Crucially there is *no* ``job`` field — roles are expected to
emerge from what agents repeatedly choose to do.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .types import Resource

# Survival tuning constants (the human-designed "biology").
MAX_VITAL = 100
HUNGER_PER_TICK = 4          # hunger accrues every tick
ENERGY_PER_TICK = 2          # energy drains every tick
STARVING_HUNGER = 90         # above this, health decays
EXHAUSTED_ENERGY = 10        # below this, health decays
HEALTH_DECAY = 6             # health lost per tick while starving/exhausted
HEALTH_REGEN = 2             # health recovered per tick while well-fed & rested


@dataclass
class Personality:
    """A handful of [0, 1] traits that colour an agent's decisions.

    These bias a heuristic policy today and are injected into the LLM prompt
    later, but they never *force* an action — they are dispositions, not rules.
    """

    greed: float = 0.5        # tendency to hoard vs. share
    sociability: float = 0.5  # tendency to seek out others
    caution: float = 0.5      # tendency to prioritise survival margins
    curiosity: float = 0.5    # tendency to explore vs. exploit


@dataclass
class MemoryEvent:
    tick: int
    kind: str
    detail: str


@dataclass
class Agent:
    id: str
    name: str
    pos: tuple[int, int]
    personality: Personality = field(default_factory=Personality)
    goal: str = "survive"

    hunger: int = 0
    energy: int = MAX_VITAL
    health: int = MAX_VITAL
    alive: bool = True
    born_tick: int = 0
    died_tick: int | None = None
    cause_of_death: str | None = None

    inventory: dict[Resource, int] = field(default_factory=dict)
    # agent_id -> trust score in [-1, 1]; grows with kept promises, decays on
    # betrayal. Unused until v0.6 but carried from the start so history exists.
    relationships: dict[str, float] = field(default_factory=dict)
    memory: deque[MemoryEvent] = field(default_factory=lambda: deque(maxlen=64))

    # -- inventory helpers -----------------------------------------------
    def held(self, resource: Resource) -> int:
        return self.inventory.get(resource, 0)

    def add(self, resource: Resource, amount: int) -> None:
        if amount <= 0:
            return
        self.inventory[resource] = self.held(resource) + amount

    def remove(self, resource: Resource, amount: int) -> bool:
        if self.held(resource) < amount:
            return False
        self.inventory[resource] -= amount
        if self.inventory[resource] == 0:
            del self.inventory[resource]
        return True

    def remember(self, tick: int, kind: str, detail: str) -> None:
        self.memory.append(MemoryEvent(tick, kind, detail))

    def kill(self, tick: int, cause: str) -> None:
        if not self.alive:
            return
        self.alive = False
        self.died_tick = tick
        self.cause_of_death = cause
        self.remember(tick, "death", cause)

    def hurt(self, tick: int, amount: int, cause: str) -> None:
        """Apply external damage (e.g. a predator) and check for death."""
        self.health = max(0, self.health - amount)
        if self.health <= 0:
            self.kill(tick, cause)

    # -- metabolism -------------------------------------------------------
    def metabolize(self, tick: int) -> None:
        """Apply passive survival dynamics for one tick."""
        if not self.alive:
            return
        self.hunger = min(MAX_VITAL, self.hunger + HUNGER_PER_TICK)
        self.energy = max(0, self.energy - ENERGY_PER_TICK)

        starving = self.hunger >= STARVING_HUNGER
        exhausted = self.energy <= EXHAUSTED_ENERGY
        if starving or exhausted:
            self.health = max(0, self.health - HEALTH_DECAY)
        elif self.hunger < 50 and self.energy > 40:
            self.health = min(MAX_VITAL, self.health + HEALTH_REGEN)

        if self.health <= 0:
            self.kill(tick, "starvation" if starving else "exhaustion")

    @property
    def wealth(self) -> int:
        return sum(self.inventory.values())
