"""Environmental pressure: night, exposure, and roaming predators.

This is the part of the substrate that makes the world *dangerous*, and danger
is what gives cooperation and shelter a payoff. None of it scripts a social
response — it only sets the incentives:

* **Night** drains extra energy from anyone caught in the open.
* **Predators** appear at night, hunt the nearest exposed agent, and bite —
  but a bite is halved when the target stands in a group (safety in numbers)
  and avoided entirely inside a shelter.

So agents who band together or build shelters survive the night; loners bleed.
Whether they actually do either is left to their policy.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .agent import Agent
from .world import World

GROUP_RADIUS = 1          # neighbours within this range count as "a group"
GROUP_SIZE_SAFE = 2       # this many living neighbours halves predator damage
PREDATOR_BITE = 14        # health lost to a bite in the open, alone
PREDATOR_REACH = 1        # chebyshev distance at which a predator can bite
NIGHT_EXPOSURE = 3        # extra energy drained per night tick when unsheltered


@dataclass
class Predator:
    id: str
    pos: tuple[int, int]


@dataclass
class Ecology:
    """Owns predators and applies night/exposure effects each tick."""

    max_predators_per_10: float = 1.5   # predators scale with population
    _predators: dict[str, Predator] = field(default_factory=dict)
    _seq: int = 0

    @property
    def predators(self) -> list[Predator]:
        return list(self._predators.values())

    def update(self, world: World, agents: dict[str, Agent], rng: random.Random) -> list[dict]:
        """Advance predators and apply exposure. Returns events for metrics/UI."""
        events: list[dict] = []
        living = [a for a in agents.values() if a.alive]

        if not world.is_night:
            self._predators.clear()   # predators retreat at daybreak
            return events

        self._spawn(world, len(living), rng)
        events += self._apply_exposure(world, living)
        events += self._hunt(world, living, rng)
        return events

    # -- internals --------------------------------------------------------
    def _spawn(self, world: World, n_living: int, rng: random.Random) -> None:
        cap = max(1, round(n_living / 10 * self.max_predators_per_10))
        while len(self._predators) < cap:
            self._seq += 1
            edge = rng.choice(
                [(rng.randrange(world.width), 0), (rng.randrange(world.width), world.height - 1),
                 (0, rng.randrange(world.height)), (world.width - 1, rng.randrange(world.height))]
            )
            self._predators[f"p{self._seq}"] = Predator(f"p{self._seq}", edge)

    def _apply_exposure(self, world: World, living: list[Agent]) -> list[dict]:
        for a in living:
            if world.shelter_at(a.pos) is None:
                a.energy = max(0, a.energy - NIGHT_EXPOSURE)
        return []

    def _hunt(self, world: World, living: list[Agent], rng: random.Random) -> list[dict]:
        events: list[dict] = []
        exposed = [a for a in living if world.shelter_at(a.pos) is None]
        for pred in self._predators.values():
            if not exposed:
                # wander if no prey is in the open
                pred.pos = _step(pred.pos, rng.choice(exposed or living).pos, world) \
                    if living else pred.pos
                continue
            target = min(exposed, key=lambda a: _chebyshev(pred.pos, a.pos))
            pred.pos = _step(pred.pos, target.pos, world)
            if _chebyshev(pred.pos, target.pos) <= PREDATOR_REACH:
                neighbours = sum(
                    1 for o in living
                    if o.id != target.id and _chebyshev(o.pos, target.pos) <= GROUP_RADIUS
                )
                bite = PREDATOR_BITE // 2 if neighbours >= GROUP_SIZE_SAFE else PREDATOR_BITE
                target.hurt(world.tick, bite, "predator")
                events.append(
                    {"type": "attack", "a": pred.id, "b": target.id,
                     "text": f"🐺 {target.name} bitten ({bite} dmg"
                             + (", but the group blunted it)" if neighbours >= GROUP_SIZE_SAFE else ")")}
                )
        return events


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _step(frm: tuple[int, int], to: tuple[int, int], world: World) -> tuple[int, int]:
    dx = (to[0] > frm[0]) - (to[0] < frm[0])
    dy = (to[1] > frm[1]) - (to[1] < frm[1])
    nxt = (frm[0] + dx, frm[1] + dy)
    return nxt if world.in_bounds(nxt) else frm
