"""The world substrate: a grid of tiles carrying regenerating resource nodes.

Humans design *this* — the map and the resources. Nothing here encodes jobs,
economy, or law; those are meant to emerge from agent behaviour.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .types import Resource


@dataclass
class ResourceNode:
    """A depletable, slowly regenerating source of one resource on a tile."""

    resource: Resource
    amount: int
    capacity: int
    regen_per_tick: float = 0.5
    _regen_accum: float = 0.0

    def harvest(self, requested: int) -> int:
        """Remove up to ``requested`` units, returning how many were taken."""
        taken = min(requested, self.amount)
        self.amount -= taken
        return taken

    def regenerate(self) -> None:
        if self.amount >= self.capacity:
            return
        self._regen_accum += self.regen_per_tick
        grown = int(self._regen_accum)
        if grown:
            self._regen_accum -= grown
            self.amount = min(self.capacity, self.amount + grown)


@dataclass
class Tile:
    x: int
    y: int
    node: ResourceNode | None = None


@dataclass
class Structure:
    """A shelter raised by an agent. Protects occupants from night threats and
    speeds rest. Ownership is recorded (the seed of property/territory) but the
    shelter physically protects whoever stands on it."""

    pos: tuple[int, int]
    owner_id: str
    built_tick: int
    durability: int = 100


@dataclass
class World:
    """A finite 2D grid with a day/night cycle. Edges block movement (no wrap)."""

    width: int
    height: int
    tiles: dict[tuple[int, int], Tile] = field(default_factory=dict)
    structures: dict[tuple[int, int], Structure] = field(default_factory=dict)
    tick: int = 0
    day_length: int = 24          # ticks per full day
    night_fraction: float = 0.4   # the last 40% of each day is night

    def __post_init__(self) -> None:
        for x in range(self.width):
            for y in range(self.height):
                self.tiles.setdefault((x, y), Tile(x, y))

    # -- day / night ------------------------------------------------------
    @property
    def time_of_day(self) -> int:
        return self.tick % self.day_length

    @property
    def is_night(self) -> bool:
        return self.time_of_day >= self.day_length * (1 - self.night_fraction)

    def shelter_at(self, pos: tuple[int, int]) -> Structure | None:
        return self.structures.get(pos)

    # -- geometry ---------------------------------------------------------
    def in_bounds(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def tile(self, pos: tuple[int, int]) -> Tile:
        return self.tiles[pos]

    def neighbours(self, pos: tuple[int, int], radius: int = 1) -> list[Tile]:
        x, y = pos
        out = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                p = (x + dx, y + dy)
                if p != pos and self.in_bounds(p):
                    out.append(self.tiles[p])
        return out

    # -- lifecycle --------------------------------------------------------
    def step(self) -> None:
        """Advance world processes by one tick (resource regrowth)."""
        self.tick += 1
        for tile in self.tiles.values():
            if tile.node is not None:
                tile.node.regenerate()

    def total_resources(self) -> dict[Resource, int]:
        totals: dict[Resource, int] = {r: 0 for r in Resource}
        for tile in self.tiles.values():
            if tile.node is not None:
                totals[tile.node.resource] += tile.node.amount
        return totals


def make_scattered_world(
    width: int = 12,
    height: int = 12,
    density: float = 0.18,
    seed: int | None = None,
) -> World:
    """Create a world with resource nodes scattered at random tiles.

    ``density`` is the approximate fraction of tiles carrying a node. Food is
    over-represented so that pure survival is possible but not trivial.
    """
    rng = random.Random(seed)
    world = World(width, height)
    weighted = [Resource.FOOD, Resource.FOOD, Resource.WOOD, Resource.STONE]
    for tile in world.tiles.values():
        if rng.random() < density:
            resource = rng.choice(weighted)
            capacity = rng.randint(6, 14)
            tile.node = ResourceNode(
                resource=resource,
                amount=capacity,
                capacity=capacity,
                regen_per_tick=0.3 if resource == Resource.FOOD else 0.15,
            )
    return world
