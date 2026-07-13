"""Core value types shared across the simulation.

The simulation deliberately keeps the *substrate* (world, resources, survival
rules, the set of possible actions) hard-coded here, while every *decision*
about which action to take is left to an agent policy (a heuristic today, an
LLM later). This module defines that fixed substrate vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Resource(str, Enum):
    """Things an agent can hold in its inventory.

    FOOD/WOOD/STONE are *raw* materials the world produces at resource nodes.
    TOOL is a *crafted* good — it never appears as a world node; it only comes
    into existence when an agent crafts one (see ``actions.py``). Keeping it in
    the same enum lets inventories, trades and gifts treat it uniformly.
    """

    FOOD = "food"
    WOOD = "wood"
    STONE = "stone"
    TOOL = "tool"

    @property
    def is_raw(self) -> bool:
        return self in (Resource.FOOD, Resource.WOOD, Resource.STONE)


class Direction(str, Enum):
    """The four cardinal moves on the grid."""

    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"

    @property
    def delta(self) -> tuple[int, int]:
        return {
            Direction.NORTH: (0, 1),
            Direction.SOUTH: (0, -1),
            Direction.EAST: (1, 0),
            Direction.WEST: (-1, 0),
        }[self]


class ActionType(str, Enum):
    """The complete action vocabulary given to agents.

    Not every action is meaningful in every development milestone (see
    ``docs/07_roadmap.md``). Actions past the current milestone are accepted by
    the parser but rejected at execution time with a clear reason, so agents can
    *discover* the boundary of what is possible rather than being silently
    ignored.
    """

    MOVE = "move"          # v0.1  reposition on the grid
    GATHER = "gather"      # v0.2  collect from a resource node on the current tile
    EAT = "eat"            # v0.1  consume FOOD from inventory to reduce hunger
    REST = "rest"          # v0.1  recover energy (faster inside a shelter)
    IDLE = "idle"          # v0.1  do nothing this tick
    SPEAK = "speak"        # v0.3  send a message to a nearby agent
    TRADE = "trade"        # v0.3  offer an exchange of resources to a nearby agent
    GIVE = "give"          # v0.4  hand resources to a nearby agent with nothing back
    CRAFT = "craft"        # v0.4  combine materials into a tool
    BUILD = "build"        # v0.4  spend materials to raise a shelter on this tile
    PROPOSE_RULE = "propose_rule"  # v0.7  suggest a norm to nearby agents


@dataclass(frozen=True)
class Action:
    """A single decision emitted by a policy for one agent on one tick."""

    type: ActionType
    args: dict[str, Any] = field(default_factory=dict)
    # Free-form rationale the agent may attach; useful for logging emergent
    # behaviour and for other agents' observations when actions are public.
    reason: str = ""

    def __str__(self) -> str:
        if self.args:
            arg_str = ", ".join(f"{k}={v}" for k, v in self.args.items())
            return f"{self.type.value}({arg_str})"
        return f"{self.type.value}()"


@dataclass
class ActionResult:
    """Outcome of attempting to execute an action, fed back to the agent."""

    ok: bool
    message: str = ""
    # Optional structured payload, e.g. amount gathered, item received in trade.
    data: dict[str, Any] = field(default_factory=dict)
