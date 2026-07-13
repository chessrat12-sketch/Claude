"""Execution of agent actions against the world.

This is the *rules engine*: it decides what an action does and whether it is
currently allowed. It never decides *which* action an agent takes — that is the
policy's job. Actions belonging to a future milestone return a failing
``ActionResult`` explaining why, so the boundary of the possible is legible to
agents rather than hidden.
"""

from __future__ import annotations

from .agent import Agent
from .types import Action, ActionResult, ActionType, Direction, Resource
from .world import World

# The milestone this build implements. Actions above this are parsed but
# refused at execution time. Bump as the roadmap in docs/07_roadmap.md advances.
IMPLEMENTED_ACTIONS = {
    ActionType.MOVE,
    ActionType.GATHER,
    ActionType.EAT,
    ActionType.REST,
    ActionType.IDLE,
}

EAT_HUNGER_RELIEF = 35
REST_ENERGY_GAIN = 30
GATHER_PER_ACTION = 3


class ActionExecutor:
    """Applies actions to (world, agents) and returns per-action feedback."""

    def __init__(self, world: World, agents: dict[str, Agent]) -> None:
        self.world = world
        self.agents = agents

    def execute(self, agent: Agent, action: Action) -> ActionResult:
        if not agent.alive:
            return ActionResult(False, "agent is dead")

        if action.type not in IMPLEMENTED_ACTIONS:
            return ActionResult(
                False,
                f"action '{action.type.value}' is not available in this "
                f"milestone yet",
            )

        handler = {
            ActionType.MOVE: self._move,
            ActionType.GATHER: self._gather,
            ActionType.EAT: self._eat,
            ActionType.REST: self._rest,
            ActionType.IDLE: self._idle,
        }[action.type]
        result = handler(agent, action)
        agent.remember(self.world.tick, action.type.value, result.message)
        return result

    # -- handlers ---------------------------------------------------------
    def _move(self, agent: Agent, action: Action) -> ActionResult:
        raw = action.args.get("direction")
        try:
            direction = Direction(raw)
        except ValueError:
            return ActionResult(False, f"unknown direction '{raw}'")
        dx, dy = direction.delta
        target = (agent.pos[0] + dx, agent.pos[1] + dy)
        if not self.world.in_bounds(target):
            return ActionResult(False, f"edge of the world to the {direction.value}")
        agent.pos = target
        return ActionResult(True, f"moved {direction.value} to {target}")

    def _gather(self, agent: Agent, _action: Action) -> ActionResult:
        tile = self.world.tile(agent.pos)
        if tile.node is None or tile.node.amount <= 0:
            return ActionResult(False, "nothing to gather here")
        taken = tile.node.harvest(GATHER_PER_ACTION)
        agent.add(tile.node.resource, taken)
        return ActionResult(
            True,
            f"gathered {taken} {tile.node.resource.value}",
            {"resource": tile.node.resource.value, "amount": taken},
        )

    def _eat(self, agent: Agent, _action: Action) -> ActionResult:
        if not agent.remove(Resource.FOOD, 1):
            return ActionResult(False, "no food to eat")
        agent.hunger = max(0, agent.hunger - EAT_HUNGER_RELIEF)
        return ActionResult(True, "ate 1 food", {"hunger": agent.hunger})

    def _rest(self, agent: Agent, _action: Action) -> ActionResult:
        agent.energy = min(100, agent.energy + REST_ENERGY_GAIN)
        return ActionResult(True, "rested", {"energy": agent.energy})

    def _idle(self, _agent: Agent, _action: Action) -> ActionResult:
        return ActionResult(True, "idled")
