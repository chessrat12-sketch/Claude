"""Execution of agent actions against the world.

This is the *rules engine*: it decides what an action does and whether it is
currently allowed. It never decides *which* action an agent takes — that is the
policy's job. Actions belonging to a future milestone return a failing
``ActionResult`` explaining why, so the boundary of the possible is legible to
agents rather than hidden.
"""

from __future__ import annotations

from .agent import Agent
from .interactions import Interactions, Message, Offer
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
    ActionType.SPEAK,
    ActionType.TRADE,
}

EAT_HUNGER_RELIEF = 35
REST_ENERGY_GAIN = 30
GATHER_PER_ACTION = 3
INTERACT_RADIUS = 2       # how close two agents must be to speak or trade
TRUST_ON_TRADE = 0.1      # trust each party gains from a completed exchange


class ActionExecutor:
    """Applies actions to (world, agents) and returns per-action feedback.

    Side effects that other systems care about (a trade completing, a message
    being sent) are appended to ``events`` for the engine to drain into metrics
    and the render snapshot each tick.
    """

    def __init__(
        self,
        world: World,
        agents: dict[str, Agent],
        interactions: Interactions | None = None,
    ) -> None:
        self.world = world
        self.agents = agents
        self.interactions = interactions or Interactions()
        self.events: list[dict] = []

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
            ActionType.SPEAK: self._speak,
            ActionType.TRADE: self._trade,
        }[action.type]
        result = handler(agent, action)
        agent.remember(self.world.tick, action.type.value, result.message)
        return result

    # -- survival handlers ------------------------------------------------
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

    # -- social handlers --------------------------------------------------
    def _speak(self, agent: Agent, action: Action) -> ActionResult:
        listener = self.agents.get(action.args.get("to"))
        text = str(action.args.get("message", "")).strip()[:200]
        if listener is None or not listener.alive:
            return ActionResult(False, "no such listener")
        if _chebyshev(agent.pos, listener.pos) > INTERACT_RADIUS:
            return ActionResult(False, "too far to be heard")
        if not text:
            return ActionResult(False, "nothing to say")
        self.interactions.post_message(Message(agent.id, listener.id, text, self.world.tick))
        self.events.append(
            {"type": "speak", "a": agent.id, "b": listener.id,
             "text": f"{agent.name}→{listener.name}: {text}"}
        )
        return ActionResult(True, f"said to {listener.name}: {text}")

    def _trade(self, agent: Agent, action: Action) -> ActionResult:
        # Two forms share the TRADE verb: propose (has 'to') or respond (has 'offer').
        if "offer" in action.args:
            return self._respond_offer(agent, action)
        return self._propose_trade(agent, action)

    def _propose_trade(self, agent: Agent, action: Action) -> ActionResult:
        partner = self.agents.get(action.args.get("to"))
        if partner is None or not partner.alive:
            return ActionResult(False, "no such trade partner")
        if _chebyshev(agent.pos, partner.pos) > INTERACT_RADIUS:
            return ActionResult(False, "too far to trade")
        give = _parse_bundle(action.args.get("give"))
        receive = _parse_bundle(action.args.get("receive"))
        if not give or not receive:
            return ActionResult(False, "a trade needs both 'give' and 'receive'")
        for resource, n in give.items():
            if agent.held(resource) < n:
                return ActionResult(False, f"you lack {n} {resource.value} to offer")
        offer = Offer(
            self.interactions.next_offer_id(), agent.id, partner.id, give, receive,
            self.world.tick,
        )
        self.interactions.post_offer(offer)
        return ActionResult(
            True,
            f"offered {_fmt(give)} for {_fmt(receive)} to {partner.name}",
            {"offer": offer.id},
        )

    def _respond_offer(self, agent: Agent, action: Action) -> ActionResult:
        offer_id = action.args.get("offer")
        accept = bool(action.args.get("accept", False))
        offer = self.interactions.take_offer(agent.id, offer_id)
        if offer is None:
            return ActionResult(False, f"offer '{offer_id}' is gone")
        if not accept:
            return ActionResult(True, f"declined offer {offer_id}")

        proposer = self.agents.get(offer.frm)
        if proposer is None or not proposer.alive:
            return ActionResult(False, "the proposer is no longer around")
        if _chebyshev(agent.pos, proposer.pos) > INTERACT_RADIUS:
            return ActionResult(False, "the proposer moved out of range")
        # Re-validate both sides still hold the goods (state moved on since offer).
        for resource, n in offer.give.items():
            if proposer.held(resource) < n:
                return ActionResult(False, "proposer no longer has the goods")
        for resource, n in offer.receive.items():
            if agent.held(resource) < n:
                return ActionResult(False, f"you lack {n} {resource.value} to pay")

        # Swap: proposer gives 'give' and gets 'receive'; acceptor is the mirror.
        for resource, n in offer.give.items():
            proposer.remove(resource, n)
            agent.add(resource, n)
        for resource, n in offer.receive.items():
            agent.remove(resource, n)
            proposer.add(resource, n)

        _bump_trust(proposer, agent.id, TRUST_ON_TRADE)
        _bump_trust(agent, proposer.id, TRUST_ON_TRADE)
        give_j = {r.value: n for r, n in offer.give.items()}
        receive_j = {r.value: n for r, n in offer.receive.items()}
        self.events.append(
            {"type": "trade", "a": proposer.id, "b": agent.id,
             "give": give_j, "receive": receive_j,
             "text": f"{proposer.name} ⇄ {agent.name}: {_fmt(offer.give)} for {_fmt(offer.receive)}"}
        )
        return ActionResult(True, "trade completed", {"give": give_j, "receive": receive_j})


# -- helpers --------------------------------------------------------------
def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _parse_bundle(raw) -> dict[Resource, int]:
    """Coerce a {resource: amount} mapping (string or enum keys) into enums."""
    if not isinstance(raw, dict):
        return {}
    out: dict[Resource, int] = {}
    for key, amount in raw.items():
        try:
            resource = key if isinstance(key, Resource) else Resource(str(key).lower())
            n = int(amount)
        except (ValueError, TypeError):
            continue
        if n > 0:
            out[resource] = n
    return out


def _fmt(bundle: dict[Resource, int]) -> str:
    return ", ".join(f"{n} {r.value}" for r, n in bundle.items()) or "nothing"


def _bump_trust(agent: Agent, other_id: str, delta: float) -> None:
    current = agent.relationships.get(other_id, 0.0)
    agent.relationships[other_id] = max(-1.0, min(1.0, current + delta))
