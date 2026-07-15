"""Policies turn an observation into an action.

Two implementations ship today:

* ``HeuristicPolicy`` — a transparent, offline survival controller. It is *not*
  the research subject; it exists to validate that the world's survival dynamics
  are actually solvable and to give a baseline to compare emergent LLM
  behaviour against.
* ``LLMPolicy`` — serialises the observation into a prompt, asks an
  ``LLMBackend`` for a JSON action, and parses it. This is the intended driver
  of the experiment; swap the backend to point at a real inference server.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from .agent import Agent, Personality
from .llm_client import EchoBackend, LLMBackend
from .types import Action, ActionType, Direction, Resource


class Policy(Protocol):
    def decide(self, observation: dict[str, Any]) -> Action: ...


# --------------------------------------------------------------------------
# Heuristic baseline
# --------------------------------------------------------------------------
class HeuristicPolicy:
    """A survival-first controller, lightly flavoured by personality.

    Priority order: don't starve, don't collapse from exhaustion, stock food,
    otherwise wander toward the nearest resource. Deliberately simple and
    legible — every branch maps to a survival pressure defined in ``agent.py``.
    """

    def __init__(self, personality: Personality | None = None) -> None:
        self.personality = personality or Personality()

    def decide(self, obs: dict[str, Any]) -> Action:
        me = obs["self"]
        hunger, energy = me["hunger"], me["energy"]
        inv = me["inventory"]
        food = inv.get(Resource.FOOD.value, 0)
        wood = inv.get(Resource.WOOD.value, 0)
        stone = inv.get(Resource.STONE.value, 0)
        food_val = Resource.FOOD.value

        # 1. Eat before starving if we can.
        if hunger >= 60 and food > 0:
            return Action(ActionType.EAT, reason="hunger is high")

        # 2. Survive the night. Threats hunt exposed loners, so seek cover:
        #    rest in a shelter, move to one, or huddle with others (safety in
        #    numbers halves a bite). This is where cooperation earns its keep.
        if obs.get("is_night") and obs.get("nearby_threats"):
            if me.get("sheltered"):
                return Action(ActionType.REST, reason="sheltering through the night")
            shelter = self._nearest(me["pos"], obs.get("nearby_shelters", []))
            if shelter is not None:
                return self._step_toward(me["pos"], shelter)
            if obs["nearby_agents"]:
                mate = self._nearest(me["pos"], obs["nearby_agents"])
                if _manhattan(me["pos"], mate["pos"]) > 1:
                    return self._step_toward(me["pos"], mate)
            return Action(ActionType.REST, reason="hunkering down")

        # 3. Recover energy before collapse (in shelter if we're on one).
        if energy <= 25:
            return Action(ActionType.REST, reason="energy is low")

        # 4. Generosity: help a visibly-suffering neighbour we don't distrust.
        #    Costs a little food, buys reputation (their trust in us jumps).
        if food >= 6:
            for other in obs["nearby_agents"]:
                if other.get("health", 100) < 40 and other.get("trust", 0) >= 0:
                    return Action(
                        ActionType.GIVE,
                        {"to": other["id"], "items": {food_val: 1}},
                        reason="helping a starving neighbour",
                    )

        # 5. Build a shelter once we can afford it and none is nearby — this is
        #    what gives wood/stone their value and makes trade rational.
        need_shelter = not me.get("sheltered") and not obs.get("nearby_shelters")
        if need_shelter and wood >= 3 and stone >= 1:
            return Action(ActionType.BUILD, reason="raising a shelter")

        # 6. No shelter yet and short on the materials for one? With a little
        #    food banked, go deliberately mine the material we lack.
        if need_shelter and food >= 3 and hunger < 65 and (wood < 3 or stone < 1):
            want = Resource.STONE.value if stone < 1 else Resource.WOOD.value
            on_node = obs.get("on_tile_node")
            if on_node is not None and on_node["resource"] == want:
                return Action(ActionType.GATHER, reason=f"mining {want} for a shelter")
            node = self._nearest(me["pos"], [n for n in obs["nearby_nodes"] if n["resource"] == want])
            if node is not None:
                return self._step_toward(me["pos"], node)

        # 7. Craft a tool from surplus materials to gather faster (division of
        #    labour: some specialise in tools).
        if wood >= 4 and stone >= 2 and inv.get(Resource.TOOL.value, 0) == 0:
            return Action(ActionType.CRAFT, reason="crafting a tool")

        # 8. Weigh any standing offers. Two reasons a trade is worth taking:
        #     (a) it brings food we need, or (b) we have a food surplus and can
        #     bank materials (now genuinely valuable — shelters & tools need
        #     them). Complementary needs are what let a trade clear.
        for offer in obs.get("pending_offers", []):
            gain, cost = offer["give"], offer["receive"]
            if not all(inv.get(r, 0) >= n for r, n in cost.items()):
                continue
            gains_food = gain.get(food_val, 0) > 0
            food_paid = cost.get(food_val, 0)
            takes_material = any(r != food_val for r in gain)
            wants_food_now = gains_food and (hunger >= 40 or food < 4)
            surplus_buyer = takes_material and (food - food_paid) >= 6
            if wants_food_now or surplus_buyer:
                return Action(
                    ActionType.TRADE,
                    {"offer": offer["id"], "accept": True},
                    reason="food need" if wants_food_now else "banking materials",
                )

        # 9. Short on food but holding more material than we need to build?
        #    Offer the excess for food. Kept above build needs (3 wood/1 stone)
        #    so trading never starves our own shelter plans.
        if food < 4 and hunger < 70 and obs["nearby_agents"]:
            surplus = next(
                (r for r in (Resource.WOOD.value, Resource.STONE.value) if inv.get(r, 0) >= 6),
                None,
            )
            if surplus is not None:
                partner = obs["nearby_agents"][0]["id"]
                return Action(
                    ActionType.TRADE,
                    {"to": partner, "give": {surplus: 2}, "receive": {Resource.FOOD.value: 1}},
                    reason="trading surplus material for food",
                )

        # 9. Standing on a node worth taking? Gather it.
        on_node = obs.get("on_tile_node")
        if on_node is not None:
            want_food = food < 3 + int(self.personality.caution * 4)
            if on_node["resource"] == Resource.FOOD.value and want_food:
                return Action(ActionType.GATHER, reason="topping up food")
            if on_node["resource"] != Resource.FOOD.value:
                return Action(ActionType.GATHER, reason="collecting materials")

        # 10. Head toward the nearest useful node.
        target = self._nearest_node(me["pos"], obs["nearby_nodes"], prefer_food=food < 3)
        if target is not None:
            return self._step_toward(me["pos"], target)

        # 11. Nothing in sight: wander (curiosity picks the direction).
        idx = (obs["tick"] + int(self.personality.curiosity * 3)) % 4
        return Action(ActionType.MOVE, {"direction": list(Direction)[idx].value}, "exploring")

    def _nearest(self, pos, items):
        """Nearest of a list of dicts that each carry a 'pos'."""
        return min(items, key=lambda it: _manhattan(pos, it["pos"])) if items else None

    def _nearest_node(self, pos, nodes, prefer_food):
        candidates = nodes
        if prefer_food:
            food_nodes = [n for n in nodes if n["resource"] == Resource.FOOD.value]
            candidates = food_nodes or nodes
        if not candidates:
            return None
        return min(candidates, key=lambda n: _manhattan(pos, n["pos"]))

    def _step_toward(self, pos, node) -> Action:
        px, py = pos
        tx, ty = node["pos"]
        if abs(tx - px) >= abs(ty - py) and tx != px:
            d = Direction.EAST if tx > px else Direction.WEST
        elif ty != py:
            d = Direction.NORTH if ty > py else Direction.SOUTH
        else:
            d = Direction.EAST if tx > px else Direction.WEST
        return Action(ActionType.MOVE, {"direction": d.value}, "moving to resource")


# --------------------------------------------------------------------------
# LLM policy
# --------------------------------------------------------------------------
SYSTEM_INSTRUCTIONS = """You are an autonomous being in a small shared world, \
deciding for yourself. You survive by managing hunger and energy. At night, \
predators hunt anyone alone in the open; grouping up or sheltering keeps you \
safe. Wood and stone can be crafted into a tool (faster gathering) or a \
shelter. Others live nearby — talking, trading, giving, or building trust may \
help you, but nothing forces it.

Reply with ONE action as strict JSON, nothing else:
{"action":"<move|gather|eat|rest|idle|speak|trade|give|craft|build>","args":{...},"reason":"<short>"}

Args:
- move: {"direction":"north|south|east|west"}
- speak: {"to":"<id>","message":"<text>"}
- trade (offer): {"to":"<id>","give":{"wood":2},"receive":{"food":1}}
- trade (respond): {"offer":"<id>","accept":true|false}
- give: {"to":"<id>","items":{"food":1}}
- craft: {} (2 wood + 1 stone -> tool)
- build: {} (3 wood + 1 stone -> shelter)
- gather/eat/rest/idle: no args."""


class LLMPolicy:
    """Drives an agent by prompting an LLM backend for a JSON action."""

    def __init__(self, agent: Agent, backend: LLMBackend | None = None) -> None:
        self.agent = agent
        self.backend = backend or EchoBackend()

    def build_prompt(self, obs: dict[str, Any]) -> str:
        p = self.agent.personality
        persona = (
            f"Your name is {self.agent.name}. Your goal: {self.agent.goal}. "
            f"Traits — greed {p.greed:.1f}, sociability {p.sociability:.1f}, "
            f"caution {p.caution:.1f}, curiosity {p.curiosity:.1f}."
        )
        # Compact (no indent) JSON — cuts prompt size by roughly a third versus
        # pretty-printing, which matters a lot on free-tier token-per-minute
        # limits (e.g. Groq's free tier) when several agents call every tick.
        obs_json = json.dumps(obs, ensure_ascii=False, separators=(",", ":"))
        # "/no_think" is Qwen3's documented switch to skip its extended
        # <think>...</think> reasoning trace and answer directly. Harmless
        # stray text to any other model. Without it, Qwen3 can spend the
        # whole output budget "thinking" and get truncated before ever
        # emitting the action JSON (-> "unparseable model output").
        return (
            f"{SYSTEM_INSTRUCTIONS}\n\n{persona}\n\n"
            f"Current observation:\n{obs_json}\n\n"
            "Your action: /no_think"
        )

    def decide(self, obs: dict[str, Any]) -> Action:
        try:
            raw = self.backend.complete(self.build_prompt(obs))
        except Exception as e:  # noqa: BLE001 — a bad response must never crash the sim
            print(f"[llm] {self.agent.name}: backend error, idling this tick — {e}")
            return Action(ActionType.IDLE, reason=f"backend error: {e}")
        return parse_action(raw)


def parse_action(raw: str) -> Action:
    """Parse a model's text into an ``Action``, tolerating chatty output.

    Falls back to IDLE on anything unparseable so a single bad completion never
    crashes the simulation.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return Action(ActionType.IDLE, reason="unparseable model output")
    try:
        data = json.loads(match.group(0))
        atype = ActionType(str(data.get("action", "idle")).lower())
    except (json.JSONDecodeError, ValueError):
        return Action(ActionType.IDLE, reason="invalid action in model output")
    args = data.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    return Action(atype, args, str(data.get("reason", "")))


def _manhattan(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
