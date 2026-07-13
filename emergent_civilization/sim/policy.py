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
        food = me["inventory"].get(Resource.FOOD.value, 0)

        # 1. Eat before starving if we can.
        if hunger >= 60 and food > 0:
            return Action(ActionType.EAT, reason="hunger is high")

        # 2. Recover energy before collapse.
        if energy <= 25:
            return Action(ActionType.REST, reason="energy is low")

        # 3. Standing on a node worth taking? Gather it.
        on_node = obs.get("on_tile_node")
        if on_node is not None:
            want_food = food < 3 + int(self.personality.caution * 4)
            if on_node["resource"] == Resource.FOOD.value and want_food:
                return Action(ActionType.GATHER, reason="topping up food")
            if on_node["resource"] != Resource.FOOD.value:
                return Action(ActionType.GATHER, reason="collecting materials")

        # 4. Head toward the nearest useful node.
        target = self._nearest_node(me["pos"], obs["nearby_nodes"], prefer_food=food < 3)
        if target is not None:
            return self._step_toward(me["pos"], target)

        # 5. Nothing in sight: wander (curiosity picks the direction).
        idx = (obs["tick"] + int(self.personality.curiosity * 3)) % 4
        return Action(ActionType.MOVE, {"direction": list(Direction)[idx].value}, "exploring")

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
SYSTEM_INSTRUCTIONS = """You are an autonomous being in a small shared world.
You are not told what to do. You have your own memory, personality and goals.
You survive by managing hunger and energy, and you may gather, move, rest, or
eat. Other beings live here too; over time you may find it useful to cooperate,
trade, build trust, or propose shared rules — but nothing forces you to.

Respond with ONE action as strict JSON and nothing else:
{"action": "<move|gather|eat|rest|idle>", "args": {...}, "reason": "<short>"}
For move, args must be {"direction": "north|south|east|west"}."""


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
        return (
            f"{SYSTEM_INSTRUCTIONS}\n\n{persona}\n\n"
            f"Current observation:\n{json.dumps(obs, ensure_ascii=False, indent=2)}\n\n"
            "Your action:"
        )

    def decide(self, obs: dict[str, Any]) -> Action:
        raw = self.backend.complete(self.build_prompt(obs))
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
