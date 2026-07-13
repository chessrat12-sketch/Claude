"""Builds the partial, egocentric view of the world handed to a policy.

Agents never see global state. They receive only what is local and personal:
their own vitals and inventory, nearby tiles, nearby agents, and a slice of
their memory. This dict is both the input to the heuristic policy and the
payload serialised into an LLM prompt, so the two policies reason over
identical information.
"""

from __future__ import annotations

from typing import Any

from .agent import Agent
from .world import World

VISION_RADIUS = 2


def build_observation(world: World, agent: Agent, agents: dict[str, Agent]) -> dict[str, Any]:
    here = world.tile(agent.pos)
    nearby_nodes = []
    for tile in [here, *world.neighbours(agent.pos, VISION_RADIUS)]:
        if tile.node is not None and tile.node.amount > 0:
            nearby_nodes.append(
                {
                    "pos": [tile.x, tile.y],
                    "resource": tile.node.resource.value,
                    "amount": tile.node.amount,
                    "here": tile.x == agent.pos[0] and tile.y == agent.pos[1],
                }
            )

    nearby_agents = []
    for other in agents.values():
        if other.id == agent.id or not other.alive:
            continue
        if _chebyshev(other.pos, agent.pos) <= VISION_RADIUS:
            nearby_agents.append(
                {
                    "id": other.id,
                    "name": other.name,
                    "pos": list(other.pos),
                    "trust": round(agent.relationships.get(other.id, 0.0), 2),
                }
            )

    return {
        "tick": world.tick,
        "self": {
            "id": agent.id,
            "name": agent.name,
            "pos": list(agent.pos),
            "hunger": agent.hunger,
            "energy": agent.energy,
            "health": agent.health,
            "goal": agent.goal,
            "inventory": {r.value: n for r, n in agent.inventory.items()},
        },
        "on_tile_node": (
            {"resource": here.node.resource.value, "amount": here.node.amount}
            if here.node is not None and here.node.amount > 0
            else None
        ),
        "nearby_nodes": nearby_nodes,
        "nearby_agents": nearby_agents,
        "recent_memory": [
            {"tick": e.tick, "kind": e.kind, "detail": e.detail}
            for e in list(agent.memory)[-6:]
        ],
    }


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
