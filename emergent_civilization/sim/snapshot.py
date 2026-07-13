"""Serialises the whole world into one render-friendly snapshot.

Unlike an agent *observation* (partial, egocentric), a snapshot is the global
god's-eye view — but it is strictly for *rendering* (Unity, the browser viewer).
Agents never see it, so it cannot leak global knowledge into decisions.

The schema avoids nested maps so Unity's ``JsonUtility`` can parse it directly:
inventories are lists of {resource, amount} pairs, everything else is scalars
or flat arrays. See ``unity/README_UNITY.md`` for the matching C# models.
"""

from __future__ import annotations

from .agent import Agent
from .types import Resource
from .world import World


def _inventory_list(agent: Agent) -> list[dict]:
    return [{"resource": r.value, "amount": n} for r, n in agent.inventory.items()]


def world_snapshot(
    world: World,
    agents: dict[str, Agent],
    events: list[dict] | None = None,
    stats: dict | None = None,
    predators: list | None = None,
) -> dict:
    nodes = []
    for tile in world.tiles.values():
        if tile.node is not None and tile.node.amount > 0:
            nodes.append(
                {
                    "x": tile.x,
                    "y": tile.y,
                    "resource": tile.node.resource.value,
                    "amount": tile.node.amount,
                    "capacity": tile.node.capacity,
                }
            )

    agent_views = []
    for a in agents.values():
        last = a.memory[-1].kind if a.memory else "idle"
        agent_views.append(
            {
                "id": a.id,
                "name": a.name,
                "x": a.pos[0],
                "y": a.pos[1],
                "hunger": a.hunger,
                "energy": a.energy,
                "health": a.health,
                "alive": a.alive,
                "wealth": a.wealth,
                "sheltered": world.shelter_at(a.pos) is not None,
                "tools": a.held(Resource.TOOL),
                "lastAction": last,
                "inventory": _inventory_list(a),
            }
        )

    structures = [
        {"x": s.pos[0], "y": s.pos[1], "owner": s.owner_id, "durability": s.durability}
        for s in world.structures.values()
    ]
    threats = [{"id": p.id, "x": p.pos[0], "y": p.pos[1]} for p in (predators or [])]

    return {
        "tick": world.tick,
        "width": world.width,
        "height": world.height,
        "timeOfDay": world.time_of_day,
        "dayLength": world.day_length,
        "isNight": world.is_night,
        "nodes": nodes,
        "structures": structures,
        "threats": threats,
        "agents": agent_views,
        "events": [
            {"type": e["type"], "a": e.get("a", ""), "b": e.get("b", ""), "text": e.get("text", "")}
            for e in (events or [])
        ],
        "stats": stats or {},
    }
