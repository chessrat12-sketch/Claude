"""Tests for the survival substrate and the offline simulation loop."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import Agent, HeuristicPolicy, Simulation, World
from sim.actions import ActionExecutor
from sim.agent import MAX_VITAL
from sim.policy import parse_action
from sim.types import Action, ActionType, Resource
from sim.world import ResourceNode, make_scattered_world


def _tile_world() -> World:
    world = World(3, 3)
    world.tile((1, 1)).node = ResourceNode(Resource.FOOD, amount=10, capacity=10)
    return world


def test_metabolism_kills_without_food():
    agent = Agent(id="x", name="X", pos=(0, 0))
    world = World(2, 2)
    for tick in range(200):
        agent.metabolize(tick)
        if not agent.alive:
            break
    assert not agent.alive
    assert agent.died_tick is not None


def test_eating_reduces_hunger():
    world = _tile_world()
    agent = Agent(id="x", name="X", pos=(1, 1), hunger=80)
    agent.add(Resource.FOOD, 2)
    ex = ActionExecutor(world, {"x": agent})
    result = ex.execute(agent, Action(ActionType.EAT))
    assert result.ok
    assert agent.hunger < 80
    assert agent.held(Resource.FOOD) == 1


def test_gather_collects_and_depletes_node():
    world = _tile_world()
    agent = Agent(id="x", name="X", pos=(1, 1))
    ex = ActionExecutor(world, {"x": agent})
    result = ex.execute(agent, Action(ActionType.GATHER))
    assert result.ok
    assert agent.held(Resource.FOOD) == result.data["amount"]
    assert world.tile((1, 1)).node.amount < 10


def test_move_blocked_at_edge():
    world = World(2, 2)
    agent = Agent(id="x", name="X", pos=(0, 0))
    ex = ActionExecutor(world, {"x": agent})
    result = ex.execute(agent, Action(ActionType.MOVE, {"direction": "south"}))
    assert not result.ok
    assert agent.pos == (0, 0)


def test_future_action_is_refused_not_crashed():
    world = World(2, 2)
    agent = Agent(id="x", name="X", pos=(0, 0))
    ex = ActionExecutor(world, {"x": agent})
    result = ex.execute(agent, Action(ActionType.TRADE, {"with": "y"}))
    assert not result.ok
    assert "not available" in result.message


def test_parse_action_tolerates_chatty_output():
    a = parse_action('Sure! Here you go: {"action": "gather", "reason": "hungry"}')
    assert a.type == ActionType.GATHER
    b = parse_action("no json here")
    assert b.type == ActionType.IDLE


def test_node_regenerates_up_to_capacity():
    node = ResourceNode(Resource.FOOD, amount=0, capacity=5, regen_per_tick=1.0)
    for _ in range(10):
        node.regenerate()
    assert node.amount == 5


def test_simulation_runs_and_survivors_remain():
    world = make_scattered_world(width=12, height=12, density=0.25, seed=3)
    agents = [Agent(id=f"a{i}", name=f"A{i}", pos=(i % 12, (i * 2) % 12)) for i in range(8)]
    sim = Simulation(world, agents, lambda ag: HeuristicPolicy(ag.personality), seed=1)
    metrics = sim.run(ticks=120)
    summary = metrics.summary(world, sim.agents)
    assert summary["ticks"] > 0
    # The heuristic baseline should keep at least some agents alive — if not,
    # the world's survival dynamics are unsolvable and need retuning.
    assert summary["alive"] >= 1
    assert summary["action_counts"]


if __name__ == "__main__":
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    raise SystemExit(1 if failures else 0)
