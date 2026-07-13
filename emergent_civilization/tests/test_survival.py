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
    # PROPOSE_RULE belongs to a later milestone (v0.7) and must be refused.
    result = ex.execute(agent, Action(ActionType.PROPOSE_RULE, {"text": "share food"}))
    assert not result.ok
    assert "not available" in result.message


def test_trade_completes_and_swaps_inventory():
    world = World(3, 3)
    seller = Agent(id="s", name="Seller", pos=(1, 1))
    buyer = Agent(id="b", name="Buyer", pos=(1, 1))
    seller.add(Resource.WOOD, 2)
    buyer.add(Resource.FOOD, 3)
    agents = {"s": seller, "b": buyer}
    ex = ActionExecutor(world, agents)
    # Seller proposes 2 wood for 1 food; buyer accepts.
    propose = ex.execute(seller, Action(ActionType.TRADE,
                         {"to": "b", "give": {"wood": 2}, "receive": {"food": 1}}))
    assert propose.ok
    offer_id = propose.data["offer"]
    accept = ex.execute(buyer, Action(ActionType.TRADE, {"offer": offer_id, "accept": True}))
    assert accept.ok
    assert buyer.held(Resource.WOOD) == 2 and seller.held(Resource.WOOD) == 0
    assert seller.held(Resource.FOOD) == 1 and buyer.held(Resource.FOOD) == 2
    # A completed exchange builds mutual trust.
    assert seller.relationships["b"] > 0 and buyer.relationships["s"] > 0
    assert any(e["type"] == "trade" for e in ex.events)


def test_trade_rejected_when_partner_out_of_range():
    world = World(6, 6)
    a = Agent(id="a", name="A", pos=(0, 0))
    b = Agent(id="b", name="B", pos=(5, 5))
    a.add(Resource.WOOD, 2)
    ex = ActionExecutor(world, {"a": a, "b": b})
    result = ex.execute(a, Action(ActionType.TRADE,
                        {"to": "b", "give": {"wood": 2}, "receive": {"food": 1}}))
    assert not result.ok
    assert "too far" in result.message


def test_build_shelter_consumes_materials():
    world = World(3, 3)
    a = Agent(id="a", name="A", pos=(1, 1))
    a.add(Resource.WOOD, 3)
    a.add(Resource.STONE, 1)
    ex = ActionExecutor(world, {"a": a})
    result = ex.execute(a, Action(ActionType.BUILD))
    assert result.ok
    assert world.shelter_at((1, 1)) is not None
    assert world.structures[(1, 1)].owner_id == "a"
    assert a.held(Resource.WOOD) == 0 and a.held(Resource.STONE) == 0
    # Can't build a second on the same tile.
    a.add(Resource.WOOD, 3); a.add(Resource.STONE, 1)
    assert not ex.execute(a, Action(ActionType.BUILD)).ok


def test_craft_tool_and_gather_boost():
    world = World(3, 3)
    world.tile((1, 1)).node = ResourceNode(Resource.WOOD, amount=20, capacity=20)
    a = Agent(id="a", name="A", pos=(1, 1))
    a.add(Resource.WOOD, 2); a.add(Resource.STONE, 1)
    ex = ActionExecutor(world, {"a": a})
    assert ex.execute(a, Action(ActionType.CRAFT)).ok
    assert a.held(Resource.TOOL) == 1 and a.held(Resource.WOOD) == 0
    # With a tool, a gather yields more than the base amount.
    got = ex.execute(a, Action(ActionType.GATHER))
    assert got.data["amount"] > 3


def test_give_transfers_and_builds_trust():
    world = World(3, 3)
    giver = Agent(id="g", name="Giver", pos=(1, 1))
    poor = Agent(id="p", name="Poor", pos=(1, 1))
    giver.add(Resource.FOOD, 3)
    ex = ActionExecutor(world, {"g": giver, "p": poor})
    result = ex.execute(giver, Action(ActionType.GIVE, {"to": "p", "items": {"food": 2}}))
    assert result.ok
    assert poor.held(Resource.FOOD) == 2 and giver.held(Resource.FOOD) == 1
    assert poor.relationships["g"] > 0   # generosity earns the receiver's trust


def test_predators_appear_at_night_and_can_bite():
    from sim.ecology import Ecology
    world = World(6, 6, day_length=10, night_fraction=0.5)
    world.tick = 8  # inside the night window (>= 5)
    assert world.is_night
    victim = Agent(id="v", name="V", pos=(3, 3))
    eco = Ecology()
    rng = __import__("random").Random(0)
    # Run several night ticks; a lone exposed agent should eventually be bitten.
    bitten = False
    for _ in range(15):
        events = eco.update(world, {"v": victim}, rng)
        if any(e["type"] == "attack" for e in events):
            bitten = True
            break
    assert eco.predators  # predators spawned at night
    assert bitten and victim.health < 100


def test_shelter_prevents_predator_exposure():
    from sim.ecology import Ecology
    from sim.world import Structure
    world = World(6, 6, day_length=10, night_fraction=0.5)
    world.tick = 8
    world.structures[(3, 3)] = Structure((3, 3), "v", 0)
    victim = Agent(id="v", name="V", pos=(3, 3))
    eco = Ecology()
    rng = __import__("random").Random(0)
    for _ in range(15):
        eco.update(world, {"v": victim}, rng)
    assert victim.health == 100  # never bitten while sheltered


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
