"""Emergent Civilization — server-side simulation core.

This package is the human-authored *substrate* of the experiment: a world, a
survival metabolism, a fixed action vocabulary, and the plumbing to let any
policy (heuristic or LLM) drive an agent. It contains no code for jobs, prices,
laws, or alliances — those are meant to emerge from agent behaviour.
"""

from .agent import Agent, Personality
from .engine import Simulation
from .metrics import Metrics
from .policy import HeuristicPolicy, LLMPolicy, Policy
from .types import Action, ActionResult, ActionType, Direction, Resource
from .world import World, make_scattered_world

__all__ = [
    "Agent",
    "Personality",
    "Simulation",
    "Metrics",
    "HeuristicPolicy",
    "LLMPolicy",
    "Policy",
    "Action",
    "ActionResult",
    "ActionType",
    "Direction",
    "Resource",
    "World",
    "make_scattered_world",
]
