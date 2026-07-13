"""Ephemeral social state that lives *between* agents, not inside one.

Speaking and trading need a place to hold in-flight messages and open offers
until the recipient gets a turn to observe and respond. This is deliberately a
thin transport — it carries what agents say and offer, but encodes no notion of
a market, a price, or a contract. Those must emerge from repeated use.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .types import Resource

OFFER_TTL = 3  # ticks an unanswered offer survives before expiring


@dataclass
class Message:
    frm: str
    to: str
    text: str
    tick: int


@dataclass
class Offer:
    id: str
    frm: str
    to: str
    give: dict[Resource, int]      # what the proposer hands over
    receive: dict[Resource, int]   # what the proposer wants back
    created_tick: int


class Interactions:
    """Per-recipient inboxes (messages) and offer books (open trades)."""

    def __init__(self) -> None:
        self.inbox: dict[str, list[Message]] = defaultdict(list)
        self.offers: dict[str, list[Offer]] = defaultdict(list)
        self._seq = 0

    def next_offer_id(self) -> str:
        self._seq += 1
        return f"o{self._seq}"

    def post_message(self, msg: Message) -> None:
        self.inbox[msg.to].append(msg)

    def post_offer(self, offer: Offer) -> None:
        self.offers[offer.to].append(offer)

    def take_offer(self, recipient: str, offer_id: str) -> Offer | None:
        book = self.offers.get(recipient, [])
        for offer in book:
            if offer.id == offer_id:
                book.remove(offer)
                return offer
        return None

    def clear_inbox(self, agent_id: str) -> None:
        self.inbox[agent_id].clear()

    def expire(self, now: int) -> None:
        for recipient, book in self.offers.items():
            self.offers[recipient] = [
                o for o in book if now - o.created_tick < OFFER_TTL
            ]
