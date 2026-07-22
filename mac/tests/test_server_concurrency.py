"""The seam must never run twice at once.

The trigger loop, the heartbeat loop and the console all call rank_vocabulary
against ONE shared MLX model instance. Concurrent generate() calls on it are
not safe, and a warm diverse board takes tens of seconds -- long enough for the
heartbeat to wake up mid-generation and start a second one.
"""

from __future__ import annotations

import asyncio

import pytest

import server
from context_state import ContextState


class ConcurrencySpy:
    """A ranker that records how many callers are inside it simultaneously."""

    def __init__(self) -> None:
        self.live = 0
        self.max_live = 0
        self.calls = 0

    def rank(self, system_prompt, user_prompt, max_tokens=None):
        self.live += 1
        self.calls += 1
        self.max_live = max(self.max_live, self.live)
        try:
            # Stand in for a slow generation. rank() runs in a worker thread, so
            # a real sleep here is what lets a second caller overlap if the lock
            # is missing.
            import time as _t
            _t.sleep(0.05)
            return [
                {"word": "I want water", "symbol_keyword": "water",
                 "function": "request", "topic": "drink"},
                {"word": "it is loud", "symbol_keyword": "loud",
                 "function": "comment", "topic": "room"},
            ]
        finally:
            self.live -= 1


def _app(spy: ConcurrencySpy) -> dict:
    app = {"state": dict(server.make_app()["state"])}
    app["state"].update(
        {
            "context": ContextState(),
            "personal_context": None,
            "ranker": spy,
            "board_strategy": "diverse",
            "clients": set(),
            "boards": {},
            "inference_lock": asyncio.Lock(),
        }
    )
    return app


def test_concurrent_pushes_never_overlap_in_the_seam():
    async def scenario():
        spy = ConcurrencySpy()
        app = _app(spy)
        # Exactly the trigger-loop / heartbeat-loop collision.
        await asyncio.gather(
            server._produce_and_broadcast(app),
            server._produce_and_broadcast(app),
        )
        return spy

    spy = asyncio.run(scenario())

    assert spy.calls == 2, "both pushes should have run"
    assert spy.max_live == 1, (
        f"{spy.max_live} concurrent generations on one model instance -- "
        "the seam lock is not serializing"
    )


def test_push_timestamp_moves_before_generation_finishes():
    """last_push_ts must be claimed at generation START.

    If it only moved on completion, the heartbeat loop would wake during a long
    generation, still see a stale timestamp, and stack a second one.
    """

    async def scenario():
        spy = ConcurrencySpy()
        app = _app(spy)
        app["state"]["last_push_ts"] = 0.0

        task = asyncio.ensure_future(server._produce_and_broadcast(app))
        # Yield until the seam is actually running, then read the timestamp.
        while spy.live == 0 and not task.done():
            await asyncio.sleep(0.005)
        claimed = app["state"]["last_push_ts"]
        await task
        return claimed

    assert asyncio.run(scenario()) > 0.0
