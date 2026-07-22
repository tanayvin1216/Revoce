"""End-to-end: a real aiohttp server, a real WebSocket client, a real tap.

Everything else in the suite stops at the seam. This is the only test that
exercises the wire the device actually talks over — connect, receive a
vocab_update, tap a word, and check the interaction was logged with the context
that produced the board. The whole server layer (ws_handler, _record_tap,
_produce_and_broadcast, board retention) had no coverage before this.

The LLM is faked deliberately: this tests the transport and the server's state
machine, not the model. Everything between the socket and the seam is real.

Driven with asyncio.run rather than async fixtures so no pytest-asyncio /
pytest-aiohttp plugin is needed — same approach as test_server_concurrency.py.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

import rank_vocabulary  # noqa: E402
import server  # noqa: E402
from context_state import ContextState  # noqa: E402


class FakeRanker:
    """Deterministic stand-in for the LLM. Returns a usable diverse pool."""

    def rank(self, system_prompt, user_prompt, max_tokens=None):
        return [
            {"word": "i want water", "symbol_keyword": "water",
             "function": "request", "topic": "drink"},
            {"word": "it is too loud", "symbol_keyword": "loud",
             "function": "comment", "topic": "room"},
            {"word": "i feel sick", "symbol_keyword": "sick",
             "function": "share-information", "topic": "body"},
            {"word": "can we go home", "symbol_keyword": "home",
             "function": "request", "topic": "place"},
        ]


@asynccontextmanager
async def running_server(tap_log):
    """A live server: perception off, fake LLM, temp tap log."""
    app = server.make_app()
    state = app["state"]
    # Stand the app up the way on_startup does, minus perception and warm-up.
    state["context"] = ContextState()
    state["ranker"] = FakeRanker()
    state["personal_context"] = {"identity": {"name": "Jordan"}}
    state["board_strategy"] = "diverse"      # exercise the seam, not the ckpt
    state["inference_lock"] = asyncio.Lock()

    cli = TestClient(TestServer(app))
    await cli.start_server()
    try:
        yield cli, app
    finally:
        await cli.close()


def test_client_receives_a_board_and_a_tap_is_logged(tmp_path, monkeypatch):
    """The full round trip the device performs."""
    log = tmp_path / "taps.jsonl"
    monkeypatch.setattr(server, "TAP_LOG_PATH", log)

    async def scenario():
        async with running_server(log) as (cli, app):
            async with cli.session.ws_connect(cli.make_url("/ws")) as ws:
                await server._produce_and_broadcast(app)

                msg = await ws.receive_json(timeout=15)
                assert msg["type"] == "vocab_update"
                assert len(msg["items"]) == 10

                words = [i["word"].lower() for i in msg["items"]]
                for core in rank_vocabulary.GUARANTEED_CORE:
                    assert core.lower() in words, f"live board missing {core!r}"

                # Tap a dynamic word, exactly as index.html does.
                await ws.send_json({"type": "tap",
                                    "word": msg["items"][4]["word"],
                                    "source": "dynamic", "client_ts": 1.0,
                                    "board_id": msg["board_id"]})
                # Push again so we know the tap was processed before we assert.
                await server._produce_and_broadcast(app)
                await ws.receive_json(timeout=15)
                return msg["board_id"]

    board_id = asyncio.run(scenario())

    entry = json.loads(log.read_text().strip().splitlines()[0])
    assert entry["board_id"] == board_id
    assert entry["context_snapshot"] is not None, "tap lost its provenance"
    assert entry["board_items"], "tap did not record the board it came from"


def test_board_page_and_status_endpoints_serve(tmp_path, monkeypatch):
    """A judge opening the URL must get the board, not a stack trace."""
    monkeypatch.setattr(server, "TAP_LOG_PATH", tmp_path / "taps.jsonl")

    async def scenario():
        async with running_server(None) as (cli, _app):
            r = await cli.get("/")
            assert r.status == 200
            body = await r.text()

            s = await cli.get("/api/phase-status")
            assert s.status == 200
            return body, await s.json()

    body, status = asyncio.run(scenario())
    assert "<html" in body.lower()
    assert "phases" in status


def test_a_dead_client_does_not_stop_the_others(tmp_path, monkeypatch):
    """One iPad sleeping must not freeze every other display."""
    monkeypatch.setattr(server, "TAP_LOG_PATH", tmp_path / "taps.jsonl")

    async def scenario():
        async with running_server(None) as (cli, app):
            ws1 = await cli.session.ws_connect(cli.make_url("/ws"))
            ws2 = await cli.session.ws_connect(cli.make_url("/ws"))
            await ws1.close()                # ws1 is now dead

            await server._produce_and_broadcast(app)

            msg = await ws2.receive_json(timeout=15)
            await ws2.close()
            return msg

    assert asyncio.run(scenario())["type"] == "vocab_update"


def test_remembered_boards_are_capped(tmp_path, monkeypatch):
    """boards{} holds a full context snapshot per entry, so it must not grow
    without bound on a device that runs for hours."""
    monkeypatch.setattr(server, "TAP_LOG_PATH", tmp_path / "taps.jsonl")
    monkeypatch.setattr(server, "MAX_REMEMBERED_BOARDS", 3)

    async def scenario():
        async with running_server(None) as (_cli, app):
            for _ in range(6):
                await server._produce_and_broadcast(app)
            return len(app["state"]["boards"])

    assert asyncio.run(scenario()) <= 3
