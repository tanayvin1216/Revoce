"""Phase-6 integration smoke: the whole host path in one process.

server boots (no perception, heuristic or student ranker) -> a board client
connects over the WebSocket -> receives a vocab_update with the pinned core
words -> taps a word -> the tap lands in the tap log WITH the context snapshot
that produced the board. This is the contract the iPad relies on and the one
CLAUDE.md calls irreplaceable (log every tap from the first UI).
"""
from __future__ import annotations

import asyncio
import json

import pytest
from aiohttp.test_utils import TestClient, TestServer


async def _boot(monkeypatch, tmp_path, strategy: str):
    monkeypatch.setenv("VOICELINK_PERCEPTION", "0")
    monkeypatch.setenv("VOICELINK_RANKER", "heuristic")
    monkeypatch.setenv("VOICELINK_BOARD_STRATEGY", strategy)
    import importlib
    import ranker_config
    importlib.reload(ranker_config)
    import server
    monkeypatch.setattr(server, "TAP_LOG_PATH", tmp_path / "taps.jsonl")
    monkeypatch.setattr(server, "PROFILE_PATH", tmp_path / "absent_profile.json")
    app = server.make_app()
    client = TestClient(TestServer(app))
    await client.start_server()
    return client, app


async def _exercise(client, app, tmp_path):
    async with client.ws_connect("/ws") as ws:
        msg = json.loads((await ws.receive(timeout=60)).data)
        assert msg["type"] == "vocab_update"
        words = [it["word"].lower() for it in msg["items"]]
        assert len(words) >= 4 and len(words) == len(set(words))
        for core in ("yes", "no"):
            assert core in words, words
        assert all(it.get("symbol") for it in msg["items"])
        await ws.send_str(json.dumps({"type": "tap", "word": msg["items"][0]["word"],
                                      "source": "test", "client_ts": 1.0,
                                      "board_id": msg["board_id"]}))
        # Give the server loop a tick to write the log.
        for _ in range(50):
            if (tmp_path / "taps.jsonl").exists():
                break
            await asyncio.sleep(0.02)
    entry = json.loads((tmp_path / "taps.jsonl").read_text().splitlines()[-1])
    assert entry["board_id"] == msg["board_id"]
    assert entry["context_snapshot"] is not None and "context" in entry["context_snapshot"]
    assert [it["word"] for it in entry["board_items"]] == [it["word"] for it in msg["items"]]
    r = await client.get("/")
    assert r.status == 200 and "ws" in (await r.text())
    r = await client.get("/api/phase-status")
    assert r.status == 200


async def _run(monkeypatch, tmp_path, strategy):
    client, app = await _boot(monkeypatch, tmp_path, strategy)
    try:
        assert app["state"]["board_strategy"] == strategy
        await _exercise(client, app, tmp_path)
    finally:
        await client.close()


def test_server_board_tap_roundtrip_diverse(monkeypatch, tmp_path):
    asyncio.run(_run(monkeypatch, tmp_path, "diverse"))


def test_server_board_tap_roundtrip_student(monkeypatch, tmp_path):
    """The distilled student served through the real server — the device path."""
    pytest.importorskip("torch")
    asyncio.run(_run(monkeypatch, tmp_path, "student"))
