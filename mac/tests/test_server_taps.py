"""Tap-handling tests for the server.

_record_tap decides two things that reach the user directly: whether the
interaction actually gets logged, and whether the held question survives the
tap. Both were untested.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import server
from context_state import ContextState


def _app(tmp_path: Path) -> dict:
    ctx = ContextState()
    ctx.set_active_question("where would you like to sit?")
    return {
        "state": {
            "boards": {
                "b1": {
                    "produced_from": {"scene": "a bus", "transcript": "", "context": {}},
                    "items": [{"word": "wait", "symbol": "x"}],
                }
            },
            "context": ctx,
        }
    }


def _tap(word: str) -> dict:
    return {"word": word, "source": "dynamic", "client_ts": 1.0, "board_id": "b1"}


@pytest.fixture(autouse=True)
def tap_log(tmp_path, monkeypatch):
    path = tmp_path / "taps.jsonl"
    monkeypatch.setattr(server, "TAP_LOG_PATH", path)
    return path


def test_answering_tap_clears_the_held_question(tmp_path):
    app = _app(tmp_path)

    server._record_tap(app, _tap("yes"), "peer")

    assert app["state"]["context"].snapshot()[2]["active_question"] is None


@pytest.mark.parametrize("word", ["wait", "I need help", "  WAIT  "])
def test_holding_taps_keep_the_question_alive(tmp_path, word):
    """The whole point of "wait" is to buy time to compose an answer.

    Clearing the question here would drop the answer vocabulary the user was
    reaching for and let the next trigger rebuild the board from ambient
    chatter -- the tap meant to hold the moment would destroy it.
    """
    app = _app(tmp_path)

    server._record_tap(app, _tap(word), "peer")

    assert app["state"]["context"].snapshot()[2]["active_question"] == (
        "where would you like to sit?"
    )


def test_tap_is_logged_with_the_context_that_produced_the_board(tmp_path, tap_log):
    app = _app(tmp_path)

    server._record_tap(app, _tap("yes"), "peer")

    entry = json.loads(tap_log.read_text(encoding="utf-8").strip())
    assert entry["word"] == "yes"
    assert entry["context_snapshot"]["scene"] == "a bus"


def test_unwritable_tap_log_does_not_take_down_the_connection(tmp_path, monkeypatch):
    """An OSError here used to propagate out through the WebSocket handler,
    closing the connection on every tap. The word was already spoken
    client-side, so dropping the log row is the graceful outcome."""
    app = _app(tmp_path)
    monkeypatch.setattr(
        server, "TAP_LOG_PATH", tmp_path / "no-such-dir" / "taps.jsonl"
    )

    server._record_tap(app, _tap("yes"), "peer")  # must not raise

    assert app["state"]["context"].snapshot()[2]["active_question"] is None
