"""
VoiceLink Mac server — the ONLY way the wearable and the web app reach
intelligence.

Responsibilities:
  * Serve the placeholder AAC web page over PLAIN HTTP (same origin as the
    WebSocket, so a ws:// connection isn't blocked as mixed content).
  * Push vocabulary boards to the page over a WebSocket   (server -> client).
  * Receive taps from the page over the same WebSocket     (client -> server)
    and LOG every one with the context snapshot that produced the board.

THE SEAM: this server is the ONLY caller of rank_vocabulary(). Models and
perception live behind that function — never in a route, the UI, or here.

================================================================================
WIRE MESSAGE SCHEMA  (STABLE CONTRACT — everything downstream depends on this)
================================================================================
server -> client:
  {
    "type": "vocab_update",
    "v": 1,                          # schema version
    "board_id": "<uuid4>",           # identifies this exact board snapshot
    "server_ts": <float epoch sec>,  # when the server produced it
    "items": [ {"word": <str>, "symbol": <str>}, ... ]   # ranked, best first
  }

client -> server:
  {
    "type": "tap",
    "word": <str>,                   # REQUIRED: the word that was tapped
    "client_ts": <number>,           # REQUIRED: browser timestamp (ms epoch)
    "board_id": "<uuid4>",           # OPTIONAL: which board it came from (for logging)
    "source": "core" | "dynamic"     # OPTIONAL (additive): stable-core vs dynamic strip
  }

These two message types are FIXED. Add fields if needed; never repurpose
existing ones.
================================================================================
"""

from __future__ import annotations

import asyncio
import json
import socket
import time
import uuid
from pathlib import Path

from aiohttp import WSMsgType, web

from context_state import ContextState
from rank_vocabulary import rank_vocabulary
from symbol_map import SYMBOLS_DIR
import perception.config as perception_config
from perception.runner import Perception

# ------------------------------- CONFIG -------------------------------------
HOST = "0.0.0.0"
PORT = 8080
SCHEMA_VERSION = 1
WEB_DIR = Path(__file__).parent / "web"
TAP_LOG_PATH = Path(__file__).parent / "taps.jsonl"
MAX_REMEMBERED_BOARDS = 200

# Trigger-driven push: fire on (scene_changed OR addressed), debounced.
TRIGGER_POLL_S = 0.5        # how often we check for a trigger condition
TRIGGER_DEBOUNCE_S = 2.0    # minimum gap between consecutive pushes
HEARTBEAT_INTERVAL_S = 30.0  # slow liveness push when no trigger fires

PROFILE_PATH = Path(__file__).parent / "personal_context.json"
# ----------------------------------------------------------------------------


async def _make_board(app: web.Application) -> tuple[dict, dict]:
    """Produce one vocab_update message by calling THE SEAM via asyncio.to_thread.

    The LLM inference call blocks ~2.5s; offloading to a thread keeps the event
    loop free for WS heartbeats and incoming taps.

    Returns (board_message, produced_from).
    """
    state = app["state"]
    scene, transcript, context = state["context"].snapshot()
    context["personal_context"] = state["personal_context"]

    ranker = state["ranker"]
    items = await asyncio.to_thread(rank_vocabulary, scene, transcript, context, ranker)

    board = {
        "type": "vocab_update",
        "v": SCHEMA_VERSION,
        "board_id": str(uuid.uuid4()),
        "server_ts": time.time(),
        "items": items,
    }
    produced_from = {"scene": scene, "transcript": transcript, "context": context}
    return board, produced_from


def _record_tap(app: web.Application, msg: dict, peer: str) -> None:
    """Print and LOG a tap with the context snapshot that produced its board."""
    word = msg.get("word")
    source = msg.get("source")
    client_ts = msg.get("client_ts")
    board_id = msg.get("board_id")
    print(f"[tap] {source}:{word!r} client_ts={client_ts} board_id={board_id} from {peer}")

    board_record = app["state"]["boards"].get(board_id, {})
    entry = {
        "server_recv_ts": time.time(),
        "word": word,
        "source": source,
        "client_ts": client_ts,
        "board_id": board_id,
        "context_snapshot": board_record.get("produced_from"),
        "board_items": board_record.get("items"),
        "peer": peer,
    }
    with TAP_LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


async def _handle_client_message(app: web.Application, raw: str, peer: str) -> None:
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[ws] dropped non-JSON message from {peer}")
        return
    if msg.get("type") == "tap":
        _record_tap(app, msg, peer)
    else:
        print(f"[ws] ignored message type={msg.get('type')!r} from {peer}")


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)

    clients = request.app["state"]["clients"]
    clients.add(ws)
    peer = request.remote
    print(f"[ws] connected: {peer}  ({len(clients)} client(s))")

    latest = request.app["state"]["latest_board"]
    if latest is not None:
        await ws.send_str(json.dumps(latest))

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                await _handle_client_message(request.app, msg.data, peer)
            elif msg.type == WSMsgType.ERROR:
                print(f"[ws] error from {peer}: {ws.exception()}")
    finally:
        clients.discard(ws)
        print(f"[ws] disconnected: {peer}  ({len(clients)} client(s))")
    return ws


async def index_handler(request: web.Request) -> web.FileResponse:
    return web.FileResponse(WEB_DIR / "index.html")


async def _produce_and_broadcast(app: web.Application) -> None:
    board, produced_from = await _make_board(app)

    state = app["state"]
    boards = state["boards"]
    boards[board["board_id"]] = {"items": board["items"], "produced_from": produced_from}
    while len(boards) > MAX_REMEMBERED_BOARDS:
        del boards[next(iter(boards))]

    state["latest_board"] = board
    payload = json.dumps(board)

    dead = []
    for ws in list(state["clients"]):
        try:
            await ws.send_str(payload)
        except (ConnectionResetError, RuntimeError):
            dead.append(ws)
    for ws in dead:
        state["clients"].discard(ws)

    print(f"[push] vocab_update board={board['board_id'][:8]} "
          f"items={len(board['items'])} -> {len(state['clients'])} client(s)")


async def trigger_loop(app: web.Application) -> None:
    """Poll ContextState every TRIGGER_POLL_S; fire on scene_changed OR addressed,
    debounced so the LLM isn't hammered."""
    state = app["state"]
    while True:
        await asyncio.sleep(TRIGGER_POLL_S)
        try:
            _, _, context = state["context"].snapshot()
            scene_changed = bool(context.get("scene_changed"))
            addressed = bool(context.get("addressed"))
            fire = scene_changed or addressed
            if not fire:
                continue

            # Build a dedup signature from the relevant timestamps
            sig = (
                round(context.get("vision_ts", 0.0), 1) if scene_changed else None,
                round(context.get("audio_ts", 0.0), 1) if addressed else None,
            )
            now = time.time()
            if (
                sig != state["last_trigger_sig"]
                and now - state["last_push_ts"] >= TRIGGER_DEBOUNCE_S
            ):
                await _produce_and_broadcast(app)
                state["last_trigger_sig"] = sig
                state["last_push_ts"] = now
        except Exception as exc:
            print(f"[trigger_loop] error: {exc}")


async def heartbeat_loop(app: web.Application) -> None:
    """Push a liveness board every HEARTBEAT_INTERVAL_S when no trigger fires.

    Ensures a newly-connected board isn't stale if no trigger event has fired
    since the last push.
    """
    state = app["state"]
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_S)
        try:
            if time.time() - state["last_push_ts"] >= HEARTBEAT_INTERVAL_S:
                await _produce_and_broadcast(app)
                state["last_push_ts"] = time.time()
        except Exception as exc:
            print(f"[heartbeat_loop] error: {exc}")


def _lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


async def on_startup(app: web.Application) -> None:
    # Load personal context profile (safe: returns minimal stub if file absent)
    from personal_context_loader import load_profile
    from ranker import make_ranker
    import ranker_config

    app["state"]["personal_context"] = load_profile(PROFILE_PATH)
    app["state"]["ranker"] = make_ranker(ranker_config)

    # Perception: shared context state + concurrent workers
    app["state"]["context"] = ContextState()
    if perception_config.PERCEPTION_ENABLED:
        perception = Perception(app["state"]["context"])
        perception.start(asyncio.get_running_loop())
        app["state"]["perception"] = perception

    # Trigger-driven push replaces the old blind 4s push loop.
    app["state"]["trigger_task"] = asyncio.create_task(trigger_loop(app))
    app["state"]["heartbeat_task"] = asyncio.create_task(heartbeat_loop(app))

    # Push an initial board now so a client connecting before any trigger fires
    # sees words immediately, not a blank strip for up to HEARTBEAT_INTERVAL_S.
    await _produce_and_broadcast(app)
    app["state"]["last_push_ts"] = time.time()

    ip = _lan_ip()
    print("=" * 64)
    print("VoiceLink server up. LLM ranker active (or heuristic fallback).")
    print("Board fires on scene_changed / addressed (debounced 2s).")
    print(f"  Open the AAC board on the iPad/phone (same Wi-Fi):")
    print(f"      http://{ip}:{PORT}/")
    print(f"  Local:  http://127.0.0.1:{PORT}/")
    print(f"  Taps logged to: {TAP_LOG_PATH}")
    print("=" * 64)


async def on_cleanup(app: web.Application) -> None:
    state = app["state"]
    for key in ("trigger_task", "heartbeat_task"):
        task = state.get(key)
        if task is not None:
            task.cancel()
    if state.get("perception") is not None:
        await state["perception"].stop()
    for ws in list(state["clients"]):
        await ws.close()


def make_app() -> web.Application:
    app = web.Application()
    app["state"] = {
        "clients": set(),
        "boards": {},
        "latest_board": None,
        "context": None,
        "perception": None,
        # Task 5 additions — all initialised here before the app freezes
        "personal_context": None,
        "ranker": None,
        "last_trigger_sig": None,
        "last_push_ts": 0.0,
        "trigger_task": None,
        "heartbeat_task": None,
    }
    app.add_routes([
        web.get("/", index_handler),
        web.get("/ws", ws_handler),
    ])
    # Static route for locally-cached ARASAAC pictograms — same origin, no internet
    app.router.add_static("/symbols/", path=SYMBOLS_DIR, show_index=False)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


if __name__ == "__main__":
    web.run_app(make_app(), host=HOST, port=PORT, print=None)
