"""
VoiceLink Mac server — the ONLY way the wearable and the web app reach
intelligence.

Responsibilities:
  * Serve the placeholder AAC web page over PLAIN HTTP (same origin as the
    WebSocket, so a ws:// connection isn't blocked as mixed content).
  * Push vocabulary boards to the page over a WebSocket   (server -> client).
  * Receive taps from the page over the same WebSocket     (client -> server)
    and LOG every one with the context snapshot that produced the board.
    (CLAUDE.md: this interaction data is irreplaceable and cannot be recovered
    retroactively — so we log it from the very first version of the UI.)

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
existing ones. ("source" is such an additive field — vocab_update is untouched.)
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
import perception.config as perception_config
from perception.runner import Perception

# ------------------------------- CONFIG -------------------------------------
HOST = "0.0.0.0"        # bind all interfaces so the iPad on the LAN can reach us
PORT = 8080             # PLAIN HTTP (not HTTPS) — keeps ws:// from being blocked
PUSH_INTERVAL_S = 4.0   # how often the server pushes a fresh board (liveness/demo)
SCHEMA_VERSION = 1
WEB_DIR = Path(__file__).parent / "web"
TAP_LOG_PATH = Path(__file__).parent / "taps.jsonl"   # append-only tap log
MAX_REMEMBERED_BOARDS = 200   # keep recent boards so taps can be logged with context
# ----------------------------------------------------------------------------


def _make_board(app: web.Application) -> tuple[dict, dict]:
    """Produce one vocab_update message by calling THE SEAM.

    Reads the latest perception understanding from the shared ContextState and
    passes it straight into rank_vocabulary(). Perception writes that state on
    its own threads; here we only read a snapshot.

    Returns (board_message, produced_from) where produced_from is the snapshot
    of inputs that produced the board — kept so taps can be logged with context.
    """
    scene, transcript, context = app["state"]["context"].snapshot()

    items = rank_vocabulary(scene, transcript, context)   # <-- the only model seam

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
    """Print and LOG a tap with the context snapshot that produced its board.

    This is the irreplaceable interaction data from CLAUDE.md: every tap, with a
    snapshot of what the system "saw" when it offered that board.
    """
    word = msg.get("word")
    source = msg.get("source")        # "core" | "dynamic" (additive tap field)
    client_ts = msg.get("client_ts")
    board_id = msg.get("board_id")
    print(f"[tap] {source}:{word!r} client_ts={client_ts} board_id={board_id} from {peer}")

    # Join the tap to the context that produced the board the user was viewing.
    # produced_from = {scene, transcript, context} — currently placeholders, but
    # the snapshot is captured NOW so it becomes real training data once
    # perception feeds rank_vocabulary(). This cannot be recreated later.
    board_record = app["state"]["boards"].get(board_id, {})
    entry = {
        "server_recv_ts": time.time(),
        "word": word,
        "source": source,
        "client_ts": client_ts,
        "board_id": board_id,
        "context_snapshot": board_record.get("produced_from"),  # {scene, transcript, context}
        "board_items": board_record.get("items"),               # the board that was shown
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
    ws = web.WebSocketResponse(heartbeat=30)  # heartbeat keeps iPad sleep from dropping us
    await ws.prepare(request)

    clients = request.app["state"]["clients"]
    clients.add(ws)
    peer = request.remote
    print(f"[ws] connected: {peer}  ({len(clients)} client(s))")

    # Send the latest board immediately so a freshly opened page isn't blank.
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
    board, produced_from = _make_board(app)

    state = app["state"]
    boards = state["boards"]
    boards[board["board_id"]] = {"items": board["items"], "produced_from": produced_from}
    # bounded memory: drop oldest boards (dict preserves insertion order)
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


async def push_loop(app: web.Application) -> None:
    """Periodically produce a board via the seam and push it to all clients."""
    while True:
        await _produce_and_broadcast(app)
        await asyncio.sleep(PUSH_INTERVAL_S)


def _lan_ip() -> str:
    """Best-effort LAN IP so we can print the URL to open on the iPad."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no packets actually sent; picks the route's source IP
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


async def on_startup(app: web.Application) -> None:
    # The push task is created here; all mutable state lives in app["state"]
    # (set in make_app before the app freezes) and is only ever mutated in place.
    app["state"]["push_task"] = asyncio.create_task(push_loop(app))

    # Perception: shared context state + concurrent workers that feed it.
    # (perception writes the context; _make_board reads it — they never call
    # each other directly, and perception never touches rank_vocabulary/the UI.)
    app["state"]["context"] = ContextState()
    if perception_config.PERCEPTION_ENABLED:
        perception = Perception(app["state"]["context"])
        perception.start(asyncio.get_running_loop())
        app["state"]["perception"] = perception

    ip = _lan_ip()
    print("=" * 64)
    print("VoiceLink server up — plain HTTP. rank_vocabulary is still a STUB;")
    print("perception (Whisper + gated VLM) feeds the context it will consume.")
    print("  Open the AAC board on the iPad/phone (same Wi-Fi):")
    print(f"      http://{ip}:{PORT}/")
    print(f"  Local:  http://127.0.0.1:{PORT}/")
    print(f"  Taps logged to: {TAP_LOG_PATH}")
    print("=" * 64)


async def on_cleanup(app: web.Application) -> None:
    state = app["state"]
    if state["push_task"] is not None:
        state["push_task"].cancel()
    if state.get("perception") is not None:
        await state["perception"].stop()
    for ws in list(state["clients"]):
        await ws.close()


def make_app() -> web.Application:
    app = web.Application()
    # All mutable runtime state lives in this one dict so we never reassign an
    # app[...] key after startup (aiohttp freezes the Application mapping).
    app["state"] = {
        "clients": set(),       # connected WebSocket responses
        "boards": {},           # board_id -> {items, produced_from}  (bounded)
        "latest_board": None,   # last board pushed (sent to new clients on connect)
        "push_task": None,      # background broadcast loop
        "context": None,        # ContextState — perception writes, _make_board reads
        "perception": None,     # Perception runner (wearable client + workers)
    }
    app.add_routes([
        web.get("/", index_handler),
        web.get("/ws", ws_handler),
    ])
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


if __name__ == "__main__":
    web.run_app(make_app(), host=HOST, port=PORT, print=None)
