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
from typing import Any

from aiohttp import WSMsgType, web

import privacy
from context_state import ContextState
from perception.wearable_client import CAPTURE_PAUSED
from rank_vocabulary import rank_vocabulary, topics_from_context
from symbol_map import SYMBOLS_DIR
import perception.config as perception_config
from perception.runner import Perception

# ------------------------------- CONFIG -------------------------------------
HOST = "0.0.0.0"
PORT = 8080
SCHEMA_VERSION = 1
WEB_DIR = Path(__file__).parent / "web"
TAP_LOG_PATH = Path(__file__).parent / "taps.jsonl"
PHASE_STATUS_PATH = Path(__file__).parent.parent / "phase_status.json"
MAX_REMEMBERED_BOARDS = 200

# Core taps that ask to HOLD the conversation rather than answer it. Tapping
# these must not clear the active question -- see _record_tap.
_QUESTION_HOLDING_TAPS = {"wait", "i need help"}

# How long to wait on a single client's send before treating it as dead.
WS_SEND_TIMEOUT_S = 5.0

# Trigger-driven push: fire on (scene_changed OR addressed), debounced.
TRIGGER_POLL_S = 0.5        # how often we check for a trigger condition
TRIGGER_DEBOUNCE_S = 2.0    # minimum gap between consecutive pushes
HEARTBEAT_INTERVAL_S = 30.0  # slow liveness push when no trigger fires

PROFILE_PATH = Path(__file__).parent / "personal_context.json"
SAMPLE_PROFILE_PATH = Path(__file__).parent / "personal_context.sample.json"
# Console image uploads: lift aiohttp's 1 MB default so real phone photos fit.
# The VLM downscales past 768px anyway, so this only bounds the upload itself.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
# ----------------------------------------------------------------------------


class _CaptureRanker:
    """Records the pool returned by the ranker during a seam call.

    The console must not call models directly. It wraps the same ranker instance
    passed into rank_vocabulary(), so any captured candidates were produced only
    inside the seam call.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.candidates: list[dict] = []
        self.elapsed_ms = 0.0

    def rank(self, system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> list[dict]:
        start = time.perf_counter()
        try:
            if max_tokens is None:
                result = self._inner.rank(system_prompt, user_prompt)
            else:
                try:
                    result = self._inner.rank(system_prompt, user_prompt, max_tokens=max_tokens)
                except TypeError:
                    result = self._inner.rank(system_prompt, user_prompt)
        finally:
            self.elapsed_ms += (time.perf_counter() - start) * 1000
        self.candidates = [dict(item) for item in result if isinstance(item, dict)]
        return result


def _console_scenarios() -> list[dict]:
    return [
        {
            "id": "airport",
            "label": "Airport",
            "scene": "a crowded airport gate with boarding screens, luggage, and a loud announcement",
            "transcript": "Boarding group B can line up now. Do you want to use the bathroom before we board?",
            "profile_name": "Jordan",
        },
        {
            "id": "photosynthesis",
            "label": "Photosynthesis classroom",
            "scene": "a biology classroom with a whiteboard diagram of sunlight, leaves, water, and carbon dioxide",
            "transcript": "How do plants make sugar during photosynthesis?",
            "profile_name": "Jordan",
        },
        {
            "id": "doctor",
            "label": "Doctor",
            "scene": "a quiet exam room with a doctor, a blood pressure cuff, and a pain chart",
            "transcript": "Can you show me where it hurts and tell me if the pain is sharp or dull?",
            "profile_name": "Jordan",
        },
    ]


def _phase_status_payload(path: Path = PHASE_STATUS_PATH) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        return {
            "updated_ts": None,
            "phases": [],
            "error": f"could not read phase status: {exc}",
        }


def _load_json_file(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _profile_options() -> list[dict]:
    options: list[dict] = []
    for label, path in (("local", PROFILE_PATH), ("sample", SAMPLE_PROFILE_PATH)):
        profile = _load_json_file(path)
        if profile is None:
            continue
        identity = profile.get("identity", {})
        name = identity.get("name") or label
        options.append({"id": label, "name": name, "profile": profile})
    if not options:
        options.append({"id": "minimal", "name": "User", "profile": {}})
    return options


def _profile_for_name(name: str | None, fallback: dict | None = None) -> dict:
    profiles = _profile_options()
    target = (name or "").strip().lower()
    for option in profiles:
        if target in {option["id"].lower(), option["name"].lower()}:
            return option["profile"]
    return fallback or profiles[0]["profile"]


async def _get_scene_describer(state: dict):
    """Lazily build ONE real VLM scene describer for the console image route.

    Built on first image use (off the event loop) so text-only console runs never
    pay for the VLM load. Independent of perception, so the console is a complete
    standalone simulator: image -> real VLM -> seam -> board.
    """
    if state.get("scene_describer") is None:
        import perception.config as perception_config
        from perception.scene_describer import make_scene_describer
        state["scene_describer"] = await asyncio.to_thread(
            make_scene_describer, perception_config
        )
    return state["scene_describer"]


async def _get_transcriber(state: dict):
    """Lazily build ONE real transcriber for the console audio route (mirrors
    _get_scene_describer's lazy-VLM pattern) so text-only console runs never
    pay for the whisper model load."""
    if state.get("transcriber") is None:
        from perception.transcriber import make_transcriber
        state["transcriber"] = await asyncio.to_thread(make_transcriber, perception_config)
    return state["transcriber"]


async def _run_console_payload(state: dict, payload: dict, image_bytes: bytes | None = None) -> dict:
    transcript = str(payload.get("transcript") or "").strip() or None

    # When an image is supplied, the REAL VLM produces the scene (the full
    # pipeline). Otherwise the scene comes from the typed text (seam-only).
    vlm_ms = 0.0
    if image_bytes:
        describer = await _get_scene_describer(state)
        vlm_start = time.perf_counter()
        scene = await asyncio.to_thread(describer.describe, image_bytes)
        vlm_ms = (time.perf_counter() - vlm_start) * 1000
        scene = (scene or "").strip() or None
    else:
        scene = str(payload.get("scene") or "").strip() or None

    profile = _profile_for_name(payload.get("profile_name"), state.get("personal_context"))
    context = {
        "speaker": "partner" if transcript else None,
        "addressed": bool(transcript),
        "last_utterance": transcript,
        "recent_transcript": transcript or "",
        "scene_changed": bool(scene),
        "audio_ts": time.time() if transcript else 0.0,
        "vision_ts": time.time() if scene else 0.0,
        "personal_context": profile,
    }

    capture_ranker = _CaptureRanker(state["ranker"])
    # The console may pick the strategy per run (so you can compare student vs
    # diverse live); otherwise fall back to the server default.
    requested = str(payload.get("strategy") or "").strip()
    strategy = requested if requested in ("diverse", "topk", "student") \
        else state.get("board_strategy", "diverse")
    topics = topics_from_context(scene, transcript, context)
    start = time.perf_counter()
    # Same model instance the push loops use — take the lock. A console "Fire"
    # during a live generation would otherwise be a third concurrent caller.
    lock = state.get("inference_lock")
    if lock is None:
        board = await asyncio.to_thread(
            rank_vocabulary, scene, transcript, context, capture_ranker,
            strategy=strategy, topics=topics,
        )
    else:
        async with lock:
            board = await asyncio.to_thread(
                rank_vocabulary, scene, transcript, context, capture_ranker,
                strategy=strategy, topics=topics,
            )
    total_ms = (time.perf_counter() - start) * 1000
    pool_ms = capture_ranker.elapsed_ms

    return {
        "scene_description": scene or "not available",
        "transcript": transcript or "",
        "profile": (profile.get("identity", {}) or {}).get("name") or "User",
        "strategy": strategy,
        "topics": topics,
        "candidate_pool": capture_ranker.candidates,
        "board": board,
        "latency_ms": {
            "whisper": 0.0,
            "vlm": round(vlm_ms, 1),
            "pool_gen": round(pool_ms, 1),
            "selection": round(max(total_ms - pool_ms, 0.0), 1),
            "total": round(total_ms, 1),
        },
    }


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
    strategy = state["board_strategy"]
    topics = topics_from_context(scene, transcript, context)
    items = await asyncio.to_thread(
        rank_vocabulary, scene, transcript, context, ranker,
        strategy=strategy, topics=topics,
    )

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
    try:
        with TAP_LOG_PATH.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as exc:
        # The word was already spoken client-side, so losing the log row is the
        # graceful outcome. Letting this escape would tear down the WebSocket on
        # every tap and churn the connection instead.
        print(f"[tap] WARNING could not write tap log: {exc}")

    # Most taps answer the held question, so the board should move on. But two
    # of the core words are not answers: "wait" means "give me time" and
    # "I need help" is a call for support. An AAC user taps those precisely
    # BECAUSE they need the moment held — clearing the question on them drops
    # the answer vocabulary they were composing with and regenerates the board
    # from ambient chatter. The tap meant to hold the moment destroyed it.
    ctx = app["state"].get("context")
    if ctx is not None and str(word or "").strip().lower() not in _QUESTION_HOLDING_TAPS:
        ctx.clear_active_question()


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


async def console_handler(request: web.Request) -> web.FileResponse:
    return web.FileResponse(WEB_DIR / "console.html")


async def phase_status_handler(request: web.Request) -> web.Response:
    return web.json_response(_phase_status_payload(request.app["state"]["phase_status_path"]))


async def console_scenarios_handler(request: web.Request) -> web.Response:
    return web.json_response({"scenarios": _console_scenarios()})


async def console_profiles_handler(request: web.Request) -> web.Response:
    profiles = [
        {"id": option["id"], "name": option["name"]}
        for option in _profile_options()
    ]
    return web.json_response({"profiles": profiles})


async def console_run_handler(request: web.Request) -> web.Response:
    # Two body shapes: multipart (image upload -> real VLM) or JSON (text scene).
    content_type = request.headers.get("Content-Type") or ""
    image_bytes: bytes | None = None
    if content_type.startswith("multipart/"):
        payload: dict = {}
        reader = await request.multipart()
        async for part in reader:
            if part.name == "image":
                data = await part.read(decode=False)
                image_bytes = data or None
            else:
                payload[part.name] = await part.text()
    else:
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON"}, status=400)
        if not isinstance(payload, dict):
            return web.json_response({"error": "payload must be an object"}, status=400)
    result = await _run_console_payload(request.app["state"], payload, image_bytes=image_bytes)
    return web.json_response(result)


async def console_transcribe_handler(request: web.Request) -> web.Response:
    """Console mic-recording route: browser audio (any container av can read,
    e.g. webm/opus) in -> real whisper transcript out. Independent of the
    live wearable audio path; lets you test whisper from the console alone."""
    reader = await request.multipart()
    audio_bytes: bytes | None = None
    async for part in reader:
        if part.name == "audio":
            audio_bytes = await part.read(decode=False)
    if not audio_bytes:
        return web.json_response({"error": "missing 'audio' field"}, status=400)

    import io
    from faster_whisper.audio import decode_audio
    try:
        pcm = await asyncio.to_thread(
            decode_audio, io.BytesIO(audio_bytes), perception_config.SAMPLE_RATE
        )
    except Exception as exc:
        return web.json_response({"error": f"could not decode audio: {exc}"}, status=400)

    transcriber = await _get_transcriber(request.app["state"])
    start = time.perf_counter()
    result = await asyncio.to_thread(transcriber.transcribe, pcm, perception_config.SAMPLE_RATE)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return web.json_response({
        "text": result.text,
        "avg_logprob": round(result.avg_logprob, 3),
        "no_speech_prob": round(result.no_speech_prob, 3),
        "latency_ms": round(elapsed_ms, 1),
    })


async def privacy_status_handler(request: web.Request) -> web.Response:
    """Data-management status: tap-log size/age, profile location, proof that no
    audio/video artifacts exist on disk. No model call (plain file inspection)."""
    payload = await asyncio.to_thread(privacy.status, TAP_LOG_PATH, PROFILE_PATH)
    payload["capture_paused"] = CAPTURE_PAUSED.is_set()
    return web.json_response(payload)


async def privacy_pause_handler(request: web.Request) -> web.Response:
    """Software kill switch: POST toggles the pause flag that makes the wearable
    client drop every incoming frame/audio chunk (see perception/wearable_client.py).
    The physical button on the wearable remains the primary kill switch."""
    if CAPTURE_PAUSED.is_set():
        CAPTURE_PAUSED.clear()
    else:
        CAPTURE_PAUSED.set()
    print(f"[privacy] capture {'PAUSED' if CAPTURE_PAUSED.is_set() else 'RESUMED'} via /api/privacy/pause")
    return web.json_response({"capture_paused": CAPTURE_PAUSED.is_set()})


async def _produce_and_broadcast(app: web.Application) -> None:
    state = app["state"]

    # Claim the slot BEFORE generating, not after. A warm diverse board takes
    # ~30s; if last_push_ts only moved on completion, the heartbeat loop would
    # wake mid-generation, still see a stale timestamp, and start a second
    # generation on the same model.
    async with state["inference_lock"]:
        state["last_push_ts"] = time.time()
        board, produced_from = await _make_board(app)

    boards = state["boards"]
    boards[board["board_id"]] = {"items": board["items"], "produced_from": produced_from}
    while len(boards) > MAX_REMEMBERED_BOARDS:
        del boards[next(iter(boards))]

    state["latest_board"] = board
    payload = json.dumps(board)

    # Send to every client concurrently, with a timeout. Sequential awaits meant
    # one iPad that slept with a full TCP buffer froze the board for everyone
    # until aiohttp's 30s heartbeat finally reaped it.
    async def _send(ws) -> bool:
        try:
            await asyncio.wait_for(ws.send_str(payload), timeout=WS_SEND_TIMEOUT_S)
            return True
        except (ConnectionResetError, RuntimeError, asyncio.TimeoutError):
            return False

    targets = list(state["clients"])
    results = await asyncio.gather(*(_send(ws) for ws in targets))
    for ws, ok in zip(targets, results):
        if not ok:
            state["clients"].discard(ws)

    state["last_push_ts"] = time.time()
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
                state["last_trigger_sig"] = sig
                await _produce_and_broadcast(app)
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


async def _warm_up_ranker(app: web.Application) -> None:
    """Pay the serving path's one-time cost at STARTUP, not on the first click.

    Warms whatever board_strategy actually serves, which matters because the
    cost differs by three orders of magnitude:

      "student" (default) — loads MiniLM + the checkpoint. A few seconds.
      "diverse"           — compiles MLX/Metal kernels for a ~520-token
                            generation. ~110s cold, and the board then costs
                            ~30s every time.

    This runs BEFORE the socket binds, so it is also how long the board page
    shows "connecting". That was a two-minute wall when the LLM path was the
    default; on the student path it is seconds. Never blocks on failure — a
    HeuristicRanker warm-up is instant and harmless.
    """
    warm_context = {
        "addressed": True,
        "last_utterance": "can you tell me how this works?",
        "personal_context": app["state"]["personal_context"],
    }
    try:
        t0 = time.perf_counter()
        await asyncio.to_thread(
            rank_vocabulary,
            "a classroom with a whiteboard and students at their desks",
            "Can you explain how this works?",
            warm_context,
            app["state"]["ranker"],
            strategy=app["state"]["board_strategy"],
            topics=["science", "lesson", "classroom"],
        )
        print(f"\n[server] ranker warm-up done in {time.perf_counter() - t0:.1f}s "
              f"(first board will be warm)", flush=True)
    except Exception as exc:
        print(f"\n[server] ranker warm-up skipped: {exc}", flush=True)


async def on_startup(app: web.Application) -> None:
    # Load personal context profile (safe: returns minimal stub if file absent)
    from personal_context_loader import load_profile
    from ranker import make_ranker
    import ranker_config

    app["state"]["personal_context"] = load_profile(PROFILE_PATH)
    app["state"]["ranker"] = make_ranker(ranker_config)

    # Tap-log retention: VOICELINK_TAP_RETENTION_DAYS=N drops taps older than N
    # days at startup. Default 0 = keep forever (tap data is irreplaceable).
    privacy.apply_retention(TAP_LOG_PATH)

    # Server board strategy (config-flagged; "diverse" by default, "topk" fallback).
    strategy = getattr(ranker_config, "BOARD_STRATEGY", "diverse")
    app["state"]["board_strategy"] = (
        strategy if strategy in ("diverse", "topk", "student") else "diverse")
    print(f"[server] board strategy: {app['state']['board_strategy']}")

    # Bind the seam lock to the running loop before anything can call the seam.
    app["state"]["inference_lock"] = asyncio.Lock()

    # Perception: shared context state + concurrent workers
    app["state"]["context"] = ContextState()
    if perception_config.PERCEPTION_ENABLED:
        perception = Perception(app["state"]["context"])
        perception.start(asyncio.get_running_loop())
        app["state"]["perception"] = perception

    # Pay the one-time model kernel-compile cost now, before the first click.
    await _warm_up_ranker(app)

    # Trigger-driven push replaces the old blind 4s push loop.
    app["state"]["trigger_task"] = asyncio.create_task(trigger_loop(app))
    app["state"]["heartbeat_task"] = asyncio.create_task(heartbeat_loop(app))

    # Push an initial board now so a client connecting before any trigger fires
    # sees words immediately, not a blank strip for up to HEARTBEAT_INTERVAL_S.
    await _produce_and_broadcast(app)

    ip = _lan_ip()
    print("=" * 64)
    print("VoiceLink server up. LLM ranker active (or heuristic fallback).")
    print("Board fires on scene_changed / addressed (debounced 2s).")
    print("  Open the AAC board on the iPad/phone (same Wi-Fi):")
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
    app = web.Application(client_max_size=MAX_UPLOAD_BYTES)
    app["state"] = {
        "clients": set(),
        "boards": {},
        "latest_board": None,
        "context": None,
        "perception": None,
        # Task 5 additions — all initialised here before the app freezes
        "personal_context": None,
        "ranker": None,
        "board_strategy": "diverse",  # set from ranker_config in on_startup
        "scene_describer": None,      # lazy VLM for the console image route
        "transcriber": None,          # lazy whisper for the console audio route
        "last_trigger_sig": None,
        "last_push_ts": 0.0,
        # Serializes every seam call. One MLX model instance is shared by the
        # trigger loop, the heartbeat loop and the console; concurrent
        # generate() calls on it are not safe. Created lazily in on_startup so
        # it binds to the running loop.
        "inference_lock": None,
        "trigger_task": None,
        "heartbeat_task": None,
        "phase_status_path": PHASE_STATUS_PATH,
    }
    app.add_routes([
        web.get("/", index_handler),
        web.get("/console", console_handler),
        web.get("/ws", ws_handler),
        web.get("/api/phase-status", phase_status_handler),
        web.get("/api/console/scenarios", console_scenarios_handler),
        web.get("/api/console/profiles", console_profiles_handler),
        web.post("/api/console/run", console_run_handler),
        web.post("/api/console/transcribe", console_transcribe_handler),
        web.get("/api/privacy/status", privacy_status_handler),
        web.post("/api/privacy/pause", privacy_pause_handler),
    ])
    # Static route for locally-cached ARASAAC pictograms — same origin, no internet
    app.router.add_static("/symbols/", path=SYMBOLS_DIR, show_index=False)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


if __name__ == "__main__":
    web.run_app(make_app(), host=HOST, port=PORT, print=None)
