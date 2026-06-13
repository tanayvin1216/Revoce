"""
Perception runner — wires the wearable client + the two concurrent workers and
points all of their output at the shared ContextState.

Concurrency model:
  * wearable_client  -> asyncio task on the server's event loop (fast demux only)
  * AudioWorker      -> its own thread  (Whisper)
  * VisionWorker     -> its own thread  (change detector + gated VLM)
The two workers are independent threads, so a slow VLM call never stalls audio.
Backends are built INSIDE the worker threads so heavy model loads don't block
the asyncio startup.
"""

from __future__ import annotations

import asyncio
import queue
import threading

from . import config as default_cfg
from .audio_worker import AudioWorker
from .scene_describer import make_scene_describer
from .transcriber import make_transcriber
from .vision_worker import FrameHolder, VisionWorker
from .wearable_client import wearable_client


class Perception:
    def __init__(self, context_state, cfg=default_cfg) -> None:
        self._ctx = context_state
        self._cfg = cfg
        self._stop = threading.Event()
        self._audio_q: "queue.Queue[bytes]" = queue.Queue(maxsize=200)
        self._frames = FrameHolder()
        self._audio_worker: AudioWorker | None = None
        self._vision_worker: VisionWorker | None = None
        self._client_task: asyncio.Task | None = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        c = self._cfg
        print(f"[perception] transcriber={c.TRANSCRIBER}({c.WHISPER_MODEL})  "
              f"scene={c.SCENE_DESCRIBER}({c.VLM_MODEL})")
        print(f"[perception] wearable feed: {c.WEARABLE_WS_URL}")

        self._audio_worker = AudioWorker(self._audio_q, lambda: make_transcriber(c), self._ctx, c)
        self._vision_worker = VisionWorker(self._frames, lambda: make_scene_describer(c), self._ctx, c)
        self._audio_worker.start()
        self._vision_worker.start()
        self._client_task = loop.create_task(
            wearable_client(c.WEARABLE_WS_URL, self._audio_q, self._frames, self._stop)
        )

    async def stop(self) -> None:
        self._stop.set()
        if self._client_task:
            self._client_task.cancel()
        if self._audio_worker:
            self._audio_worker.stop()
        if self._vision_worker:
            self._vision_worker.stop()
