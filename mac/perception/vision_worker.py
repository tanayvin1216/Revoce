"""
Vision worker — gated VLM on the wearable's 0x01 JPEG stream.

Runs on its OWN thread. It always processes only the NEWEST frame (a single-slot
holder, never a backlog), runs the cheap change detector, and fires the
expensive VLM ONLY when the scene meaningfully changed AND a minimum interval
has passed. The VLM call blocks THIS thread only — the audio worker is
unaffected, so audio never stalls on vision.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from .change_detector import FrameChangeDetector


class FrameHolder:
    """Single-slot latest-frame holder (thread-safe). We always describe the
    newest frame, never a queue of stale ones."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None

    def set(self, jpeg_bytes: bytes) -> None:
        with self._lock:
            self._jpeg = jpeg_bytes

    def get(self) -> bytes | None:
        with self._lock:
            return self._jpeg


class VisionWorker(threading.Thread):
    def __init__(self, frame_holder: FrameHolder, describer_factory, context_state, cfg) -> None:
        super().__init__(name="vision", daemon=True)
        self._holder = frame_holder
        self._make_describer = describer_factory
        self._ctx = context_state
        self._cfg = cfg
        self._stop = threading.Event()
        self._detector = FrameChangeDetector(cfg.CHANGE_THRESHOLD)
        self._last_vlm_t = 0.0

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        cfg = self._cfg
        vlm = self._make_describer()   # build backend here (off the event loop)
        while not self._stop.is_set():
            time.sleep(cfg.VISION_POLL_S)
            jpeg = self._holder.get()
            if jpeg is None:
                continue
            gray_small = self._decode_small(jpeg, cfg.CHANGE_GRAY_SIZE)
            if gray_small is None:
                continue

            changed, score = self._detector.is_changed(gray_small)
            if not changed:
                continue
            now = time.time()
            if (now - self._last_vlm_t) < cfg.VLM_MIN_INTERVAL_S:
                # changed, but throttled — thermal/latency discipline
                continue
            self._last_vlm_t = now

            print(f"[vision] scene changed (diff={score:.1f}) -> running VLM…")
            try:
                desc = vlm.describe(jpeg)   # EXPENSIVE — blocks only this thread
            except Exception as e:
                print(f"[vision] VLM error: {e}")
                continue
            self._ctx.update_vision(description=desc, scene_changed=True)
            print(f"[vision] scene: «{desc}»")

    @staticmethod
    def _decode_small(jpeg_bytes: bytes, size: int) -> "np.ndarray | None":
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
