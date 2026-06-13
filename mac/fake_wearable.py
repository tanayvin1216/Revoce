"""
Fake wearable — a dev stand-in for the ESP32-S3 capture device.

It speaks the SAME WebSocket protocol as the real firmware (binary messages:
0x01 + JPEG frame, 0x02 + 16 kHz mono int16 PCM) so the perception layer can be
exercised end-to-end without the hardware. Frames hold steady for a few seconds
then change (so you can watch the VLM gate fire only on changes); audio comes in
loud "speech" bursts separated by silence (so the audio worker segments
utterances).

Run:
    python3 mac/fake_wearable.py            # serves ws://0.0.0.0:8090/
Then point perception at it:
    VOICELINK_WEARABLE_URL=ws://127.0.0.1:8090/ python3 mac/server.py
"""

from __future__ import annotations

import asyncio
import math

import cv2
import numpy as np
from aiohttp import web

PORT = 8090
SAMPLE_RATE = 16000
MSG_FRAME = 0x01
MSG_AUDIO = 0x02

# Distinct solid-color frames; the change detector should fire once per change.
_COLORS = [(40, 90, 160), (60, 160, 80), (170, 70, 70), (200, 180, 60)]
_FRAME_HOLD_S = 5.0      # how long each scene stays before changing
_FRAME_SEND_S = 1.5      # how often a frame is sent (like the real device cadence)

_BURST_S = 1.0           # length of a "speech" burst
_GAP_S = 0.8             # silence between bursts (> VAD_SILENCE_MS ends an utterance)


def _jpeg_for(color: tuple[int, int, int]) -> bytes:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = color
    cv2.putText(img, "fake", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


async def _send_frames(ws) -> None:
    t = 0.0
    while not ws.closed:
        idx = int(t // _FRAME_HOLD_S) % len(_COLORS)
        await ws.send_bytes(bytes([MSG_FRAME]) + _jpeg_for(_COLORS[idx]))
        await asyncio.sleep(_FRAME_SEND_S)
        t += _FRAME_SEND_S


async def _send_audio(ws) -> None:
    chunk = 1024  # samples per message (~64 ms), like the firmware
    phase = 0.0
    while not ws.closed:
        # one loud burst (a "partner" utterance)
        n_burst = int(SAMPLE_RATE * _BURST_S)
        sent = 0
        while sent < n_burst and not ws.closed:
            k = min(chunk, n_burst - sent)
            samples = np.empty(k, dtype=np.int16)
            for i in range(k):
                samples[i] = int(6000 * math.sin(phase))
                phase += 2 * math.pi * 300 / SAMPLE_RATE
            await ws.send_bytes(bytes([MSG_AUDIO]) + samples.tobytes())
            sent += k
            await asyncio.sleep(k / SAMPLE_RATE)
        # silence (ends the utterance)
        n_gap = int(SAMPLE_RATE * _GAP_S)
        sent = 0
        while sent < n_gap and not ws.closed:
            k = min(chunk, n_gap - sent)
            await ws.send_bytes(bytes([MSG_AUDIO]) + (b"\x00\x00" * k))
            sent += k
            await asyncio.sleep(k / SAMPLE_RATE)


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    print(f"[fake-wearable] client connected: {request.remote}")
    tasks = [asyncio.create_task(_send_frames(ws)), asyncio.create_task(_send_audio(ws))]
    try:
        async for _ in ws:   # drain anything the client sends
            pass
    finally:
        for t in tasks:
            t.cancel()
        print("[fake-wearable] client disconnected")
    return ws


def main() -> None:
    app = web.Application()
    app.add_routes([web.get("/", ws_handler)])
    print(f"[fake-wearable] serving ws://0.0.0.0:{PORT}/  (Ctrl-C to stop)")
    web.run_app(app, host="0.0.0.0", port=PORT, print=None)


if __name__ == "__main__":
    main()
