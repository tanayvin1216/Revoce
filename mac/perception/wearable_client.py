"""
Wearable WebSocket client.

The wearable is the WS SERVER (SoftAP); the Mac connects to it as a CLIENT.
This coroutine runs on the server's asyncio event loop and does ONLY the fast
path: demux binary messages (0x01 frame / 0x02 audio) into the audio queue and
the latest-frame holder. It never runs inference, so the network read is never
blocked by Whisper or the VLM. Reconnects with backoff (the wearable's button
can pause/resume the stream at any time).
"""

from __future__ import annotations

import asyncio

import aiohttp

MSG_FRAME = 0x01
MSG_AUDIO = 0x02


async def wearable_client(url, audio_q, frame_holder, stop_event) -> None:
    backoff = 0.5
    while not stop_event.is_set():
        try:
            async with aiohttp.ClientSession() as session:
                # max_msg_size=0 -> no cap (VGA JPEG frames can be tens of KB)
                async with session.ws_connect(url, max_msg_size=0, heartbeat=20) as ws:
                    print(f"[wearable] connected: {url}")
                    backoff = 0.5
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.BINARY and len(msg.data) >= 1:
                            tag = msg.data[0]
                            payload = msg.data[1:]
                            if tag == MSG_AUDIO:
                                _enqueue_audio(audio_q, payload)
                            elif tag == MSG_FRAME:
                                frame_holder.set(payload)
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[wearable] disconnected ({e}); retrying in {backoff:.1f}s")
        if stop_event.is_set():
            break
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 5.0)


def _enqueue_audio(audio_q, payload: bytes) -> None:
    """Non-blocking enqueue; if the worker has fallen behind, drop the oldest
    chunk to stay near real-time rather than build unbounded latency."""
    try:
        audio_q.put_nowait(payload)
    except Exception:
        try:
            audio_q.get_nowait()
            audio_q.put_nowait(payload)
        except Exception:
            pass
