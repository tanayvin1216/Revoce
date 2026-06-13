#!/usr/bin/env python3
"""VoiceLink capture-device test client.

Connects to the wearable's WebSocket (SoftAP, default ws://192.168.4.1:81/),
shows the latest JPEG frame in a window, and prints a live audio level meter.

Use it to verify the firmware end to end:
  1. Join the device's Wi-Fi network ("VoiceLink-Capture", pass "voicelink").
  2. Run this client.
  3. You should see live frames + a responsive meter. Pressing the device's
     button should cleanly stop the stream (window/meter freeze) and resume it.

Requires:
  pip install opencv-python numpy websocket-client

Usage:
  python test_client.py [ws://HOST:PORT/]
"""

import sys
import threading

import cv2
import numpy as np
import websocket

DEFAULT_URL = "ws://192.168.4.1:81/"

MSG_FRAME = 0x01
MSG_AUDIO = 0x02

METER_WIDTH = 50          # characters in the level bar
PCM_FULL_SCALE = 32768.0  # int16 max magnitude

_latest_frame = None
_frame_lock = threading.Lock()


def _on_message(ws, message):
    # Text frames (e.g. handshake noise) are ignored — we only care about binary.
    if not isinstance(message, (bytes, bytearray)) or len(message) < 1:
        return

    msg_type = message[0]
    payload = memoryview(message)[1:]

    if msg_type == MSG_FRAME:
        arr = np.frombuffer(payload, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            with _frame_lock:
                global _latest_frame
                _latest_frame = img

    elif msg_type == MSG_AUDIO:
        pcm = np.frombuffer(payload, dtype=np.int16)
        if pcm.size:
            _print_meter(pcm)


def _print_meter(pcm):
    rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))
    peak = int(np.max(np.abs(pcm)))
    level = min(int((rms / PCM_FULL_SCALE) * METER_WIDTH * 4), METER_WIDTH)  # x4: speech is well below full scale
    bar = "#" * level + "-" * (METER_WIDTH - level)
    sys.stdout.write(f"\raudio [{bar}] rms={rms:7.1f} peak={peak:6d}")
    sys.stdout.flush()


def _on_open(ws):
    print("[client] connected — waiting for frames + audio...")


def _on_error(ws, error):
    print(f"\n[client] error: {error}")


def _on_close(ws, status, msg):
    print("\n[client] connection closed")


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    print(f"[client] connecting to {url}")

    ws = websocket.WebSocketApp(
        url,
        on_open=_on_open,
        on_message=_on_message,
        on_error=_on_error,
        on_close=_on_close,
    )

    # The WebSocket runs in a background thread; OpenCV's GUI must run on the
    # main thread, so we poll the latest frame and display it here.
    thread = threading.Thread(target=ws.run_forever, daemon=True)
    thread.start()

    window = "VoiceLink capture"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while True:
            with _frame_lock:
                frame = _latest_frame
            if frame is not None:
                cv2.imshow(window, frame)
            if (cv2.waitKey(30) & 0xFF) == ord("q"):
                break
    finally:
        ws.close()
        cv2.destroyAllWindows()
        print("\n[client] bye")


if __name__ == "__main__":
    main()
