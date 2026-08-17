#!/usr/bin/env python3
"""Capture audio from the ESP32 I2S recorder sketch into a playable WAV file.

Usage:
    python3 record.py                    # auto-detect the board, write mic.wav
    python3 record.py -o kitchen.wav     # choose the output name
    python3 record.py -p /dev/cu.usbserial-0001 -b 115200

Close the PlatformIO / Arduino serial monitor first — only one program can hold
the port, and a monitor still attached is the usual reason this hangs.
"""

from __future__ import annotations

import argparse
import sys
import time
import wave

import serial
from serial.tools import list_ports

BAUD = 921600  # must match SERIAL_BAUD in recorder.cpp
RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes per sample (16-bit)


def find_port() -> str:
    """Return the single likely USB-serial device, or exit with a clear message."""
    candidates = [
        p.device
        for p in list_ports.comports()
        if "usbserial" in p.device or "usbmodem" in p.device or "wchusb" in p.device
    ]
    if not candidates:
        sys.exit(
            "No USB serial device found. Plug the board in (and use a DATA cable, "
            "not a charge-only one), then re-run. Override with --port."
        )
    if len(candidates) > 1:
        sys.exit(f"Multiple ports found: {candidates}. Pick one with --port.")
    return candidates[0]


def wait_for_ready(port: serial.Serial, timeout_s: float = 15.0) -> None:
    """Block until the sketch announces READY, echoing its banner as it arrives."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        line = port.readline().decode("utf-8", "replace").strip()
        if line:
            print(f"  board: {line}")
        if line == "READY":
            return
    sys.exit("Board never said READY. Is the `recorder` env flashed, and the baud right?")


def read_length_header(port: serial.Serial, timeout_s: float = 30.0) -> int:
    """Consume lines until the LEN:<bytes> header, and return that byte count."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        line = port.readline().decode("utf-8", "replace").strip()
        if line:
            print(f"  board: {line}")
        if line.startswith("LEN:"):
            return int(line[4:])
    sys.exit("Timed out waiting for the audio header.")


def read_exactly(port: serial.Serial, length: int, timeout_s: float = 60.0) -> bytes:
    """Read exactly `length` bytes, tolerating the gaps between serial chunks."""
    chunks: list[bytes] = []
    remaining = length
    deadline = time.monotonic() + timeout_s
    while remaining > 0 and time.monotonic() < deadline:
        chunk = port.read(remaining)
        if chunk:
            chunks.append(chunk)
            remaining -= len(chunk)
    if remaining > 0:
        sys.exit(f"Short read: got {length - remaining} of {length} bytes.")
    return b"".join(chunks)


def write_wav(path: str, data: bytes) -> None:
    with wave.open(path, "wb") as out:
        out.setnchannels(CHANNELS)
        out.setsampwidth(SAMPLE_WIDTH)
        out.setframerate(RATE)
        out.writeframes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--port", help="serial device (default: auto-detect)")
    parser.add_argument("-b", "--baud", type=int, default=BAUD)
    parser.add_argument("-o", "--out", default="mic.wav")
    args = parser.parse_args()

    device = args.port or find_port()
    print(f"Opening {device} at {args.baud}...")

    with serial.Serial(device, args.baud, timeout=1) as port:
        time.sleep(2.0)  # opening the port resets the board; let it boot
        port.reset_input_buffer()
        wait_for_ready(port)

        print("Sending record command — SPEAK NOW.")
        port.write(b"r")

        length = read_length_header(port)
        data = read_exactly(port, length)

    write_wav(args.out, data)
    seconds = length / SAMPLE_WIDTH / RATE
    print(f"Wrote {args.out} — {seconds:.1f} seconds. Play it: afplay {args.out}")


if __name__ == "__main__":
    main()
