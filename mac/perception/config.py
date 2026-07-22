"""
Perception configuration. Backend/runtime swaps are ONE-LINE changes here
(or via the env overrides below, handy for dev without models/hardware).
"""

from __future__ import annotations

import os

# --- Wearable feed (the ESP32-S3 is the WS SERVER; the Mac connects as client) ---
# Override for dev: VOICELINK_WEARABLE_URL=ws://127.0.0.1:8090/  (fake_wearable.py)
WEARABLE_WS_URL = os.environ.get("VOICELINK_WEARABLE_URL", "ws://192.168.4.1:81/")

PERCEPTION_ENABLED = os.environ.get("VOICELINK_PERCEPTION", "1") != "0"

# --- Audio / Whisper ---------------------------------------------------------
# "faster-whisper" | "openvino" | "mock"  <- one-line swap
#
# "openvino" serves the INT8 IR in intel/artifacts/whisper-base.en-ov-int8:
# measured 0.205s vs faster-whisper's 0.324s on a 6.25s clip, and 81 MB vs
# 141 MB on disk (BENCHMARKS.md). It is not the default on the Mac because its
# working runtime (optimum-intel) pins transformers<5.1 and mlx-vlm needs >=5.5
# — a dev-box conflict that does not exist on the Intel deployment target.
TRANSCRIBER = os.environ.get("VOICELINK_TRANSCRIBER", "faster-whisper")
WHISPER_MODEL = os.environ.get("VOICELINK_WHISPER_MODEL", "base.en")  # -> "small.en", etc.
WHISPER_DEVICE = "cpu"            # faster-whisper on Apple Silicon runs on CPU
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_OV_DIR = os.environ.get("VOICELINK_WHISPER_OV_DIR")  # None -> intel/artifacts/
WHISPER_OV_DEVICE = os.environ.get("VOICELINK_WHISPER_OV_DEVICE", "CPU")  # CPU|GPU|NPU
SAMPLE_RATE = 16000               # must match the wearable's mic stream

# Energy VAD — detects utterance boundaries so we know when one is COMPLETE.
VAD_FRAME_MS = 30
VAD_RMS_THRESHOLD = 350           # int16 RMS above which a frame counts as speech
VAD_SILENCE_MS = 600              # trailing silence that ends an utterance
VAD_MIN_UTTERANCE_MS = 300        # ignore blips shorter than this
PARTNER_RMS_THRESHOLD = 800       # near-field loudness => "partner" vs "ambient" (heuristic)

# --- Vision / VLM ------------------------------------------------------------
# "mlx-vlm" | "mock" | (later) "openvino"  <- one-line swap
SCENE_DESCRIBER = os.environ.get("VOICELINK_SCENE_DESCRIBER", "mlx-vlm")
VLM_MODEL = os.environ.get("VOICELINK_VLM_MODEL", "mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
VLM_PROMPT = (
    "You help a nonverbal person communicate. In ONE short sentence, describe "
    "this scene focusing on what they might want to talk about or ask for. "
    "No preamble, no list."
)
VLM_MAX_TOKENS = 64
VLM_MAX_IMAGE_DIM = 768           # downscale frames past this before the VLM (token/latency cap)

# Cheap change detector — gates the expensive VLM (thermal/latency discipline).
CHANGE_GRAY_SIZE = 64             # downscale to NxN grayscale for the diff
CHANGE_THRESHOLD = 12.0           # mean abs gray diff (0..255) to call it "changed"
VLM_MIN_INTERVAL_S = 2.0          # never fire the VLM more often than this
VISION_POLL_S = 0.5              # how often to check the latest frame
