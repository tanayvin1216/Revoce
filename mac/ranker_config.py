"""
Ranker backend configuration — one line selects the runtime.

Override any value with environment variables:
  VOICELINK_BOARD_STRATEGY "diverse" | "topk"
  VOICELINK_STUDENT_BACKEND "torch" | "openvino"
  VOICELINK_RANKER         "local-llm" | "heuristic"
  VOICELINK_LLM_MODEL      HuggingFace repo id (must already be cached)
  VOICELINK_LLM_MAX_TOKENS maximum tokens to generate
  VOICELINK_LLM_TEMP       sampling temperature (0.0 = greedy, recommended)
"""

from __future__ import annotations

import os

# Board strategy the SERVER runs through the seam:
#   "student" — the distilled MLP + MMR selection (default). ~44ms end to end.
#   "diverse" — over-generate a wide pool with the LLM, then MMR-select.
#               Same selection policy, ~30s warm / ~110s cold on MLX.
#   "topk"    — research baseline only; does NOT pin the safety words.
#
# The default is "student" because the product claim is a board that keeps up
# with a conversation, and 30 seconds does not. A partner asks a question, and
# by the time the LLM path answers, the moment is gone — which is precisely the
# failure mode VoiceLink exists to fix. The measured gap is 43.6ms vs 6.8s.
#
# This is safe to default: rank_vocabulary's student path falls back to the
# diverse LLM path automatically if the phase-4 artifacts are missing, and
# _student_board now says so loudly instead of silently degrading.
BOARD_STRATEGY: str = os.environ.get("VOICELINK_BOARD_STRATEGY", "student")

# Backend the seam loads for strategy="student":
#   "torch"    — student.pt via PyTorch (default)
#   "openvino" — student_int8.xml (fallback student_fp32.xml) via the OpenVINO
#                CPU plugin; same logits, swapped runtime (the Intel path)
STUDENT_BACKEND: str = os.environ.get("VOICELINK_STUDENT_BACKEND", "torch")

# Which student ARCHITECTURE the seam serves for strategy="student":
#   "v2" — the 94-column classifier over a closed, frequency-floored vocabulary
#          (default: it is the model whose gates we can stand behind)
#   "v3" — the open-vocabulary bi-encoder scoring (context, phrase) pairs
#
# v3 is NOT the default and should not be until it clears coverage. It wins the
# diversity gates decisively and reproducibly (distinct-functions and
# same-function-pair both resolved PASS under 5-fold CV), but its
# cross-validated coverage-of-intent margin is -0.0397 against the bar — i.e.
# the best-powered estimate says it is BELOW the coverage bar, not above it.
# Shipping it by default would trade a measured regression in "did the board
# offer something the moment actually called for" against a measured gain in
# board variety. That is the user's core need, so v2 keeps the default.
# See evaluation/phase4/V3_BIENCODER_REPORT.md.
STUDENT_VERSION: str = os.environ.get("VOICELINK_STUDENT_VERSION", "v2")

RANKER: str = os.environ.get("VOICELINK_RANKER", "local-llm")
LLM_MODEL: str = os.environ.get(
    "VOICELINK_LLM_MODEL", "mlx-community/Qwen2.5-3B-Instruct-4bit"
)
LLM_MAX_TOKENS: int = int(os.environ.get("VOICELINK_LLM_MAX_TOKENS", "300"))
LLM_TEMP: float = float(os.environ.get("VOICELINK_LLM_TEMP", "0.0"))
