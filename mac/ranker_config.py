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
#   "diverse" — over-generate a wide pool, then MMR-select for coverage (default)
#   "topk"    — single-shot "most likely" ranking (the Task-5 fallback)
# This flips only the server default. rank_vocabulary()'s own signature default
# stays "topk", so the eval baseline (evaluation/harness.py) is untouched.
BOARD_STRATEGY: str = os.environ.get("VOICELINK_BOARD_STRATEGY", "diverse")

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
