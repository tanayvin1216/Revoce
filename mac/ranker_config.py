"""
Ranker backend configuration — one line selects the runtime.

Override any value with environment variables:
  VOICELINK_BOARD_STRATEGY "diverse" | "topk"
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

RANKER: str = os.environ.get("VOICELINK_RANKER", "local-llm")
LLM_MODEL: str = os.environ.get(
    "VOICELINK_LLM_MODEL", "mlx-community/Qwen2.5-3B-Instruct-4bit"
)
LLM_MAX_TOKENS: int = int(os.environ.get("VOICELINK_LLM_MAX_TOKENS", "300"))
LLM_TEMP: float = float(os.environ.get("VOICELINK_LLM_TEMP", "0.0"))
