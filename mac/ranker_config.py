"""
Ranker backend configuration — one line selects the runtime.

Override any value with environment variables:
  VOICELINK_RANKER         "local-llm" | "heuristic"
  VOICELINK_LLM_MODEL      HuggingFace repo id (must already be cached)
  VOICELINK_LLM_MAX_TOKENS maximum tokens to generate
  VOICELINK_LLM_TEMP       sampling temperature (0.0 = greedy, recommended)
"""

from __future__ import annotations

import os

RANKER: str = os.environ.get("VOICELINK_RANKER", "local-llm")
LLM_MODEL: str = os.environ.get(
    "VOICELINK_LLM_MODEL", "mlx-community/Qwen2.5-3B-Instruct-4bit"
)
LLM_MAX_TOKENS: int = int(os.environ.get("VOICELINK_LLM_MAX_TOKENS", "300"))
LLM_TEMP: float = float(os.environ.get("VOICELINK_LLM_TEMP", "0.0"))
