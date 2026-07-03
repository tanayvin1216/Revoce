"""The ONE hedge/epistemic-stall phrase cluster (review fix S3/L1).

infer.py's board cap and metrics3's same-function scoring previously kept two
hand-maintained, out-of-sync lists — so the cap missed phrases the metric
penalized ("maybe", "no idea"). Both now import from here; keep additions here
and nowhere else.
"""

from __future__ import annotations

HEDGE_PHRASES: frozenset[str] = frozenset({
    # "i don't know" family
    "i don't know", "i dont know", "i don't know yet", "i dont know yet",
    "i am not sure", "i'm not sure", "im not sure", "not sure",
    "i am not sure yet", "i'm not sure yet", "unsure",
    # broader epistemic stalls (previously only in metrics3)
    "no idea", "i can't remember", "i cant remember", "maybe",
    "i don't care", "i dont care", "i need more time",
})


def normalize(text: str) -> str:
    return text.strip().lower().replace("’", "'").rstrip(".!?,")


def is_hedge(text: str) -> bool:
    return normalize(text) in HEDGE_PHRASES
