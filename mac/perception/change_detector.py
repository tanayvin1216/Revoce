"""
Cheap frame change detector — the gate in front of the expensive VLM.

This is the thermal/latency discipline: the VLM must NOT run on every frame.
We compare a downscaled grayscale frame against the last keyframe and only
report "changed" when the mean absolute difference clears a threshold. Pure and
deterministic given its inputs (easy to unit-test).
"""

from __future__ import annotations

import numpy as np


class FrameChangeDetector:
    def __init__(self, threshold: float) -> None:
        self._threshold = threshold
        self._reference: np.ndarray | None = None

    def is_changed(self, gray_small: np.ndarray) -> tuple[bool, float]:
        """Return (changed, score). The first frame is always a change. When a
        change is reported, the reference re-baselines so slow drift doesn't
        keep re-firing."""
        ref = self._reference
        if ref is None or ref.shape != gray_small.shape:
            self._reference = gray_small
            return True, float("inf")
        score = float(np.mean(np.abs(gray_small.astype(np.int16) - ref.astype(np.int16))))
        changed = score >= self._threshold
        if changed:
            self._reference = gray_small
        return changed, score
