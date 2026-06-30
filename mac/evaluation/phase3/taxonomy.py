"""Communication-function taxonomy for the Phase-3 reframe.

The Phase-3 spec names NINE communicative functions an AAC board should be able
to launch. They are NOT the same nine as the Phase-1 set in
`coverage_metrics.FUNCTION_PROTOTYPES` (request / reject / comment / question /
share-information / express-feeling / social / direct-attention / clarify). The
Phase-1 set is left intact for the demoted exact-match/recall path; this module
defines the spec's nine as fresh embedding prototypes.

v3 -> phase-1 mapping (for reference; the two are scored independently):
    question-back          ~ question
    specific-answer        ~ share-information   (a concrete on-topic answer)
    I-dont-know            ~ clarify             (epistemic "no idea")
    request-help           ~ request
    comment                ~ comment
    refuse/redirect        ~ reject
    emotion                ~ express-feeling
    request-clarification  ~ clarify             ("say that again")
    social                 ~ social

A board is classified the SAME way both arms are measured: each item is assigned
to its embedding-nearest prototype (argmax cosine), so "distinct functions" is a
fair, generation-tag-free count of how many different moves the board offers.
"""

from __future__ import annotations

import numpy as np

# The spec's exact nine functions, each anchored by a few short first-person
# exemplars so the MiniLM prototype sits in the right region of the space.
# MULTI-EXEMPLAR prototypes. The first cut used ONE concatenated sentence per
# function and classified by nearest centroid. That centroid is a blunt instrument:
# the "I don't know" anchor became an attractor for any short token, so "yes" AND a
# real answer like "carbon dioxide" both snapped onto it, and "wait" onto "refuse".
# The fix is the standard one — several short exemplars per function, classify a
# board item by its single NEAREST exemplar (max over exemplars), not by a smeared
# centroid. This is a correctness fix to the measuring instrument, applied
# identically to BOTH arms; it is not per-arm tuning. The spec's nine have no
# affirmation or direct-attention bucket, so "yes"/"okay" and "wait"/"hold on" are
# folded into "comment" (a reactive acknowledgement), where they honestly belong.
FUNCTION_EXEMPLARS_V3 = {
    "question-back": ["what about you?", "why is that?", "can you tell me more?", "how come?"],
    "specific-answer": ["it is the blue one", "in the back", "at three o'clock",
                        "carbon dioxide", "my head hurts", "a window seat", "home"],
    "I-dont-know": ["I don't know", "I'm not sure", "no idea", "I can't remember", "maybe"],
    "request-help": ["I need help", "can you help me", "please do this for me"],
    "comment": ["yes", "okay", "sure", "I agree", "that's right", "sounds good",
                "wait", "hold on a moment", "that is nice", "I like it"],
    "refuse/redirect": ["no", "I don't want that", "stop", "not that one", "no thank you"],
    "emotion": ["I feel happy", "I am tired", "I am scared", "this makes me sad", "I'm nervous"],
    "request-clarification": ["what do you mean?", "can you repeat that?", "I didn't understand"],
    "social": ["hello", "thank you", "good morning", "nice to see you", "goodbye"],
}

_proto_vecs = None        # cached (M, 384) exemplar matrix; reset by the test fixture
_proto_funcs: list[str] = []   # length-M function label per exemplar row
_proto_names = list(FUNCTION_EXEMPLARS_V3.keys())


def _exemplars() -> tuple[np.ndarray, list[str]]:
    """Lazy, cached (exemplar_vecs (M,384), func_per_row). Rebuilt when the global
    `_proto_vecs` cache is cleared (the test fixture sets it to None)."""
    global _proto_vecs, _proto_funcs
    if _proto_vecs is None:
        from evaluation.embed import embed  # lazy so tests can monkeypatch it
        flat: list[str] = []
        funcs: list[str] = []
        for name, exs in FUNCTION_EXEMPLARS_V3.items():
            flat.extend(exs)
            funcs.extend([name] * len(exs))
        _proto_vecs = embed(flat)
        _proto_funcs = funcs
    return _proto_vecs, _proto_funcs


def _prototypes() -> np.ndarray:
    """The exemplar matrix (M, 384). Kept for callers/tests that want raw vectors."""
    return _exemplars()[0]


def classify_functions(board_vecs: np.ndarray) -> tuple[int, list[str]]:
    """How many of the spec's 9 functions a board spans, by NEAREST EXEMPLAR.
    Returns (distinct_count, per-item labels). Same contract as
    coverage_metrics.function_coverage, but max-pooled over multi-exemplar v3 protos."""
    if board_vecs.shape[0] == 0:
        return 0, []
    vecs, funcs = _exemplars()
    sims = board_vecs @ vecs.T                   # (B, M), inputs normalized -> cosine
    idx = sims.argmax(axis=1)
    labels = [funcs[i] for i in idx]
    return len(set(labels)), labels
