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
FUNCTION_PROTOTYPES_V3 = {
    "question-back": "What about you? Why is that? Can you tell me more? How come?",
    "specific-answer": "It is the blue one. In the back. At three o'clock. The water.",
    "I-dont-know": "I don't know. I'm not sure. No idea. I can't remember.",
    "request-help": "I need help. Can you help me. Please do this for me. I want that.",
    "comment": "That is nice. I like it. Good point. That looks fun. Interesting.",
    "refuse/redirect": "No. I don't want that. Stop. Let's talk about something else.",
    "emotion": "I feel happy. I am tired. I am scared. This makes me sad.",
    "request-clarification": "What do you mean? Can you repeat that? I didn't understand.",
    "social": "Hello. Thank you. Good morning. Nice to see you. Goodbye.",
}

_proto_vecs = None
_proto_names = list(FUNCTION_PROTOTYPES_V3.keys())


def _prototypes() -> np.ndarray:
    """Lazy, cached prototype matrix (9, 384). Mirrors coverage_metrics._prototypes."""
    global _proto_vecs
    if _proto_vecs is None:
        from evaluation.embed import embed  # lazy so tests can monkeypatch it
        _proto_vecs = embed(list(FUNCTION_PROTOTYPES_V3.values()))
    return _proto_vecs


def classify_functions(board_vecs: np.ndarray) -> tuple[int, list[str]]:
    """How many of the spec's 9 functions a board spans, by embedding-nearest
    prototype. Returns (distinct_count, per-item labels). Same shape/contract as
    coverage_metrics.function_coverage, but against the v3 prototypes."""
    if board_vecs.shape[0] == 0:
        return 0, []
    sims = board_vecs @ _prototypes().T          # (B, 9), inputs normalized -> cosine
    idx = sims.argmax(axis=1)
    labels = [_proto_names[i] for i in idx]
    return len(set(labels)), labels
