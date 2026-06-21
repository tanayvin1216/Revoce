"""
Plausible-set ORACLE — the honest bridge for single-ground-truth data.

We only have ONE logged utterance per turn, but the objective is COVERAGE of the
many ways the user could plausibly reply. The oracle enumerates that set with a
STRUCTURALLY INDEPENDENT model role:
  - different system prompt (a conversation partner listing what COULD be said)
  - higher temperature (0.8) for spread vs the greedy generator
  - it NEVER sees the board, the generator prompt, or rank_vocabulary

Frozen to cache/oracle-<seed>.jsonl once, before any board is scored, so it cannot
be tuned to a result. A sanity gate checks the real GT is itself inside the set.

DELIBERATELY NOT imported by rank_vocabulary — independence is structural.
"""

from __future__ import annotations

import json

import numpy as np

_ORACLE_SYSTEM = """\
You are mapping out a conversation. Given the situation and the last thing said, list the
genuinely DIFFERENT ways a person could reply next — agree, disagree, ask something,
change the subject, share a fact, express a feeling, deflect, joke. Aim for breadth, not
the single most likely reply.

Output ONLY a JSON array of short first-person strings (1-6 words each), 10 to 12 items,
all clearly different from each other. No prose, no code fences.\
"""

_ORACLE_USER = """\
Scene: {scene}
Conversation just now: {transcript}
Most recent thing said: {last_utterance}

List 10-12 distinct possible replies as a JSON array of strings.\
"""


def _parse_strings(text: str) -> list[str]:
    import json
    import re
    t = text.strip()
    if "```" in t:
        t = t[t.find("```") + 3:]
        t = t[t.find("\n") + 1:] if "\n" in t else t
        t = t[: t.find("```")] if "```" in t else t
    out: list[str] = []
    start, end = t.find("["), t.rfind("]")
    if start != -1 and end > start:
        try:
            arr = json.loads(t[start:end + 1])
            out = [str(x).strip() for x in arr if str(x).strip()]
        except json.JSONDecodeError:
            out = []
    if not out:  # salvage quoted strings on truncation
        out = [m.strip() for m in re.findall(r'"([^"\n]{2,60})"', t)]
    return out


def dedup_by_embedding(phrases: list[str], embed_fn, thr: float = 0.8) -> list[str]:
    """Greedy drop of near-duplicates (cos >= thr) so the set is genuinely distinct."""
    kept: list[str] = []
    if not phrases:
        return kept
    vecs = embed_fn(phrases)
    kept_idx: list[int] = []
    for i in range(len(phrases)):
        if kept_idx and (vecs[i] @ vecs[kept_idx].T).max() >= thr:
            continue
        kept_idx.append(i)
        kept.append(phrases[i])
    return kept


def generate_oracle_set(scene, transcript, last_utterance, oracle_ranker, embed_fn) -> list[str]:
    """One independent, high-temperature enumeration, deduped to a distinct set."""
    user = _ORACLE_USER.format(
        scene=scene or "not available",
        transcript=transcript or "none",
        last_utterance=last_utterance or "none",
    )
    try:
        # The oracle emits an array of STRINGS, so parse the raw text directly
        # (the dict-oriented ranker parser would drop them).
        if hasattr(oracle_ranker, "raw"):
            text = oracle_ranker.raw(_ORACLE_SYSTEM, user, 0.8, 400)
        else:
            text = oracle_ranker.rank(_ORACLE_SYSTEM, user, max_tokens=400)
            text = text if isinstance(text, str) else json.dumps(text)
        phrases = _parse_strings(text)
    except Exception:
        phrases = []
    phrases = [p for p in phrases if p]
    return dedup_by_embedding(phrases, embed_fn, thr=0.8)


def sanity_gate(gt_vecs: np.ndarray, set_vecs_list, thr: float = 0.55) -> float:
    """Fraction of turns where the GT is itself covered by its plausible set.
    Must be high (>=0.9) for SetCov to be trustworthy."""
    hits = 0
    for gtv, sv in zip(gt_vecs, set_vecs_list):
        if sv.shape[0] and float((sv @ gtv).max()) >= thr:
            hits += 1
    return hits / len(gt_vecs) if len(gt_vecs) else 0.0
