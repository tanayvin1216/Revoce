"""
Phase-1 data prep: export a reproducible sample of REAL AACConversations test
turns (English, AAC-user) for the diverse-coverage eval.

Outputs (written next to this file):
  examples.json       flat sample for the headline coverage eval (single ground
                      truth per turn — directly comparable to the 33% baseline).
  conversations.json  multi-turn sequences for the production simulation.

Run:  HF_TOKEN=... python3 mac/evaluation/phase1/export_real.py [--n 120]
Requires gated access to willwade/AACConversations.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
_AAC_MARKER = "aac"


def _as_list(field):
    """context_speakers/context_utterances arrive as STRING reprs of lists,
    e.g. "['Therapist']". Parse to a real list; tolerate already-list / junk."""
    if isinstance(field, list):
        return field
    if not field:
        return []
    try:
        val = ast.literal_eval(field)
        return val if isinstance(val, list) else [str(val)]
    except (ValueError, SyntaxError):
        return [str(field)]


def _transcript(speakers, utterances):
    """Build a readable transcript by zipping the PARSED lists (not characters).

    The previous version zipped the raw strings, which iterates characters and
    produces a transposed, truncated scramble. Parsing first is the fix.
    """
    sp, ut = _as_list(speakers), _as_list(utterances)
    lines = [f"{s}: {u}" for s, u in zip(sp, ut)]
    return "\n".join(lines) if lines else None


def main(n: int, seed: int) -> None:
    from datasets import load_dataset

    ds = load_dataset("willwade/AACConversations", split="test")
    ds = ds.filter(
        lambda r: str(r.get("language_code", "")).lower().startswith("en")
        and _AAC_MARKER in str(r.get("speaker", "")).lower()
    )
    print(f"english aac-user turns in test split: {len(ds)}")

    # --- Flat sample (deterministic) for the headline eval ---
    sample = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    examples = []
    for r in sample:
        gt = (r.get("utterance") or "").strip()
        if not gt:
            continue
        examples.append({
            "conversation_id": r.get("conversation_id"),
            "turn_number": r.get("turn_number"),
            "scene": r.get("scene"),
            "transcript": _transcript(r.get("context_speakers"), r.get("context_utterances")),
            "last_utterance": (_as_list(r.get("context_utterances")) or [None])[-1],
            "ground_truth": gt,
        })
    (HERE / "examples.json").write_text(json.dumps(examples, indent=2))
    print(f"wrote examples.json: {len(examples)} turns")

    # --- Multi-turn conversations for the production sim ---
    by_conv = defaultdict(list)
    for r in ds:
        gt = (r.get("utterance") or "").strip()
        if not gt:
            continue
        by_conv[r.get("conversation_id")].append({
            "turn_number": r.get("turn_number"),
            "scene": r.get("scene"),
            "transcript": _transcript(r.get("context_speakers"), r.get("context_utterances")),
            "last_utterance": (_as_list(r.get("context_utterances")) or [None])[-1],
            "ground_truth": gt,
        })
    multi = []
    for cid, turns in by_conv.items():
        if len(turns) >= 3:
            turns_sorted = sorted(turns, key=lambda t: (t["turn_number"] or 0))
            multi.append({"conversation_id": cid, "turns": turns_sorted})
    multi.sort(key=lambda c: -len(c["turns"]))
    multi = multi[:6]
    (HERE / "conversations.json").write_text(json.dumps(multi, indent=2))
    print(f"wrote conversations.json: {len(multi)} conversations "
          f"({sum(len(c['turns']) for c in multi)} aac turns)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    main(**vars(p.parse_args()))
