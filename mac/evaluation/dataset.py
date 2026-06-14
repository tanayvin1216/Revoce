"""
Eval example sources.

Three sources, one shape (EvalExample):
  - "dataset": willwade/AACConversations test split (GATED — needs HF auth).
  - "fixture": a tiny bundled synthetic set (fixture.json) so the harness runs
    with no download/auth, for plumbing checks and demos.
  - "real":    persisted taps from Task 3 (taps.jsonl) — the clean hook to
    evaluate against REAL consented interaction logs once a pilot produces them.

Every source maps to (scene, transcript, context, ground_truth) — exactly the
inputs rank_vocabulary() consumes — so the same baseline runs over all three.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# The dataset is multilingual; the AAC user is labelled per language ("AAC User",
# "Utente CAA", "AACユーザー", ...). Our system is English-only, so we evaluate on
# English AAC-user turns: language_code == en AND "aac" in the speaker label.
_AAC_MARKER = "aac"


@dataclass
class EvalExample:
    scene: str | None
    transcript: str | None
    context: dict
    ground_truth: str
    meta: dict = field(default_factory=dict)


def _build_context(scene, context_speakers, context_utterances, ground_truth):
    """Construct rank_vocabulary inputs from a dialogue history.

    The AAC user is about to respond, so treat them as addressed; the last
    context turn is the partner's last utterance. No personal profile is
    available for this dataset, so personal_context is empty (honest baseline).
    """
    speakers = context_speakers or []
    utterances = context_utterances or []
    lines = [f"{s}: {u}" for s, u in zip(speakers, utterances)]
    transcript = "\n".join(lines) if lines else None
    last_utt = utterances[-1] if utterances else None
    context = {
        "addressed": True,
        "speaker": "partner",
        "last_utterance": last_utt,
        "recent_transcript": transcript or "",
        "scene_changed": False,
        "personal_context": {},
    }
    return EvalExample(scene=scene or None, transcript=transcript,
                       context=context, ground_truth=ground_truth)


def from_dataset(split: str, limit: int, seed: int) -> list[EvalExample]:
    """Load the GATED willwade/AACConversations split. Requires HF auth
    (HF_TOKEN env var or `huggingface-cli login`) AND acceptance of the dataset
    terms on its Hugging Face page."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("pip install datasets") from e

    try:
        ds = load_dataset("willwade/AACConversations", split=split)
    except Exception as e:
        raise RuntimeError(
            "Could not load willwade/AACConversations (it is GATED).\n"
            "  1) Accept the terms at https://huggingface.co/datasets/willwade/AACConversations\n"
            "  2) Authenticate: `huggingface-cli login`  or  export HF_TOKEN=hf_...\n"
            f"  underlying error: {e}"
        ) from e

    # English AAC-user turns only: their utterance is the ground truth.
    ds = ds.filter(
        lambda r: str(r.get("language_code", "")).lower().startswith("en")
        and _AAC_MARKER in str(r.get("speaker", "")).lower()
    )
    if limit and limit < len(ds):
        ds = ds.shuffle(seed=seed).select(range(limit))

    out = []
    for r in ds:
        gt = (r.get("utterance") or "").strip()
        if not gt:
            continue
        out.append(_build_context(
            r.get("scene"), r.get("context_speakers"), r.get("context_utterances"), gt
        ))
    return out


def from_fixture(path: Path, limit: int) -> list[EvalExample]:
    """Tiny synthetic fixture in the real dataset's shape (no auth needed)."""
    rows = json.loads(Path(path).read_text())
    out = []
    for r in rows:
        gt = (r.get("utterance") or "").strip()
        if not gt:
            continue
        out.append(_build_context(
            r.get("scene"), r.get("context_speakers"), r.get("context_utterances"), gt
        ))
    return out[:limit] if limit else out


def from_real_logs(path: Path, limit: int) -> list[EvalExample]:
    """REAL interaction logs: Task 3 taps (taps.jsonl). Ground truth = the word
    the user actually tapped; inputs = the context snapshot that produced that
    board. This is the path that matters once a consented pilot exists."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        gt = (rec.get("word") or "").strip()
        snap = rec.get("context_snapshot") or {}
        if not gt:
            continue
        ctx = dict(snap.get("context") or {})
        ctx.setdefault("personal_context", {})
        out.append(EvalExample(
            scene=snap.get("scene"),
            transcript=snap.get("transcript"),
            context=ctx,
            ground_truth=gt,
            meta={"source": "real_tap"},
        ))
    return out[:limit] if limit else out
