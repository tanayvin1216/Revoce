"""Phase-3 LLM JUDGE — a PROXY signal, never the gate.

The four metrics in metrics3.py are the gate. This judge is a secondary, human-
shaped read: it scores boards 1-5 on four axes and exists mainly to (a) catch
failure modes the embedding metrics miss and (b) be calibrated against humans
later. Every artifact it emits is stamped PROXY so no one mistakes it for ground
truth.

Backends are pluggable (JudgeBackend ABC). The COMMITTED, reproducible default is
the local Qwen model run as a 3x temperature-varied ensemble (QwenJudge); a Claude
or human backend can drop in behind the same interface. The aggregate flags an
ensemble that is too lenient (mean > 4.5) or suspiciously uniform (near-zero
variance) — the classic "LLM judge says 9/10 to everything" failure.

    python3 evaluation/phase3/judge.py --judge-sample 20   # Mac / Qwen only

`compare_to_human(judge_jsonl, human_jsonl)` is the calibration hook: stubbed now,
ready to wire to real human ratings without touching the harness.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAC = HERE.parent.parent
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))

PROXY_LABEL = "PROXY — LLM judge, NOT ground truth; gate on metrics3.py"

JUDGE_PATH = HERE / "judge_results.jsonl"
BOARDS_PATH = HERE / "boards.jsonl"

AXES = ("coverage_of_intent", "diversity", "relevance", "authorship")

# Tightened rubric: explicit LOW anchors so the model has permission to score 1-2.
_JUDGE_SYSTEM = """\
You rate an AAC vocabulary board for ONE conversational moment. The board should
COVER the different things the user might want to say, not guess one. Score each
axis 1-5 and give a ONE-LINE reason. Be harsh; most boards are not a 5.

Axes (with low anchors):
- coverage_of_intent: do the buttons cover the plausible replies?
    1 = none of the likely replies are present; 5 = every plausible direction is.
- diversity: are the buttons DIFFERENT moves?
    1 = mostly paraphrases of one reply; 5 = all distinct moves.
- relevance: do the buttons fit THIS moment?
    1 = generic/off-topic; 5 = sharply on-topic.
- authorship: are these things the USER could say (first person, speakable)?
    1 = descriptions of the scene / words for the other person; 5 = all user speech.

Output ONLY a JSON object:
{"coverage_of_intent": {"score": n, "reason": "..."},
 "diversity": {"score": n, "reason": "..."},
 "relevance": {"score": n, "reason": "..."},
 "authorship": {"score": n, "reason": "..."}}"""

_JUDGE_USER = """\
Scene: {scene}
Said to the user: {transcript}
Board buttons: {board}

Rate the board now."""


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class JudgeBackend:
    """A judge backend turns (scene, transcript, board) into a per-axis score dict."""

    name = "abstract"

    def score(self, scene: str, transcript: str, board: list[str]) -> dict:
        raise NotImplementedError


def _parse_scores(text: str) -> dict:
    """Pull the {axis: {score, reason}} object out of raw model text."""
    t = (text or "").strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        obj = json.loads(t[start:end + 1])
    except json.JSONDecodeError:
        return {}
    out: dict = {}
    for axis in AXES:
        node = obj.get(axis) or {}
        try:
            score = float(node.get("score"))
        except (TypeError, ValueError):
            continue
        out[axis] = {"score": max(1.0, min(5.0, score)),
                     "reason": str(node.get("reason") or "").strip()}
    return out


class QwenJudge(JudgeBackend):
    """Committed default: local Qwen, 3x temperature-varied ensemble (averaged)."""

    name = "qwen-ensemble-3x"

    def __init__(self, temps=(0.2, 0.5, 0.8)) -> None:
        from evaluation.phase1.gen_phase1 import MlxRanker  # lazy: imports mlx
        self._ranker = MlxRanker()
        self._temps = temps

    def score(self, scene: str, transcript: str, board: list[str]) -> dict:
        user = _JUDGE_USER.format(scene=scene or "not available",
                                  transcript=transcript or "none",
                                  board=", ".join(board) or "none")
        runs = []
        for temp in self._temps:
            try:
                raw = self._ranker.raw(_JUDGE_SYSTEM, user, temp, max_tokens=400)
                parsed = _parse_scores(raw)
            except Exception:  # noqa: BLE001
                parsed = {}
            if parsed:
                runs.append(parsed)
        return _average_runs(runs)


def _average_runs(runs: list[dict]) -> dict:
    """Average per-axis scores across ensemble runs; keep one sample reason."""
    out: dict = {}
    for axis in AXES:
        scores = [r[axis]["score"] for r in runs if axis in r]
        if not scores:
            continue
        reason = next((r[axis]["reason"] for r in runs if axis in r and r[axis]["reason"]), "")
        out[axis] = {"score": sum(scores) / len(scores), "reason": reason}
    return out


# ---------------------------------------------------------------------------
# Aggregation + leniency check
# ---------------------------------------------------------------------------
def aggregate(judgements: list[dict]) -> dict:
    """Mean per axis + an overall mean, with a too-lenient flag. PROXY-labeled."""
    import numpy as np

    per_axis: dict[str, float] = {}
    all_scores: list[float] = []
    for axis in AXES:
        scores = [j[axis]["score"] for j in judgements if axis in j]
        if scores:
            per_axis[axis] = float(np.mean(scores))
            all_scores.append(per_axis[axis])
    overall = float(np.mean(all_scores)) if all_scores else 0.0
    variance = float(np.var([s for j in judgements for a in AXES
                             if a in j for s in [j[a]["score"]]])) if judgements else 0.0
    too_lenient = overall > 4.5 or variance < 0.05
    return {
        "label": PROXY_LABEL,
        "n": len(judgements),
        "per_axis": per_axis,
        "overall": overall,
        "variance": variance,
        "too_lenient": bool(too_lenient),
    }


def run_judge(board_rows: list[dict], backend: JudgeBackend, sample: int) -> dict:
    """Judge up to `sample` boards (across strategies), write JSONL, return aggregate."""
    rows = board_rows[: sample * len(("diverse", "topk"))] if sample else board_rows
    judgements: list[dict] = []
    with open(JUDGE_PATH, "w") as f:
        for row in rows:
            j = backend.score(row.get("scene", ""), row.get("transcript", ""),
                              row.get("board", []))
            if not j:
                continue
            rec = {"label": PROXY_LABEL, "id": row.get("id"),
                   "strategy": row.get("strategy"), "backend": backend.name, "axes": j}
            f.write(json.dumps(rec) + "\n")
            judgements.append(j)
    agg = aggregate(judgements)
    print(f"[{PROXY_LABEL}]")
    print(json.dumps(agg, indent=2))
    if agg["too_lenient"]:
        print("WARNING: judge looks too lenient (mean>4.5 or near-zero variance).")
    return agg


# ---------------------------------------------------------------------------
# Human-calibration hook (stub) — wired later without touching the harness.
# ---------------------------------------------------------------------------
def compare_to_human(judge_jsonl: str | Path, human_jsonl: str | Path) -> dict:
    """Compare LLM-judge scores against human ratings keyed by (id, strategy).

    STUB: returns a structurally-correct empty report until real human ratings
    exist. When wired, this is where Spearman/agreement vs humans lands — proving
    (or disproving) that the PROXY tracks human judgement."""
    jp, hp = Path(judge_jsonl), Path(human_jsonl)
    if not jp.exists() or not hp.exists():
        return {"label": PROXY_LABEL, "status": "no-human-data",
                "judge_exists": jp.exists(), "human_exists": hp.exists(),
                "per_axis_correlation": {}}
    return {"label": PROXY_LABEL, "status": "not-implemented",
            "per_axis_correlation": {}}


def main() -> None:
    p = argparse.ArgumentParser(description="Phase-3 LLM judge (PROXY)")
    p.add_argument("--judge-sample", type=int, default=20,
                   help="boards per strategy to judge")
    args = p.parse_args()
    rows = [json.loads(line) for line in BOARDS_PATH.read_text().splitlines() if line.strip()]
    run_judge(rows, QwenJudge(), args.judge_sample)


if __name__ == "__main__":
    main()
