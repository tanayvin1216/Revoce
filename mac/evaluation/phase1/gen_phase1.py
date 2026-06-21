"""
Phase-1 GENERATION (the expensive half) — run once, cache everything.

For each of N real AAC turns produces three cached artifacts:
  cache/baseline.jsonl  the CURRENT production board (old top-k prompt, greedy)
  cache/pools.jsonl     the diverse candidate POOL (new prompt, temp 0.7)
  cache/oracle.jsonl    an INDEPENDENT plausible-set (oracle role, temp 0.8)

The model is loaded ONCE and shared across all three roles (different sampler /
prompt per role). Resumable: existing lines are skipped on re-run. Scoring,
lambda-sweep and plots all run downstream from these caches for free.

    python3 evaluation/phase1/gen_phase1.py --n 120
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAC = HERE.parent.parent
sys.path.insert(0, str(MAC))

from evaluation.phase1.prep import transcript_text, extract_topics  # noqa: E402
from evaluation.phase1 import oracle as oracle_mod                   # noqa: E402

CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
MODEL_ID = "mlx-community/Qwen2.5-3B-Instruct-4bit"


class MlxRanker:
    """One loaded model; per-call temperature. Exposes .rank(system,user,...)."""

    def __init__(self) -> None:
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
        self._model, self._tok = load(MODEL_ID)
        self._generate = generate
        self._make_sampler = make_sampler

    def raw(self, system: str, user: str, temp: float, max_tokens: int) -> str:
        toks = self._tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=True, add_generation_prompt=True,
        )
        return self._generate(self._model, self._tok, toks,
                              max_tokens=max_tokens, sampler=self._make_sampler(temp=temp))

    def rank(self, system: str, user: str, max_tokens: int | None = None):
        from ranker import _parse_json_array
        temp = getattr(self, "_temp", 0.0)
        return _parse_json_array(self.raw(system, user, temp, max_tokens or 300))


def _ctx(ex: dict) -> dict:
    return {"addressed": True, "last_utterance": ex.get("last_utterance"),
            "personal_context": {}}


def _done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


def main(n: int) -> None:
    from rank_vocabulary import rank_vocabulary, generate_pool
    from evaluation.embed import embed

    examples = json.loads((HERE / "examples.json").read_text())[:n]
    ranker = MlxRanker()
    print(f"model loaded; {len(examples)} turns", flush=True)

    paths = {k: CACHE / f"{k}.jsonl" for k in ("baseline", "pools", "oracle")}
    done = {k: _done_ids(p) for k, p in paths.items()}
    files = {k: open(p, "a") for k, p in paths.items()}

    for i, ex in enumerate(examples):
        eid = f"{ex['conversation_id']}-{ex['turn_number']}-{i}"
        scene = ex["scene"]
        tr = transcript_text(ex["transcript"])
        topics = extract_topics(scene, ex["transcript"])
        ctx = _ctx(ex)
        gt = ex["ground_truth"]
        t0 = time.time()

        # 1) baseline board (current production: old prompt, greedy)
        if eid not in done["baseline"]:
            ranker._temp = 0.0
            try:
                board = rank_vocabulary(scene, tr, ctx, ranker=ranker, strategy="topk")
                words = [b["word"] for b in board][:7]
            except Exception as e:
                words = []
                print(f"  baseline err {eid}: {e}", flush=True)
            files["baseline"].write(json.dumps({"id": eid, "gt": gt, "board": words}) + "\n")
            files["baseline"].flush()

        # 2) diverse pool (new prompt, temp 0.7 for spread)
        if eid not in done["pools"]:
            ranker._temp = 0.7
            t1 = time.time()
            try:
                pool = generate_pool(scene, tr, ctx, ranker=ranker, topics=topics)
            except Exception as e:
                pool = []
                print(f"  pool err {eid}: {e}", flush=True)
            files["pools"].write(json.dumps({
                "id": eid, "gt": gt, "scene": scene, "transcript": tr,
                "last_utterance": ex.get("last_utterance"), "topics": topics,
                "pool": pool, "pool_latency_s": round(time.time() - t1, 2),
            }) + "\n")
            files["pools"].flush()

        # 3) oracle plausible-set (independent role, temp 0.8)
        if eid not in done["oracle"]:
            ranker._temp = 0.8
            pset = oracle_mod.generate_oracle_set(
                scene, tr, ex.get("last_utterance"), ranker, embed)
            files["oracle"].write(json.dumps({"id": eid, "gt": gt, "plausible": pset}) + "\n")
            files["oracle"].flush()

        print(f"[{i+1}/{len(examples)}] {eid} {time.time()-t0:.1f}s gt={gt!r}", flush=True)

    for f in files.values():
        f.close()
    print("GENERATION COMPLETE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=120)
    main(**vars(p.parse_args()))
