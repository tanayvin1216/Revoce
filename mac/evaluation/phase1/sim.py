"""
Phase-1 PRODUCTION SIMULATION — replay 6 real multi-turn conversations.

Walks each conversation turn by turn as the device would: at every turn it
generates the diverse pool (the one ~3-4s LLM call), MMR-selects the 7-board
(sub-ms), and records what the user would have experienced —

  covered      : was the user's real reply launchable from the 7 buttons?
  soft         : nearest-button similarity (how close even on a miss)
  ild          : board diversity that turn
  pool_lat_s   : model cost (the only slow part)
  select_lat_s : MMR cost (must be sub-frame -> proves non-blocking)
  source       : warm | stub  (degrade path when generation collapses)

Baseline board is generated alongside for the coverage-vs-mode-seeking contrast.
Writes sim_results.json for the timeline + latency graphs.

    python3 evaluation/phase1/sim.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAC = HERE.parent.parent
sys.path.insert(0, str(MAC))

from evaluation.phase1.prep import transcript_text, extract_topics  # noqa: E402
from evaluation.phase1.gen_phase1 import MlxRanker                   # noqa: E402
from evaluation.embed import embed                                   # noqa: E402
from evaluation import coverage_metrics as cm                        # noqa: E402

PROD_LAMBDA = 0.6
BOARD_K = 7


def main() -> None:
    from rank_vocabulary import generate_pool, select_board, rank_vocabulary, _STUB_WORDS

    convs = json.loads((HERE / "conversations.json").read_text())
    ranker = MlxRanker()
    print(f"model loaded; {len(convs)} conversations", flush=True)

    out_convs = []
    for c in convs:
        turns_out = []
        for t in c["turns"]:
            scene = t["scene"]
            tr = transcript_text(t["transcript"])
            topics = extract_topics(scene, t["transcript"])
            ctx = {"addressed": True, "last_utterance": t.get("last_utterance"),
                   "personal_context": {}}
            gt = t["ground_truth"]

            # diverse: time pool (LLM) and select (MMR) separately
            ranker._temp = 0.7
            t0 = time.time()
            pool = generate_pool(scene, tr, ctx, ranker=ranker, topics=topics)
            pool_lat = time.time() - t0
            t1 = time.time()
            board = select_board(pool, scene, tr, ctx, k=BOARD_K, lam=PROD_LAMBDA)
            select_lat = time.time() - t1
            words = [b["word"] for b in board]
            source = "stub" if {w.lower() for w in words} <= {s["word"].lower() for s in _STUB_WORDS} else "warm"

            # baseline board (mode-seeking) for contrast
            ranker._temp = 0.0
            base = rank_vocabulary(scene, tr, ctx, ranker=ranker, strategy="topk")
            base_words = [b["word"] for b in base][:BOARD_K]

            gtv = embed([gt])[0]
            dm = cm.board_metrics(words, gt, board_vecs=embed(words), gt_vec=gtv)
            bm = cm.board_metrics(base_words, gt, board_vecs=embed(base_words), gt_vec=gtv)

            turns_out.append({
                "turn": t["turn_number"], "gt": gt, "transcript": tr[:120],
                "diverse_board": words, "baseline_board": base_words,
                "covered": dm["covered"], "soft": dm["soft"], "ild": dm["ild"],
                "base_covered": bm["covered"], "base_ild": bm["ild"],
                "pool_lat_s": round(pool_lat, 3), "select_lat_s": round(select_lat, 4),
                "source": source,
            })
            print(f"  conv {c['conversation_id']} turn {t['turn_number']}: "
                  f"cov={dm['covered']} base={bm['covered']} pool={pool_lat:.1f}s", flush=True)
        out_convs.append({"conversation_id": c["conversation_id"], "turns": turns_out})

    flat = [t for c in out_convs for t in c["turns"]]
    summary = {
        "n_turns": len(flat),
        "coverage": float(np.mean([t["covered"] for t in flat])),
        "base_coverage": float(np.mean([t["base_covered"] for t in flat])),
        "miss_rate": 1.0 - float(np.mean([t["covered"] for t in flat])),
        "ild": float(np.mean([t["ild"] for t in flat])),
        "base_ild": float(np.mean([t["base_ild"] for t in flat])),
        "pool_lat_p50": float(np.percentile([t["pool_lat_s"] for t in flat], 50)),
        "pool_lat_p95": float(np.percentile([t["pool_lat_s"] for t in flat], 95)),
        "select_lat_p50_ms": float(np.percentile([t["select_lat_s"] for t in flat], 50)) * 1000,
        "select_lat_p95_ms": float(np.percentile([t["select_lat_s"] for t in flat], 95)) * 1000,
        "stub_rate": float(np.mean([t["source"] == "stub" for t in flat])),
    }
    (HERE / "sim_results.json").write_text(
        json.dumps({"summary": summary, "conversations": out_convs}, indent=2))
    print("\nSIM SUMMARY:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
