"""
Phase-1 SCORING (the free half) — read caches, compute the diverse-coverage suite.

Reads cache/{baseline,pools,oracle}.jsonl produced by gen_phase1.py. The pool +
oracle generations are frozen; everything here is embedding + numpy, so the whole
lambda sweep and every metric re-runs in seconds with no model calls.

Arms compared on the SAME turns (paired):
  baseline   : current production board (old top-k prompt)
  diverse@L  : MMR over the diverse pool at lambda L (sweep)
  diverse*   : the chosen production lambda (default 0.6)

Writes results.json consumed by plots.py and the write-up.

    python3 evaluation/phase1/score_phase1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAC = HERE.parent.parent
sys.path.insert(0, str(MAC))

from evaluation.embed import embed                                   # noqa: E402
from evaluation import coverage_metrics as cm                        # noqa: E402
from selection import build_query_vector, mmr_select, intra_list_diversity  # noqa: E402

CACHE = HERE / "cache"
LAMBDAS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
PROD_LAMBDA = 0.6
BOARD_K = 7


def _load(name: str) -> dict:
    out = {}
    for line in (CACHE / f"{name}.jsonl").read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def _board_score(board: list[str], gt: str, plausible_vecs, gt_vec) -> dict:
    bv = embed(board) if board else np.zeros((0, 384), dtype=np.float32)
    m = cm.board_metrics(board, gt, board_vecs=bv, gt_vec=gt_vec)
    m["setcov"] = cm.set_coverage(bv, plausible_vecs)
    m["dcs"] = cm.dcs(m["setcov"], m["ild"])
    return m


def _aggregate(per_turn: list[dict]) -> dict:
    keys = ["covered", "soft", "ild", "dup", "setcov", "dcs", "fcov"]
    agg = {}
    for k in keys:
        vals = [t[k] for t in per_turn]
        agg[k] = float(np.mean(vals))
    agg["miss_rate"] = 1.0 - agg["covered"]
    agg["n"] = len(per_turn)
    return agg


def main() -> None:
    baseline = _load("baseline")
    pools = _load("pools")
    oracle = _load("oracle")
    ids = [i for i in pools if i in baseline and i in oracle]
    print(f"scoring {len(ids)} paired turns")

    # Pre-embed oracle plausible sets + ground truths once.
    gt_vecs, plaus_vecs, oracle_gt_hit = {}, {}, []
    for i in ids:
        gt = pools[i]["gt"]
        gtv = embed([gt])[0]
        gt_vecs[i] = gtv
        pset = oracle[i]["plausible"]
        pv = embed(pset) if pset else np.zeros((0, 384), dtype=np.float32)
        plaus_vecs[i] = pv
        oracle_gt_hit.append(float((pv @ gtv).max()) >= 0.55 if pv.shape[0] else False)
    oracle_sanity = float(np.mean(oracle_gt_hit))

    # ---- Baseline arm ----
    base_turns = []
    for i in ids:
        base_turns.append(_board_score(baseline[i]["board"][:BOARD_K],
                                       baseline[i]["gt"], plaus_vecs[i], gt_vecs[i]))

    # ---- Diverse arm: MMR over the cached pool, swept over lambda (free) ----
    sweep = {}
    pool_cache = {}  # i -> (words, vecs, query, functions)
    for i in ids:
        rec = pools[i]
        words = [p["word"] for p in rec["pool"]]
        funcs = [p.get("function", "") for p in rec["pool"]]
        vecs = embed(words) if words else np.zeros((0, 384), dtype=np.float32)
        query = build_query_vector(embed, rec["scene"], rec["transcript"],
                                   {"last_utterance": rec.get("last_utterance")})
        pool_cache[i] = (words, vecs, query, funcs)

    for lam in LAMBDAS:
        turns = []
        for i in ids:
            words, vecs, query, funcs = pool_cache[i]
            idx = mmr_select(vecs, query, BOARD_K, lam, funcs)
            board = [words[j] for j in idx]
            turns.append(_board_score(board, pools[i]["gt"], plaus_vecs[i], gt_vecs[i]))
        sweep[f"{lam:.1f}"] = {"agg": _aggregate(turns), "per_turn": turns}

    base_agg = _aggregate(base_turns)
    prod = sweep[f"{PROD_LAMBDA:.1f}"]

    # ---- Significance: baseline vs production-lambda (paired) ----
    deltas_setcov = [p["setcov"] - b["setcov"] for p, b in zip(prod["per_turn"], base_turns)]
    deltas_ild = [p["ild"] - b["ild"] for p, b in zip(prod["per_turn"], base_turns)]
    deltas_cov = [float(p["covered"]) - float(b["covered"]) for p, b in zip(prod["per_turn"], base_turns)]
    stats = {
        "setcov_ci": cm.bootstrap_ci(deltas_setcov),
        "ild_ci": cm.bootstrap_ci(deltas_ild),
        "cov_ci": cm.bootstrap_ci(deltas_cov),
        "mcnemar": cm.mcnemar([b["covered"] for b in base_turns],
                              [p["covered"] for p in prod["per_turn"]]),
        "delta_setcov": float(np.mean(deltas_setcov)),
        "delta_ild": float(np.mean(deltas_ild)),
        "delta_cov": float(np.mean(deltas_cov)),
    }

    # pool diversity (how diverse is generation itself, before selection)
    pool_ild = [intra_list_diversity(pool_cache[i][1]) for i in ids if pool_cache[i][1].shape[0] >= 2]
    pool_sizes = [len(pool_cache[i][0]) for i in ids]

    results = {
        "n": len(ids),
        "ids": ids,
        "oracle_sanity": oracle_sanity,
        "board_k": BOARD_K,
        "prod_lambda": PROD_LAMBDA,
        "baseline": {"agg": base_agg, "per_turn": base_turns},
        "sweep": sweep,
        "stats": stats,
        "pool": {"mean_ild": float(np.mean(pool_ild)) if pool_ild else 0.0,
                 "mean_size": float(np.mean(pool_sizes))},
        "examples": _example_boards(ids[:6], baseline, pools, pool_cache),
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    _print_summary(results)


def _example_boards(ids, baseline, pools, pool_cache) -> list:
    out = []
    for i in ids:
        words, vecs, query, funcs = pool_cache[i]
        idx = mmr_select(vecs, query, BOARD_K, PROD_LAMBDA, funcs)
        out.append({
            "gt": pools[i]["gt"],
            "transcript": pools[i]["transcript"],
            "baseline": baseline[i]["board"][:BOARD_K],
            "diverse": [words[j] for j in idx],
        })
    return out


def _print_summary(r: dict) -> None:
    b, p = r["baseline"]["agg"], r["sweep"][f"{r['prod_lambda']:.1f}"]["agg"]
    print("\n" + "=" * 68)
    print(f"VoiceLink Phase-1 diverse-coverage  (N={r['n']}, k={r['board_k']})")
    print("=" * 68)
    print(f"oracle sanity (GT in plausible set): {r['oracle_sanity']*100:.0f}%")
    print(f"diverse-pool mean ILD={r['pool']['mean_ild']:.2f}  mean size={r['pool']['mean_size']:.1f}")
    print(f"{'metric':<16}{'baseline':>12}{'diverse@'+str(r['prod_lambda']):>12}{'delta':>10}")
    for k, lbl in [("covered", "Coverage@7"), ("setcov", "SetCov@7"),
                   ("ild", "ILD@7"), ("dup", "dup_rate"), ("fcov", "FuncCov(/9)"),
                   ("soft", "Soft-Cov"), ("dcs", "DCS")]:
        print(f"{lbl:<16}{b[k]:>12.3f}{p[k]:>12.3f}{p[k]-b[k]:>+10.3f}")
    s = r["stats"]
    print(f"\nΔSetCov 95% CI: [{s['setcov_ci'][0]:+.3f},{s['setcov_ci'][1]:+.3f}]"
          f"   ΔILD 95% CI: [{s['ild_ci'][0]:+.3f},{s['ild_ci'][1]:+.3f}]")
    print(f"McNemar: new fixed {s['mcnemar']['new_only']}, broke {s['mcnemar']['base_only']}")
    print("=" * 68)


if __name__ == "__main__":
    main()
