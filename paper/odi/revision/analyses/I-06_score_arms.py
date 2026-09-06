#!/usr/bin/env python3
"""I-06 step 2 — same-pool selector ablation over the frozen teacher pools.

For every stored pool (I-06_pools.jsonl, one per Phase-3 scenario) build boards
that differ ONLY in the dynamic-slot selector. Every arm pins the same four
core entries in GUARANTEED_CORE order, fills exactly six dynamic slots from the
same pool, and returns exactly ten entries, so board size and core are equal
by construction:

  relevance    pinned 4 + the six pool items with the highest clipped cosine to
               the situation query (selection.build_query_vector). No MMR
               penalty, no answer quota, no function tie-break. Ties keep pool
               order (stable sort).
  mmr_plain    pinned 4 + selection.mmr_select(lambda=0.6) with NO function
               tie-break and NO answer quota — the same quota/tie-break
               settings as the relevance arm, so the pair isolates the MMR
               penalty alone.
  full_policy  pinned 4 + the shipped constrained policy, exactly
               rank_vocabulary.select_board (answer quota <= 5 at lambda=0.35,
               then general MMR at lambda=0.6 with the unseen-function
               tie-break within 0.02).
  mmr_tiebreak SUPPLEMENTARY: mmr_select(lambda=0.6) WITH the function
               tie-break but without the answer quota; separates the two
               constraints of the full policy.

Scoring goes through the Phase-3 harness itself (selftest.score_boards /
selftest._score_one, metrics3, the torch MiniLM embedder) at three explicit
windows: full@7 (the paper's Table 1 window), full@10 (whole board) and
dynamic-only@6 (core stripped, six entries). Same-function-pair rate is added
from metrics3 on the identical window. Paired differences between arms carry
10,000-resample percentile bootstrap 95% intervals (seed 0), as in I-03.

The historical cached boards (boards.jsonl: diversity-first and direct LLM
generation) are re-scored in the same process as a ruler check; they were
produced from a DIFFERENT teacher realisation in June and are reported as a
reference, not as a fourth arm of this ablation.

Run from mac/:
    VOICELINK_PERCEPTION=0 VOICELINK_EMBED_BACKEND=torch HF_HUB_OFFLINE=1 \
      /usr/local/bin/python3 ../paper/odi/revision/analyses/I-06_score_arms.py \
      --pools ../paper/odi/revision/analyses/I-06_pools.jsonl \
      --output ../paper/odi/revision/analyses/I-06_arms_scores.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VOICELINK_PERCEPTION", "0")
os.environ.setdefault("VOICELINK_EMBED_BACKEND", "torch")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
MAC = HERE.parents[3] / "mac"
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))
os.chdir(MAC)

import numpy as np  # noqa: E402

import rank_vocabulary as rv  # noqa: E402
from evaluation import embed as embed_mod  # noqa: E402
from evaluation.coverage_metrics import bootstrap_ci  # noqa: E402
from evaluation.embed import embed  # noqa: E402
from evaluation.phase3 import metrics3, selftest  # noqa: E402
from selection import DEFAULT_LAMBDA, build_query_vector, mmr_select  # noqa: E402

ARMS = ["relevance", "mmr_plain", "full_policy", "mmr_tiebreak"]
PRIMARY_ARMS = ["relevance", "mmr_plain", "full_policy"]
N_DYNAMIC = rv.BOARD_SIZE - len(rv.GUARANTEED_CORE)  # 6
WINDOWS = {"full7": (7, False), "full10": (10, False), "dyn6": (6, True)}
PAIRS = [("full_policy", "relevance"), ("mmr_plain", "relevance"),
         ("full_policy", "mmr_plain"), ("mmr_tiebreak", "mmr_plain"),
         ("full_policy", "mmr_tiebreak")]
BOOT_N, BOOT_SEED = 10_000, 0


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pool_hash(pool: list[dict]) -> str:
    canon = [[p.get("word", ""), p.get("function", ""), p.get("topic", "")] for p in pool]
    return hashlib.sha256(json.dumps(canon, ensure_ascii=False).encode()).hexdigest()


# --------------------------------------------------------------------------
# Arm construction
# --------------------------------------------------------------------------
def build_arms(row: dict) -> tuple[dict[str, list[str]], dict]:
    pool = row["pool"]
    scene = row["scene"] or None
    transcript = row["transcript"] or None
    context = row["context"]
    core_idx, rest_idx = rv._split_core(pool)
    assert len(core_idx) == len(rv.GUARANTEED_CORE), (row["id"], core_idx)
    assert len(rest_idx) >= N_DYNAMIC, (row["id"], len(rest_idx))
    pinned = [pool[i]["word"] for i in core_idx]
    assert [w.lower() for w in pinned] == [w.lower() for w in rv.GUARANTEED_CORE]

    words = [pool[i]["word"] for i in rest_idx]
    functions = [pool[i].get("function", "") for i in rest_idx]
    cand = embed(words)
    query = build_query_vector(embed, scene, transcript, context)
    qnorm = float(np.linalg.norm(query))
    rel = np.clip(cand @ query, 0.0, 1.0) if qnorm > 0 else np.zeros(len(words), np.float32)

    # relevance-only: stable descending sort so ties keep pool order
    order_rel = np.argsort(-rel, kind="stable")[:N_DYNAMIC]
    arm_rel = [words[i] for i in order_rel]

    # pure MMR, no quota, no function tie-break
    order_mmr = mmr_select(cand, query, N_DYNAMIC, DEFAULT_LAMBDA, None)
    arm_mmr = [words[i] for i in order_mmr]

    # MMR + function tie-break, no quota (supplementary)
    order_tb = mmr_select(cand, query, N_DYNAMIC, DEFAULT_LAMBDA, functions)
    arm_tb = [words[i] for i in order_tb]

    # the shipped policy, verbatim
    full = rv.select_board(pool, scene, transcript, context, k=rv.BOARD_SIZE, lam=DEFAULT_LAMBDA)
    full_words = [b["word"] for b in full]
    assert len(full_words) == rv.BOARD_SIZE, (row["id"], full_words)
    assert [w.lower() for w in full_words[:4]] == [w.lower() for w in rv.GUARANTEED_CORE]
    arm_full_dyn = full_words[4:]

    pool_rest_lower = {w.lower() for w in words}
    arms = {"relevance": pinned + arm_rel, "mmr_plain": pinned + arm_mmr,
            "full_policy": pinned + arm_full_dyn, "mmr_tiebreak": pinned + arm_tb}
    for name, board in arms.items():
        assert len(board) == rv.BOARD_SIZE, (row["id"], name, board)
        assert len({w.lower() for w in board}) == rv.BOARD_SIZE, (row["id"], name, board)
        for w in board[4:]:
            assert w.lower() in pool_rest_lower, (row["id"], name, w)  # no stub padding

    n_answer = sum(1 for f in functions if f == "answer")
    quota = min(n_answer, max(N_DYNAMIC - 1, 1)) if n_answer else 0
    diag = {
        "n_rest": len(rest_idx), "n_answer_tagged": n_answer, "answer_quota": quota,
        "query_norm": qnorm,
        "rel_top6_mean": float(np.mean(rel[order_rel])),
        "rel_mmr_mean": float(np.mean(rel[order_mmr])),
        "rel_full_mean": float(np.mean([rel[words.index(w)] for w in arm_full_dyn])),
        "rel_tb_mean": float(np.mean(rel[order_tb])),
        "full_equals_tiebreak_dyn": sorted(arm_full_dyn) == sorted(arm_tb),
    }
    return arms, diag


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score_row(board: list[str], moves: list[str], gold: str, k: int, dynamic_only: bool) -> dict:
    """selftest._score_one on a synthetic row, plus same-function-pair rate on
    the identical window (mirrors _score_one's window logic)."""
    row = {"board": board, "acceptable_moves": moves, "gold_utterance": gold}
    s = selftest._score_one(row, metric_k=k, dynamic_only=dynamic_only)
    b = board
    if dynamic_only:
        core = {w.lower() for w in rv.GUARANTEED_CORE}
        b = [w for w in b if w.lower() not in core]
    top = b[:k]
    s["same_function_pair"] = metrics3.same_function_pair_rate(top, k) if top else 0.0
    s["n_evaluated"] = len(top)
    return s


def aggregate(scored: list[dict]) -> dict:
    agg = selftest._aggregate(scored)
    agg["same_function_pair"] = float(np.mean([s["same_function_pair"] for s in scored]))
    agg["mean_evaluated_length"] = float(np.mean([s["n_evaluated"] for s in scored]))
    agg["coverage_count"] = int(sum(1 for s in scored if s["coverage_of_intent"]))
    return agg


def paired(a: list[dict], b: list[dict]) -> dict:
    """a minus b, paired by row, with bootstrap CIs."""
    out = {}
    for m in ("distinct_functions", "near_dup", "same_function_pair"):
        d = [float(x[m]) - float(y[m]) for x, y in zip(a, b)]
        lo, hi = bootstrap_ci(d, BOOT_N, BOOT_SEED)
        out[m] = {"mean": float(np.mean(d)), "ci95": [lo, hi]}
    d = [float(bool(x["coverage_of_intent"])) - float(bool(y["coverage_of_intent"])) for x, y in zip(a, b)]
    lo, hi = bootstrap_ci(d, BOOT_N, BOOT_SEED)
    counts = {"11": 0, "01": 0, "10": 0, "00": 0}
    for x, y in zip(a, b):
        counts[f"{int(bool(x['coverage_of_intent']))}{int(bool(y['coverage_of_intent']))}"] += 1
    out["coverage_of_intent"] = {"mean": float(np.mean(d)), "ci95": [lo, hi],
                                 "paired_counts_a_b": counts}
    return out


def harness_aggregate(rows: list[dict], k: int, dynamic_only: bool) -> dict:
    """Run selftest.score_boards itself (the harness entry point) on rows
    relabelled strategy='diverse' so the aggregate comes from the harness."""
    relabelled = [dict(r, strategy="diverse") for r in rows]
    res = selftest.score_boards(relabelled, metric_k=k, dynamic_only=dynamic_only)
    return res["overall"]["diverse"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--boards-out", type=Path, default=None,
                    help="jsonl of every arm board in boards.jsonl schema (+pool_hash)")
    args = ap.parse_args(argv)
    if args.output.exists():
        ap.error(f"{args.output} exists; refuse to overwrite")
    boards_out = args.boards_out or args.output.with_name(args.output.stem + "_boards.jsonl")
    if boards_out.exists():
        ap.error(f"{boards_out} exists; refuse to overwrite")

    t0 = time.perf_counter()
    torch_backend = embed_mod.assert_backend("torch")

    pools = [json.loads(l) for l in args.pools.read_text().splitlines() if l.strip()]
    ids = [p["id"] for p in pools]
    assert len(ids) == len(set(ids)), "duplicate scenario ids in pools"
    checks = {
        "n_pools": len(pools),
        "all_status_ok": all(p["status"] == "ok" for p in pools),
        "status_counts": {},
        "pool_hash_recomputed_matches": all(pool_hash(p["pool"]) == p["pool_hash"] for p in pools),
        "min_n_rest": min(p["n_rest"] for p in pools),
        "teacher_exceptions": [p["id"] for p in pools if p["teacher"].get("exception")],
        "embed_backend": torch_backend,
    }
    for p in pools:
        checks["status_counts"][p["status"]] = checks["status_counts"].get(p["status"], 0) + 1
    # A pool is usable when the teacher call succeeded (no exception, >=1 parsed
    # item) and at least six non-core items exist, so no arm needs stub padding.
    # 'thin_pool' (fewer than six LLM items, the sixth being the deterministic
    # "I don't know") is informational: such pools give every arm the same six
    # items and dilute, never inflate, selector differences.
    usable = [p for p in pools
              if not p["teacher"].get("exception") and (p["teacher"].get("n_parsed") or 0) > 0
              and p["n_rest"] >= N_DYNAMIC]
    checks["n_usable"] = len(usable)
    checks["n_thin_pool_included"] = sum(1 for p in usable if p["status"] == "thin_pool")
    checks["n_rest_eq_6_all_arms_same_set"] = sum(1 for p in usable if p["n_rest"] == N_DYNAMIC)
    checks["excluded_ids"] = [p["id"] for p in pools if p not in usable]

    # scenario identity: the pools must cover the same 120 ids as boards.jsonl
    hist = selftest.load_boards()
    hist_ids = sorted({r["id"] for r in hist})
    checks["ids_match_boards_jsonl"] = sorted(ids) == hist_ids

    rv._mmr_fallback_warned = False
    per_row: list[dict] = []
    arm_rows: dict[str, list[dict]] = {a: [] for a in ARMS}
    for p in usable:
        arms, diag = build_arms(p)
        rec = {"id": p["id"], "category": p["category"], "pool_hash": p["pool_hash"],
               "answer_mode": p["answer_mode"], "arms": arms, "diag": diag, "scores": {}}
        for a in ARMS:
            row = {"id": p["id"], "category": p["category"], "strategy": a,
                   "scene": p["scene"], "transcript": p["transcript"],
                   "active_question": p["active_question"], "is_question": p["is_question"],
                   "acceptable_moves": p["acceptable_moves"], "gold_utterance": p["gold_utterance"],
                   "board": arms[a], "pool_hash": p["pool_hash"]}
            arm_rows[a].append(row)
            rec["scores"][a] = {w: score_row(arms[a], p["acceptable_moves"], p["gold_utterance"], k, dyn)
                                for w, (k, dyn) in WINDOWS.items()}
        # dynamic-set overlaps between arms
        dyn = {a: {w.lower() for w in arms[a][4:]} for a in ARMS}
        rec["jaccard"] = {f"{x}_vs_{y}": len(dyn[x] & dyn[y]) / len(dyn[x] | dyn[y])
                          for x, y in PAIRS}
        per_row.append(rec)
    checks["mmr_fallback_triggered"] = bool(rv._mmr_fallback_warned)
    assert not rv._mmr_fallback_warned, "MMR fell back to function-coverage order"

    # ---- aggregates per arm and window (harness + per-row cross-check) ----
    aggregates: dict[str, dict] = {}
    harness: dict[str, dict] = {}
    for w, (k, dyn_only) in WINDOWS.items():
        aggregates[w] = {}
        harness[w] = {}
        for a in ARMS:
            scored = [r["scores"][a][w] for r in per_row]
            aggregates[w][a] = aggregate(scored)
            harness[w][a] = harness_aggregate(arm_rows[a], k, dyn_only)
            for m in ("distinct_functions", "near_dup", "coverage_of_intent", "exact_match"):
                assert abs(harness[w][a][m] - aggregates[w][a][m]) < 1e-12, (w, a, m)

    paired_stats = {w: {f"{x}_minus_{y}": paired([r["scores"][x][w] for r in per_row],
                                                 [r["scores"][y][w] for r in per_row])
                        for x, y in PAIRS} for w in WINDOWS}

    # ---- diagnostics ------------------------------------------------------
    n_no_answer = sum(1 for r in per_row if r["diag"]["n_answer_tagged"] == 0)
    diagnostics = {
        "n_scenarios_answer_mode": sum(1 for r in per_row if r["answer_mode"]),
        "n_pools_with_answer_tags": sum(1 for r in per_row if r["diag"]["n_answer_tagged"] > 0),
        "n_pools_without_answer_tags": n_no_answer,
        "full_equals_tiebreak_when_no_answers": sum(
            1 for r in per_row if r["diag"]["n_answer_tagged"] == 0 and r["diag"]["full_equals_tiebreak_dyn"]),
        "mean_n_rest": float(np.mean([r["diag"]["n_rest"] for r in per_row])),
        "mean_answer_quota": float(np.mean([r["diag"]["answer_quota"] for r in per_row])),
        "mean_relevance_of_dynamic": {
            "relevance": float(np.mean([r["diag"]["rel_top6_mean"] for r in per_row])),
            "mmr_plain": float(np.mean([r["diag"]["rel_mmr_mean"] for r in per_row])),
            "full_policy": float(np.mean([r["diag"]["rel_full_mean"] for r in per_row])),
            "mmr_tiebreak": float(np.mean([r["diag"]["rel_tb_mean"] for r in per_row])),
        },
        "mean_jaccard_dynamic": {key: float(np.mean([r["jaccard"][key] for r in per_row]))
                                 for key in per_row[0]["jaccard"]},
        "identical_dynamic_sets": {key: int(sum(1 for r in per_row if r["jaccard"][key] == 1.0))
                                   for key in per_row[0]["jaccard"]},
        "answer_tagged_in_dynamic_mean": {
            a: float(np.mean([sum(1 for w in r["arms"][a][4:]
                                  if next((q.get("function") for q in usable_pool(usable, r["id"])
                                           if q["word"].lower() == w.lower()), "") == "answer")
                              for r in per_row])) for a in ARMS},
        "per_category_full7": {},
    }
    for cat in sorted({r["category"] for r in per_row}):
        diagnostics["per_category_full7"][cat] = {
            a: aggregate([r["scores"][a]["full7"] for r in per_row if r["category"] == cat]) for a in ARMS}

    # ---- historical cached boards, same ruler, same process ----------------
    historical = {}
    for w, (k, dyn_only) in WINDOWS.items():
        res = selftest.score_boards(hist, metric_k=k, dynamic_only=dyn_only)
        historical[w] = {"diversity_first_cached": res["overall"]["diverse"],
                         "direct_llm_generation_cached": res["overall"]["topk"]}
    hist_lengths = {}
    for r in hist:
        hist_lengths.setdefault(r["strategy"], {})
        hist_lengths[r["strategy"]][len(r["board"])] = hist_lengths[r["strategy"]].get(len(r["board"]), 0) + 1

    # ---- write ------------------------------------------------------------
    with boards_out.open("x") as f:
        for a in ARMS:
            for row in arm_rows[a]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    out = {
        "issue": "I-06", "step": "score_arms",
        "arms": {
            "relevance": "pinned core + top-6 clipped cosine to query; no MMR, no quota, no tie-break",
            "mmr_plain": "pinned core + mmr_select(lambda=0.6, functions=None); no quota, no tie-break",
            "full_policy": "pinned core + rank_vocabulary.select_board (answer quota<=5 @ lambda=0.35, then MMR lambda=0.6 with function tie-break)",
            "mmr_tiebreak": "SUPPLEMENTARY: pinned core + mmr_select(lambda=0.6, functions=tags); no quota",
        },
        "windows": {w: {"k": k, "dynamic_only": d} for w, (k, d) in WINDOWS.items()},
        "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED, "type": "paired percentile, per-row"},
        "checks": checks,
        "aggregates": aggregates,
        "harness_aggregates_match": True,
        "paired": paired_stats,
        "diagnostics": diagnostics,
        "historical_cached_reference": historical,
        "historical_board_lengths": hist_lengths,
        "inputs": {
            "pools": str(args.pools), "pools_sha256": sha256_file(args.pools),
            "boards.jsonl_sha256": sha256_file(MAC / "evaluation/phase3/boards.jsonl"),
            "rank_vocabulary.py_sha256": sha256_file(MAC / "rank_vocabulary.py"),
            "selection.py_sha256": sha256_file(MAC / "selection.py"),
            "selftest.py_sha256": sha256_file(MAC / "evaluation/phase3/selftest.py"),
            "metrics3.py_sha256": sha256_file(MAC / "evaluation/phase3/metrics3.py"),
            "taxonomy.py_sha256": sha256_file(MAC / "evaluation/phase3/taxonomy.py"),
        },
        "boards_out": str(boards_out),
        "wall_seconds": time.perf_counter() - t0,
        "per_row": per_row,
    }
    args.output.write_text(json.dumps(out, indent=1, ensure_ascii=False))

    # ---- console summary ----------------------------------------------------
    print(f"pools={checks['n_pools']} usable={checks['n_usable']} status={checks['status_counts']} "
          f"hash_ok={checks['pool_hash_recomputed_matches']} fallback={checks['mmr_fallback_triggered']} "
          f"embed={torch_backend}")
    for w in WINDOWS:
        print(f"\n[{w}] {'arm':<14}{'len':>6}{'DF':>9}{'near-dup':>10}{'SFP':>9}{'cov':>8}{'cov_n':>7}{'exact':>8}")
        for a in ARMS:
            g = aggregates[w][a]
            print(f"      {a:<14}{g['mean_evaluated_length']:>6.2f}{g['distinct_functions']:>9.4f}"
                  f"{g['near_dup']:>10.4f}{g['same_function_pair']:>9.4f}{g['coverage_of_intent']:>8.4f}"
                  f"{g['coverage_count']:>7d}{g['exact_match']:>8.3f}")
        for lab, arm in (("hist diverse", "diversity_first_cached"), ("hist direct", "direct_llm_generation_cached")):
            g = historical[w][arm]
            print(f"      {lab:<14}{'':>6}{g['distinct_functions']:>9.4f}{g['near_dup']:>10.4f}"
                  f"{'':>9}{g['coverage_of_intent']:>8.4f}{'':>7}{g['exact_match']:>8.3f}")
        for key, st in paired_stats[w].items():
            print(f"      {key:<32} DF {st['distinct_functions']['mean']:+.4f} "
                  f"[{st['distinct_functions']['ci95'][0]:+.4f},{st['distinct_functions']['ci95'][1]:+.4f}]  "
                  f"nd {st['near_dup']['mean']:+.4f} [{st['near_dup']['ci95'][0]:+.4f},{st['near_dup']['ci95'][1]:+.4f}]  "
                  f"sfp {st['same_function_pair']['mean']:+.4f} [{st['same_function_pair']['ci95'][0]:+.4f},{st['same_function_pair']['ci95'][1]:+.4f}]  "
                  f"cov {st['coverage_of_intent']['mean']:+.4f} [{st['coverage_of_intent']['ci95'][0]:+.4f},{st['coverage_of_intent']['ci95'][1]:+.4f}] "
                  f"{st['coverage_of_intent']['paired_counts_a_b']}")
    print(f"\ndiagnostics: {json.dumps({k: v for k, v in diagnostics.items() if k != 'per_category_full7'}, indent=1)}")
    print(f"wrote {args.output} and {boards_out} in {out['wall_seconds']:.1f}s")
    return 0


def usable_pool(usable: list[dict], sid: str) -> list[dict]:
    for p in usable:
        if p["id"] == sid:
            return p["pool"]
    return []


if __name__ == "__main__":
    raise SystemExit(main())
