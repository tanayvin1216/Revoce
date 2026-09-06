"""I-09 — template-family siblings across the train/val/test split (read-only analysis).

Phase-3 scenarios are generated from templates by cycling filler names
(gen_teacher.py:57-75, ids `<category>-<prompt>-<filler>`), and the Phase-1 examples
corpus contributes several turns of the same conversation (ids `p1-<conv>-<turn>`).
`data.dedup_rows` (cosine > 0.95) and the random `split_indices` do not keep such
families inside one split, so part of the "held-out" 84 has a near-copy in train/val.

This script records, for the paper's checkpoint/vocabulary/splits (or any replacement
passed on the command line):
  1. the family map: every gathered row -> teacher_data id, family id, source, split;
  2. which test rows have a same-family sibling in train, in val, in train-or-val;
  3. per-test-row max MiniLM cosine of `data.context_text(row)` against training rows
     only and against train+val rows, with the argmax reference row (index, id, family);
  4. Table-2 metrics on the no-sibling and sibling subsets at @7 and @10, computed by
     `evaluate.held_out_eval` itself (a subset splits file is written under analyses/
     and passed as `splits_path`; the fingerprint guard still applies) so the ruler is
     identical to the paper's, including the paired-bootstrap intervals (seed 0).
     @10 is obtained by setting `evaluate.METRIC_K = 10` in-process (the module-level
     constant every window in held_out_eval reads) — nothing under mac/ is modified.

Family definition (the checker's, issues_gpt_response.md:157): join rows to
teacher_data.jsonl by context text, drop the last id component for Phase-3 template
variants and for p1 conversation turns; v2 rows are singletons (one authored row each).

Outputs (next to this script, suffix <tag>):
  I-09_family_map<tag>.json         all 559 rows: idx, id, family, source, split
  I-09_similarity_pairs<tag>.jsonl  one line per test row with both max-cosine pairs
  I-09_subset_metrics<tag>.json     held_out_eval output for full/no-sibling/sibling x @7/@10
  I-09_splits_nosib<tag>.json, I-09_splits_sib<tag>.json  the subset splits handed to held_out_eval

Run (from mac/):
  VOICELINK_PERCEPTION=0 /usr/local/bin/python3 ../paper/odi/revision/analyses/I-09_siblings.py
Refresh on an I-05 checkpoint:
  ... I-09_siblings.py --checkpoint <new.pt> --vocab <new_vocab.json> --splits <splits.json> --tag _i05
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAC = HERE.parents[3] / "mac"
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))

import torch  # noqa: E402

from evaluation.embed import embed  # noqa: E402
from evaluation.phase4 import data as d, evaluate as ev, train as tm  # noqa: E402
from evaluation.phase4.student import StudentRanker  # noqa: E402

TEACHER_DATA = MAC / "evaluation" / "phase4" / "teacher_data.jsonl"
SIM_THRESHOLD = 0.90


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family_of(row_id: str) -> str:
    """Phase-3 `<cat>-<p>-<f>` -> `<cat>-<p>`; p1 `p1-<conv>-<turn>` -> `p1-<conv>`;
    v2 rows are singletons (`v2-<n>` stays as is)."""
    if row_id.startswith("v2-"):
        return row_id
    return row_id.rsplit("-", 1)[0]


def source_of(row_id: str) -> str:
    if row_id.startswith("v2-"):
        return "v2"
    if row_id.startswith("p1-"):
        return "phase1"
    return "phase3"


def load_student_from(ckpt_path: Path, vocab_path: Path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = StudentRanker(ck["context_dim"], ck["hidden_dim"], ck["vocab_size"], ck["dropout"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    vocab, _ = d.load_vocab(vocab_path)
    return model, vocab, ck.get("guaranteed_core", []), int(ck.get("k_dyn", 6)), ck


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=ev.CKPT)
    ap.add_argument("--vocab", type=Path, default=d.VOCAB_PATH)
    ap.add_argument("--splits", type=Path, default=d.SPLITS_PATH)
    ap.add_argument("--tag", default="")
    ap.add_argument("--k", type=int, default=10, help="board size handed to student_board")
    args = ap.parse_args()
    t0 = time.time()

    # ---- rows, splits, ids, families -------------------------------------------
    rows = tm.gather_rows()
    splits = json.loads(args.splits.read_text())
    fp = d.rows_fingerprint(rows)
    if splits.get("fingerprint") and splits["fingerprint"] != fp:
        raise SystemExit(f"fingerprint mismatch: splits {splits['fingerprint']} vs rows {fp}")
    teacher = [json.loads(l) for l in TEACHER_DATA.read_text().splitlines() if l.strip()]
    id_by_ctx = {d.context_text(r): r["id"] for r in teacher}
    ids = [id_by_ctx.get(d.context_text(r)) for r in rows]
    if any(i is None for i in ids):
        raise SystemExit("some gathered rows do not join back to teacher_data.jsonl")
    split_of: dict[int, str] = {}
    for key in ("train", "val", "test"):
        for i in splits[key]:
            split_of[i] = key
    fams = [family_of(i) for i in ids]
    fam_map = [{"idx": i, "id": ids[i], "family": fams[i], "source": source_of(ids[i]),
                "category": rows[i].get("category"), "split": split_of.get(i)}
               for i in range(len(rows))]
    fam_in = {key: {fams[i] for i in splits[key]} for key in ("train", "val", "test")}
    fam_size = collections.Counter(fams)

    # ---- sibling flags per test row ----------------------------------------------
    test_idx = list(splits["test"])
    sib_train = [i for i in test_idx if fams[i] in fam_in["train"]]
    sib_val = [i for i in test_idx if fams[i] in fam_in["val"]]
    sib_any = [i for i in test_idx if fams[i] in fam_in["train"] or fams[i] in fam_in["val"]]
    sib_test_only = [i for i in test_idx if fams[i] not in fam_in["train"]
                     and fams[i] not in fam_in["val"] and fam_size[fams[i]] > 1]
    nosib = [i for i in test_idx if i not in set(sib_any)]

    # ---- cosines: test vs train-only and vs train+val ------------------------------
    vecs = embed([d.context_text(r) for r in rows])          # (559, 384), L2-normalised
    train_idx = list(splits["train"])
    trval_idx = list(splits["train"]) + list(splits["val"])
    pairs = []
    for i in test_idx:
        rec = {"test_idx": i, "test_id": ids[i], "family": fams[i], "source": source_of(ids[i]),
               "sibling_in_train": fams[i] in fam_in["train"],
               "sibling_in_val": fams[i] in fam_in["val"]}
        for label, ref in (("train", train_idx), ("trainval", trval_idx)):
            sims = vecs[ref] @ vecs[i]
            j = int(np.argmax(sims))
            rec[f"max_cos_{label}"] = round(float(sims[j]), 6)
            rec[f"argmax_{label}_idx"] = ref[j]
            rec[f"argmax_{label}_id"] = ids[ref[j]]
            rec[f"argmax_{label}_split"] = split_of[ref[j]]
            rec[f"argmax_{label}_same_family"] = fams[ref[j]] == fams[i]
        # every same-family sibling outside the test split, with its cosine
        rec["siblings"] = [{"idx": j, "id": ids[j], "split": split_of[j],
                            "cos": round(float(vecs[j] @ vecs[i]), 6)}
                           for j in range(len(rows)) if j != i and fams[j] == fams[i]
                           and split_of[j] != "test"]
        pairs.append(rec)
    cos_train = np.array([p["max_cos_train"] for p in pairs])
    cos_trval = np.array([p["max_cos_trainval"] for p in pairs])
    above_train = [p for p in pairs if p["max_cos_train"] > SIM_THRESHOLD]
    above_trval = [p for p in pairs if p["max_cos_trainval"] > SIM_THRESHOLD]

    # ---- Table-2 metrics on subsets via held_out_eval itself -----------------------
    model, vocab, core, k_dyn, ck = load_student_from(args.checkpoint, args.vocab)
    subset_defs = {"full": test_idx, "no_sibling": nosib, "sibling": sib_any}
    subset_paths = {}
    for name, idx in subset_defs.items():
        if name == "full":
            subset_paths[name] = args.splits
            continue
        p = HERE / f"I-09_splits_{'nosib' if name == 'no_sibling' else 'sib'}{args.tag}.json"
        p.write_text(json.dumps({"fingerprint": fp, "train": splits["train"], "val": splits["val"],
                                 "test": idx, "note": f"I-09 {name} subset of the test split; "
                                 "train/val unchanged; for held_out_eval only"}, indent=2))
        subset_paths[name] = p
    metrics: dict = {}
    original_k = ev.METRIC_K
    original_dyn = ev.dynamic_words
    try:
        for K in (7, 10):
            ev.METRIC_K = K
            # `dynamic_words(board, core, k=METRIC_K)` bound its default at import time,
            # so the coverage window must be re-pointed explicitly as well (I-03 note:
            # a --metric-k flag has to do the same or coverage silently stays @7).
            ev.dynamic_words = (lambda board, core_words, k=None, _K=K:
                                original_dyn(board, core_words, _K))
            for name, path in subset_paths.items():
                res = ev.held_out_eval(model, vocab, args.k, k_dyn, core, splits_path=path)
                metrics[f"{name}@{K}"] = res
    finally:
        ev.METRIC_K = original_k
        ev.dynamic_words = original_dyn

    def compact(m: dict) -> dict:
        dr, cov = m["diversity_retention"], m["coverage_of_intent"]
        return {"n": dr["n_heldout"], "n_cov": cov["n_scored_dynamic"],
                "student_df": dr["student_distinct_functions"], "teacher_df": dr["teacher_distinct_functions"],
                "retention": dr["retention_ratio"], "df_margin": dr["distinct_functions_margin"],
                "student_near_dup": dr["student_near_dup"], "teacher_near_dup": dr["teacher_near_dup"],
                "student_sfp": dr["student_same_function_pair"], "teacher_sfp": dr["teacher_same_function_pair"],
                "sfp_margin": dr["same_function_pair_margin"], "board_variety": dr["board_variety"],
                "student_cov_dyn": cov["student_dynamic"], "teacher_cov_dyn": cov["teacher_dynamic"],
                "cov_dyn_ratio": (round(cov["student_dynamic"] / cov["teacher_dynamic"], 3)
                                  if cov["student_dynamic"] is not None and cov["teacher_dynamic"] else None),
                "cov_margin": cov["margin"], "student_cov_all": cov["student"], "teacher_cov_all": cov["teacher"],
                "dyn_word_agreement": m["agreement"]["dynamic_overlap"],
                "diversity_verdict": dr["verdict"], "coverage_verdict": cov["verdict"]}

    def composition(idx: list[int]) -> dict:
        return dict(collections.Counter(source_of(ids[i]) for i in idx))

    summary = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                   text=True, cwd=MAC).stdout.strip(),
        "artifacts": {
            "checkpoint": {"path": str(args.checkpoint), "sha256": sha256(args.checkpoint),
                           "config_hash": ck.get("config_hash"), "k_dyn": k_dyn, "core": core},
            "vocab": {"path": str(args.vocab), "sha256": sha256(args.vocab), "n": len(vocab),
                      "vocab_sha16": d.vocab_sha(vocab)},
            "splits": {"path": str(args.splits), "sha256": sha256(args.splits), "fingerprint": fp,
                       "sizes": {k: len(splits[k]) for k in ("train", "val", "test")}},
            "teacher_data": {"path": str(TEACHER_DATA), "sha256": sha256(TEACHER_DATA), "n": len(teacher)},
        },
        "family_definition": "teacher_data id without its last '-' component for phase3 and p1; v2 ids are singletons",
        "families": {"n_rows": len(rows), "n_families": len(fam_size),
                     "n_multi_row_families": sum(1 for v in fam_size.values() if v > 1),
                     "rows_by_source": composition(list(range(len(rows)))),
                     "test_by_source": composition(test_idx)},
        "siblings": {
            "test_with_sibling_in_train": len(sib_train),
            "test_with_sibling_in_val": len(sib_val),
            "test_with_sibling_in_train_or_val": len(sib_any),
            "test_with_sibling_in_train_or_val_by_source": composition(sib_any),
            "test_with_sibling_only_inside_test": len(sib_test_only),
            "no_sibling_subset_n": len(nosib),
            "no_sibling_by_source": composition(nosib),
            "sibling_test_ids": [ids[i] for i in sib_any],
            "no_sibling_test_ids": [ids[i] for i in nosib],
        },
        "similarity": {
            "threshold": SIM_THRESHOLD,
            "test_above_threshold_vs_train": len(above_train),
            "test_above_threshold_vs_trainval": len(above_trval),
            "median_max_cos_vs_train": round(float(np.median(cos_train)), 6),
            "median_max_cos_vs_trainval": round(float(np.median(cos_trval)), 6),
            "above_train_ids": [(p["test_id"], p["argmax_train_id"], p["max_cos_train"]) for p in above_train],
            "above_trainval_ids": [(p["test_id"], p["argmax_trainval_id"], p["max_cos_trainval"]) for p in above_trval],
            "above_train_same_family": sum(p["argmax_train_same_family"] for p in above_train),
            "above_trainval_same_family": sum(p["argmax_trainval_same_family"] for p in above_trval),
            "above_trainval_sibling_flag": sum(1 for p in above_trval
                                               if p["sibling_in_train"] or p["sibling_in_val"]),
            "max_cos_trainval_nosib_max": round(float(max(p["max_cos_trainval"] for p in pairs
                                                          if not (p["sibling_in_train"] or p["sibling_in_val"]))), 6),
            "max_cos_trainval_sib_min": round(float(min(p["max_cos_trainval"] for p in pairs
                                                        if p["sibling_in_train"] or p["sibling_in_val"])), 6),
        },
        "metrics_compact": {k: compact(v) for k, v in metrics.items()},
        "metrics_full": metrics,
        "wall_clock_s": round(time.time() - t0, 1),
    }
    out = HERE / f"I-09_subset_metrics{args.tag}.json"
    out.write_text(json.dumps(summary, indent=2))
    (HERE / f"I-09_family_map{args.tag}.json").write_text(json.dumps(fam_map, indent=1))
    with (HERE / f"I-09_similarity_pairs{args.tag}.jsonl").open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print(json.dumps({k: v for k, v in summary.items() if k not in ("metrics_full",)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
