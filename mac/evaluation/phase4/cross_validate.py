"""K-fold cross-validated gates for the v3 bi-encoder.

    VOICELINK_PERCEPTION=0 python -m evaluation.phase4.cross_validate

Why this exists
---------------
The single-split gates score coverage on 71 scenarios. A power analysis on the
observed per-scenario spread (SD 0.448 of the paired margin) puts the minimum
detectable effect at **0.149 coverage points** — the gate literally cannot see a
difference smaller than 15 points, and the v3-vs-bar margin is 0.003. That is
why the interval came back INCONCLUSIVE, and no amount of student work fixes a
measurement that blunt.

The corpus is not the constraint. 471 of 559 teacher rows carry authored
`acceptable_moves`; we were scoring 71 of them, because coverage was only
evaluated on the 15% test split and the other 328 scorable rows sat in train.

Cross-validation uses all of them without leaking: fit K models, each scoring
only the fold it never saw, then pool the per-scenario margins. Every scorable
row contributes exactly one honest observation. The bi-encoder trains in ~2s, so
K folds cost seconds.

This does NOT replace the held-out gate — the single split stays the headline
number so it remains comparable to v2 and to the phase-5 history. This is the
instrument that can actually resolve a difference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluation.phase3 import metrics3
from evaluation.phase4 import bank as bank_mod
from evaluation.phase4 import data as d
from evaluation.phase4 import evaluate as ev
from evaluation.phase4 import infer_biencoder as ib
from evaluation.phase4 import train_biencoder as tb
from evaluation.phase4.train import gather_rows, load_config
from rank_vocabulary import GUARANTEED_CORE

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "phase4_v3_cv.json"


def _folds(n: int, k: int, seed: int) -> list[np.ndarray]:
    idx = np.random.default_rng(seed).permutation(n)
    return [np.sort(f) for f in np.array_split(idx, k)]


def cross_validate(k_folds: int = 5, seed: int = 0) -> dict:
    cfg, _ = load_config()
    rows = gather_rows()
    tau = float(cfg["vocab"].get("bank_dedup_tau", cfg["vocab"]["dedup_tau"]))
    linkage = str(cfg["vocab"].get("bank_linkage", "complete"))

    # The bank is built from the WHOLE corpus, exactly as in production: it is a
    # candidate list, not a label set, and the model never sees fold labels
    # through it. Rebuilding it per fold would change the candidate space
    # between folds and make the pooled numbers incomparable.
    phrases, canonical = bank_mod.build_phrase_bank(rows, tau, 1, linkage)
    phrase2idx = {p: i for i, p in enumerate(phrases)}
    X = d.context_features(rows)
    P = bank_mod.phrase_features(phrases)
    pos = [bank_mod.positives_for_row(r, canonical, phrase2idx) for r in rows]
    Xt, Pt = torch.from_numpy(X), torch.from_numpy(P)

    k, k_dyn = cfg["board"]["k"], cfg["board"]["k_dyn"]
    core = list(GUARANTEED_CORE)
    moves_by = ev._acceptable_moves_lookup()
    folds = _folds(len(rows), k_folds, seed)

    s_cov, t_cov, s_df, t_df, s_sfp, t_sfp = [], [], [], [], [], []
    per_fold = []

    for fi, held in enumerate(folds):
        rest = np.setdiff1d(np.arange(len(rows)), held)
        # A small validation slice carved from the training part only, so
        # early stopping never sees the held-out fold.
        cut = max(1, int(0.15 * len(rest)))
        va, tr = rest[:cut].tolist(), rest[cut:].tolist()

        model, vloss = tb.fit(cfg, Xt, Pt, pos, len(phrases), tr, va,
                              seed=seed + fi, quiet=True)
        index = ib.build_bank_index(model, phrases)

        fold_cov = []
        for i in held.tolist():
            r = rows[i]
            sb = ib.student_board(model, index, r["scene"], r["transcript"],
                                  r.get("is_question", False), k, k_dyn, core)
            sdf, _, ssfp = ev._df_dup_sfp(sb)
            tdf, _, tsfp = ev._df_dup_sfp(r["board"])
            s_df.append(sdf); t_df.append(tdf)
            s_sfp.append(ssfp); t_sfp.append(tsfp)

            mv = moves_by.get((r.get("scene", "").strip(),
                               r.get("transcript", "").strip()))
            if not mv:
                continue
            honest = [m for m in mv if not metrics3.is_hedge(m)]
            if not honest:
                continue
            sv = 1.0 if metrics3.coverage_of_intent(
                ev.dynamic_words(sb, core), honest, ev.METRIC_K) else 0.0
            tv = 1.0 if metrics3.coverage_of_intent(
                ev.dynamic_words(r["board"], core), honest, ev.METRIC_K) else 0.0
            s_cov.append(sv); t_cov.append(tv); fold_cov.append(sv)

        per_fold.append({"fold": fi, "n_held": len(held),
                         "n_scored": len(fold_cov),
                         "coverage": round(float(np.mean(fold_cov)), 4)
                         if fold_cov else None,
                         "val_loss": round(vloss, 5)})
        print(f"  fold {fi}: {len(held)} held, {len(fold_cov)} scorable, "
              f"coverage {per_fold[-1]['coverage']}", flush=True)

    return {
        "k_folds": k_folds,
        "seed": seed,
        "n_rows": len(rows),
        "n_scored_coverage": len(s_cov),
        "bank_size": len(phrases),
        "per_fold": per_fold,
        "coverage": {
            "student": round(float(np.mean(s_cov)), 4),
            "teacher": round(float(np.mean(t_cov)), 4),
            "margin": ev._margin_ci(s_cov, t_cov,
                                    factor=ev.DYNAMIC_COVERAGE_GATE),
        },
        "distinct_functions": {
            "student": round(float(np.mean(s_df)), 4),
            "teacher": round(float(np.mean(t_df)), 4),
            "margin": ev._margin_ci(s_df, t_df, factor=0.90),
        },
        "same_function_pair": {
            "student": round(float(np.mean(s_sfp)), 4),
            "teacher": round(float(np.mean(t_sfp)), 4),
            "margin": ev._margin_ci(s_sfp, t_sfp, higher_is_better=False),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"cross-validating v3 over {args.folds} folds...", flush=True)
    res = cross_validate(args.folds, args.seed)
    RESULTS.write_text(json.dumps(res, indent=2))

    bar = "=" * 78
    print("\n" + bar)
    print(f"v3 bi-encoder — {res['k_folds']}-fold cross-validated gates "
          f"(every row scored by a model that never saw it)")
    print(bar)
    print(f"{'metric':<24}{'v3':>9}{'teacher':>9}  {'margin':>9}  "
          f"{'95% CI':>22}  verdict")
    for label, key in (("coverage-of-intent", "coverage"),
                       ("distinct-functions", "distinct_functions"),
                       ("same-function-pair", "same_function_pair")):
        m = res[key]["margin"]
        lo, hi = m["ci"]
        print(f"{label:<24}{res[key]['student']:>9}{res[key]['teacher']:>9}  "
              f"{m['margin']:>+9.4f}  [{lo:>+9.4f},{hi:>+9.4f}]  {m['verdict']}")
    print(bar)
    print(f"coverage scored on n={res['n_scored_coverage']} "
          f"(single-split gate uses 71)")
    print(bar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
