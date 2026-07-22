"""Train the v3 bi-encoder student.

Objective: multi-label BCE over (context, phrase) scores, exactly the same
"which phrases belong on the board right now" framing as v2 — so the phase-3
gates still mean what they meant. What changes is the candidate axis. v2 scored
a fixed 94-column head; this scores a sampled candidate set drawn from an open
bank, which is what lets the bank grow without touching the weights.

Negative sampling: each batch's candidate set is the union of the batch's own
positives (in-batch negatives — hard, because they are phrases the teacher DID
choose, just for a different context) plus `n_random_negatives` drawn from the
rest of the bank. Learning to separate "right phrase, wrong situation" is the
whole job, so the in-batch half matters more than the random half.

Reuses gather_rows(), the split discipline, and the row fingerprint from the v2
pipeline unchanged — same corpus, same partition, so v2 and v3 numbers are
comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from evaluation.phase4 import bank as bank_mod
from evaluation.phase4 import data as d
from evaluation.phase4.student_biencoder import BiEncoderRanker
from evaluation.phase4.train import gather_rows, load_config, seed_everything

HERE = Path(__file__).resolve().parent
CKPT_PATH = HERE / "student_biencoder.pt"
SPLITS_PATH = HERE / "splits_biencoder.json"


def _batches(n: int, bs: int, rng: np.random.Generator):
    idx = rng.permutation(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def _candidate_set(pos_lists: list[list[int]], bank_size: int,
                   n_random: int, rng: np.random.Generator) -> list[int]:
    """Union of the batch's positives plus random negatives from the bank."""
    cands: set[int] = set()
    for p in pos_lists:
        cands.update(p)
    target = min(bank_size, len(cands) + n_random)
    if n_random > 0 and len(cands) < target:
        # Sample without replacement so a small bank can't be dominated by
        # repeats of the same negative.
        draw = rng.choice(bank_size, size=min(bank_size, target * 2),
                          replace=False).tolist()
        for c in draw:
            if len(cands) >= target:
                break
            cands.add(int(c))
    return sorted(cands)


def _targets(pos_lists: list[list[int]], cands: list[int]) -> torch.Tensor:
    pos_of = {c: j for j, c in enumerate(cands)}
    Y = np.zeros((len(pos_lists), len(cands)), dtype=np.float32)
    for i, p in enumerate(pos_lists):
        for idx in p:
            j = pos_of.get(idx)
            if j is not None:
                Y[i, j] = 1.0
    return torch.from_numpy(Y)


def train(min_freq: int = 1) -> dict:
    cfg, cfg_hash = load_config()
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    rng = np.random.default_rng(seed)

    rows = gather_rows()
    if not rows:
        raise SystemExit("no teacher rows — run gen_teacher first")

    # --- the bank: no frequency floor -------------------------------------
    tau = float(cfg["vocab"].get("bank_dedup_tau", cfg["vocab"]["dedup_tau"]))
    linkage = str(cfg["vocab"].get("bank_linkage", "complete"))
    phrases, canonical = bank_mod.build_phrase_bank(
        rows, dedup_tau=tau, min_freq=min_freq, linkage=linkage)
    phrase2idx = {p: i for i, p in enumerate(phrases)}
    bank_mod.save_bank(phrases, canonical)

    # How much the old frequency floor would have thrown away, measured.
    floored, _ = bank_mod.build_phrase_bank(
        rows, dedup_tau=tau, min_freq=int(cfg["vocab"]["min_freq"]),
        linkage=linkage)
    print(f"[bank] {len(phrases)} phrases (no floor) vs {len(floored)} at "
          f"min_freq={cfg['vocab']['min_freq']} — the floor would discard "
          f"{100 * (1 - len(floored) / max(1, len(phrases))):.1f}%", flush=True)

    # --- features ----------------------------------------------------------
    X = d.context_features(rows)                       # (R, 769)
    P = bank_mod.phrase_features(phrases)              # (N, 384)
    pos = [bank_mod.positives_for_row(r, canonical, phrase2idx) for r in rows]

    splits = d.split_indices(len(rows), cfg["train"]["val_frac"],
                             cfg["train"]["test_frac"], seed)
    splits["fingerprint"] = d.rows_fingerprint(rows)
    splits["bank_sha"] = bank_mod.bank_sha(phrases)
    SPLITS_PATH.write_text(json.dumps(splits, indent=2))

    tr, va = splits["train"], splits["val"]
    print(f"[train] rows={len(rows)} bank={len(phrases)} cfg={cfg_hash} "
          f"train={len(tr)} val={len(va)} test={len(splits['test'])}", flush=True)

    Xt = torch.from_numpy(X)
    Pt = torch.from_numpy(P)

    model = BiEncoderRanker(
        context_dim=int(cfg["model"]["context_dim"]),
        phrase_dim=P.shape[1],
        hidden_dim=int(cfg["model"].get("hidden_dim", 256)),
        proj_dim=int(cfg["model"].get("proj_dim", 128)),
        dropout=float(cfg["model"]["dropout"]),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]),
                            weight_decay=float(cfg["train"]["weight_decay"]))
    lossf = nn.BCEWithLogitsLoss()
    n_random = int(cfg["train"].get("n_random_negatives", 128))

    best = float("inf")
    best_state = None
    bad = 0
    t0 = time.time()

    for epoch in range(int(cfg["train"]["epochs"])):
        model.train()
        tot = 0.0
        nb = 0
        for b in _batches(len(tr), int(cfg["train"]["batch_size"]), rng):
            ridx = [tr[i] for i in b]
            pl = [pos[i] for i in ridx]
            if not any(pl):
                continue
            cands = _candidate_set(pl, len(phrases), n_random, rng)
            if not cands:
                continue
            scores = model(Xt[ridx], Pt[cands])
            loss = lossf(scores, _targets(pl, cands))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1

        # Validation scores against the FULL bank — no sampling, so the number
        # is comparable across epochs and not a function of the draw.
        model.eval()
        with torch.no_grad():
            vpl = [pos[i] for i in va]
            allc = list(range(len(phrases)))
            vscores = model(Xt[va], Pt)
            vloss = float(lossf(vscores, _targets(vpl, allc)))

        if vloss < best - 1e-5:
            best, bad = vloss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

        if epoch % 5 == 0 or bad >= int(cfg["train"]["patience"]):
            print(f"  epoch {epoch:3d} train={tot / max(1, nb):.4f} "
                  f"val={vloss:.4f} best={best:.4f}", flush=True)
        if bad >= int(cfg["train"]["patience"]):
            print(f"  early stop at epoch {epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({
        "state_dict": model.state_dict(),
        "context_dim": model.context_dim,
        "phrase_dim": model.phrase_dim,
        "proj_dim": model.proj_dim,
        "hidden_dim": int(cfg["model"].get("hidden_dim", 256)),
        "dropout": float(cfg["model"]["dropout"]),
        "bank_sha": bank_mod.bank_sha(phrases),
        "bank_size": len(phrases),
        "config_sha": cfg_hash,
        "val_loss": best,
    }, CKPT_PATH)

    print(f"[train] saved {CKPT_PATH.name} val_loss={best:.4f} "
          f"in {time.time() - t0:.1f}s", flush=True)
    return {"val_loss": best, "bank_size": len(phrases),
            "floored_size": len(floored)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-freq", type=int, default=1,
                    help="frequency floor for the bank; 1 = no floor (default)")
    args = ap.parse_args()
    train(min_freq=args.min_freq)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
