"""Train the student (Phase-4 build order, step 3).

Distillation target: context_embedding -> multi-hot(teacher's diverse board) over a
fixed vocab, optimised with BCE. The teacher rows come from BOTH the Phase-3 boards
(already cached) and the growing teacher_data.jsonl, so training can start on the
120 we have and improve as the background gen adds more.

ML hygiene (CLAUDE.md §6): all hyperparams from config.yaml, every RNG seeded, the
config hashed and logged, the train/val/test split persisted. Writes student.pt +
vocab.json + splits.json and re-exports the TRAINED model to student.onnx.

    VOICELINK_PERCEPTION=0 python3 evaluation/phase4/train.py
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
MAC = HERE.parent.parent
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))

from evaluation.phase4 import data as d                 # noqa: E402
from evaluation.phase4.export_onnx import export_student  # noqa: E402
from evaluation.phase4.student import StudentRanker      # noqa: E402

CKPT = HERE / "student.pt"
PHASE3_BOARDS = MAC / "evaluation" / "phase3" / "boards.jsonl"
TEACHER_DATA = HERE / "teacher_data.jsonl"


def load_config() -> tuple[dict, str]:
    import yaml
    text = (HERE / "config.yaml").read_text()
    cfg = yaml.safe_load(text)
    cfg_hash = hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    return cfg, cfg_hash


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gather_rows() -> list[dict]:
    """Teacher boards from every available source, deduped by context."""
    rows: list[dict] = []
    for src in (PHASE3_BOARDS, TEACHER_DATA):
        if src.exists():
            rows.extend(d.load_teacher_rows(src))
    seen: set[str] = set()
    out: list[dict] = []
    for r in rows:
        key = d.context_text(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _batches(n: int, bs: int, rng: np.random.Generator):
    idx = rng.permutation(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def train() -> dict:
    cfg, cfg_hash = load_config()
    seed_everything(cfg["seed"])
    from rank_vocabulary import GUARANTEED_CORE

    rows = gather_rows()
    if len(rows) < 20:
        raise SystemExit(f"only {len(rows)} teacher rows — generate more first")
    vocab, word2idx = d.build_vocab(rows, cfg["vocab"]["min_freq"],
                                    cfg["vocab"]["max_size"], GUARANTEED_CORE)
    d.save_vocab(vocab)
    X, Y = d.make_xy(rows, word2idx)
    splits = d.split_indices(len(rows), cfg["train"]["val_frac"],
                             cfg["train"]["test_frac"], cfg["seed"])
    d.save_splits(splits)

    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Y)
    tr, va = splits["train"], splits["val"]

    model = StudentRanker(cfg["model"]["context_dim"], cfg["model"]["hidden_dim"],
                          len(vocab), cfg["model"]["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                           weight_decay=cfg["train"]["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(cfg["seed"])

    print(f"config_hash={cfg_hash}  rows={len(rows)}  vocab={len(vocab)}  "
          f"train={len(tr)} val={len(va)} test={len(splits['test'])}", flush=True)

    best_val, best_state = float("inf"), None
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        for b in _batches(len(tr), cfg["train"]["batch_size"], rng):
            sel = [tr[i] for i in b]
            opt.zero_grad()
            loss = loss_fn(model(Xt[sel]), Yt[sel])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xt[va]), Yt[va]).item() if va else float("nan")
        if va and vloss < best_val:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  val_bce={vloss:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({
        "state_dict": model.state_dict(),
        "context_dim": cfg["model"]["context_dim"],
        "hidden_dim": cfg["model"]["hidden_dim"],
        "vocab_size": len(vocab), "dropout": cfg["model"]["dropout"],
        "config_hash": cfg_hash, "n_rows": len(rows),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, CKPT)
    export_student(model)   # re-export the TRAINED weights to student.onnx
    print(f"saved {CKPT.name} + student.onnx  best_val_bce={best_val:.4f}", flush=True)
    return {"rows": len(rows), "vocab": len(vocab), "best_val_bce": best_val,
            "config_hash": cfg_hash}


if __name__ == "__main__":
    train()
