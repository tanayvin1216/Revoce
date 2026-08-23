"""Train the student (Phase-4 build order, step 3).

v2 distillation target: context_embedding -> multi-hot(teacher's DYNAMIC board words)
over the dynamic vocab (core excluded). Loss = BCE with pos_weight + frequency
logit-adjustment + auxiliary listwise term. Early stopping on val loss.

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
import torch.nn.functional as F

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
TAU = 1.0  # logit-adjustment temperature


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
    """Teacher boards from every available source.

    Pipeline: exact-text dedup -> category cap (<=15%) -> embedding cosine dedup
    (>0.95 threshold). Applied before the split so the fingerprint reflects the
    final preprocessed set — evaluate.py compares against this fingerprint.
    """
    rows: list[dict] = []
    # TEACHER_DATA first so freshly-regenerated boards win the context dedup over
    # any stale duplicate in PHASE3_BOARDS (e.g. after an answer-mode prompt fix).
    for src in (TEACHER_DATA, PHASE3_BOARDS):
        if src.exists():
            rows.extend(d.load_teacher_rows(src))
    seen: set[str] = set()
    exact_deduped: list[dict] = []
    for r in rows:
        key = d.context_text(r)
        if key not in seen:
            seen.add(key)
            exact_deduped.append(r)
    capped = d.cap_categories(exact_deduped)
    return d.dedup_rows(capped)


def _batches(n: int, bs: int, rng: np.random.Generator):
    idx = rng.permutation(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def _pos_weight(Y: np.ndarray) -> torch.Tensor:
    """Per-class weight = clip(N_neg / N_pos, 1, 20) to up-weight rare positives."""
    n_pos = Y.sum(axis=0).clip(1e-9)
    n_neg = len(Y) - n_pos
    return torch.from_numpy(np.clip(n_neg / n_pos, 1.0, 20.0).astype(np.float32))


def _log_priors(Y: np.ndarray) -> torch.Tensor:
    """Log of per-class appearance rate — used for logit-adjustment at train time."""
    prior = np.maximum(Y.sum(axis=0) / max(len(Y), 1), 1e-9)
    return torch.from_numpy(np.log(prior).astype(np.float32))


def _distil_loss(logits: torch.Tensor, targets: torch.Tensor,
                 pw: torch.Tensor, log_p: torch.Tensor,
                 bce_fn: nn.BCEWithLogitsLoss) -> torch.Tensor:
    """BCE with logit-adjustment + auxiliary listwise term.

    Logit adjustment (tau * log prior) is applied ONLY here — never at inference
    or export — so the exported model is unmodified.
    """
    adj = logits - TAU * log_p
    bce = bce_fn(adj, targets)
    lsp = F.log_softmax(logits, dim=-1)
    n_pos = targets.sum(dim=-1, keepdim=True).clamp(1.0)
    l_list = ((-lsp * targets).sum(dim=-1) / n_pos.squeeze(-1)).mean()
    return bce + 0.3 * l_list


def train() -> dict:
    cfg, cfg_hash = load_config()
    seed_everything(cfg["seed"])
    from rank_vocabulary import GUARANTEED_CORE

    rows = gather_rows()
    if len(rows) < 20:
        raise SystemExit(f"only {len(rows)} teacher rows — generate more first")
    vocab, word2idx = d.build_vocab(rows, cfg["vocab"]["min_freq"],
                                    cfg["vocab"]["max_size"])
    # RC-2: collapse near-synonym clusters (the 7-hedge wall) to one column each;
    # dropped variants keep contributing label mass via the canonical map.
    canonical: dict[str, str] = {}
    dedup_tau = cfg["vocab"].get("dedup_tau")
    if dedup_tau:
        n_before = len(vocab)
        vocab, canonical = d.dedup_vocab(vocab, dedup_tau)
        word2idx = {w: i for i, w in enumerate(vocab)}
        print(f"vocab dedup tau={dedup_tau}: {n_before} -> {len(vocab)} "
              f"({len(canonical)} variants remapped)", flush=True)
    d.save_vocab(vocab)
    d.save_canonical(canonical)
    X, Y = d.make_xy(rows, word2idx, canonical)
    splits = d.split_indices(len(rows), cfg["train"]["val_frac"],
                             cfg["train"]["test_frac"], cfg["seed"])
    splits["fingerprint"] = d.rows_fingerprint(rows)
    d.save_splits(splits)

    # Device: VOICELINK_DEVICE or auto (hpu > cuda > mps > cpu). The MLP is tiny,
    # so this is about running the TRAINING stage on Intel Gaudi, not speed; the
    # checkpoint is always saved on CPU so every backend loads it.
    from ranker import torch_device
    dev = torch_device()
    print(f"training device: {dev}", flush=True)
    Xt = torch.from_numpy(X).to(dev)
    Yt = torch.from_numpy(Y).to(dev)
    tr, va = splits["train"], splits["val"]

    pw = _pos_weight(Y).to(dev)
    log_p = _log_priors(Y).to(dev)
    model = StudentRanker(cfg["model"]["context_dim"], cfg["model"]["hidden_dim"],
                          len(vocab), cfg["model"]["dropout"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                           weight_decay=cfg["train"]["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    rng = np.random.default_rng(cfg["seed"])

    print(f"config_hash={cfg_hash}  rows={len(rows)}  vocab={len(vocab)}  "
          f"train={len(tr)} val={len(va)} test={len(splits['test'])}", flush=True)

    patience = cfg["train"]["patience"]
    best_val, best_state, no_improve = float("inf"), None, 0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        for b in _batches(len(tr), cfg["train"]["batch_size"], rng):
            sel = [tr[i] for i in b]
            opt.zero_grad()
            _distil_loss(model(Xt[sel]), Yt[sel], pw, log_p, loss_fn).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = _distil_loss(
                model(Xt[va]), Yt[va], pw, log_p, loss_fn).item() if va else float("nan")
        if va:
            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                print(f"  early stop epoch {epoch}  best_val={best_val:.4f}", flush=True)
                break
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  val_loss={vloss:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu").eval()  # export + checkpoint on CPU regardless of training device

    torch.save({
        "state_dict": model.state_dict(),
        "context_dim": cfg["model"]["context_dim"],
        "hidden_dim": cfg["model"]["hidden_dim"],
        "vocab_size": len(vocab), "vocab_sha": d.vocab_sha(vocab),
        "dropout": cfg["model"]["dropout"],
        "config_hash": cfg_hash, "n_rows": len(rows),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "guaranteed_core": GUARANTEED_CORE,
        "k_dyn": cfg["board"]["k_dyn"],
    }, CKPT)
    export_student(model)
    print(f"saved {CKPT.name} + student.onnx  best_val={best_val:.4f}", flush=True)
    return {"rows": len(rows), "vocab": len(vocab), "best_val": best_val,
            "config_hash": cfg_hash}


if __name__ == "__main__":
    train()
