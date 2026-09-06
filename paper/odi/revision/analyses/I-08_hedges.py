"""I-08 — recount the hedge statistics on the evaluated checkpoint (read-only analysis).

The paper (main.tex:477) quotes 2.1 hedges per top-seven slate / >=2 on 72.6 % of boards
(direct score-based selection) and 0.012 vs teacher 0.321 hedges per board. Those four
numbers come from docs/claude/working-notes/phase5-competition-ml/FINAL_REVIEW.md
(2026-07-03, the pre-dedup 94-word student). This script recounts them on whatever
checkpoint/vocabulary is passed (default: the artifacts the paper evaluates,
config_hash d6ce2e56d50b) for three arms and three windows.

Arms (same 4 pinned core entries first in every case):
  raw       — core + numpy.argsort(-logits)[:k_dyn] of the student head, no selector
              (the paper's "direct score-based selection");
  selector  — infer.student_board(...): pool 18 -> answer reservation -> MMR -> hedge cap
              (the shipped student policy);
  teacher   — the cached teacher board for the same held-out row (r["board"]).
Windows:
  @7    — board[:7]   (4 core + 3 dynamic; the Phase-3 metric window);
  @10   — board[:10]  (whole board);
  dyn@6 — the non-core entries of board[:10] (identical to @10 for hedge counts because
          no core entry is a hedge; reported so the equal-window I-03 tables can cite it).
Hedge membership is evaluation.hedges.is_hedge — the ONE list the cap and metrics3 share.

Nothing is written into mac/. Outputs go next to this script:
  I-08_hedges_summary[<tag>].json, I-08_hedges_per_row[<tag>].jsonl

Run (from mac/):
  VOICELINK_PERCEPTION=0 /usr/local/bin/python3 ../paper/odi/revision/analyses/I-08_hedges.py
Refresh on an I-05 checkpoint:
  ... I-08_hedges.py --checkpoint <new.pt> --vocab <new_vocab.json> --splits <splits.json> --tag _i05
"""

from __future__ import annotations

import argparse
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

from evaluation.hedges import HEDGE_PHRASES, is_hedge  # noqa: E402
from evaluation.phase4 import data as d, evaluate as e, infer  # noqa: E402
from evaluation.phase4.student import StudentRanker  # noqa: E402

WINDOWS = ("@7", "@10", "dyn@6")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_student(ckpt: Path, vocab_path: Path):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = StudentRanker(ck["context_dim"], ck["hidden_dim"], ck["vocab_size"], ck["dropout"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    vocab, _ = d.load_vocab(vocab_path)
    assert len(vocab) == ck["vocab_size"], (len(vocab), ck["vocab_size"])
    return model, vocab, ck.get("guaranteed_core", []), int(ck.get("k_dyn", 6)), ck


def raw_board(model, vocab, core_words, scene, transcript, is_q, k, k_dyn) -> list[str]:
    feats = d.row_features(scene, transcript, is_q)[None, :]
    with torch.no_grad():
        logits = model(torch.from_numpy(feats))[0].numpy()
    top = np.argsort(-logits)[:k_dyn].tolist()
    return d.assemble_board(core_words, [vocab[i] for i in top], k)


def windows(board: list[str], core_lower: set[str]) -> dict[str, list[str]]:
    b10 = board[:10]
    return {"@7": board[:7], "@10": b10,
            "dyn@6": [w for w in b10 if w.lower() not in core_lower]}


def hedge_count(entries: list[str]) -> int:
    return sum(1 for w in entries if is_hedge(w))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=e.CKPT)
    ap.add_argument("--vocab", type=Path, default=d.VOCAB_PATH)
    ap.add_argument("--canonical", type=Path, default=d.CANONICAL_PATH)
    ap.add_argument("--splits", type=Path, default=d.SPLITS_PATH)
    ap.add_argument("--split-key", default="test")
    ap.add_argument("--tag", default="", help="suffix for the output files (e.g. _i05)")
    args = ap.parse_args()

    t0 = time.time()
    model, vocab, core_words, k_dyn, ck = load_student(args.checkpoint, args.vocab)
    k = 10
    core_lower = {w.lower() for w in core_words}
    rows = e._held_out_rows(args.splits, args.split_key)
    splits = json.loads(args.splits.read_text())

    # Hedge cluster inside the (deduplicated) vocabulary and its canonical map.
    vocab_hedges = [w for w in vocab if is_hedge(w)]
    canonical = json.loads(args.canonical.read_text()) if args.canonical.exists() else {}
    canon_into_hedge = {src: dst for src, dst in canonical.items() if is_hedge(dst)}

    arms = ("raw", "selector", "teacher")
    per_row: list[dict] = []
    totals = {a: {w: 0 for w in WINDOWS} for a in arms}
    ge2 = {a: {w: 0 for w in WINDOWS} for a in arms}
    ge1 = {a: {w: 0 for w in WINDOWS} for a in arms}
    phrases = {a: {} for a in arms}
    for r in rows:
        scene, transcript = r.get("scene", ""), r.get("transcript", "")
        is_q = bool(r.get("is_question", False))
        boards = {
            "raw": raw_board(model, vocab, core_words, scene, transcript, is_q, k, k_dyn),
            "selector": infer.student_board(model, vocab, scene, transcript, is_q, k, k_dyn, core_words),
            "teacher": list(r["board"]),
        }
        rec = {"id": r.get("id"), "category": r.get("category"), "is_question": is_q,
               "context_sha": hashlib.sha256(d.context_text(r).encode()).hexdigest()[:16],
               "boards": boards, "hedges": {}}
        for a in arms:
            wins = windows(boards[a], core_lower)
            rec["hedges"][a] = {}
            for wname, entries in wins.items():
                n = hedge_count(entries)
                rec["hedges"][a][wname] = n
                totals[a][wname] += n
                ge2[a][wname] += int(n >= 2)
                ge1[a][wname] += int(n >= 1)
                if wname == "@10":
                    for w in entries:
                        if is_hedge(w):
                            phrases[a][w] = phrases[a].get(w, 0) + 1
        per_row.append(rec)

    n = len(rows)
    summary = {
        "issue": "I-08",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=MAC,
                                   capture_output=True, text=True).stdout.strip(),
        "artifacts": {
            "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
            "checkpoint_config_hash": ck.get("config_hash"),
            "vocab": str(args.vocab), "vocab_sha256": sha256(args.vocab),
            "vocab_sha16": d.vocab_sha(vocab), "vocab_size": len(vocab),
            "canonical_sha256": sha256(args.canonical) if args.canonical.exists() else None,
            "splits": str(args.splits), "splits_fingerprint": splits.get("fingerprint"),
            "split_key": args.split_key, "n_rows": n,
            "teacher_data_sha256": sha256(d.HERE / "teacher_data.jsonl"),
        },
        "policy": {"HEDGE_CAP": infer.HEDGE_CAP, "HEDGE_CAP_WITH_OPEN": infer.HEDGE_CAP_WITH_OPEN,
                   "ANSWER_SLOTS": infer.ANSWER_SLOTS, "FILL_LAM": infer.FILL_LAM,
                   "OPEN_ANSWERS": infer.OPEN_ANSWERS, "CONTEXT_UNION": infer.CONTEXT_UNION,
                   "pool_n": min(len(vocab), infer._pool_mult() * k_dyn), "k": k, "k_dyn": k_dyn,
                   "core_words": core_words},
        "hedge_list": sorted(HEDGE_PHRASES),
        "vocab_hedge_entries": vocab_hedges,
        "canonical_sources_mapped_to_hedges": canon_into_hedge,
        "windows": {a: {w: {"total": totals[a][w], "mean": totals[a][w] / n,
                            "boards_ge2": ge2[a][w], "frac_ge2": ge2[a][w] / n,
                            "boards_ge1": ge1[a][w], "frac_ge1": ge1[a][w] / n}
                        for w in WINDOWS} for a in arms},
        "hedge_phrases_seen_at10": phrases,
        "historical_paper_numbers": {
            "source": "docs/claude/working-notes/phase5-competition-ml/FINAL_REVIEW.md (2026-07-03, "
                      "116->94-entry vocabulary, pre-tau=0.80-dedup-of-current-vocab student)",
            "raw_hedges_per_top7": 2.107, "raw_frac_ge2": 0.726,
            "selector_hedges_per_board": 0.012, "teacher_hedges_per_board": 0.321},
        "wall_clock_s": round(time.time() - t0, 1),
    }
    out_json = HERE / f"I-08_hedges_summary{args.tag}.json"
    out_rows = HERE / f"I-08_hedges_per_row{args.tag}.jsonl"
    out_json.write_text(json.dumps(summary, indent=2))
    out_rows.write_text("\n".join(json.dumps(r) for r in per_row) + "\n")

    print(f"n={n} ckpt={ck.get('config_hash')} vocab={len(vocab)} k_dyn={k_dyn} "
          f"wall={summary['wall_clock_s']}s")
    print(f"vocab hedge entries: {vocab_hedges}")
    print(f"{'arm':9s} {'window':7s} {'total':>5s} {'mean':>7s} {'>=2':>4s} {'frac>=2':>8s} {'>=1':>4s}")
    for a in arms:
        for w in WINDOWS:
            s = summary["windows"][a][w]
            print(f"{a:9s} {w:7s} {s['total']:5d} {s['mean']:7.3f} {s['boards_ge2']:4d} "
                  f"{s['frac_ge2']:8.3f} {s['boards_ge1']:4d}")
    print("wrote", out_json, out_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
