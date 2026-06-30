"""Evaluate the student against the Phase-3 ruler (build order, step 4).

Three numbers, three bars — the "wrapper critique is dead" definition:

  (1) DIVERSITY RETENTION (make-or-break): student distinct-functions@7 >= 90% of
      the teacher's AND student near-duplicate rate <= 8%. Measured over the SAME
      Phase-3 scenario set, with the SAME metrics3 functions.
  (2) TEACHER-STUDENT AGREEMENT (supporting): top-7 set-overlap on HELD-OUT contexts
      (the persisted test split). PASS >= 70%.
  (3) SPEEDUP (the Intel story): end-to-end board latency, student vs teacher pipeline,
      same machine. PASS >= 10x.

Verdict rule (no wiggle): the wrapper critique is dead IFF (1) and (3) pass. (2) is
supporting evidence. A fast student that ranks worse is a worse product — say so.

    VOICELINK_PERCEPTION=0 python3 evaluation/phase4/evaluate.py --teacher-samples 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MAC = HERE.parent.parent
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))

from evaluation.phase3 import metrics3, scengen, selftest  # noqa: E402
from evaluation.phase4 import data as d, train as train_mod  # noqa: E402
from evaluation.phase4.student import StudentRanker          # noqa: E402

CKPT = HERE / "student.pt"
PHASE3_BOARDS = MAC / "evaluation" / "phase3" / "boards.jsonl"
REPORT = HERE / "PHASE4_REPORT.md"
METRIC_K = metrics3.TOPK  # 7


def load_student() -> tuple[StudentRanker, list[str]]:
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = StudentRanker(ck["context_dim"], ck["hidden_dim"], ck["vocab_size"],
                          ck["dropout"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    vocab, _ = d.load_vocab()
    return model, vocab


def student_board(model, vocab, scene, transcript, k) -> list[str]:
    from evaluation.embed import embed
    emb = torch.from_numpy(embed([f"{scene} || {transcript}".strip()]))
    idxs = model.board_indices(emb, k)
    return [vocab[i] for i in idxs]


def _df_and_dup(board: list[str]) -> tuple[int, float]:
    from evaluation.embed import embed
    top = board[:METRIC_K]
    vecs = embed(top) if top else embed([])
    return metrics3.distinct_functions_at(board, METRIC_K, board_vecs=vecs), \
        metrics3.near_dup_rate(vecs)


def diversity_retention(model, vocab, k) -> dict:
    """(1) student vs teacher distinct-functions + near-dup over Phase-3 scenarios."""
    scenarios = scengen.load_scenarios()
    teacher_by_id = {}
    if PHASE3_BOARDS.exists():
        for line in PHASE3_BOARDS.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("strategy") == "diverse":
                    teacher_by_id[r["id"]] = r["board"]

    s_df, s_dup, t_df, t_dup = [], [], [], []
    distinct_boards: set[tuple] = set()
    for sc in scenarios:
        sb = student_board(model, vocab, sc.get("scene", ""), sc.get("transcript", ""), k)
        distinct_boards.add(tuple(sorted(w.lower() for w in sb[:METRIC_K])))
        df, dup = _df_and_dup(sb)
        s_df.append(df)
        s_dup.append(dup)
        tb = teacher_by_id.get(sc["id"])
        if tb:
            tdf, tdup = _df_and_dup(tb)
            t_df.append(tdf)
            t_dup.append(tdup)

    sd, td = float(np.mean(s_df)), float(np.mean(t_df)) if t_df else float("nan")
    sdup = float(np.mean(s_dup))
    retention = sd / td if td else float("nan")
    # Degeneracy guard: a student that emits ONE diverse-looking board for every
    # context would still score high distinct-functions + low near-dup but be
    # useless. Report how many DISTINCT top-7 boards it actually produced.
    board_variety = len(distinct_boards) / len(scenarios) if scenarios else 0.0
    return {
        "student_distinct_functions": round(sd, 3),
        "teacher_distinct_functions": round(td, 3),
        "retention_ratio": round(retention, 3),
        "student_near_dup": round(sdup, 4),
        "teacher_near_dup": round(float(np.mean(t_dup)), 4) if t_dup else None,
        "distinct_student_boards": len(distinct_boards),
        "board_variety": round(board_variety, 3),
        "pass": bool(retention >= 0.90 and sdup <= 0.08),
        "need": "student distinct-fn >= 90% of teacher AND student near-dup <= 0.08",
    }


def agreement(model, vocab, k) -> dict:
    """(2) top-7 set-overlap with the teacher on HELD-OUT (test-split) contexts."""
    rows = train_mod.gather_rows()
    splits = json.loads(d.SPLITS_PATH.read_text())
    test = [i for i in splits.get("test", []) if i < len(rows)]
    overlaps = []
    for i in test:
        r = rows[i]
        sb = set(w.lower() for w in
                 student_board(model, vocab, r["scene"], r["transcript"], k)[:METRIC_K])
        tb = set(w.lower() for w in r["board"][:METRIC_K])
        if tb:
            overlaps.append(len(sb & tb) / len(tb))
    mean = float(np.mean(overlaps)) if overlaps else 0.0
    return {"set_overlap_top7": round(mean, 3), "n_heldout": len(overlaps),
            "pass": bool(mean >= 0.70), "need": "top-7 set-overlap >= 0.70"}


def speedup(model, vocab, k, teacher_samples: int) -> dict:
    """(3) end-to-end board latency: student (embed+MLP+topk) vs teacher (LLM+MMR)."""
    from evaluation.embed import embed
    scenarios = scengen.load_scenarios()

    # Student: warm, then time the full board path.
    _ = student_board(model, vocab, scenarios[0]["scene"], scenarios[0]["transcript"], k)
    s_times = []
    for sc in scenarios[:50]:
        t0 = time.time()
        emb = torch.from_numpy(embed([f"{sc.get('scene','')} || {sc.get('transcript','')}".strip()]))
        idxs = model.board_indices(emb, k)
        _ = [vocab[i] for i in idxs]
        s_times.append(time.time() - t0)
    student_ms = float(np.median(s_times)) * 1000

    # Teacher: time the real diverse seam on a small sample (one LLM load).
    t_times = []
    note = ""
    if teacher_samples > 0:
        try:
            from evaluation.phase1.gen_phase1 import MlxRanker
            ranker = MlxRanker()
            ranker._temp = 0.0
            for sc in scenarios[:teacher_samples]:
                t0 = time.time()
                selftest.run_seam(sc, "diverse", ranker)
                t_times.append(time.time() - t0)
        except Exception as e:  # noqa: BLE001
            note = f"teacher latency unmeasured ({e})"
    teacher_s = float(np.median(t_times)) if t_times else float("nan")
    ratio = (teacher_s * 1000 / student_ms) if (t_times and student_ms) else float("nan")
    return {
        "student_latency_ms": round(student_ms, 3),
        "teacher_latency_s": round(teacher_s, 2) if t_times else None,
        "speedup_x": round(ratio, 1) if t_times else None,
        "teacher_samples": len(t_times), "note": note,
        "pass": bool(t_times and ratio >= 10.0),
        "need": "student >= 10x faster than teacher",
    }


def write_report(res: dict) -> None:
    g1, g2, g3 = res["diversity_retention"], res["agreement"], res["speedup"]
    wrapper_dead = g1["pass"] and g3["pass"]
    lines = [
        "# Phase-4 distillation v1 — evaluation report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  |  "
        f"student rows={res['meta']['n_rows']} vocab={res['meta']['vocab']} "
        f"config_hash={res['meta']['config_hash']}",
        "",
        "## The three numbers",
        "",
        "| # | metric | student | teacher | bar | verdict |",
        "|---|--------|---------|---------|-----|---------|",
        f"| ① | distinct-functions@7 | {g1['student_distinct_functions']} | "
        f"{g1['teacher_distinct_functions']} | ≥90% retention | "
        f"{'PASS' if g1['pass'] else 'FAIL'} (ret {g1['retention_ratio']}) |",
        f"| ① | near-duplicate rate | {g1['student_near_dup']} | "
        f"{g1['teacher_near_dup']} | ≤0.08 | "
        f"{'PASS' if g1['student_near_dup'] <= 0.08 else 'FAIL'} |",
        f"| ① | distinct student boards (degeneracy guard) | "
        f"{g1.get('distinct_student_boards')} distinct "
        f"(variety {g1.get('board_variety')}) | — | not a global constant | — |",
        f"| ② | top-7 set-overlap (held-out) | {g2['set_overlap_top7']} | — | ≥0.70 | "
        f"{'PASS' if g2['pass'] else 'FAIL'} (n={g2['n_heldout']}) |",
        f"| ③ | speedup | {g3['student_latency_ms']} ms | "
        f"{g3['teacher_latency_s']} s | ≥10× | "
        f"{'PASS' if g3['pass'] else 'FAIL'} ({g3['speedup_x']}×) |",
        "",
        f"## Verdict: {'the wrapper critique is dead' if wrapper_dead else 'NOT crossed — iterate'}",
        "",
        "Rule: dead IFF ① diversity retention AND ③ speedup both pass; ② is supporting.",
        "",
    ]
    if wrapper_dead:
        lines.append(
            "> We distilled a slow LLM+MMR diverse-ranking pipeline into a small trained "
            f"model that retains {int(round(g1['retention_ratio']*100))}% of its "
            f"communicative diversity at a {g3['speedup_x']}× speedup.")
    elif g3["pass"] and not g1["pass"]:
        lines.append(
            "> The v1 student achieved the speedup but did NOT retain teacher diversity "
            "(mode-collapse). Iterating on diversity loss / more teacher data. Not a win.")
    else:
        lines.append("> v1 did not clear the bars; see numbers above. Honest finding, iterate.")
    lines += [
        "",
        "## Honesty note",
        "v1 is validated against TEACHER behaviour on SYNTHETIC scenarios; the teacher is "
        "itself an LLM (Qwen) over curated/synthetic contexts. A student that passes here "
        "may be imitating the teacher imitating GPT. Human/pilot validation on consented "
        "interaction logs is the next step and the number that ultimately matters.",
    ]
    if g3.get("note"):
        lines.append(f"\nLatency note: {g3['note']}")
    REPORT.write_text("\n".join(lines) + "\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-samples", type=int, default=8,
                   help="how many scenarios to time the real teacher on (0 = skip)")
    args = p.parse_args()

    import yaml
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    k = cfg["board"]["k"]
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model, vocab = load_student()

    res = {
        "meta": {"n_rows": ck.get("n_rows"), "vocab": ck.get("vocab_size"),
                 "config_hash": ck.get("config_hash")},
        "diversity_retention": diversity_retention(model, vocab, k),
        "agreement": agreement(model, vocab, k),
        "speedup": speedup(model, vocab, k, args.teacher_samples),
    }
    (HERE / "phase4_results.json").write_text(json.dumps(res, indent=2))
    write_report(res)

    g1, g3 = res["diversity_retention"], res["speedup"]
    bar = "=" * 74
    print("\n" + bar)
    print("VoiceLink Phase-4 — student vs teacher")
    print(bar)
    print(json.dumps(res, indent=2))
    print(bar)
    wrapper_dead = g1["pass"] and g3["pass"]
    print(f">>> WRAPPER CRITIQUE DEAD: {'YES' if wrapper_dead else 'NO'} "
          f"(①={'PASS' if g1['pass'] else 'FAIL'} ③={'PASS' if g3['pass'] else 'FAIL'}) <<<")
    print(bar)
    return 0 if wrapper_dead else 1


if __name__ == "__main__":
    raise SystemExit(main())
