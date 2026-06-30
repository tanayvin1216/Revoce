"""Evaluate the student against the Phase-3 ruler (build order, step 4).

Three numbers, three bars — the "wrapper critique is dead" definition:

  (1) DIVERSITY RETENTION (make-or-break): student distinct-functions@7 >= 90% of
      the teacher's AND student near-duplicate rate <= 8% AND board-variety >= 50%.
      Measured on the HELD-OUT test split with the SAME metrics3 functions.
  (2) TEACHER-STUDENT AGREEMENT (supporting, v2): dynamic_overlap = fraction of
      teacher's non-core top-7 words the student's dynamic predictions include.
      GATE >= 0.45. Also reports full_overlap_top7 for continuity.
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
REPORT = HERE / "PHASE4_REPORT.md"
METRIC_K = metrics3.TOPK  # 7
BOARD_VARIETY_FLOOR = 0.5
DYNAMIC_OVERLAP_GATE = 0.45


def load_student() -> tuple[StudentRanker, list[str], list[str], int]:
    """Returns (model, dynamic_vocab, core_words, k_dyn)."""
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = StudentRanker(ck["context_dim"], ck["hidden_dim"], ck["vocab_size"],
                          ck["dropout"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    vocab, _ = d.load_vocab()
    core_words = ck.get("guaranteed_core", [])
    k_dyn = ck.get("k_dyn", 6)
    return model, vocab, core_words, k_dyn


def student_board(model, vocab, scene, transcript, is_question, k, k_dyn, core_words) -> list[str]:
    """Assemble full board: core pinned first, student dynamic top-k_dyn after.

    Uses the v2.2 richer 769-d feature (separate scene/transcript embeds + is_question),
    the SAME builder training used, so train and inference see identical inputs."""
    emb = torch.from_numpy(d.row_features(scene, transcript, is_question)).unsqueeze(0)
    idxs = model.board_indices(emb, k_dyn)
    dynamic_words = [vocab[i] for i in idxs]
    return d.assemble_board(core_words, dynamic_words, k)


def _df_and_dup(board: list[str]) -> tuple[int, float]:
    from evaluation.embed import embed
    top = board[:METRIC_K]
    vecs = embed(top) if top else embed([])
    return metrics3.distinct_functions_at(board, METRIC_K, board_vecs=vecs), \
        metrics3.near_dup_rate(vecs)


def _held_out_rows() -> list[dict]:
    """Test-split rows with fingerprint guard against stale partitions."""
    rows = train_mod.gather_rows()
    splits = json.loads(d.SPLITS_PATH.read_text())
    fp = splits.get("fingerprint")
    if fp and fp != d.rows_fingerprint(rows):
        raise SystemExit(
            "splits.json fingerprint mismatch — teacher data changed since training; "
            "retrain before evaluating (otherwise the held-out split is not held-out).")
    return [rows[i] for i in splits.get("test", []) if i < len(rows)]


def held_out_eval(model, vocab, k, k_dyn, core_words) -> dict:
    """(1) diversity retention + (2) agreement on the held-out test split."""
    core_lower = {w.lower() for w in core_words}
    held = _held_out_rows()
    s_df, s_dup, t_df, t_dup = [], [], [], []
    full_overlaps: list[float] = []
    dynamic_overlaps: list[float] = []
    distinct_boards: set[tuple] = set()
    dynamic_triples: set[frozenset] = set()

    for r in held:
        sb = student_board(model, vocab, r["scene"], r["transcript"],
                           r.get("is_question", False), k, k_dyn, core_words)
        distinct_boards.add(tuple(sorted(w.lower() for w in sb[:METRIC_K])))

        sb_dynamic = [w for w in sb if w.lower() not in core_lower]
        if len(sb_dynamic) >= 3:
            dynamic_triples.add(frozenset(w.lower() for w in sb_dynamic[:3]))

        sdf, sdup = _df_and_dup(sb)
        tdf, tdup = _df_and_dup(r["board"])
        s_df.append(sdf); s_dup.append(sdup)
        t_df.append(tdf); t_dup.append(tdup)

        sset = {w.lower() for w in sb[:METRIC_K]}
        tset = {w.lower() for w in r["board"][:METRIC_K]}
        if tset:
            full_overlaps.append(len(sset & tset) / len(tset))

        teacher_dyn = {w.lower() for w in r["board"][:METRIC_K]
                       if w.lower() not in core_lower}
        student_dyn = {w.lower() for w in sb_dynamic}
        if teacher_dyn:
            dynamic_overlaps.append(len(student_dyn & teacher_dyn) / len(teacher_dyn))

    n = len(held)
    sd = float(np.mean(s_df)) if s_df else 0.0
    td = float(np.mean(t_df)) if t_df else float("nan")
    sdup_mean = float(np.mean(s_dup)) if s_dup else 0.0
    retention = sd / td if td else float("nan")
    variety = len(distinct_boards) / n if n else 0.0
    dyn_variety = len(dynamic_triples) / n if n else 0.0
    full_overlap = float(np.mean(full_overlaps)) if full_overlaps else 0.0
    dyn_overlap = float(np.mean(dynamic_overlaps)) if dynamic_overlaps else 0.0

    diversity_pass = bool(retention >= 0.90 and sdup_mean <= 0.08
                          and variety >= BOARD_VARIETY_FLOOR)
    return {
        "diversity_retention": {
            "n_heldout": n,
            "student_distinct_functions": round(sd, 3),
            "teacher_distinct_functions": round(td, 3),
            "retention_ratio": round(retention, 3),
            "student_near_dup": round(sdup_mean, 4),
            "teacher_near_dup": round(float(np.mean(t_dup)), 4) if t_dup else None,
            "distinct_student_boards": len(distinct_boards),
            "board_variety": round(variety, 3),
            "dynamic_variety": round(dyn_variety, 3),
            "pass": diversity_pass,
            "need": (f"HELD-OUT: distinct-fn >= 90% of teacher AND near-dup <= 0.08 "
                     f"AND board-variety >= {BOARD_VARIETY_FLOOR}"),
        },
        "agreement": {
            "full_overlap_top7": round(full_overlap, 3),
            "dynamic_overlap": round(dyn_overlap, 3),
            "n_heldout": len(dynamic_overlaps),
            "pass": bool(dyn_overlap >= DYNAMIC_OVERLAP_GATE),
            "need": f"HELD-OUT: dynamic_overlap >= {DYNAMIC_OVERLAP_GATE}",
        },
    }


def speedup(model, vocab, k, k_dyn, core_words, teacher_samples: int) -> dict:
    """(3) end-to-end board latency: student (embed+MLP+assemble) vs teacher."""
    from evaluation.embed import embed
    scenarios = scengen.load_scenarios()

    _ = student_board(model, vocab, scenarios[0]["scene"], scenarios[0]["transcript"],
                      scenarios[0].get("is_question", False), k, k_dyn, core_words)
    s_times = []
    for sc in scenarios[:50]:
        t0 = time.time()
        emb = torch.from_numpy(d.row_features(
            sc.get("scene", ""), sc.get("transcript", ""), sc.get("is_question", False))
        ).unsqueeze(0)
        idxs = model.board_indices(emb, k_dyn)
        d.assemble_board(core_words, [vocab[i] for i in idxs], k)
        s_times.append(time.time() - t0)
    student_ms = float(np.median(s_times)) * 1000

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
    t_iqr = ([round(float(np.percentile(t_times, 25)), 2),
              round(float(np.percentile(t_times, 75)), 2)] if t_times else None)
    return {
        "student_latency_ms": round(student_ms, 3),
        "student_latency_iqr_ms": [round(float(np.percentile(s_times, 25)) * 1000, 2),
                                   round(float(np.percentile(s_times, 75)) * 1000, 2)],
        "teacher_latency_s": round(teacher_s, 2) if t_times else None,
        "teacher_latency_iqr_s": t_iqr,
        "speedup_x": round(ratio, 1) if t_times else None,
        "teacher_samples": len(t_times), "student_samples": len(s_times), "note": note,
        "pass": bool(t_times and ratio >= 10.0),
        "need": "student >= 10x faster than teacher",
    }


def write_report(res: dict) -> None:
    g1, g2, g3 = res["diversity_retention"], res["agreement"], res["speedup"]
    wrapper_dead = g1["pass"] and g3["pass"]
    lines = [
        "# Phase-4 distillation v2 — evaluation report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  |  "
        f"student rows={res['meta']['n_rows']} vocab={res['meta']['vocab']} "
        f"config_hash={res['meta']['config_hash']}",
        "",
        f"## The three numbers (① and ② on the HELD-OUT split, n={g1['n_heldout']})",
        "",
        "| # | metric | student | teacher | bar | verdict |",
        "|---|--------|---------|---------|-----|---------|",
        f"| ① | distinct-functions@7 (held-out) | {g1['student_distinct_functions']} | "
        f"{g1['teacher_distinct_functions']} | ≥90% retention | "
        f"{'PASS' if g1['retention_ratio'] >= 0.90 else 'FAIL'} (ret {g1['retention_ratio']}) |",
        f"| ① | near-duplicate rate (held-out) | {g1['student_near_dup']} | "
        f"{g1['teacher_near_dup']} | ≤0.08 | "
        f"{'PASS' if g1['student_near_dup'] <= 0.08 else 'FAIL'} |",
        f"| ① | board-variety (degeneracy guard) | "
        f"{g1.get('distinct_student_boards')} distinct of {g1['n_heldout']} "
        f"(variety {g1.get('board_variety')}) | — | ≥{BOARD_VARIETY_FLOOR} | "
        f"{'PASS' if g1.get('board_variety', 0) >= BOARD_VARIETY_FLOOR else 'FAIL'} |",
        f"| ① | dynamic-variety (distinct triples) | {g1.get('dynamic_variety')} | — | — | — |",
        f"| ① | **combined** | — | — | all three above | "
        f"**{'PASS' if g1['pass'] else 'FAIL'}** |",
        f"| ② | dynamic_overlap (gate) | {g2['dynamic_overlap']} | — "
        f"| ≥{DYNAMIC_OVERLAP_GATE} | "
        f"{'PASS' if g2['pass'] else 'FAIL'} (n={g2['n_heldout']}) |",
        f"| ② | full_overlap_top7 (continuity) | {g2['full_overlap_top7']} | — | — | — |",
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
            "> The v2 student achieved the speedup but did NOT retain teacher diversity "
            "(mode-collapse). Iterating on diversity loss / more teacher data. Not a win.")
    else:
        lines.append("> v2 did not clear the bars; see numbers above. Honest finding, iterate.")
    lines += [
        "",
        "## Honesty note",
        "v2 is validated against TEACHER behaviour on SYNTHETIC scenarios; the teacher is "
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
    model, vocab, core_words, k_dyn = load_student()

    held = held_out_eval(model, vocab, k, k_dyn, core_words)
    res = {
        "meta": {"n_rows": ck.get("n_rows"), "vocab": ck.get("vocab_size"),
                 "config_hash": ck.get("config_hash")},
        "diversity_retention": held["diversity_retention"],
        "agreement": held["agreement"],
        "speedup": speedup(model, vocab, k, k_dyn, core_words, args.teacher_samples),
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
