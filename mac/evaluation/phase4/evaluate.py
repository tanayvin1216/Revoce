"""Evaluate the student against the Phase-3 ruler (build order, step 4).

Three numbers, three bars — the "wrapper critique is dead" definition:

  (1) DIVERSITY RETENTION (make-or-break): student distinct-functions@7 >= 90% of
      the teacher's AND student near-duplicate rate <= 8% AND same-function-pair
      rate <= 10% AND board-variety >= 50%. Measured on the HELD-OUT test split
      with the SAME metrics3 functions.
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
# Honest gates (INVESTIGATION_useless_boards.md §3): coverage is gated on the
# DYNAMIC words inside the top-7 (the pinned core self-certified 15.5% of the
# old passes), and redundancy on same-function pairs (near_dup's 0.85 threshold
# averaged the hedge wall away).
DYNAMIC_COVERAGE_GATE = 0.90
SFP_GATE = 0.10


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


try:
    # Track-1's MMR + answer-aware inference; the one board builder post-merge.
    from evaluation.phase4.infer import student_board  # noqa: E402
except ImportError:
    def student_board(model, vocab, scene, transcript, is_question, k, k_dyn,
                      core_words) -> list[str]:
        """Fallback (raw-topk) board assembly while infer.py does not exist:
        core pinned first, student dynamic top-k_dyn after.

        Uses the v2.2 richer 769-d feature (separate scene/transcript embeds +
        is_question), the SAME builder training used, so train and inference see
        identical inputs."""
        emb = torch.from_numpy(
            d.row_features(scene, transcript, is_question)).unsqueeze(0)
        idxs = model.board_indices(emb, k_dyn)
        dynamic = [vocab[i] for i in idxs]
        return d.assemble_board(core_words, dynamic, k)


def dynamic_words(board: list[str], core_words: list[str],
                  k: int = METRIC_K) -> list[str]:
    """The non-core words inside the top-k window — what the model actually
    contributed to the visible board. The honest coverage gate scores THESE."""
    core_lower = {w.lower() for w in core_words}
    return [w for w in board[:k] if w.lower() not in core_lower]


def _margin_ci(student: list[float], teacher: list[float],
               factor: float = 1.0, higher_is_better: bool = True) -> dict:
    """Bootstrap CI on the PAIRED margin by which a relative gate is met.

    The phase-4 gates are all of the form "student >= factor * teacher" (or
    "student <= teacher" for same-function-pair). Reporting only the point
    estimate is how a gate gets called PASS on a margin of a quarter of one
    scenario — which is exactly what v3's coverage did. This computes the
    per-scenario margin and its 95% CI, so a pass whose interval straddles zero
    is reported as INCONCLUSIVE rather than as a pass.

    Returns margin (mean), ci_low, ci_high, and a verdict.
    """
    from evaluation.coverage_metrics import bootstrap_ci

    if not student or not teacher or len(student) != len(teacher):
        return {"margin": None, "ci": None, "verdict": "not-scored"}

    if higher_is_better:
        deltas = [s - factor * t for s, t in zip(student, teacher)]
    else:                                  # e.g. sfp: teacher - student >= 0
        deltas = [factor * t - s for s, t in zip(student, teacher)]

    lo, hi = bootstrap_ci(deltas)
    mean = float(np.mean(deltas))
    if lo > 0:
        verdict = "PASS"                   # reliably on the right side
    elif hi < 0:
        verdict = "FAIL"                   # reliably on the wrong side
    else:
        verdict = "INCONCLUSIVE"           # point estimate is not evidence
    return {"margin": round(mean, 4),
            "ci": [round(lo, 4), round(hi, 4)],
            "n": len(deltas),
            "verdict": verdict}


def _df_dup_sfp(board: list[str]) -> tuple[int, float, float]:
    """distinct-functions@7, near-dup rate, same-function-pair rate — one embed."""
    from evaluation.embed import embed
    top = board[:METRIC_K]
    vecs = embed(top) if top else embed([])
    return (metrics3.distinct_functions_at(board, METRIC_K, board_vecs=vecs),
            metrics3.near_dup_rate(vecs),
            metrics3.same_function_pair_rate(board, METRIC_K, board_vecs=vecs))


def _held_out_rows(splits_path: Path | None = None,
                   split_key: str = "test") -> list[dict]:
    """Held-out rows with fingerprint guard against stale partitions.

    split_key exists so hyperparameters can be tuned on "val" and the final
    number reported once on "test". Sweeping on the test split would make the
    reported gate a training number.
    """
    rows = train_mod.gather_rows()
    splits = json.loads(Path(splits_path or d.SPLITS_PATH).read_text())
    fp = splits.get("fingerprint")
    if fp and fp != d.rows_fingerprint(rows):
        raise SystemExit(
            "splits.json fingerprint mismatch — teacher data changed since training; "
            "retrain before evaluating (otherwise the held-out split is not held-out).")
    return [rows[i] for i in splits.get(split_key, []) if i < len(rows)]


def _acceptable_moves_lookup() -> dict[tuple, list[str]]:
    """(scene, transcript) -> acceptable_moves, from the authored scenario corpora.

    The held-out rows come from teacher_data (no moves stored), so we re-attach the
    authored acceptable_moves by matching context. This is what coverage-of-intent —
    the 'is the board GOOD' metric — scores against, instead of teacher word-match."""
    lookup: dict[tuple, list[str]] = {}
    sources = [scengen.load_scenarios()]
    v2 = HERE / "v2_scenarios.jsonl"
    if v2.exists():
        sources.append([json.loads(l) for l in v2.read_text().splitlines() if l.strip()])
    for rows in sources:
        for s in rows:
            moves = [m for m in (s.get("acceptable_moves") or []) if m and m.strip()]
            if moves:
                lookup[(s.get("scene", "").strip(), s.get("transcript", "").strip())] = moves
    return lookup


def held_out_eval(model, vocab, k, k_dyn, core_words,
                  board_fn=None, splits_path: Path | None = None,
                  split_key: str = "test") -> dict:
    """(1) diversity retention + (2) agreement + (4) coverage-of-intent, held-out.

    board_fn lets a DIFFERENT student be scored by exactly this gate code —
    it takes (scene, transcript, is_question) and returns a board. The v3
    bi-encoder uses it (see evaluate_biencoder.py). Reusing this function
    rather than reimplementing the metrics is the whole point: if v3 is
    compared to v2, it has to be the same ruler.
    """
    core_lower = {w.lower() for w in core_words}
    held = _held_out_rows(splits_path, split_key)
    moves_by_ctx = _acceptable_moves_lookup()
    if board_fn is None:
        def board_fn(scene, transcript, is_q):
            return student_board(model, vocab, scene, transcript, is_q,
                                 k, k_dyn, core_words)
    s_df, s_dup, t_df, t_dup = [], [], [], []
    s_sfp, t_sfp = [], []
    full_overlaps: list[float] = []
    dynamic_overlaps: list[float] = []
    s_cov: list[float] = []
    t_cov: list[float] = []
    s_cov_dyn: list[float] = []
    t_cov_dyn: list[float] = []
    distinct_boards: set[tuple] = set()
    dynamic_triples: set[frozenset] = set()

    for r in held:
        sb = board_fn(r["scene"], r["transcript"], r.get("is_question", False))
        moves = moves_by_ctx.get((r.get("scene", "").strip(), r.get("transcript", "").strip()))
        if moves:
            s_cov.append(1.0 if metrics3.coverage_of_intent(sb, moves, METRIC_K) else 0.0)
            t_cov.append(1.0 if metrics3.coverage_of_intent(r["board"], moves, METRIC_K) else 0.0)
            # The HONEST coverage: only the model's own words inside the top-7
            # window, scored against non-hedge moves (a hedge is never the
            # intent the gate should certify).
            honest_moves = [m for m in moves if not metrics3.is_hedge(m)]
            if honest_moves:
                s_cov_dyn.append(1.0 if metrics3.coverage_of_intent(
                    dynamic_words(sb, core_words), honest_moves, METRIC_K) else 0.0)
                t_cov_dyn.append(1.0 if metrics3.coverage_of_intent(
                    dynamic_words(r["board"], core_words), honest_moves, METRIC_K) else 0.0)
        distinct_boards.add(tuple(sorted(w.lower() for w in sb[:METRIC_K])))

        sb_dynamic = [w for w in sb if w.lower() not in core_lower]
        if len(sb_dynamic) >= 3:
            dynamic_triples.add(frozenset(w.lower() for w in sb_dynamic[:3]))

        sdf, sdup, ssfp = _df_dup_sfp(sb)
        tdf, tdup, tsfp = _df_dup_sfp(r["board"])
        s_df.append(sdf); s_dup.append(sdup); s_sfp.append(ssfp)
        t_df.append(tdf); t_dup.append(tdup); t_sfp.append(tsfp)

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
    ssfp_mean = float(np.mean(s_sfp)) if s_sfp else 0.0
    retention = sd / td if td else float("nan")
    variety = len(distinct_boards) / n if n else 0.0
    dyn_variety = len(dynamic_triples) / n if n else 0.0
    full_overlap = float(np.mean(full_overlaps)) if full_overlaps else 0.0
    dyn_overlap = float(np.mean(dynamic_overlaps)) if dynamic_overlaps else 0.0
    cov_dyn = float(np.mean(s_cov_dyn)) if s_cov_dyn else 0.0

    # Review-opus adjudication 1: the sfp gate is RELATIVE to the teacher on the
    # same full-top-7 window — ~0.048 of every board's sfp is a fixed
    # core-pinning artifact ("yes"/"wait" both classify `comment`) the model
    # cannot control, and the absolute 0.10 bar came from a wrong estimate
    # (measured teacher = 0.168). SFP_GATE remains the fallback when the
    # teacher arm is unavailable.
    tsfp_mean = float(np.mean(t_sfp)) if t_sfp else None
    sfp_bar = tsfp_mean if tsfp_mean is not None else SFP_GATE
    diversity_pass = bool(retention >= 0.90 and sdup_mean <= 0.08
                          and ssfp_mean <= sfp_bar
                          and variety >= BOARD_VARIETY_FLOOR)
    return {
        "diversity_retention": {
            "n_heldout": n,
            "student_distinct_functions": round(sd, 3),
            "teacher_distinct_functions": round(td, 3),
            "retention_ratio": round(retention, 3),
            "student_near_dup": round(sdup_mean, 4),
            "teacher_near_dup": round(float(np.mean(t_dup)), 4) if t_dup else None,
            "student_same_function_pair": round(ssfp_mean, 4),
            "teacher_same_function_pair": round(float(np.mean(t_sfp)), 4) if t_sfp else None,
            "distinct_student_boards": len(distinct_boards),
            "board_variety": round(variety, 3),
            "dynamic_variety": round(dyn_variety, 3),
            "pass": diversity_pass,
            # Point estimates above; these say whether the margin survives
            # resampling. A "pass" with an INCONCLUSIVE interval is a coin flip.
            "distinct_functions_margin": _margin_ci(s_df, t_df, factor=0.90),
            "same_function_pair_margin": _margin_ci(s_sfp, t_sfp,
                                                    higher_is_better=False),
            "need": (f"HELD-OUT: distinct-fn >= 90% of teacher AND near-dup <= 0.08 "
                     f"AND same-function-pair <= teacher's ({round(sfp_bar, 4)}) "
                     f"AND board-variety >= {BOARD_VARIETY_FLOOR}"),
        },
        # Review-opus: agreement is REPORT-ONLY. The student intentionally
        # diverges from the teacher's exact word choice (different pools, same
        # MMR discipline); coverage-of-intent is the product bar, so a word-
        # parroting gate would contradict the report's own thesis.
        "agreement": {
            "full_overlap_top7": round(full_overlap, 3),
            "dynamic_overlap": round(dyn_overlap, 3),
            "n_heldout": len(dynamic_overlaps),
            "pass": None,
            "need": "report-only (coverage-of-intent is the gated bar)",
        },
        "coverage_of_intent": {
            "student": round(float(np.mean(s_cov)), 3) if s_cov else None,
            "teacher": round(float(np.mean(t_cov)), 3) if t_cov else None,
            "n_scored": len(s_cov),
            # The gated number: dynamic-within-top-7 words vs non-hedge moves.
            # The full-board number above is reported but NOT gated — 15.5% of
            # its passes came from the pinned core matching a move (§3).
            "student_dynamic": round(cov_dyn, 3) if s_cov_dyn else None,
            "teacher_dynamic": round(float(np.mean(t_cov_dyn)), 3) if t_cov_dyn else None,
            "n_scored_dynamic": len(s_cov_dyn),
            # Review-opus adjudication 2: gate RELATIVE to the teacher — the
            # absolute 0.90 bar sits above the teacher's own dynamic coverage
            # (0.845), which no distilled student can be required to beat.
            "pass": bool(s_cov_dyn and t_cov_dyn
                         and cov_dyn >= DYNAMIC_COVERAGE_GATE
                         * float(np.mean(t_cov_dyn))),
            "margin": _margin_ci(s_cov_dyn, t_cov_dyn,
                                 factor=DYNAMIC_COVERAGE_GATE),
            "need": (f"HELD-OUT: student DYNAMIC top-7 non-hedge coverage >= "
                     f"{DYNAMIC_COVERAGE_GATE:.0%} of the teacher's"),
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
        # Codex P3: time the SAME MMR/answer-aware path production serves —
        # raw board_indices() timing certified a path the device no longer runs.
        student_board(model, vocab, sc.get("scene", ""), sc.get("transcript", ""),
                      sc.get("is_question", False), k, k_dyn, core_words)
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
    g4 = res.get("coverage_of_intent", {})
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
        f"| ① | same-function-pair rate (held-out) | {g1['student_same_function_pair']} | "
        f"{g1['teacher_same_function_pair']} | ≤ teacher | "
        f"{'PASS' if g1['student_same_function_pair'] <= (g1['teacher_same_function_pair'] or SFP_GATE) else 'FAIL'} |",
        f"| ① | board-variety (degeneracy guard) | "
        f"{g1.get('distinct_student_boards')} distinct of {g1['n_heldout']} "
        f"(variety {g1.get('board_variety')}) | — | ≥{BOARD_VARIETY_FLOOR} | "
        f"{'PASS' if g1.get('board_variety', 0) >= BOARD_VARIETY_FLOOR else 'FAIL'} |",
        f"| ① | dynamic-variety (distinct triples) | {g1.get('dynamic_variety')} | — | — | — |",
        f"| ① | **combined** | — | — | all three above | "
        f"**{'PASS' if g1['pass'] else 'FAIL'}** |",
        f"| ② | dynamic_overlap (report-only) | {g2['dynamic_overlap']} | — "
        f"| — | — (n={g2['n_heldout']}) |",
        f"| ② | full_overlap_top7 (continuity) | {g2['full_overlap_top7']} | — | — | — |",
        f"| ③ | speedup | {g3['student_latency_ms']} ms | "
        f"{g3['teacher_latency_s']} s | ≥10× | "
        f"{'PASS' if g3['pass'] else 'FAIL'} ({g3['speedup_x']}×) |",
        f"| ④ | coverage-of-intent, full board (reported, NOT gated) | {g4.get('student')} | "
        f"{g4.get('teacher')} | — | — (n={g4.get('n_scored')}) |",
        f"| ④ | **coverage-of-intent, DYNAMIC (the real bar)** | **{g4.get('student_dynamic')}** | "
        f"{g4.get('teacher_dynamic')} | ≥{DYNAMIC_COVERAGE_GATE:.0%} of teacher | "
        f"{'PASS' if g4.get('pass') else 'FAIL'} (n={g4.get('n_scored_dynamic')}) |",
        "",
        "## The number that matters: DYNAMIC coverage-of-intent",
        "",
        f"Does the board the MODEL built give the user a way to express their intent? "
        f"Student **{g4.get('student_dynamic')}** vs teacher **{g4.get('teacher_dynamic')}** "
        f"(held-out, n={g4.get('n_scored_dynamic')}; non-core words in the top-7, scored "
        "against non-hedge acceptable moves). The full-board number "
        f"({g4.get('student')}) is reported for continuity but is inflated by the 4 "
        "pinned core words self-certifying (INVESTIGATION_useless_boards.md §3). "
        "Coverage — not `dynamic_overlap` word-parroting — is the right product "
        "question; the dynamic slice is the honest version of it.",
        "",
        f"## Verdict (literal ①+③ rule): {'PASS' if wrapper_dead else 'NOT crossed'}",
        "",
        "Rule: dead IFF ① diversity retention AND ③ speedup both pass; ② is supporting. "
        "We read ④ coverage-of-intent as the honest 'is the board good' signal.",
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
        "coverage_of_intent": held["coverage_of_intent"],
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
