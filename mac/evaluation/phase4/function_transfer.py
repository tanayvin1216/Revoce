"""What survives distillation: communicative FUNCTION vs lexical CHOICE.

The Phase-4 gates already say two things about the distilled student on the
held-out split: it keeps 95.3% of the teacher's distinct-functions@7, and it
shares only 6.7% of the teacher's dynamic (non-core) words. This script breaks
that pair of numbers down PER communicative function, using exactly the same
instruments the gates use:

  * boards  — the teacher's cached held-out boards vs the student's boards
              built by the production `infer.student_board` (embed + MLP + MMR);
  * labels  — `evaluation.phase3.taxonomy.classify_functions` (nearest exemplar
              over the spec's nine functions), applied identically to both arms;
  * window  — the top-7 visible window (`metrics3.TOPK`), as every gate.

Per function f it reports
  presence   : fraction of held-out boards whose top-7 contains >=1 item of f
               (teacher and student), and the student/teacher ratio = retention;
  slots      : mean number of top-7 items labelled f per board, both arms;
  lexical    : of the teacher's DYNAMIC top-7 items labelled f, the fraction the
               student's FULL board also contains (exact lower-cased match) — the
               per-function analogue of the gate's dynamic_overlap, which likewise
               scores the teacher's top-7 against the student's whole board.
Plus two paired, per-board agreement scores whose contrast is the paper's claim:
  function_jaccard : |F_s ∩ F_t| / |F_s ∪ F_t| over the SET of functions in each top-7;
  word_jaccard     : the same over the set of dynamic WORDS in each top-7.
Retention gets a 10,000-resample paired bootstrap CI on the presence margin,
the same construction as `evaluate._margin_ci`.

The overall rows reproduce the committed gate numbers (retention 0.953,
dynamic_overlap 0.067) so the breakdown is anchored to VERIFIED_NUMBERS.md.

    cd mac && VOICELINK_PERCEPTION=0 python3 evaluation/phase4/function_transfer.py

Writes evaluation/phase4/function_transfer.json and prints a markdown table.
The arithmetic lives in `transfer_table`, which takes plain boards and a
labeller so it can be unit-tested with no model (tests/test_function_transfer.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
MAC = HERE.parent.parent
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))

from evaluation.phase3 import metrics3, taxonomy  # noqa: E402
from evaluation.coverage_metrics import bootstrap_ci  # noqa: E402

OUT = HERE / "function_transfer.json"
METRIC_K = metrics3.TOPK
FUNCTIONS = list(taxonomy.FUNCTION_EXEMPLARS_V3.keys())

Labeller = Callable[[Sequence[str]], list[str]]


def _jaccard(a: set, b: set) -> float | None:
    if not a and not b:
        return None
    return len(a & b) / len(a | b)


def transfer_table(pairs: Sequence[tuple[list[str], list[str]]],
                   label: Labeller, core_words: Sequence[str],
                   k: int = METRIC_K, functions: Sequence[str] = FUNCTIONS,
                   n_boot: int = 10000) -> dict:
    """Pure arithmetic over (student_board, teacher_board) pairs.

    `label(words) -> list[str]` assigns one function per word; it is called once
    per board window. `core_words` are the pinned safety words, excluded from
    every lexical count (the gate's dynamic_overlap excludes them too) but kept
    in the function counts, exactly as distinct_functions@7 keeps them.
    """
    core = {w.lower() for w in core_words}
    n = len(pairs)
    # per-board indicators
    s_present = {f: [] for f in functions}
    t_present = {f: [] for f in functions}
    s_slots = {f: [] for f in functions}
    t_slots = {f: [] for f in functions}
    lex_hit = defaultdict(int)     # teacher dynamic items of f found in student top-7
    lex_tot = defaultdict(int)     # teacher dynamic items of f
    s_distinct, t_distinct = [], []
    fn_jacc, word_jacc = [], []
    dyn_overlap, full_overlap = [], []
    # Positional confusion over the DYNAMIC slots of the top-k window: both
    # arms pin the same core words first, so slot i>len(core) in the teacher's
    # window is paired with slot i in the student's (same MMR rank). Counts
    # teacher function -> student function.
    n_core = len(core)
    confusion = {f: {g: 0 for g in functions} for f in functions}

    for sb, tb in pairs:
        s_top = [w for w in sb[:k]]
        t_top = [w for w in tb[:k]]
        s_lab = label(s_top) if s_top else []
        t_lab = label(t_top) if t_top else []
        s_set_f, t_set_f = set(s_lab), set(t_lab)
        s_distinct.append(len(s_set_f))
        t_distinct.append(len(t_set_f))
        s_words = {w.lower() for w in s_top}
        t_words = {w.lower() for w in t_top}
        s_dyn = {w for w in s_words if w not in core}
        t_dyn = {w for w in t_words if w not in core}
        # The gate's dynamic_overlap is lenient towards the student: the
        # teacher's top-7 dynamic words are looked up in the student's WHOLE
        # board (all k slots), not just its top-7. Reproduce that exactly.
        s_all_dyn = {w.lower() for w in sb if w.lower() not in core}
        j = _jaccard(s_set_f, t_set_f)
        if j is not None:
            fn_jacc.append(j)
        j = _jaccard(s_dyn, t_dyn)
        if j is not None:
            word_jacc.append(j)
        if t_dyn:
            dyn_overlap.append(len(s_all_dyn & t_dyn) / len(t_dyn))
        if t_words:
            full_overlap.append(len(s_words & t_words) / len(t_words))
        for f in functions:
            s_present[f].append(1.0 if f in s_set_f else 0.0)
            t_present[f].append(1.0 if f in t_set_f else 0.0)
            s_slots[f].append(float(sum(1 for l in s_lab if l == f)))
            t_slots[f].append(float(sum(1 for l in t_lab if l == f)))
        for w, l in zip(t_top, t_lab):
            wl = w.lower()
            if wl in core:
                continue
            lex_tot[l] += 1
            if wl in s_all_dyn:
                lex_hit[l] += 1
        for i in range(n_core, min(len(t_lab), len(s_lab))):
            if t_top[i].lower() in core or s_top[i].lower() in core:
                continue
            if t_lab[i] in confusion and s_lab[i] in confusion[t_lab[i]]:
                confusion[t_lab[i]][s_lab[i]] += 1

    per_function = {}
    for f in functions:
        sp = float(np.mean(s_present[f])) if n else 0.0
        tp = float(np.mean(t_present[f])) if n else 0.0
        deltas = [s - t for s, t in zip(s_present[f], t_present[f])]
        lo, hi = bootstrap_ci(deltas, n=n_boot) if n else (0.0, 0.0)
        per_function[f] = {
            "teacher_presence": round(tp, 4),
            "student_presence": round(sp, 4),
            "retention": round(sp / tp, 3) if tp else None,
            "presence_margin": round(float(np.mean(deltas)), 4) if n else None,
            "presence_margin_ci95": [round(lo, 4), round(hi, 4)],
            "teacher_slots_per_board": round(float(np.mean(t_slots[f])), 3) if n else 0.0,
            "student_slots_per_board": round(float(np.mean(s_slots[f])), 3) if n else 0.0,
            "teacher_dynamic_items": int(lex_tot[f]),
            "lexical_overlap": (round(lex_hit[f] / lex_tot[f], 3)
                                if lex_tot[f] else None),
        }

    sd = float(np.mean(s_distinct)) if n else 0.0
    td = float(np.mean(t_distinct)) if n else 0.0
    return {
        "n_heldout": n,
        "k": k,
        "overall": {
            "student_distinct_functions": round(sd, 3),
            "teacher_distinct_functions": round(td, 3),
            "retention_ratio": round(sd / td, 3) if td else None,
            "dynamic_overlap": round(float(np.mean(dyn_overlap)), 3) if dyn_overlap else None,
            "full_overlap_top7": round(float(np.mean(full_overlap)), 3) if full_overlap else None,
            "function_jaccard": round(float(np.mean(fn_jacc)), 3) if fn_jacc else None,
            "function_jaccard_ci95": [round(x, 3) for x in bootstrap_ci(fn_jacc, n=n_boot)] if fn_jacc else None,
            "word_jaccard": round(float(np.mean(word_jacc)), 3) if word_jacc else None,
            "word_jaccard_ci95": [round(x, 3) for x in bootstrap_ci(word_jacc, n=n_boot)] if word_jacc else None,
            "functions_with_zero_lexical_overlap": sorted(
                f for f in functions if lex_tot[f] and lex_hit[f] == 0),
        },
        "per_function": per_function,
        "confusion_dynamic_slots": confusion,
    }


def word_breakdown(pairs: Sequence[tuple[list[str], list[str]]], label: Labeller,
                   core_words: Sequence[str], vocab: Sequence[str],
                   canonical: dict[str, str], k: int = METRIC_K,
                   functions: Sequence[str] = FUNCTIONS, top: int = 8) -> dict:
    """Per function: the teacher's and student's DYNAMIC top-k words, and how
    many of the teacher's are representable in the student's label space
    (directly, or through the canonical-merge map). This is what separates a
    policy loss (words representable, policy withholds them) from a label-space
    loss (words not representable)."""
    from collections import Counter
    core = {w.lower() for w in core_words}
    vs = {w.lower() for w in vocab}
    tw = {f: Counter() for f in functions}
    sw = {f: Counter() for f in functions}
    for sb, tb in pairs:
        for board, bag in ((tb, tw), (sb, sw)):
            topw = list(board[:k])
            labs = label(topw) if topw else []
            for w, l in zip(topw, labs):
                if w.lower() not in core and l in bag:
                    bag[l][w.lower()] += 1
    out = {}
    for f in functions:
        items = tw[f]
        direct = sum(c for w, c in items.items() if w in vs)
        via = sum(c for w, c in items.items()
                  if w not in vs and canonical.get(w, w) in vs)
        out[f] = {
            "teacher_items": int(sum(items.values())),
            "teacher_distinct": len(items),
            "representable_direct": int(direct),
            "representable_via_canonical": int(via),
            "representable": int(direct + via),
            "teacher_top": items.most_common(top),
            "student_top": sw[f].most_common(top),
        }
    return out


def markdown(res: dict) -> str:
    o = res["overall"]
    lines = [
        f"n = {res['n_heldout']} held-out boards, top-{res['k']} window",
        "",
        "| function | teacher presence | student presence | retention | margin CI95 | "
        "teacher slots/board | student slots/board | teacher dyn items | lexical overlap |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for f, r in res["per_function"].items():
        ret = "—" if r["retention"] is None else f"{r['retention']:.3f}"
        lex = "—" if r["lexical_overlap"] is None else f"{r['lexical_overlap']:.3f}"
        ci = r["presence_margin_ci95"]
        lines.append(
            f"| {f} | {r['teacher_presence']:.3f} | {r['student_presence']:.3f} | {ret} | "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | {r['teacher_slots_per_board']:.2f} | "
            f"{r['student_slots_per_board']:.2f} | {r['teacher_dynamic_items']} | {lex} |")
    lines += [
        "",
        f"distinct-functions@7: student {o['student_distinct_functions']} vs teacher "
        f"{o['teacher_distinct_functions']} (retention {o['retention_ratio']})",
        f"dynamic_overlap {o['dynamic_overlap']} | full_overlap_top7 {o['full_overlap_top7']}",
        f"function-set Jaccard {o['function_jaccard']} {o['function_jaccard_ci95']} vs "
        f"word-set Jaccard {o['word_jaccard']} {o['word_jaccard_ci95']}",
        f"functions with zero lexical overlap: {o['functions_with_zero_lexical_overlap']}",
        "",
        "confusion over paired dynamic slots (rows: teacher function, cols: student function)",
    ]
    conf = res.get("confusion_dynamic_slots", {})
    if conf:
        funcs = list(conf.keys())
        lines.append("| teacher \\ student | " + " | ".join(funcs) + " | total |")
        lines.append("|---|" + "---|" * (len(funcs) + 1))
        for f in funcs:
            row = conf[f]
            lines.append(f"| {f} | " + " | ".join(str(row[g]) for g in funcs)
                         + f" | {sum(row.values())} |")
    return "\n".join(lines)


def _embedding_labeller() -> Labeller:
    """The production instrument: MiniLM embed -> nearest v3 exemplar."""
    from evaluation.embed import embed

    def label(words: Sequence[str]) -> list[str]:
        vecs = embed(list(words))
        return taxonomy.classify_functions(vecs)[1]
    return label


STUDENTS = ("v2", "v3", "prior", "random")


def _prior_board(train_rows: list[dict], core_words: list[str], k: int,
                 k_dyn: int) -> list[str]:
    """Teacher-frequency prior: the k_dyn most frequent DYNAMIC teacher words
    on the TRAINING split, pinned behind the core words. Same board for every
    row — the strongest context-free baseline."""
    from collections import Counter
    from evaluation.phase4 import data as d
    core = {w.lower() for w in core_words}
    c: Counter = Counter()
    for r in train_rows:
        for w in r["board"]:
            if w.lower() not in core:
                c[w.lower()] += 1
    return d.assemble_board(core_words, [w for w, _ in c.most_common(k_dyn)], k)


def board_builder(student: str = "v2", split_key: str = "test"):
    """(board_fn(row) -> board, held_rows, meta) for one student. Shared by
    collect_pairs (the breakdown) and gates_for (the Phase-4 gate code).

    student:
      v2     — the served closed-vocabulary MLP, built exactly as evaluate.py
               builds it (same checkpoint, infer.student_board, split guard);
      v3     — the open-vocabulary bi-encoder (evaluate_biencoder.load_v3,
               infer_biencoder.student_board). Same held-out rows: the test
               partition of splits_biencoder.json equals splits.json's. Same
               policy: infer_biencoder imports HEDGE_CAP and the answer reserve
               from infer.py;
      prior  — the teacher-frequency prior board (see _prior_board);
      random — core words + k_dyn words drawn from the v2 vocabulary, seed 0,
               a fresh draw per row.
    """
    import yaml
    from evaluation.phase4 import data as d
    from evaluation.phase4 import evaluate as ev

    if student not in STUDENTS:
        raise ValueError(f"student must be one of {STUDENTS}")
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    k = int(cfg["board"]["k"])
    k_dyn = int(cfg["board"]["k_dyn"])
    vocab, _ = d.load_vocab()
    core_words = list(d.GUARANTEED_CORE)
    held = ev._held_out_rows(split_key=split_key)
    meta = {"n_rows": len(held), "vocab": len(vocab), "k": k, "k_dyn": k_dyn,
            "core_words": core_words, "split": split_key, "student": student}

    if student == "v2":
        model, vocab, core_words, k_dyn = ev.load_student()
        meta.update(core_words=list(core_words), k_dyn=k_dyn)

        def board(r):
            return ev.student_board(model, vocab, r["scene"], r["transcript"],
                                    r.get("is_question", False), k, k_dyn, core_words)
    elif student == "v3":
        from evaluation.phase4 import evaluate_biencoder as evb
        from evaluation.phase4 import infer_biencoder as ib
        model, index, ck = evb.load_v3()
        meta["bank_sha"] = ck.get("bank_sha")

        def board(r):
            return ib.student_board(model, index, r.get("scene") or "",
                                    r.get("transcript") or "",
                                    r.get("is_question", False), k, k_dyn, core_words)
    elif student == "prior":
        train_rows = ev._held_out_rows(split_key="train")
        fixed = _prior_board(train_rows, core_words, k, k_dyn)
        meta["prior_board"] = fixed

        def board(r):
            return list(fixed)
    else:  # random
        rng = np.random.default_rng(0)
        pool = [w for w in vocab if w.lower() not in {c.lower() for c in core_words}]

        def board(r):
            pick = [pool[i] for i in rng.choice(len(pool), size=k_dyn, replace=False)]
            return d.assemble_board(core_words, pick, k)

    return board, held, meta


def collect_pairs(split_key: str = "test",
                  student: str = "v2") -> tuple[list[tuple[list[str], list[str]]], dict]:
    """(student_board, teacher_board) for every held-out row (see board_builder)."""
    board, held, meta = board_builder(student, split_key)
    pairs = [(list(board(r)), list(r["board"])) for r in held]
    return pairs, meta


def gates_for(student: str = "v2", split_key: str = "test") -> dict:
    """Run the UNCHANGED Phase-4 held-out gate code (evaluate.held_out_eval)
    over one student's boards, so a baseline board is scored by exactly the
    ruler the served student is scored by. No LLM: coverage-of-intent is the
    lexical/embedding rule of metrics3 against the authored acceptable moves."""
    from evaluation.phase4 import evaluate as ev
    board, held, meta = board_builder(student, split_key)

    def board_fn(scene, transcript, is_q):
        return board({"scene": scene, "transcript": transcript, "is_question": is_q})
    res = ev.held_out_eval(None, None, meta["k"], meta["k_dyn"], meta["core_words"],
                           board_fn=board_fn, split_key=split_key)
    dr, ag, cov = res["diversity_retention"], res["agreement"], res["coverage_of_intent"]
    return {
        "student": student,
        "distinct_functions": dr["student_distinct_functions"],
        "teacher_distinct_functions": dr["teacher_distinct_functions"],
        "retention_ratio": dr["retention_ratio"],
        "same_function_pair": dr["student_same_function_pair"],
        "teacher_same_function_pair": dr["teacher_same_function_pair"],
        "near_dup": dr["student_near_dup"],
        "board_variety": dr["board_variety"],
        "dynamic_overlap": ag["dynamic_overlap"],
        "coverage_dynamic": cov["student_dynamic"],
        "teacher_coverage_dynamic": cov["teacher_dynamic"],
        "coverage_margin": cov["margin"],
        "n_scored_dynamic": cov["n_scored_dynamic"],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--student", default="v2", choices=STUDENTS)
    p.add_argument("--out", default=None,
                   help="output JSON (default function_transfer.json for v2, "
                        "function_transfer_<student>.json otherwise)")
    p.add_argument("--gates", action="store_true",
                   help="score this student with the unchanged Phase-4 gate code "
                        "and write function_transfer_gates_<student>.json")
    p.add_argument("--words", action="store_true",
                   help="also print per-function teacher/student words and "
                        "representability counts (no JSON written)")
    args = p.parse_args()
    if args.out is None:
        args.out = str(OUT if args.student == "v2"
                       else HERE / f"function_transfer_{args.student}.json")
    if args.gates:
        g = gates_for(args.student, args.split)
        out = HERE / f"function_transfer_gates_{args.student}.json"
        out.write_text(json.dumps(g, indent=2))
        print(json.dumps(g, indent=2))
        print(f"wrote {out}")
        return 0
    pairs, meta = collect_pairs(args.split, args.student)
    if args.words:
        from evaluation.phase4 import data as d
        vocab, _ = d.load_vocab()
        wb = word_breakdown(pairs, _embedding_labeller(), meta["core_words"],
                            vocab, d.load_canonical())
        for f, r in wb.items():
            print(f"== {f}: teacher items {r['teacher_items']}, distinct "
                  f"{r['teacher_distinct']}, representable {r['representable']} "
                  f"({r['representable_direct']} direct + "
                  f"{r['representable_via_canonical']} via canonical)")
            print("   teacher:", r["teacher_top"])
            print("   student:", r["student_top"])
        return 0
    res = transfer_table(pairs, _embedding_labeller(), meta["core_words"])
    res["meta"] = {**meta, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "labeller": "taxonomy.classify_functions (MiniLM nearest exemplar)"}
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(markdown(res))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
