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
    ]
    return "\n".join(lines)


def _embedding_labeller() -> Labeller:
    """The production instrument: MiniLM embed -> nearest v3 exemplar."""
    from evaluation.embed import embed

    def label(words: Sequence[str]) -> list[str]:
        vecs = embed(list(words))
        return taxonomy.classify_functions(vecs)[1]
    return label


def collect_pairs(split_key: str = "test") -> tuple[list[tuple[list[str], list[str]]], dict]:
    """(student_board, teacher_board) for every held-out row, built exactly as
    evaluate.py builds them (same checkpoint, same infer.student_board, same
    split fingerprint guard)."""
    import yaml
    from evaluation.phase4 import evaluate as ev

    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    k = int(cfg["board"]["k"])
    model, vocab, core_words, k_dyn = ev.load_student()
    held = ev._held_out_rows(split_key=split_key)
    pairs = []
    for r in held:
        sb = ev.student_board(model, vocab, r["scene"], r["transcript"],
                              r.get("is_question", False), k, k_dyn, core_words)
        pairs.append((list(sb), list(r["board"])))
    meta = {"n_rows": len(held), "vocab": len(vocab), "k": k, "k_dyn": k_dyn,
            "core_words": list(core_words), "split": split_key}
    return pairs, meta


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--words", action="store_true",
                   help="also print per-function teacher/student words and "
                        "representability counts (no JSON written)")
    args = p.parse_args()
    pairs, meta = collect_pairs(args.split)
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
