"""Run the phase-3/4 gates against the v3 bi-encoder student.

    VOICELINK_PERCEPTION=0 python -m evaluation.phase4.evaluate_biencoder

This deliberately calls evaluate.held_out_eval rather than reimplementing the
metrics. v3 is being compared against the teacher on the same held-out rows,
with the same distinct-functions / near-dup / same-function-pair / coverage
definitions and the same relative-to-teacher bars that v2 was judged by. A
separate metric implementation would make any v2-vs-v3 comparison meaningless.

Two things differ and both are declared, not hidden:

  * the split file (splits_biencoder.json, written by train_biencoder) — the
    corpus was regenerated, so the v2 partition no longer applies;
  * the board producer (infer_biencoder.student_board over the phrase bank
    instead of infer.student_board over the 94-word vocab).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluation.phase4 import bank as bank_mod
from evaluation.phase4 import data as d
from evaluation.phase4 import evaluate as ev
from evaluation.phase4 import infer_biencoder as ib
from evaluation.phase4.student_biencoder import BiEncoderRanker
from rank_vocabulary import GUARANTEED_CORE

HERE = Path(__file__).resolve().parent
CKPT = d.ART / "student_biencoder.pt"
SPLITS = d.ART / "splits_biencoder.json"
RESULTS = d.ART / "phase4_v3_results.json"


def load_v3():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = BiEncoderRanker(
        context_dim=ck["context_dim"], phrase_dim=ck["phrase_dim"],
        hidden_dim=ck["hidden_dim"], proj_dim=ck["proj_dim"],
        dropout=ck["dropout"])
    model.load_state_dict(ck["state_dict"])
    model.eval()

    phrases, _ = bank_mod.load_bank()
    stamped = ck.get("bank_sha")
    actual = bank_mod.bank_sha(phrases)
    if stamped and stamped != actual:
        raise SystemExit(
            f"bank sha mismatch: checkpoint was trained against {stamped}, "
            f"phrase_bank.json is {actual} — retrain before evaluating.")
    return model, ib.build_bank_index(model, phrases), ck


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--personal-context", action="store_true",
                    help="also feed the sample profile, so name candidates are live")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    k, k_dyn = cfg["board"]["k"], cfg["board"]["k_dyn"]

    model, index, ck = load_v3()
    profile = None
    if args.personal_context:
        p = HERE.parent.parent / "personal_context.sample.json"
        if p.exists():
            profile = json.loads(p.read_text())

    def board_fn(scene, transcript, is_q):
        return ib.student_board(model, index, scene or "", transcript or "",
                                is_q, k, k_dyn, list(GUARANTEED_CORE),
                                personal_context=profile)

    held = ev.held_out_eval(model, None, k, k_dyn, list(GUARANTEED_CORE),
                            board_fn=board_fn, splits_path=SPLITS)

    res = {
        "meta": {
            "model": "v3-biencoder",
            "bank_size": ck.get("bank_size"),
            "bank_sha": ck.get("bank_sha"),
            "config_sha": ck.get("config_sha"),
            "val_loss": round(float(ck.get("val_loss", float("nan"))), 5),
            "personal_context": bool(profile),
        },
        "diversity_retention": held["diversity_retention"],
        "agreement": held["agreement"],
        "coverage_of_intent": held["coverage_of_intent"],
    }
    RESULTS.write_text(json.dumps(res, indent=2))

    dr, cov = res["diversity_retention"], res["coverage_of_intent"]
    bar = "=" * 74
    print("\n" + bar)
    print(f"VoiceLink v3 bi-encoder — held-out gates (n={dr['n_heldout']})")
    print(bar)
    rows = [
        ("distinct-functions", dr["student_distinct_functions"],
         dr["teacher_distinct_functions"], ">= 90% of teacher",
         f"{dr['retention_ratio']:.3f}"),
        ("near-dup rate", dr["student_near_dup"], dr["teacher_near_dup"],
         "<= 0.08", ""),
        ("same-function-pair", dr["student_same_function_pair"],
         dr["teacher_same_function_pair"], "<= teacher", ""),
        ("board-variety", dr["board_variety"], "-", ">= 0.5", ""),
        ("coverage-of-intent (dyn)", cov["student_dynamic"],
         cov["teacher_dynamic"], ">= 90% of teacher", ""),
    ]
    print(f"{'gate':<28}{'v3':>10}{'teacher':>10}  {'bar':<20}{'ratio':>7}")
    for name, s, t, b, extra in rows:
        print(f"{name:<28}{str(s):>10}{str(t):>10}  {b:<20}{extra:>7}")
    print(bar)
    # The verdict is the interval-aware answer; "pass" is the old point-estimate
    # boolean, kept visible so a divergence between them is obvious.
    for label, g in (("diversity", dr), ("coverage", cov)):
        v = g.get("verdict", "?")
        pt = "pass" if g["pass"] else "fail"
        note = "  <- point estimate says %s" % pt if (v == "INCONCLUSIVE") else ""
        print(f"{label + ' gate':<16}: {v}{note}")
    print(bar)
    # Point estimates alone are how a gate gets called PASS on a margin of a
    # quarter of one scenario. These are the paired margins with 95% CIs.
    print("paired margin vs the bar (95% bootstrap CI over scenarios):")
    for label, m in (("coverage-of-intent", cov.get("margin")),
                     ("distinct-functions", dr.get("distinct_functions_margin")),
                     ("same-function-pair", dr.get("same_function_pair_margin"))):
        if not m or m.get("ci") is None:
            print(f"  {label:<22} not scored")
            continue
        lo, hi = m["ci"]
        print(f"  {label:<22} {m['margin']:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
              f"n={m['n']}  -> {m['verdict']}")
    print(bar)
    print(f"agreement (report-only): full={res['agreement']['full_overlap_top7']} "
          f"dynamic={res['agreement']['dynamic_overlap']}")
    print(bar)
    # Exit non-zero unless BOTH gates are resolved passes. An inconclusive gate
    # is not a green build — it is a measurement we cannot stand behind.
    return 0 if (dr.get("verdict") == "PASS"
                 and cov.get("verdict") == "PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
