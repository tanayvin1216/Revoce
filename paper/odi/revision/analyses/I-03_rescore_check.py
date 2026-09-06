"""I-03 CHECK — independent per-row re-scoring of Tables 1 and 2 at @7, @10 and
dynamic-only @6 (historical student.pt, config_hash d6ce2e56d50b).

Deliberately does NOT import evaluate.py or selftest.py (both are being edited by
the I-03 implementer); it re-implements the windows directly over metrics3 so the
implementer's numbers are checked against an independent derivation.

    cd mac && VOICELINK_PERCEPTION=0 /usr/local/bin/python3 \
        ../paper/odi/revision/analyses/I-03_rescore_check.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

MAC = Path(__file__).resolve().parents[4] / "mac"
sys.path.insert(0, str(MAC))
OUT = Path(__file__).resolve().parent / "I-03_check_numbers.json"

from evaluation.embed import embed                                  # noqa: E402
from evaluation.coverage_metrics import bootstrap_ci                # noqa: E402
from evaluation.phase3 import metrics3, scengen                     # noqa: E402
from evaluation.phase4 import data as d, train as train_mod         # noqa: E402
from evaluation.phase4.infer import student_board                   # noqa: E402
from evaluation.phase4.student import StudentRanker                 # noqa: E402

CORE = ["yes", "no", "I need help", "wait"]
CORE_L = {w.lower() for w in CORE}


def strip_core(board):
    return [w for w in board if w.lower() not in CORE_L]


def score(board, moves, k):
    """(df, near_dup, sfp, coverage) over board[:k] — same metrics3 calls the harness uses."""
    top = board[:k]
    vecs = embed(top) if top else embed([])
    df = metrics3.distinct_functions_at(board, k, board_vecs=vecs)
    nd = metrics3.near_dup_rate(vecs)
    sfp = metrics3.same_function_pair_rate(board, k, board_vecs=vecs)
    cov = (metrics3.coverage_of_intent(board, moves, k, board_vecs=vecs)
           if moves else None)
    return df, nd, sfp, cov


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


# ---------------------------------------------------------------- Table 1
def table1():
    rows = [json.loads(l) for l in (MAC / "evaluation/phase3/boards.jsonl").read_text().splitlines() if l.strip()]
    out = {}
    for strat in ("diverse", "topk"):
        rs = [r for r in rows if r["strategy"] == strat]
        res = {"n": len(rs), "mean_len": mean([len(r["board"]) for r in rs])}
        for label, fn in (
            ("at7", lambda b: (b, 7)),
            ("at10", lambda b: (b, 10)),
            ("dyn6_equal_window", lambda b: (strip_core(b)[:6], 6)),
            ("dyn_all_remaining", lambda b: (strip_core(b), len(strip_core(b)))),
        ):
            dfs, nds, sfps, covs, lens = [], [], [], [], []
            for r in rs:
                b, k = fn(r["board"])
                df, nd, sfp, cov = score(b, r.get("acceptable_moves", []), k)
                dfs.append(df); nds.append(nd); sfps.append(sfp); lens.append(len(b[:k]))
                covs.append(1.0 if cov else 0.0)
            res[label] = {"df": mean(dfs), "near_dup": mean(nds), "sfp": mean(sfps),
                          "coverage": mean(covs), "mean_len": mean(lens)}
        out[strat] = res
    return out


# ---------------------------------------------------------------- Table 2
def load_student():
    ck = torch.load(MAC / "evaluation/phase4/student.pt", map_location="cpu", weights_only=False)
    m = StudentRanker(ck["context_dim"], ck["hidden_dim"], ck["vocab_size"], ck["dropout"])
    m.load_state_dict(ck["state_dict"]); m.eval()
    vocab, _ = d.load_vocab()
    return m, vocab, ck


def moves_lookup():
    lookup = {}
    sources = [scengen.load_scenarios()]
    v2 = MAC / "evaluation/phase4/v2_scenarios.jsonl"
    if v2.exists():
        sources.append([json.loads(l) for l in v2.read_text().splitlines() if l.strip()])
    for rs in sources:
        for s in rs:
            mv = [m for m in (s.get("acceptable_moves") or []) if m and m.strip()]
            if mv:
                lookup[(s.get("scene", "").strip(), s.get("transcript", "").strip())] = mv
    return lookup


def margin(s, t, factor=1.0, higher=True):
    deltas = [x - factor * y for x, y in zip(s, t)] if higher else [factor * y - x for x, y in zip(s, t)]
    lo, hi = bootstrap_ci(deltas)
    return {"margin": round(float(np.mean(deltas)), 4), "ci": [round(lo, 4), round(hi, 4)], "n": len(deltas),
            "verdict": "PASS" if lo > 0 else ("FAIL" if hi < 0 else "INCONCLUSIVE")}


def table2():
    model, vocab, ck = load_student()
    import yaml
    k_board = yaml.safe_load((MAC / "evaluation/phase4/config.yaml").read_text())["board"]["k"]
    k_dyn = ck.get("k_dyn", 6)
    rows = train_mod.gather_rows()
    splits = json.loads((MAC / "evaluation/phase4/splits.json").read_text())
    assert splits["fingerprint"] == d.rows_fingerprint(rows), "split fingerprint mismatch"
    held = [rows[i] for i in splits["test"]]
    lookup = moves_lookup()

    boards = []
    for r in held:
        sb = student_board(model, vocab, r["scene"], r["transcript"], r.get("is_question", False),
                           k_board, k_dyn, CORE)
        mv = lookup.get((r.get("scene", "").strip(), r.get("transcript", "").strip()))
        boards.append((sb, r["board"], mv))

    out = {"n_heldout": len(held), "config_hash": ck["config_hash"], "k_board": k_board, "k_dyn": k_dyn,
           "student_mean_len": mean([len(s) for s, _, _ in boards]),
           "teacher_mean_len": mean([len(t) for _, t, _ in boards])}

    for label, fn, kk in (("at7", lambda b: b, 7), ("at10", lambda b: b, 10),
                          ("dyn6", lambda b: strip_core(b)[:6], 6)):
        s_df, t_df, s_nd, t_nd, s_sfp, t_sfp = [], [], [], [], [], []
        s_cov, t_cov, s_cd, t_cd = [], [], [], []
        pairs = {"11": 0, "01": 0, "10": 0, "00": 0}
        distinct = set(); dyn_agree = []
        for sb, tb, mv in boards:
            sdf, snd, ssfp, _ = score(fn(sb), None, kk)
            tdf, tnd, tsfp, _ = score(fn(tb), None, kk)
            s_df.append(sdf); t_df.append(tdf); s_nd.append(snd); t_nd.append(tnd)
            s_sfp.append(ssfp); t_sfp.append(tsfp)
            distinct.add(tuple(sorted(w.lower() for w in fn(sb)[:kk])))
            # agreement: directional recall of teacher's non-core entries in the window
            t_dyn = {w.lower() for w in tb[:kk] if w.lower() not in CORE_L}
            s_dyn = {w.lower() for w in strip_core(sb)}  # all student dynamic words (harness rule)
            if t_dyn:
                dyn_agree.append(len(s_dyn & t_dyn) / len(t_dyn))
            if mv:
                # full-board coverage over the window
                s_cov.append(1.0 if metrics3.coverage_of_intent(fn(sb), mv, kk) else 0.0)
                t_cov.append(1.0 if metrics3.coverage_of_intent(fn(tb), mv, kk) else 0.0)
                honest = [m for m in mv if not metrics3.is_hedge(m)]
                if honest:
                    # dynamic coverage: non-core words inside the window, vs non-hedge moves
                    sd = [w for w in fn(sb)[:kk] if w.lower() not in CORE_L]
                    td = [w for w in fn(tb)[:kk] if w.lower() not in CORE_L]
                    sc = 1.0 if metrics3.coverage_of_intent(sd, honest, kk) else 0.0
                    tc = 1.0 if metrics3.coverage_of_intent(td, honest, kk) else 0.0
                    s_cd.append(sc); t_cd.append(tc)
                    pairs[f"{int(sc)}{int(tc)}"] += 1
        sd_, td_ = mean(s_df), mean(t_df)
        out[label] = {
            "student_df": round(sd_, 3), "teacher_df": round(td_, 3), "retention": round(sd_ / td_, 3),
            "student_near_dup": round(mean(s_nd), 4), "teacher_near_dup": round(mean(t_nd), 4),
            "student_sfp": round(mean(s_sfp), 4), "teacher_sfp": round(mean(t_sfp), 4),
            "board_variety": round(len(distinct) / len(held), 3),
            "df_margin_0.9": margin(s_df, t_df, 0.9),
            "sfp_margin": margin(s_sfp, t_sfp, higher=False),
            "coverage_full": {"student": round(mean(s_cov), 3), "teacher": round(mean(t_cov), 3), "n": len(s_cov)},
            "coverage_dynamic": {"student": round(mean(s_cd), 3), "teacher": round(mean(t_cd), 3), "n": len(s_cd),
                                 "ratio": round(mean(s_cd) / mean(t_cd), 3),
                                 "pairs_student_teacher": pairs},
            "cov_margin_0.9": margin(s_cd, t_cd, 0.9),
            "dynamic_agreement": round(mean(dyn_agree), 3),
        }
    return out


if __name__ == "__main__":
    git = subprocess.run(["git", "status", "--short", "mac/evaluation"], cwd=MAC.parent,
                         capture_output=True, text=True).stdout
    res = {"git_status_mac_evaluation": git.strip().splitlines(),
           "table1": table1(), "table2": table2()}
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
