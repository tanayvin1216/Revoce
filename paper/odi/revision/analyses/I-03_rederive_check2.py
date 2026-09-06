"""I-03 CHECK (second checker pass) — four independent verifications.

  A. My re-run of the implementer's script (I-03_numbers_check2_rerun.json) must
     equal the authoritative I-03_numbers_implement_v2.json bit-for-bit apart from
     wall_seconds.
  B. Re-derive every Table 2 aggregate, paired count, margin/CI and the disputed
     dyn6 agreement number from the implementer's stored PER-ROW records using
     only numpy + bootstrap_ci (no evaluate.py / selftest.py code).
  C. Regenerate the student boards from the historical checkpoint via infer.py
     (unedited) and re-score full@10 / dyn@6 with metrics3 directly — a third
     derivation that never touches the edited harness files.
  D. Default-path regression: evaluate.held_out_eval() and selftest.score_boards()
     with NO window arguments must reproduce the committed historical result files
     (phase4_results.json, selftest_results.json).

    cd mac && VOICELINK_PERCEPTION=0 /usr/local/bin/python3 \
        ../paper/odi/revision/analyses/I-03_rederive_check2.py
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

MAC = Path(__file__).resolve().parents[4] / "mac"
sys.path.insert(0, str(MAC))
os.environ.update(VOICELINK_PERCEPTION="0", VOICELINK_EMBED_BACKEND="torch",
                  VOICELINK_STUDENT_BACKEND="torch", HF_HUB_OFFLINE="1")
import numpy as np
import torch
from evaluation.coverage_metrics import bootstrap_ci
from evaluation.phase3 import metrics3
from evaluation.embed import embed

HERE = Path(__file__).resolve().parent
V2 = json.loads((HERE / "I-03_numbers_implement_v2.json").read_text())
RERUN = json.loads((HERE / "I-03_numbers_check2_rerun.json").read_text())
PRIOR = json.loads((HERE / "I-03_check_numbers.json").read_text())
CORE = ["yes", "no", "I need help", "wait"]
CORE_L = {w.lower() for w in CORE}
report: dict = {}


def deep_diff(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: missing on one side")
            else:
                out += deep_diff(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: len {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += deep_diff(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and isinstance(b, float):
        if not (a == b or abs(a - b) <= 1e-12):
            out.append(f"{path}: {a} vs {b}")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


# ---------------------------------------------------------------- A
diffs = deep_diff({k: v for k, v in V2.items() if k != "wall_seconds"},
                  {k: v for k, v in RERUN.items() if k != "wall_seconds"})
report["A_rerun_matches_v2"] = {"n_diffs": len(diffs), "diffs": diffs[:20]}


# ---------------------------------------------------------------- B
def margin(s, t, factor=1.0, higher=True):
    d = [x - factor * y for x, y in zip(s, t)] if higher else [factor * y - x for x, y in zip(s, t)]
    lo, hi = bootstrap_ci(d)
    return {"margin": round(float(np.mean(d)), 4), "ci": [round(lo, 4), round(hi, 4)]}


B = {}
for name in ("full10", "dyn6", "full7"):
    rows = [r for r in V2["per_row_table2"] if r["window"] == name]
    assert len(rows) == 84
    k = {"full10": 10, "dyn6": 6, "full7": 7}[name]
    s = {m: [r["student"][m] for r in rows] for m in ("df", "near_dup", "sfp")}
    t = {m: [r["teacher"][m] for r in rows] for m in ("df", "near_dup", "sfp")}
    sc = [r["student"]["dynamic_coverage"] for r in rows if r["student"]["dynamic_coverage"] is not None]
    tc = [r["teacher"]["dynamic_coverage"] for r in rows if r["teacher"]["dynamic_coverage"] is not None]
    pairs = {p: sum(f"{a}{b}" == p for a, b in zip(sc, tc)) for p in ("11", "01", "10", "00")}
    # agreement two ways: implementer rule (both windows already core-filtered /
    # student all-dynamic vs teacher non-core in window) and the prior checker's
    # dyn6 rule (teacher tb[:6] BEFORE stripping core).
    agree_impl, agree_prior = [], []
    for r in rows:
        sb_full, tb_full = r["student_board"], r["teacher_board"]
        if name == "dyn6":
            s_dyn = {w.lower() for w in r["student"]["evaluated_board"]}
            t_dyn = {w.lower() for w in r["teacher"]["evaluated_board"]}
            t_prior = {w.lower() for w in tb_full[:6] if w.lower() not in CORE_L}
            if t_prior:
                agree_prior.append(len({w.lower() for w in sb_full if w.lower() not in CORE_L} & t_prior) / len(t_prior))
        else:
            s_dyn = {w.lower() for w in sb_full if w.lower() not in CORE_L}
            t_dyn = {w.lower() for w in tb_full[:k] if w.lower() not in CORE_L}
        if t_dyn:
            agree_impl.append(len(s_dyn & t_dyn) / len(t_dyn))
    lens = sorted({(len(r["student"]["evaluated_board"]), len(r["teacher"]["evaluated_board"])) for r in rows})
    variety = len({tuple(sorted(w.lower() for w in r["student"]["evaluated_board"])) for r in rows}) / 84
    B[name] = {
        "student_df": float(np.mean(s["df"])), "teacher_df": float(np.mean(t["df"])),
        "retention": float(np.mean(s["df"]) / np.mean(t["df"])),
        "student_near_dup": float(np.mean(s["near_dup"])), "teacher_near_dup": float(np.mean(t["near_dup"])),
        "student_sfp": float(np.mean(s["sfp"])), "teacher_sfp": float(np.mean(t["sfp"])),
        "dyn_cov_student": f"{sum(sc)}/{len(sc)}", "dyn_cov_teacher": f"{sum(tc)}/{len(tc)}",
        "dyn_cov_ratio": sum(sc) / sum(tc), "pairs": pairs,
        "df_margin_0.9": margin(s["df"], t["df"], 0.9),
        "sfp_margin": margin(s["sfp"], t["sfp"], higher=False),
        "cov_margin_0.9": margin(sc, tc, 0.9),
        "cov_raw_diff": margin(sc, tc),
        "agreement_implementer_rule": round(float(np.mean(agree_impl)), 3),
        "agreement_prior_checker_rule": round(float(np.mean(agree_prior)), 3) if agree_prior else None,
        "board_variety": round(variety, 3),
        "evaluated_lengths_(student,teacher)": lens,
    }
    h = V2["table2"][name]["harness"]
    B[name]["matches_harness"] = (
        abs(B[name]["student_df"] - h["diversity_retention"]["student_distinct_functions"]) < 5e-4
        and abs(B[name]["teacher_sfp"] - h["diversity_retention"]["teacher_same_function_pair"]) < 5e-5
        and pairs == h["coverage_of_intent"]["paired_counts_student_teacher"]
        and B[name]["cov_margin_0.9"]["margin"] == h["coverage_of_intent"]["margin"]["margin"]
        and B[name]["cov_margin_0.9"]["ci"] == h["coverage_of_intent"]["margin"]["ci"]
        and B[name]["df_margin_0.9"]["ci"] == h["diversity_retention"]["distinct_functions_margin"]["ci"]
        and B[name]["sfp_margin"]["ci"] == h["diversity_retention"]["same_function_pair_margin"]["ci"]
        and B[name]["agreement_implementer_rule"] == h["agreement"]["dynamic_overlap"]
        and B[name]["board_variety"] == h["diversity_retention"]["board_variety"])
report["B_per_row_rederivation"] = B

# Table 1 from per-row records
B1 = {}
for name in ("full7", "full10", "dyn6", "dyn_all_remaining"):
    rows = [r for r in V2["per_row_table1"] if r["window"] == name]
    assert len(rows) == 240
    by = {a: [r for r in rows if r["arm"] == a] for a in ("diverse", "topk")}
    ids = [r["id"] for r in by["diverse"]]
    assert ids == [r["id"] for r in by["topk"]], "pairing order"
    B1[name] = {a: {m: float(np.mean([r[m] for r in by[a]])) for m in ("df", "near_dup", "sfp", "coverage", "length")}
                for a in by}
    B1[name]["df_gap"] = margin([r["df"] for r in by["diverse"]], [r["df"] for r in by["topk"]])
    B1[name]["cov_gap"] = margin([r["coverage"] for r in by["diverse"]], [r["coverage"] for r in by["topk"]])
    B1[name]["matches_v2"] = all(
        abs(B1[name][a][m] - V2["table1"][name]["arms"][a][m]) < 1e-12
        for a in by for m in ("df", "near_dup", "sfp", "coverage", "length"))
report["B_table1_per_row"] = B1

# Prior checker file vs v2 (the implementer's 76-quantity claim, re-checked)
prior_vs_v2 = []
for name, key in (("full7", "at7"), ("full10", "at10"), ("dyn6", "dyn6")):
    h = V2["table2"][name]["harness"]; p = PRIOR["table2"][key]
    checks = {
        "df": (p["student_df"], h["diversity_retention"]["student_distinct_functions"]),
        "sfp": (p["student_sfp"], h["diversity_retention"]["student_same_function_pair"]),
        "pairs": (p["coverage_dynamic"]["pairs_student_teacher"], h["coverage_of_intent"]["paired_counts_student_teacher"]),
        "cov_ci": (p["cov_margin_0.9"]["ci"], h["coverage_of_intent"]["margin"]["ci"]),
        "df_ci": (p["df_margin_0.9"]["ci"], h["diversity_retention"]["distinct_functions_margin"]["ci"]),
        "agreement": (p["dynamic_agreement"], h["agreement"]["dynamic_overlap"]),
    }
    prior_vs_v2.append({name: {c: (a == b) for c, (a, b) in checks.items()}})
report["prior_checker_vs_v2"] = prior_vs_v2


# ---------------------------------------------------------------- C
def score(board, moves, k):
    top = board[:k]
    vecs = embed(top) if top else embed([])
    df = metrics3.distinct_functions_at(board, k, board_vecs=vecs)
    nd = metrics3.near_dup_rate(vecs)
    sfp = metrics3.same_function_pair_rate(board, k, board_vecs=vecs)
    honest = [m for m in moves if not metrics3.is_hedge(m)]
    dyn = [w for w in top if w.lower() not in CORE_L]
    cov_dyn = (int(metrics3.coverage_of_intent(dyn, honest, k)) if honest else None)
    return df, nd, sfp, cov_dyn


from evaluation.phase4 import data as d, train as train_mod
from evaluation.phase4.infer import student_board
from evaluation.phase4.student import StudentRanker

ck = torch.load(MAC / "evaluation/phase4/student.pt", map_location="cpu", weights_only=False)
assert ck["config_hash"] == "d6ce2e56d50b"
model = StudentRanker(ck["context_dim"], ck["hidden_dim"], ck["vocab_size"], ck["dropout"])
model.load_state_dict(ck["state_dict"]); model.eval()
vocab, _ = d.load_vocab()
rows_all = train_mod.gather_rows()
splits = json.loads((MAC / "evaluation/phase4/splits.json").read_text())
assert splits["fingerprint"] == d.rows_fingerprint(rows_all)
held = [rows_all[i] for i in splits["test"]]
v2rows = {r["index"]: r for r in V2["per_row_table2"] if r["window"] == "full10"}
board_mismatch = 0
C = {}
for name, k, dyn in (("full10", 10, False), ("dyn6", 6, True)):
    acc = {"s": [], "t": []}
    for i, r in enumerate(held):
        sb = student_board(model, vocab, r["scene"], r["transcript"], r.get("is_question", False), 10, 6, CORE)
        if name == "full10" and sb != v2rows[i]["student_board"]:
            board_mismatch += 1
        moves = v2rows[i]["moves"]  # the implementer's stored moves (checked below)
        for arm, b in (("s", sb), ("t", r["board"])):
            bb = [w for w in b if w.lower() not in CORE_L][:k] if dyn else b
            acc[arm].append(score(bb, moves, k))
    sc = [x[3] for x in acc["s"] if x[3] is not None]; tc = [x[3] for x in acc["t"] if x[3] is not None]
    C[name] = {
        "student_df": float(np.mean([x[0] for x in acc["s"]])), "teacher_df": float(np.mean([x[0] for x in acc["t"]])),
        "student_sfp": float(np.mean([x[2] for x in acc["s"]])), "teacher_sfp": float(np.mean([x[2] for x in acc["t"]])),
        "teacher_near_dup": float(np.mean([x[1] for x in acc["t"]])),
        "dyn_cov": f"{sum(sc)}/{len(sc)} vs {sum(tc)}/{len(tc)}",
        "pairs": {p: sum(f"{a}{b}" == p for a, b in zip(sc, tc)) for p in ("11", "01", "10", "00")},
        "cov_margin_0.9": margin(sc, tc, 0.9),
    }
report["C_independent_regeneration"] = {"student_board_mismatches_vs_v2": board_mismatch, **C}
# moves lookup independence: the stored moves must equal a fresh lookup
from evaluation.phase3 import scengen
lookup = {}
for src in (scengen.load_scenarios(),
            [json.loads(l) for l in (MAC / "evaluation/phase4/v2_scenarios.jsonl").read_text().splitlines() if l.strip()]):
    for s in src:
        mv = [m for m in (s.get("acceptable_moves") or []) if m and m.strip()]
        if mv:
            lookup[(s.get("scene", "").strip(), s.get("transcript", "").strip())] = mv
moves_mismatch = sum(lookup.get((r["scene"].strip(), r["transcript"].strip()), []) != v2rows[i]["moves"]
                     for i, r in enumerate(held))
report["C_independent_regeneration"]["moves_mismatches_vs_v2"] = moves_mismatch
report["C_independent_regeneration"]["n_rows_with_moves"] = sum(bool(v2rows[i]["moves"]) for i in range(84))

# ---------------------------------------------------------------- D
from evaluation.phase4 import evaluate as e
from evaluation.phase3 import selftest
hist4 = json.loads((MAC / "evaluation/phase4/phase4_results.json").read_text())
model2, vocab2, core2, kd2 = e.load_student()
default_held = e.held_out_eval(model2, vocab2, 10, kd2, core2)
d_diffs = []
for sec in ("diversity_retention", "agreement", "coverage_of_intent"):
    for key, val in hist4[sec].items():
        if key == "need":
            continue
        if default_held[sec].get(key) != val:
            d_diffs.append(f"{sec}/{key}: hist={val} now={default_held[sec].get(key)}")
new_keys = {sec: sorted(set(default_held[sec]) - set(hist4[sec])) for sec in ("diversity_retention", "agreement", "coverage_of_intent")}
report["D_default_path_phase4"] = {"value_diffs_vs_historical": d_diffs, "keys_added_in_default_output": new_keys,
                                   "metric_window": default_held["metric_window"],
                                   "need_text_now": default_held["coverage_of_intent"]["need"],
                                   "need_text_hist": hist4["coverage_of_intent"]["need"]}
hist3 = json.loads((MAC / "evaluation/phase3/selftest_results.json").read_text())
default_sel = selftest.score_boards(selftest.load_boards())
report["D_default_path_phase3"] = {
    "overall_equal": default_sel["overall"] == hist3["overall"],
    "metric_window": default_sel["metric_window"],
    "keys_added": sorted(set(default_sel) - set(hist3)),
}

# pinned-set floors
from evaluation.phase3 import taxonomy
n_fn, labels = taxonomy.classify_functions(embed(CORE))
report["pinned"] = {"distinct_functions": n_fn, "labels": labels,
                    "sfp_floor_7": 1 / 21, "sfp_floor_10": 1 / 45}

out = HERE / "I-03_check2_numbers.json"
assert not out.exists()
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
