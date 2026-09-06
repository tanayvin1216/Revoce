"""Paired student-minus-baseline contrasts on the same held-out rows: dynamic-only
function Jaccard and corrected coverage of intent (test split)."""
import sys, json
import numpy as np
sys.path.insert(0, ".")
from evaluation.phase4 import function_transfer as ft
from evaluation.coverage_metrics import bootstrap_ci
label = ft._embedding_labeller()
jf = {}
for st in ["v2", "v3", "prior", "random"]:
    pairs, meta = ft.collect_pairs("test", st); core = {w.lower() for w in meta["core_words"]}
    vals = []
    for s_board, t_board in pairs:
        s7, t7 = s_board[:7], t_board[:7]; ls, lt = label(s7), label(t7)
        ds = {l for w, l in zip(s7, ls) if w.lower() not in core}; dt = {l for w, l in zip(t7, lt) if w.lower() not in core}
        j = ft._jaccard(ds, dt); vals.append(0.0 if j is None else j)
    jf[st] = vals
cov = json.load(open(ft.ART / "coverage_rule_check.json"))["test"]["corrected"]
out = {}
for a, b in [("v2", "random"), ("v3", "random"), ("v2", "prior"), ("v3", "prior"), ("v3", "v2")]:
    d = [x - y for x, y in zip(jf[a], jf[b])]; lo, hi = bootstrap_ci(d)
    dc = [x - y for x, y in zip(cov[a]["flags"], cov[b]["flags"])]; lo2, hi2 = bootstrap_ci(dc)
    out[f"{a}-{b}"] = {"JF_dyn_diff": round(float(np.mean(d)), 3), "JF_dyn_ci": [round(lo, 3), round(hi, 3)],
                       "cov_corrected_diff": round(float(np.mean(dc)), 3), "cov_corrected_ci": [round(lo2, 3), round(hi2, 3)]}
    print(f"{a}-{b}", out[f"{a}-{b}"])
json.dump(out, open(ft.ART / "paired_contrasts.json", "w"), indent=1)
