"""Dynamic-only function agreement: the same three dynamic slots of the top-7
window that word agreement uses, so both axes are measured on the same items.
Also the dynamic-only distinct-functions retention. Test split, all four arms."""
import sys, json
import numpy as np
sys.path.insert(0, ".")
from evaluation.phase4 import function_transfer as ft
from evaluation.coverage_metrics import bootstrap_ci
label = ft._embedding_labeller()
out = {}
for st in ["v2", "v3", "prior", "random"]:
    pairs, meta = ft.collect_pairs("test", st)
    core = {w.lower() for w in meta["core_words"]}
    jf_full, jf_dyn, sd, td, sd_dyn, td_dyn = [], [], [], [], [], []
    for s_board, t_board in pairs:
        s7, t7 = s_board[:7], t_board[:7]
        ls, lt = label(s7), label(t7)
        ds = {l for w, l in zip(s7, ls) if w.lower() not in core}
        dt = {l for w, l in zip(t7, lt) if w.lower() not in core}
        j = ft._jaccard(set(ls), set(lt)); jf_full.append(j if j is not None else 0.0)
        j = ft._jaccard(ds, dt)
        if j is not None: jf_dyn.append(j)
        sd.append(len(set(ls))); td.append(len(set(lt))); sd_dyn.append(len(ds)); td_dyn.append(len(dt))
    def ci(x): lo, hi = bootstrap_ci([float(v) for v in x]); return [round(lo, 3), round(hi, 3)]
    out[st] = {"n": len(pairs), "JF_full": round(float(np.mean(jf_full)), 3), "JF_full_ci": ci(jf_full),
               "JF_dynamic": round(float(np.mean(jf_dyn)), 3), "JF_dynamic_ci": ci(jf_dyn), "n_dyn_scored": len(jf_dyn),
               "distinct_full": [round(float(np.mean(sd)), 3), round(float(np.mean(td)), 3), round(float(np.mean(sd)) / float(np.mean(td)), 3)],
               "distinct_dynamic": [round(float(np.mean(sd_dyn)), 3), round(float(np.mean(td_dyn)), 3), round(float(np.mean(sd_dyn)) / float(np.mean(td_dyn)), 3)]}
    print(st, json.dumps(out[st]))
json.dump(out, open(ft.ART / "function_transfer_dynamic.json", "w"), indent=1)
