"""I-03 CHECK — how much dynamic-coverage credit comes from hedge PREDICTIONS?

The gate removes hedge TARGETS but not hedge predictions; a student board that
offers "I don't know" can still be credited if that entry matches a non-hedge
move lexically or at cosine >= 0.50. Recompute full@10 / dyn@6 dynamic coverage
with hedge predictions also removed, from the implementer's stored per-row
boards, using metrics3 directly.

    cd mac && VOICELINK_PERCEPTION=0 /usr/local/bin/python3 \
        ../paper/odi/revision/analyses/I-03_hedgecredit_check2.py
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

MAC = Path(__file__).resolve().parents[4] / "mac"
sys.path.insert(0, str(MAC))
os.environ.update(VOICELINK_PERCEPTION="0", VOICELINK_EMBED_BACKEND="torch", HF_HUB_OFFLINE="1")
from evaluation.phase3 import metrics3
from evaluation.coverage_metrics import bootstrap_ci
import numpy as np

HERE = Path(__file__).resolve().parent
V2 = json.loads((HERE / "I-03_numbers_implement_v2.json").read_text())
CORE_L = {"yes", "no", "i need help", "wait"}
rows = [r for r in V2["per_row_table2"] if r["window"] == "full10" and r["moves"]]
out = {}
for arm in ("student", "teacher"):
    base, nohedge, hedge_only_credit = [], [], 0
    for r in rows:
        honest = [m for m in r["moves"] if not metrics3.is_hedge(m)]
        if not honest:
            continue
        dyn = [w for w in r[f"{arm}_board"][:10] if w.lower() not in CORE_L]
        dyn_nh = [w for w in dyn if not metrics3.is_hedge(w)]
        a = int(metrics3.coverage_of_intent(dyn, honest, 10))
        b = int(metrics3.coverage_of_intent(dyn_nh, honest, 10)) if dyn_nh else 0
        base.append(a); nohedge.append(b)
        hedge_only_credit += int(a and not b)
    out[arm] = {"n": len(base), "dynamic_coverage": f"{sum(base)}/{len(base)}",
                "dynamic_coverage_hedge_predictions_removed": f"{sum(nohedge)}/{len(nohedge)}",
                "rows_credited_only_via_a_hedge_prediction": hedge_only_credit}
s = [int(metrics3.coverage_of_intent([w for w in r["student_board"][:10] if w.lower() not in CORE_L and not metrics3.is_hedge(w)],
                                     [m for m in r["moves"] if not metrics3.is_hedge(m)], 10)) for r in rows]
t = [int(metrics3.coverage_of_intent([w for w in r["teacher_board"][:10] if w.lower() not in CORE_L and not metrics3.is_hedge(w)],
                                     [m for m in r["moves"] if not metrics3.is_hedge(m)], 10)) for r in rows]
d = [x - 0.9 * y for x, y in zip(s, t)]
lo, hi = bootstrap_ci(d)
out["margin_0.9_hedge_predictions_removed"] = {"margin": round(float(np.mean(d)), 4), "ci": [round(lo, 4), round(hi, 4)]}
out["pairs_hedge_predictions_removed"] = {p: sum(f"{a}{b}" == p for a, b in zip(s, t)) for p in ("11", "01", "10", "00")}
p = HERE / "I-03_hedgecredit_check2.json"
assert not p.exists()
p.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
