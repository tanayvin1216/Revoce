"""Shipped vs corrected dynamic coverage-of-intent for every arm, both partitions.
Corrected = the shipped rule with pronoun contractions and bare negation/affirmation
added to the stop-list, so a shared "i'm" / "don't" / "not" is no longer a lexical hit."""
import sys, json, collections
sys.path.insert(0, ".")
import numpy as np
from evaluation import metrics as M
from evaluation.phase4 import function_transfer as ft, evaluate as ev
from evaluation.phase3 import metrics3
from evaluation.hedges import is_hedge
from evaluation.coverage_metrics import bootstrap_ci
from evaluation.phase4.evaluate import DYNAMIC_COVERAGE_GATE as FACTOR
EXTRA = {"i'm", "im", "i'll", "i've", "i'd", "it's", "that's", "don't", "dont", "doesn't", "didn't",
         "can't", "cant", "won't", "isn't", "not", "no", "yes", "m", "s", "t", "re", "ve", "ll", "d"}
BASE = set(M._STOP)
def set_rule(fixed: bool):
    M._STOP.clear(); M._STOP.update(BASE | EXTRA if fixed else BASE)
out = {}
for split in ["test", "val"]:
    moves_by_ctx = ev._acceptable_moves_lookup()
    arms = {}
    boards = {}
    held = None
    for st in ["v2", "v3", "prior", "random"]:
        board, held, meta = ft.board_builder(st, split)
        boards[st] = [list(board(r)) for r in held]
    core = {w.lower() for w in meta["core_words"]}
    boards["teacher"] = [list(r["board"]) for r in held]
    rows = []
    for i, r in enumerate(held):
        mv = moves_by_ctx.get((r["scene"].strip(), r["transcript"].strip()))
        if not mv: continue
        mv_nh = [m for m in mv if m and m.strip() and not is_hedge(m)]
        if mv_nh: rows.append((i, mv_nh))
    res = {"n_scored": len(rows)}
    for rule in ["shipped", "corrected"]:
        set_rule(rule == "corrected")
        cov = {}
        for st, bl in boards.items():
            cov[st] = [1.0 if metrics3.coverage_of_intent([w for w in bl[i][:7] if w.lower() not in core], mv, 7) else 0.0 for i, mv in rows]
        res[rule] = {}
        for st in boards:
            # Same relative gate as evaluate._margin_ci: student >= FACTOR * teacher.
            d = [s - FACTOR * t for s, t in zip(cov[st], cov["teacher"])]
            lo, hi = bootstrap_ci(d, n=10000) if st != "teacher" else (0.0, 0.0)
            res[rule][st] = {"coverage": round(float(np.mean(cov[st])), 3), "margin": round(float(np.mean(d)), 4),
                             "ci": [round(lo, 4), round(hi, 4)], "flags": cov[st],
                             "verdict": "n/a" if st == "teacher" else ("FAIL" if hi < 0 else ("PASS" if lo > 0 else "INCONCLUSIVE"))}
        # prior attribution under this rule
        att = collections.Counter()
        for i, mv in rows:
            dyn = [w for w in boards["prior"][i][:7] if w.lower() not in core]
            if metrics3.coverage_of_intent(dyn, mv, 7):
                hit = None
                for m in mv:
                    toks = M.content_tokens(m); mt = m.lower().strip()
                    for b in dyn:
                        if metrics3._lexical_hit(b, toks, mt): hit = f"lexical:{b}"; break
                    if hit: break
                att[hit or "semantic"] += 1
        res[rule]["prior_hit_attribution"] = dict(att)
    set_rule(False)
    out[split] = res
    print(split, json.dumps(res, indent=1))
json.dump(out, open(sys.argv[1], "w"), indent=1)
