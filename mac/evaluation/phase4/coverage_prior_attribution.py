"""Attribute the prior board's dynamic coverage-of-intent hits to (item, move, leg)."""
import sys, json, collections
sys.path.insert(0, ".")
from evaluation.phase4 import function_transfer as ft, evaluate as ev
from evaluation.phase3 import metrics3
from evaluation.hedges import is_hedge
from evaluation.embed import embed
board, held, meta = ft.board_builder("prior", "test")
core = {w.lower() for w in meta["core_words"]}
moves_by_ctx = ev._acceptable_moves_lookup()
leg = collections.Counter(); item_hits = collections.Counter(); examples = []
n = cov = 0
for r in held:
    mv = moves_by_ctx.get((r["scene"].strip(), r["transcript"].strip()))
    if not mv: continue
    mv_nh = [m for m in mv if m and m.strip() and not is_hedge(m)]
    dyn = [w for w in list(board(r))[:7] if w.lower() not in core]
    if not mv_nh: continue
    n += 1
    hit = metrics3.coverage_of_intent(dyn, mv_nh, 7)
    if not hit: continue
    cov += 1
    found = None
    for m in mv_nh:
        toks = metrics3.content_tokens(m); mt = m.lower().strip()
        for b in dyn:
            if metrics3._lexical_hit(b, toks, mt):
                found = ("lexical", b, m); break
        if found: break
    if not found:
        bv = embed(dyn); mvv = embed(mv_nh); sims = bv @ mvv.T
        i, j = divmod(int(sims.argmax()), sims.shape[1])
        found = ("semantic", dyn[i], mv_nh[j], round(float(sims[i, j]), 3))
    leg[found[0]] += 1; item_hits[found[1]] += 1
    if len(examples) < 12: examples.append(found)
print("n scored", n, "covered", cov, "rate", round(cov / n, 3))
print("by leg", dict(leg)); print("by item", dict(item_hits))
for e in examples: print("  ", e)
