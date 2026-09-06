"""I-02 CHECKER: independent re-derivation from the implementer's numbers.json.

Does NOT import or reuse the implementer's pool_switch / score / verifier. Run from mac/:
  VOICELINK_PERCEPTION=0 /usr/local/bin/python3 ../paper/odi/revision/analyses/I-02_independent_check.py \
      ../paper/odi/revision/analyses/I-02_pool_v1 ../paper/odi/revision/analyses/I-02_independent_check.json

Checks
 1. pool identity: every saved pool per arm is re-derived from scratch (own cosine, own train-only frequency
    counter, vocab slice, full vocab, own default_rng draws, own learned logits).
 2. board regeneration: cosine / full / frequency / random boards regenerated with the CHECKER's OWN monkeypatch of
    infer._candidate_pool (model=None) and compared with the saved boards; learned boards compared with the
    dependency artifacts (I-03 per-row table, I-05 boards.json).
 3. own scoring: DF / ND / SFP / hedges / dynamic coverage recomputed per row from the saved boards with metrics3
    only, using the paper's window definitions (evaluate.held_out_eval), compared with per_row and means.
 4. own paired bootstrap (independent implementation, seed 0 and seed 1) + exact discordant-pair counts for the
    headline coverage effects; count of intervals excluding zero; cells where cosine is worse than learned.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

MAC = Path(__file__).resolve().parents[4] / 'mac'
sys.path.insert(0, str(MAC))
os.environ.update(VOICELINK_PERCEPTION='0', HF_HUB_OFFLINE='1',
                  VOICELINK_EMBED_BACKEND='torch', VOICELINK_STUDENT_BACKEND='torch')
import numpy as np
import torch
from evaluation.phase4 import data as d, train, evaluate as e, infer
from evaluation.phase4.student import StudentRanker
from evaluation.phase3 import metrics3
from evaluation.embed import embed
from selection import build_query_vector

HERE = Path(__file__).resolve().parent
src = Path(sys.argv[1]); out = Path(sys.argv[2])
assert not out.exists(), 'refusing to overwrite'
data = json.loads((src / 'numbers.json').read_text())
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(1)
rows = train.gather_rows()
lookup = e._acceptable_moves_lookup()
report: dict = {}
WINDOWS = {'full10': (10, False), 'dyn6': (6, True), 'full7': (7, False)}


def own_counts(vocab, canonical, train_rows):
    """Own train-only canonical-aware multi-hot counter (core skipped)."""
    idx = {w.lower(): i for i, w in enumerate(vocab)}
    counts = np.zeros(len(vocab))
    for r in train_rows:
        hit = set()
        for w in r['board']:
            key = w.strip().lower()
            if key in d.CORE_SET:
                continue
            key = canonical.get(key, key)
            if key in idx:
                hit.add(idx[key])
        for i in hit:
            counts[i] += 1
    return counts


def own_boot(deltas, seed, n=10000):
    rng = np.random.default_rng(seed)
    arr = np.asarray(deltas, float)
    idx = rng.integers(0, len(arr), size=(n, len(arr)))
    means = arr[idx].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def own_score(board, moves, core, k, dyn):
    core_l = {c.lower() for c in core}
    if dyn:  # dynamic-only: strip core first, then truncate (held_out_eval)
        win = [w for w in board if w.lower() not in core_l][:k]
    else:
        win = board[:k]
    vecs = embed(win)
    df = metrics3.distinct_functions_at(win, k, board_vecs=vecs)
    nd = metrics3.near_dup_rate(vecs)
    sfp = metrics3.same_function_pair_rate(win, k, board_vecs=vecs)
    hedges = sum(metrics3.is_hedge(w) for w in win)
    honest = [m for m in moves if not metrics3.is_hedge(m)]
    dyn_words = e.dynamic_words(win, core, k=k)
    cov = int(metrics3.coverage_of_intent(dyn_words, honest, k)) if honest else None
    return dict(df=df, near_dup=nd, sfp=sfp, hedges=hedges, dynamic_coverage=cov)


for label, bundle in [('historical', e.HERE), ('I-05', HERE / 'I-05_split_first_v2')]:
    r = data[label]
    rep: dict = {}
    splits = json.loads((bundle / 'splits.json').read_text())
    assert r['test_indices'] == splits['test'] and len(splits['test']) == 84
    assert splits == json.loads(d.SPLITS_PATH.read_text())
    assert d.rows_fingerprint(rows) == splits['fingerprint'] == r['split_fingerprint']
    assert not (set(splits['train']) & set(splits['test'])) and not (set(splits['val']) & set(splits['test']))
    vocab = json.loads((bundle / 'vocab.json').read_text())
    assert vocab == r['vocab']
    canonical = d.load_canonical(bundle / 'canonical.json')
    held = [rows[i] for i in splits['test']]
    assert [x['board'] for x in held] == r['teacher_boards']
    moves = [lookup.get((x['scene'].strip(), x['transcript'].strip()), []) for x in held]
    assert moves == r['acceptable_moves']
    rep['n_rows_with_nonhedge_moves'] = sum(1 for m in moves if [x for x in m if not metrics3.is_hedge(x)])
    rep['n_rows_no_moves'] = sum(1 for m in moves if not m)

    ck = torch.load(bundle / 'student.pt', map_location='cpu', weights_only=False)
    core, kd = ck['guaranteed_core'], ck['k_dyn']
    assert ck['vocab_sha'] == d.vocab_sha(vocab) == r['vocab_sha']
    model = StudentRanker(ck['context_dim'], ck['hidden_dim'], ck['vocab_size'], ck['dropout'])
    model.load_state_dict(ck['state_dict']); model.eval()
    n = 3 * kd
    rep['vocab_size'] = len(vocab); rep['pool_n'] = n

    # ---- 1. pool identity -------------------------------------------------------------
    V = embed(vocab)
    cos_pools, learned_pools = [], []
    for x in held:
        q = build_query_vector(embed, x['scene'], x['transcript'], None)
        s = V @ q
        cos_pools.append([vocab[i] for i in np.argsort(-s, kind='quicksort')[:n]])
        feats = d.row_features(x['scene'], x['transcript'], x['is_question'])[None, :]
        with torch.no_grad():
            logits = model(torch.from_numpy(feats))[0].numpy()
        learned_pools.append([vocab[i] for i in np.argsort(-logits)[:n]])
    counts = own_counts(vocab, canonical, [rows[i] for i in splits['train']])
    assert counts.tolist() == r['training_multihot_counts']
    freq_pool = [vocab[i] for i in np.argsort(-counts, kind='quicksort')[:n]]
    cutoff = sorted(counts, reverse=True)[n - 1]
    rep['frequency_cutoff_count'] = int(cutoff)
    rep['frequency_entries_tied_at_cutoff'] = int((counts == cutoff).sum())
    rep['frequency_pool_in_vocab_order_first18'] = len(set(freq_pool) & set(vocab[:n]))
    rng = np.random.default_rng(0)
    rand_pools = [[vocab[int(i)] for i in rng.choice(len(vocab), size=n, replace=False)] for _ in held]
    expected = dict(learned=learned_pools, cosine=cos_pools, frequency=[freq_pool] * 84,
                    vocab_order=[vocab[:n]] * 84, full=[list(vocab)] * 84, random=rand_pools)
    rep['pool_identity'] = {arm: r['arms'][arm]['pools'] == expected[arm] for arm in expected}
    # test-context-independence: frequency / vocab_order / full pools are the same on every row
    rep['context_free_arms_constant'] = {arm: len({tuple(p) for p in r['arms'][arm]['pools']}) == 1
                                        for arm in ('frequency', 'vocab_order', 'full')}
    # every dynamic board entry comes from the pool or is an open-answer span (not from the vocab outside the pool)
    leak = {}
    for arm, v in r['arms'].items():
        bad = 0
        for p, b in zip(v['pools'], v['boards']):
            for w in b[4:]:
                if w not in p and w in vocab:
                    bad += 1
        leak[arm] = bad
    rep['dynamic_entries_from_vocab_outside_pool'] = leak

    # ---- 2. own monkeypatch regeneration ----------------------------------------------
    original = infer._candidate_pool
    regen = {}
    for arm in ('cosine', 'full', 'frequency', 'random'):
        it = iter(expected[arm])

        def my_pool(model_, words, scene, transcript, is_q, k_dyn, _it=it):
            assert model_ is None
            return list(next(_it))
        infer._candidate_pool = my_pool
        try:
            boards = [infer.student_board(None, vocab, x['scene'], x['transcript'], x['is_question'], 10, kd, core)
                      for x in held]
        finally:
            infer._candidate_pool = original
        regen[arm] = boards == r['arms'][arm]['boards']
    # learned: regenerated through the untouched production path with the model
    boards = [infer.student_board(model, vocab, x['scene'], x['transcript'], x['is_question'], 10, kd, core) for x in held]
    regen['learned_production_path'] = boards == r['arms']['learned']['boards']
    if label == 'I-05':
        ref = [b['student_board'] for b in json.loads((HERE / 'I-05_split_first_v2/boards.json').read_text())]
    else:
        ref = [b['student_board'] for b in json.loads((HERE / 'I-03_numbers_implement_v2.json').read_text())['per_row_table2'] if b['window'] == 'full10']
    regen['learned_equals_dependency_artifact'] = r['arms']['learned']['boards'] == ref
    rep['board_regeneration'] = regen
    assert all(len(b) == 10 and b[:4] == core for v in r['arms'].values() for b in v['boards'])

    # ---- 3. own scoring -----------------------------------------------------------------
    scoring = {}
    cells = {}
    for arm, v in r['arms'].items():
        for win, (k, dyn) in WINDOWS.items():
            per = [own_score(b, m, core, k, dyn) for b, m in zip(v['boards'], moves)]
            same = per == v['windows'][win]['per_row']
            means = {key: float(np.mean([p[key] for p in per if p[key] is not None])) for key in per[0]}
            mean_ok = all(abs(means[key] - v['windows'][win]['means'][key]) < 1e-9 for key in means)
            scoring[f'{arm}/{win}'] = dict(per_row_identical=same, means_identical=mean_ok)
            cov = [p['dynamic_coverage'] for p in per if p['dynamic_coverage'] is not None]
            cells[f'{arm}/{win}'] = dict(df=round(means['df'], 4), nd=round(means['near_dup'], 4), sfp=round(means['sfp'], 4),
                                         hedges=round(means['hedges'], 4), cov=f'{sum(cov)}/{len(cov)}')
    rep['scoring'] = scoring
    rep['cells'] = cells
    assert all(s['per_row_identical'] and s['means_identical'] for s in scoring.values())

    # ---- 4. own bootstrap + discordant pairs + multiplicity --------------------------------
    boot = {}
    n_excl = 0; n_total = 0; cosine_worse = []
    for arm in ('cosine', 'frequency', 'vocab_order', 'full', 'random'):
        for win in WINDOWS:
            a = r['arms'][arm]['windows'][win]['per_row']; b = r['arms']['learned']['windows'][win]['per_row']
            for key in ('df', 'sfp', 'dynamic_coverage'):
                x = np.array([p[key] for p in a if p[key] is not None], float)
                y = np.array([p[key] for p in b if p[key] is not None], float)
                dl = x - y
                got = r['arms'][arm]['windows'][win]['baseline_minus_learned'][key]
                ci0 = [round(c, 4) for c in own_boot(dl, 0)]
                ci1 = [round(c, 4) for c in own_boot(dl, 1)]
                ent = dict(margin=round(float(dl.mean()), 4), reported=got['margin'], reported_ci=got['ci'], own_ci_seed0=ci0, own_ci_seed1=ci1)
                if key == 'dynamic_coverage':
                    ent['baseline_only'] = int(((x == 1) & (y == 0)).sum()); ent['learned_only'] = int(((x == 0) & (y == 1)).sum())
                boot[f'{arm}/{win}/{key}'] = ent
                n_total += 1
                if got['ci'][0] > 0 or got['ci'][1] < 0:
                    n_excl += 1
                if arm == 'cosine':
                    worse = (dl.mean() < 0) if key != 'sfp' else (dl.mean() > 0)
                    if worse:
                        cosine_worse.append(f'{win}/{key} delta={dl.mean():.4f}')
    rep['paired_effects'] = boot
    rep['intervals_excluding_zero'] = f'{n_excl}/{n_total} (DF,SFP,coverage x 5 arms x 3 windows; uncorrected)'
    rep['cosine_worse_than_learned_point_estimate'] = cosine_worse
    rep['own_ci_seed0_matches_reported'] = all(v['own_ci_seed0'] == v['reported_ci'] and v['margin'] == v['reported'] for v in boot.values())
    # ND for cosine vs learned at full10 (near-duplicate increase)
    for arm in ('cosine', 'full'):
        a = r['arms'][arm]['windows']['full10']['per_row']
        rep[f'{arm}_full10_rows_with_neardup'] = int(sum(1 for p in a if p['near_dup'] > 0))
    rep['learned_full10_rows_with_neardup'] = int(sum(1 for p in r['arms']['learned']['windows']['full10']['per_row'] if p['near_dup'] > 0))
    report[label] = rep

# the two coverage windows are the same observation
for label in report:
    r = data[label]
    for arm in r['arms']:
        assert [p['dynamic_coverage'] for p in r['arms'][arm]['windows']['full10']['per_row']] == \
               [p['dynamic_coverage'] for p in r['arms'][arm]['windows']['dyn6']['per_row']]
report['full10_dyn6_coverage_identical_all_arms'] = True
with out.open('x') as f:
    json.dump(report, f, indent=1)
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk not in ('paired_effects', 'scoring', 'cells')}) for k, v in report.items()}, indent=1))
for label in ('historical', 'I-05'):
    print(label)
    for c, v in report[label]['cells'].items():
        print(' ', c, v)
    for key in ('cosine/full10/dynamic_coverage', 'cosine/full7/dynamic_coverage', 'full/full10/dynamic_coverage', 'random/full10/dynamic_coverage', 'frequency/full10/dynamic_coverage', 'cosine/full10/df', 'cosine/full10/sfp', 'cosine/dyn6/df', 'cosine/dyn6/sfp'):
        print(' ', key, report[label]['paired_effects'][key])
