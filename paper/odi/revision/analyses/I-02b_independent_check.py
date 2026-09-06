"""I-02b CHECKER: independent re-derivation from the implementer's I-02b numbers.json (three bundles).

Does NOT import or reuse the implementer's pool_switch / score / verifier. Run from mac/:
  VOICELINK_PERCEPTION=0 /usr/local/bin/python3 ../paper/odi/revision/analyses/I-02b_independent_check.py \
      ../paper/odi/revision/analyses/I-02b_pool_v2 ../paper/odi/revision/analyses/I-02b_independent_check.json

Checks (per bundle: historical, I-05, I-05b)
 1. pool identity: every saved pool per arm re-derived from scratch (own cosine, own train-only frequency counter,
    vocab slice, full vocab, own default_rng draws, own learned logits from the reloaded checkpoint).
 2. board regeneration: cosine / full / frequency / random / vocab_order boards regenerated with the CHECKER's OWN
    monkeypatch of infer._candidate_pool (model=None), compared with the saved boards; learned boards regenerated
    through the untouched production path and compared with the dependency artifacts.
 3. own scoring: DF / ND / SFP / hedges / dynamic coverage recomputed per row from the saved boards with metrics3 only,
    using held_out_eval's window rules, compared with per_row and means.
 4. own paired bootstrap (seed 0, seed 1) + exact discordant pairs + sign test for coverage; multiplicity counts.
 5. I-05b specifics: vocab overlap with historical, frequency tie at the cutoff, frequency/order overlap, hedge
    prevalence per arm, learned-minus-cosine contrasts as printed in the result table.
 6. cross-bundle continuity: historical and I-05 blocks equal I-02_pool_v1/numbers.json.
"""
from __future__ import annotations
import json, os, sys
from math import comb
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
BUNDLES = [('historical', e.HERE), ('I-05', HERE / 'I-05_split_first_v2'), ('I-05b', HERE / 'I-05b_split_first')]


def own_counts(vocab, canonical, train_rows):
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


def sign_test_p(a, b):
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def own_score(board, moves, core, k, dyn):
    core_l = {c.lower() for c in core}
    if dyn:
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


vocabs = {}
for label, bundle in BUNDLES:
    r = data[label]
    rep: dict = {}
    splits = json.loads((bundle / 'splits.json').read_text())
    assert r['test_indices'] == splits['test'] and len(splits['test']) == 84
    assert splits == json.loads(d.SPLITS_PATH.read_text())
    assert d.rows_fingerprint(rows) == splits['fingerprint'] == r['split_fingerprint']
    assert not (set(splits['train']) & set(splits['test'])) and not (set(splits['val']) & set(splits['test']))
    assert len(splits['train']) == 391 and len(splits['val']) == 84
    vocab = json.loads((bundle / 'vocab.json').read_text())
    assert vocab == r['vocab']
    vocabs[label] = vocab
    canonical = d.load_canonical(bundle / 'canonical.json')
    held = [rows[i] for i in splits['test']]
    assert [x['board'] for x in held] == r['teacher_boards']
    moves = [lookup.get((x['scene'].strip(), x['transcript'].strip()), []) for x in held]
    assert moves == r['acceptable_moves']
    rep['n_rows_with_nonhedge_moves'] = sum(1 for m in moves if [x for x in m if not metrics3.is_hedge(x)])
    rep['n_rows_no_moves'] = sum(1 for m in moves if not m)

    ck = torch.load(bundle / 'student.pt', map_location='cpu', weights_only=False)
    core, kd = ck['guaranteed_core'], ck['k_dyn']
    assert ck['vocab_sha'] == d.vocab_sha(vocab) == r['vocab_sha'] and ck['config_hash'] == r['config_hash']
    model = StudentRanker(ck['context_dim'], ck['hidden_dim'], ck['vocab_size'], ck['dropout'])
    model.load_state_dict(ck['state_dict']); model.eval()
    n = 3 * kd
    rep['config_hash'] = ck['config_hash']; rep['vocab_sha'] = r['vocab_sha']
    rep['vocab_size'] = len(vocab); rep['pool_n'] = n; rep['core'] = core

    # ---- 1. pool identity ------------------------------------------------------------------
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
    if label != 'historical':
        assert counts.tolist() == json.loads((bundle / 'preprocessing.json').read_text())['positive_counts']
    freq_pool = [vocab[i] for i in np.argsort(-counts, kind='quicksort')[:n]]
    cutoff = sorted(counts, reverse=True)[n - 1]
    rep['frequency_cutoff_count'] = int(cutoff)
    rep['frequency_entries_tied_at_cutoff'] = int((counts == cutoff).sum())
    rep['frequency_tied_entries_in_pool'] = int(sum(1 for w in freq_pool if counts[vocab.index(w)] == cutoff))
    rep['frequency_pool_overlap_with_vocab_order'] = len(set(freq_pool) & set(vocab[:n]))
    rng = np.random.default_rng(0)
    rand_pools = [[vocab[int(i)] for i in rng.choice(len(vocab), size=n, replace=False)] for _ in held]
    expected = dict(learned=learned_pools, cosine=cos_pools, frequency=[freq_pool] * 84,
                    vocab_order=[vocab[:n]] * 84, full=[list(vocab)] * 84, random=rand_pools)
    rep['pool_identity'] = {arm: r['arms'][arm]['pools'] == expected[arm] for arm in expected}
    rep['pool_sizes'] = {arm: sorted({len(p) for p in r['arms'][arm]['pools']}) for arm in expected}
    rep['context_free_arms_constant'] = {arm: len({tuple(p) for p in r['arms'][arm]['pools']}) == 1
                                        for arm in ('frequency', 'vocab_order', 'full')}
    # learned vs cosine pool overlap (how different are the two context-sensitive pools?)
    rep['learned_cosine_pool_overlap_mean'] = float(np.mean([len(set(a) & set(b)) for a, b in zip(learned_pools, cos_pools)]))
    leak = {}
    for arm, v in r['arms'].items():
        bad = 0
        for p, b in zip(v['pools'], v['boards']):
            for w in b[4:]:
                if w not in p and w in vocab:
                    bad += 1
        leak[arm] = bad
    rep['dynamic_entries_from_vocab_outside_pool'] = leak

    # ---- 2. own monkeypatch regeneration (same selector for every arm) ----------------------
    original = infer._candidate_pool
    regen = {}
    for arm in ('cosine', 'full', 'frequency', 'random', 'vocab_order'):
        it = iter(expected[arm])

        def my_pool(model_, words, scene, transcript, is_q, k_dyn, _it=it):
            assert model_ is None and words == vocab and k_dyn == kd
            return list(next(_it))
        infer._candidate_pool = my_pool
        try:
            boards = [infer.student_board(None, vocab, x['scene'], x['transcript'], x['is_question'], 10, kd, core)
                      for x in held]
        finally:
            infer._candidate_pool = original
        regen[arm] = boards == r['arms'][arm]['boards']
    assert infer._candidate_pool is original
    boards = [infer.student_board(model, vocab, x['scene'], x['transcript'], x['is_question'], 10, kd, core) for x in held]
    regen['learned_production_path'] = boards == r['arms']['learned']['boards']
    if label == 'historical':
        ref = [b['student_board'] for b in json.loads((HERE / 'I-03_numbers_implement_v2.json').read_text())['per_row_table2'] if b['window'] == 'full10']
    else:
        ref = [b['student_board'] for b in json.loads((bundle / 'boards.json').read_text())]
    regen['learned_equals_dependency_artifact'] = r['arms']['learned']['boards'] == ref
    rep['board_regeneration'] = regen
    assert all(len(b) == 10 and b[:4] == core for v in r['arms'].values() for b in v['boards'])
    rep['all_boards_10_with_core_first'] = True

    # ---- 3. own scoring ---------------------------------------------------------------------
    scoring, cells = {}, {}
    for arm, v in r['arms'].items():
        for win, (k, dyn) in WINDOWS.items():
            per = [own_score(b, m, core, k, dyn) for b, m in zip(v['boards'], moves)]
            same = per == v['windows'][win]['per_row']
            means = {key: float(np.mean([p[key] for p in per if p[key] is not None])) for key in per[0]}
            mean_ok = all(abs(means[key] - v['windows'][win]['means'][key]) < 1e-9 for key in means)
            # harness cross-check saved by the implementer
            h = v['windows'][win]['harness']
            harness_ok = abs(h['diversity_retention']['student_distinct_functions'] - means['df']) < 5e-4
            scoring[f'{arm}/{win}'] = dict(per_row_identical=same, means_identical=mean_ok, harness_df_matches=harness_ok)
            cov = [p['dynamic_coverage'] for p in per if p['dynamic_coverage'] is not None]
            cells[f'{arm}/{win}'] = dict(df=round(means['df'], 4), nd=round(means['near_dup'], 4), sfp=round(means['sfp'], 4),
                                         hedges=round(means['hedges'], 4), cov=f'{sum(cov)}/{len(cov)}')
    rep['scoring'] = scoring; rep['cells'] = cells
    assert all(s['per_row_identical'] and s['means_identical'] for s in scoring.values())
    rep['boards_with_hedge_full10'] = {arm: int(sum(1 for p in v['windows']['full10']['per_row'] if p['hedges'] > 0)) for arm, v in r['arms'].items()}

    # ---- 4. own bootstrap + discordant pairs + multiplicity -----------------------------------
    boot = {}
    n_excl = 0; n_total = 0; cosine_worse = []
    for arm in ('cosine', 'frequency', 'vocab_order', 'full', 'random'):
        for win in WINDOWS:
            a = r['arms'][arm]['windows'][win]['per_row']; b = r['arms']['learned']['windows'][win]['per_row']
            for key in ('df', 'sfp', 'dynamic_coverage', 'hedges', 'near_dup'):
                x = np.array([p[key] for p in a if p[key] is not None], float)
                y = np.array([p[key] for p in b if p[key] is not None], float)
                assert len(x) == len(y) == (71 if key == 'dynamic_coverage' else 84)
                dl = x - y
                got = r['arms'][arm]['windows'][win]['baseline_minus_learned'][key]
                assert got['n'] == len(dl)
                ci0 = [round(c, 4) for c in own_boot(dl, 0)]
                ci1 = [round(c, 4) for c in own_boot(dl, 1)]
                ent = dict(margin=round(float(dl.mean()), 4), reported=got['margin'], reported_ci=got['ci'], own_ci_seed0=ci0, own_ci_seed1=ci1)
                if key == 'dynamic_coverage':
                    bo = int(((x == 1) & (y == 0)).sum()); lo_ = int(((x == 0) & (y == 1)).sum())
                    ent['baseline_only'] = bo; ent['learned_only'] = lo_; ent['sign_test_p'] = round(sign_test_p(bo, lo_), 4)
                boot[f'{arm}/{win}/{key}'] = ent
                if key in ('df', 'sfp', 'dynamic_coverage'):
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
    rep['learned_minus_cosine_coverage'] = {win: dict(difference=-boot[f'cosine/{win}/dynamic_coverage']['reported'],
                                                      ci95=[-boot[f'cosine/{win}/dynamic_coverage']['reported_ci'][1], -boot[f'cosine/{win}/dynamic_coverage']['reported_ci'][0]],
                                                      own_seed1=[-boot[f'cosine/{win}/dynamic_coverage']['own_ci_seed1'][1], -boot[f'cosine/{win}/dynamic_coverage']['own_ci_seed1'][0]])
                                            for win in WINDOWS}
    for arm in ('cosine', 'full', 'learned'):
        rep[f'{arm}_full10_rows_with_neardup'] = int(sum(1 for p in r['arms'][arm]['windows']['full10']['per_row'] if p['near_dup'] > 0))
    report[label] = rep

for label in report:
    r = data[label]
    for arm in r['arms']:
        assert [p['dynamic_coverage'] for p in r['arms'][arm]['windows']['full10']['per_row']] == \
               [p['dynamic_coverage'] for p in r['arms'][arm]['windows']['dyn6']['per_row']]
report['full10_dyn6_coverage_identical_all_arms'] = True
report['I05b_historical_vocab_overlap'] = f"{len(set(vocabs['I-05b']) & set(vocabs['historical']))}/95"
report['I05b_I05_vocab_overlap'] = f"{len(set(vocabs['I-05b']) & set(vocabs['I-05']))}/45"
prev = json.loads((HERE / 'I-02_pool_v1/numbers.json').read_text())
report['historical_I05_blocks_equal_I02_pool_v1'] = all(data[k] == prev[k] for k in prev) and set(prev) == {'historical', 'I-05'}
with out.open('x') as f:
    json.dump(report, f, indent=1)
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk not in ('paired_effects', 'scoring', 'cells')}) for k, v in report.items()}, indent=1))
for label in ('historical', 'I-05', 'I-05b'):
    print(label)
    for c, v in report[label]['cells'].items():
        print(' ', c, v)
    for key in ('cosine/full10/dynamic_coverage', 'cosine/full7/dynamic_coverage', 'cosine/full10/df', 'cosine/full10/sfp', 'cosine/full7/df', 'cosine/full7/sfp', 'cosine/dyn6/df', 'cosine/dyn6/sfp', 'cosine/full10/hedges', 'full/full10/dynamic_coverage', 'random/full10/dynamic_coverage', 'frequency/full10/dynamic_coverage', 'vocab_order/full10/dynamic_coverage'):
        print(' ', key, report[label]['paired_effects'][key])
