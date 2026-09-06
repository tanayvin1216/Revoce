"""I-02: infer.py-owned POOL_SOURCE switch; direct board API, no monkeypatch.
Run from mac/; --out must be new. No teacher calls or timing claims. Existing outputs are never overwritten.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

MAC = Path(__file__).resolve().parents[4] / 'mac'
ROOT = MAC.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(MAC))
os.environ.update(VOICELINK_PERCEPTION='0', HF_HUB_OFFLINE='1',
                  VOICELINK_EMBED_BACKEND='torch', VOICELINK_STUDENT_BACKEND='torch')
import numpy as np
import torch
from evaluation.phase4 import data as d, train, evaluate as e, infer
from evaluation.phase4.student import StudentRanker
from evaluation.phase3 import metrics3
from evaluation.embed import embed, assert_backend
from selection import build_query_vector

ARMS = ('learned', 'cosine', 'frequency', 'vocab_order', 'full', 'random')
WINDOWS = [('full10', 10, False), ('dyn6', 6, True), ('full7', 7, False)]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, obj):
    with p.open('x') as f:
        json.dump(obj, f, indent=2)


def effect(a, b):
    result = e._margin_ci(a, b)
    result.pop('verdict')  # Signed differences are effects, not engineering gates.
    return result

def score(board, moves, core, k, dynamic_only):
    strip = lambda b: [w for w in b if w.lower() not in {c.lower() for c in core}]
    window = (strip(board) if dynamic_only else board)[:k]
    df, dup, sfp = e._df_dup_sfp(window, metric_k=k)
    targets = [m for m in moves if not metrics3.is_hedge(m)]
    return dict(df=df, near_dup=dup, sfp=sfp,
                hedges=sum(metrics3.is_hedge(w) for w in window),
                dynamic_coverage=int(metrics3.coverage_of_intent(strip(window), targets, k)) if targets else None)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pool-source', choices=('all',) + ARMS, default='all')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    start = time.perf_counter()
    torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(1)
    new = HERE / 'I-05_split_first_v2'
    paths = [MAC/'evaluation/phase4'/n for n in ('student.pt','vocab.json','canonical.json','splits.json','config.yaml','teacher_data.jsonl','v2_scenarios.jsonl','infer.py','data.py','train.py','evaluate.py','student.py','answers.py')]
    paths += [MAC/n for n in ('selection.py','rank_vocabulary.py','evaluation/embed.py','evaluation/hedges.py','evaluation/coverage_metrics.py','evaluation/phase3/metrics3.py','evaluation/phase3/taxonomy.py','evaluation/phase3/selftest.py','evaluation/phase3/boards.jsonl')]
    paths += [new/n for n in ('student.pt','vocab.json','canonical.json','splits.json','preprocessing.json','rows.json','boards.json','manifest.json','config.json')]
    paths += [e.scengen.SCENARIOS_PATH, Path(__file__).resolve(), HERE/'I-03_numbers_implement_v2.json']
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    rows = train.gather_rows()
    assert rows == json.loads((new/'rows.json').read_text())
    lookup = e._acceptable_moves_lookup()
    result = {}
    for label, bundle in [('historical', e.HERE), ('I-05', new)]:
        splits = json.loads((bundle/'splits.json').read_text())
        assert splits == json.loads(d.SPLITS_PATH.read_text())
        assert d.rows_fingerprint(rows) == splits['fingerprint']
        vocab, indices = d.load_vocab(bundle/'vocab.json')
        canonical = d.load_canonical(bundle/'canonical.json')
        freq = np.stack([d.multihot(rows[i]['board'], indices, canonical) for i in splits['train']]).sum(0)
        if label == 'I-05':
            np.testing.assert_array_equal(freq, json.loads((new/'preprocessing.json').read_text())['positive_counts'])
        ck = torch.load(bundle/'student.pt', map_location='cpu', weights_only=False)
        assert ck['vocab_sha'] == d.vocab_sha(vocab)
        model = StudentRanker(ck['context_dim'], ck['hidden_dim'], ck['vocab_size'], ck['dropout'])
        model.load_state_dict(ck['state_dict']); model.eval()
        core, kd = ck['guaranteed_core'], ck['k_dyn']
        assert kd == 6 and infer._pool_mult() == 3
        held = [rows[i] for i in splits['test']]
        moves = [lookup.get((r['scene'].strip(), r['transcript'].strip()), []) for r in held]
        arms = {}
        selected = ARMS if args.pool_source == 'all' else tuple(dict.fromkeys(('learned', args.pool_source)))
        for arm in selected:
            pools = []
            pool_rng = np.random.default_rng(args.seed)
            board_rng = np.random.default_rng(args.seed)
            boards = []
            for r in held:
                head = model if arm == 'learned' else None
                params = dict(pool_source=arm, pool_frequency=freq)
                pools.append(infer._candidate_pool(head, vocab, r['scene'], r['transcript'],
                    r['is_question'], kd, pool_rng=pool_rng, **params))
                boards.append(infer.student_board(head, vocab, r['scene'], r['transcript'],
                    r['is_question'], 10, kd, core, pool_rng=board_rng, **params))
            assert len(pools) == 84 and all(len(b) == 10 and b[:4] == core for b in boards)
            assert all(len(e.dynamic_words(b, core, k=10)) == 6 for b in boards)
            if arm == 'learned':
                reference = ([b['student_board'] for b in json.loads((new/'boards.json').read_text())] if label == 'I-05' else [b['student_board'] for b in json.loads((HERE/'I-03_numbers_implement_v2.json').read_text())['per_row_table2'] if b['window'] == 'full10'])
                assert boards == reference, 'dependency learned-board drift'
            by_context = {(r['scene'],r['transcript']): b for r,b in zip(held,boards)}
            windows = {}
            for name,k,dyn in WINDOWS:
                harness = e.held_out_eval(model,vocab,10,kd,core,board_fn=lambda s,t,q:by_context[s,t],splits_path=bundle/'splits.json',metric_k=k,dynamic_only=dyn)
                per = [score(b,m,core,k,dyn) for b,m in zip(boards,moves)]
                means = {key:float(np.mean([r[key] for r in per if r[key] is not None])) for key in per[0]}
                assert np.isclose(means['df'],harness['diversity_retention']['student_distinct_functions'],atol=.0005)
                windows[name] = dict(means=means,per_row=per,harness=harness)
            arms[arm] = dict(pools=pools,boards=boards,windows=windows)
        for arm in selected:
            for name,_,_ in WINDOWS:
                a = arms[arm]['windows'][name]
                b = arms['learned']['windows'][name]
                a['baseline_minus_learned'] = {key:effect([r[key] for r in a['per_row'] if r[key] is not None],[r[key] for r in b['per_row'] if r[key] is not None]) for key in a['means']}
        result[label] = dict(config_hash=ck['config_hash'],vocab_sha=d.vocab_sha(vocab),vocab=vocab,training_multihot_counts=freq.tolist(),split_fingerprint=splits['fingerprint'],test_indices=splits['test'],teacher_boards=[r['board'] for r in held],acceptable_moves=moves,arms=arms)
    assert assert_backend('torch') == 'torch'
    assert all(sha(ROOT/p) == h for p,h in inputs.items()), 'inputs changed during run'
    write(args.out/'numbers.json', result)
    config = dict(pool_sources=list(selected),pool_n=18,seed=args.seed,random_policy='default_rng once per arm/bundle; choice without replacement in draw order; saved test index order',ordering='numpy argsort(-scores, kind=quicksort), vocabulary order for full/vocab_order',windows=WINDOWS,bootstrap=dict(seed=0,resamples=10000,method='paired row percentile 95%, pointwise'),head_backend='torch',embedding_backend='torch',scope='quality only; no latency benchmark')
    write(args.out/'manifest.json',dict(config=config,analysis_config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),input_sha256=inputs,artifact_sha256={'numbers.json':sha(args.out/'numbers.json')},versions={x:importlib.metadata.version(x) for x in ('torch','numpy','sentence-transformers','transformers')},wall_seconds=time.perf_counter()-start))
    print(json.dumps({label:{a:{w:v['means'] for w,v in av['windows'].items()} for a,av in r['arms'].items()} for label,r in result.items()},indent=2))
if __name__ == '__main__':
    main()
