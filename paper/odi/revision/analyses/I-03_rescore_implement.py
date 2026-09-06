"""Historical I-03 re-score; no teacher calls and exclusive output creation.

Run from mac/ with VOICELINK_PERCEPTION=0 /usr/local/bin/python3 SCRIPT --output NEW.json.
Full windows truncate then strip core for dynamic coverage; dyn6 strips then truncates.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

MAC = Path(__file__).resolve().parents[4] / 'mac'
sys.path.insert(0, str(MAC))
# Fix the scoring ruler and generation backend; use only cached model weights.
os.environ.update(VOICELINK_PERCEPTION='0', VOICELINK_EMBED_BACKEND='torch',
                  VOICELINK_STUDENT_BACKEND='torch', HF_HUB_OFFLINE='1')
import numpy as np
import torch
from evaluation.phase3 import selftest, metrics3, taxonomy
from evaluation.phase4 import evaluate as e
from evaluation.embed import embed, assert_backend
from evaluation.coverage_metrics import bootstrap_ci


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(board, moves, k):
    df, dup, sfp = e._df_dup_sfp(board, metric_k=k)
    honest = [m for m in moves if not metrics3.is_hedge(m)]
    return dict(df=df, near_dup=dup, sfp=sfp, length=len(board[:k]),
                coverage=int(metrics3.coverage_of_intent(board, moves, k)) if moves else None,
                nonhedge_coverage=int(metrics3.coverage_of_intent(board, honest, k)) if honest else None)


def paired(s, t, factor=1.0, higher=True):
    return e._margin_ci(s, t, factor=factor, higher_is_better=higher)


def difference(s, t):
    # A signed effect is not a gate: negative redundancy differences are beneficial.
    result = paired(s, t)
    result.pop('verdict')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('output exists; use a distinct suffix')
    started = time.perf_counter()
    torch.manual_seed(0); np.random.seed(0)
    source_paths = [e.CKPT, e.HERE/'config.yaml', e.HERE/'vocab.json', e.HERE/'canonical.json',
                    e.HERE/'splits.json', e.HERE/'teacher_data.jsonl', e.HERE/'v2_scenarios.jsonl',
                    selftest.BOARDS_PATH, selftest.scengen.SCENARIOS_PATH,
                    Path(e.__file__), Path(selftest.__file__), Path(metrics3.__file__),
                    e.HERE/'infer.py', e.HERE/'train.py', e.HERE/'data.py',
                    MAC/'evaluation/embed.py', MAC/'evaluation/coverage_metrics.py',
                    MAC/'evaluation/phase3/taxonomy.py', MAC/'selection.py',
                    MAC/'rank_vocabulary.py', Path(__file__).resolve()]
    inputs = {str(p.relative_to(MAC.parent)): sha(p) for p in source_paths}
    ck = torch.load(e.CKPT, map_location='cpu', weights_only=False)
    assert ck['config_hash'] == 'd6ce2e56d50b', 'Not the historical checkpoint'
    model, vocab, core, kd = e.load_student()
    assert len(vocab) == 95 and kd == 6
    strip = lambda b: [w for w in b if w.lower() not in {x.lower() for x in core}]
    windows = [('full10', 10, False), ('dyn6', 6, True), ('full7', 7, False)]
    table1 = {}; per1 = []
    raw = selftest.load_boards()
    for name, k, dyn in windows + [('dyn_all_remaining', 10000, True)]:
        harness = selftest.score_boards(raw, metric_k=k, dynamic_only=dyn)
        out = dict(harness=harness, arms={}, paired_differences={})
        by_id = {}
        for r in raw:
            b = strip(r['board']) if dyn else r['board']
            score = scores(b, r.get('acceptable_moves', []), k)
            per1.append(dict(window=name, id=r['id'], arm=r['strategy'], **score))
            by_id.setdefault(r['id'], {})[r['strategy']] = score
        assert len(by_id) == 120 and all(set(v) == {'diverse', 'topk'} for v in by_id.values())
        for arm in selftest.STRATEGIES:
            out['arms'][arm] = {key: float(np.mean([v[arm][key] for v in by_id.values()]))
                                for key in ('df', 'near_dup', 'sfp', 'length', 'coverage', 'nonhedge_coverage')}
            assert np.isclose(out['arms'][arm]['df'], harness['overall'][arm]['distinct_functions'])
            assert np.isclose(out['arms'][arm]['coverage'], harness['overall'][arm]['coverage_of_intent'])
        for key in ('df', 'near_dup', 'sfp', 'coverage'):
            out['paired_differences'][key] = difference([v['diverse'][key] for v in by_id.values()],
                                                    [v['topk'][key] for v in by_id.values()])
        table1[name] = out
    held = e._held_out_rows(); lookup = e._acceptable_moves_lookup()
    assert len(held) == 84
    boards = []
    for i, r in enumerate(held):
        sb = e.student_board(model, vocab, r['scene'], r['transcript'], r.get('is_question', False),
                             10, kd, core)
        assert len(sb) == len(r['board']) == 10
        assert sb[:4] == r['board'][:4] == core
        boards.append(dict(index=i, row=r, student=sb,
                           moves=lookup.get((r['scene'].strip(), r['transcript'].strip()), [])))
    table2 = {}; per2 = []
    for name, k, dyn in windows:
        iterator = iter(boards)
        harness = e.held_out_eval(model, vocab, 10, kd, core,
                                  board_fn=lambda *unused: next(iterator)['student'],
                                  metric_k=k, dynamic_only=dyn)
        accum = {arm: [] for arm in ('student', 'teacher')}
        for b in boards:
            one = dict(window=name, index=b['index'], id=b['row'].get('id'),
                       student_board=b['student'], teacher_board=b['row']['board'], moves=b['moves'])
            for arm, raw_board in [('student', b['student']), ('teacher', b['row']['board'])]:
                board = strip(raw_board)[:k] if dyn else raw_board[:k]
                sc = scores(board, b['moves'], k)
                # Explicit k prevents the historical import-time default bug.
                dynamic = e.dynamic_words(board, core, k=k)
                honest = [m for m in b['moves'] if not metrics3.is_hedge(m)]
                sc['dynamic_coverage'] = int(metrics3.coverage_of_intent(dynamic, honest, k)) if honest else None
                sc['evaluated_board'] = board
                one[arm] = sc; accum[arm].append(sc)
            per2.append(one)
        out = dict(harness=harness, arms={}, paired_differences={})
        for arm, arr in accum.items():
            out['arms'][arm] = {key: float(np.mean([r[key] for r in arr if r[key] is not None]))
                                for key in ('df', 'near_dup', 'sfp', 'coverage', 'dynamic_coverage')}
        for key in ('df', 'near_dup', 'sfp', 'coverage', 'dynamic_coverage'):
            ss = [r[key] for r in accum['student'] if r[key] is not None]
            tt = [r[key] for r in accum['teacher'] if r[key] is not None]
            out['paired_differences'][key] = difference(ss, tt)
            if key in ('df', 'dynamic_coverage'):
                margin = paired(ss, tt, 0.9)
                reference = (harness['diversity_retention']['distinct_functions_margin'] if key == 'df'
                             else harness['coverage_of_intent']['margin'])
                assert margin == reference
        assert np.isclose(out['arms']['student']['df'], harness['diversity_retention']['student_distinct_functions'], atol=0.0005)
        assert np.isclose(out['arms']['teacher']['sfp'], harness['diversity_retention']['teacher_same_function_pair'], atol=0.00005)
        table2[name] = out
    for name in ('full10', 'dyn6'):
        assert table2[name]['harness']['coverage_of_intent']['paired_counts_student_teacher'] == {'11':58,'01':10,'10':3,'00':0}
    pinned_count, pinned_labels = taxonomy.classify_functions(embed(core))
    assert pinned_count == 3
    assert assert_backend('torch') == 'torch'
    assert all(sha(MAC.parent/p) == h for p,h in inputs.items()), 'inputs changed during run'
    result = dict(issue='I-03', config_hash=ck['config_hash'], input_sha256=inputs,
                  seed=0, bootstrap=dict(resamples=10000, seed=0, method='paired percentile, row resampling, 95%'),
                  versions={x:importlib.metadata.version(x) for x in ('torch','numpy','sentence-transformers','transformers')},
                  embedding_backend='torch', head_backend='torch',
                  pinned=dict(core=core, distinct_functions=pinned_count, labels=pinned_labels,
                              sfp_floor=dict(full7=1/21, full10=1/45, dyn6=0)),
                  table1=table1, table2=table2, per_row_table1=per1, per_row_table2=per2,
                  wall_seconds=time.perf_counter()-started)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(dict(output=str(args.output), wall_seconds=result['wall_seconds'],
                          table2={key:v['harness'] for key,v in table2.items()}), indent=2))

if __name__ == '__main__':
    main()
