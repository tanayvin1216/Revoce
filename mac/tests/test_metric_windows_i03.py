"""Regression: explicit k must reach coverage after core filtering in both arms."""
import numpy as np
import pytest
from evaluation.phase3 import selftest, metrics3
from evaluation.phase4 import evaluate as e
import evaluation.embed as embeddings


@pytest.fixture
def ruler(monkeypatch):
    monkeypatch.setattr(embeddings, 'embed', lambda b: np.eye(len(b), 384))
    monkeypatch.setattr(metrics3, 'distinct_functions_at', lambda b, k, **kw: len(b[:k]))
    monkeypatch.setattr(metrics3, 'same_function_pair_rate', lambda b, k, **kw: 0.0)
    monkeypatch.setattr(metrics3, 'coverage_of_intent', lambda b, m, k, **kw: 'target' in b[:k])
    monkeypatch.setattr(metrics3, 'exact_match', lambda b, g: 'target' in b)


def test_selftest_filters_before_truncation_and_keeps_full_lexical_hit(ruler):
    row = dict(board=['yes', 'no', 'I need help', 'wait', 'a', 'b', 'c', 'd', 'e', 'target'])
    assert not selftest._score_one(row)['coverage_of_intent']
    assert selftest._score_one(row, metric_k=10)['coverage_of_intent']
    dyn = selftest._score_one(row, metric_k=6, dynamic_only=True)
    assert dyn['distinct_functions'] == 6 and dyn['coverage_of_intent']
    assert selftest._score_one(row, metric_k=1)['exact_match']
    with pytest.raises(ValueError):
        selftest.score_boards([], metric_k=0)


def test_held_out_k_is_explicit_for_both_arms(ruler, monkeypatch):
    core = ['yes', 'no', 'I need help', 'wait']
    board = core + ['a', 'b', 'c', 'd', 'e', 'target']
    monkeypatch.setattr(e, '_held_out_rows', lambda *a: [dict(scene='s', transcript='t', board=board)])
    monkeypatch.setattr(e, '_acceptable_moves_lookup', lambda: {('s', 't'): ['target']})
    def run(k=None, dyn=False):
        return e.held_out_eval(None, [], 10, 6, core, board_fn=lambda *a: board,
                              metric_k=k, dynamic_only=dyn)
    assert run()['coverage_of_intent']['student_dynamic'] == 0
    for res in (run(10), run(6, True)):
        assert res['coverage_of_intent']['paired_counts_student_teacher'] == {'11':1,'01':0,'10':0,'00':0}
    res = run(6, True)
    assert res['diversity_retention']['student_distinct_functions'] == 6
    assert res['agreement']['dynamic_overlap'] == 1
    assert 'full_overlap_top7' not in res['agreement']
    assert 'full_overlap_top7' in run()['agreement']
    with pytest.raises(ValueError):
        run(0)


def test_dynamic_words_call_time_default(monkeypatch):
    monkeypatch.setattr(e, 'METRIC_K', 2)
    assert e.dynamic_words(['yes', 'a', 'b'], ['yes']) == ['a']
    assert e.dynamic_words(['yes', 'a', 'b'], ['yes'], k=3) == ['a', 'b']


def test_cli_requires_new_output():
    with pytest.raises(SystemExit):
        selftest.parse_args(['--metric-k', '10'])
    assert selftest.parse_args(['--metric-k', '6', '--dynamic-only', '--output', 'new.json']).metric_k == 6
    with pytest.raises(SystemExit):
        selftest.parse_args(['--metric-k', '0', '--output', 'new.json'])
