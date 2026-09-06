"""I-05 CHECKER: independent re-derivations against the implementer's bundle.
Run from mac/:  VOICELINK_PERCEPTION=0 /usr/local/bin/python3 ../paper/odi/revision/analyses/I-05_independent_check.py
Writes I-05_independent_check.json next to this file. Never writes into the bundle.
"""
import hashlib, json, os, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAC = HERE.parents[3] / 'mac'
sys.path.insert(0, str(MAC))
os.environ.update(VOICELINK_PERCEPTION='0', HF_HUB_OFFLINE='1', VOICELINK_EMBED_BACKEND='torch',
                  VOICELINK_STUDENT_BACKEND='torch', VOICELINK_DEVICE='cpu')
import numpy as np
import torch
from evaluation.phase4 import train as tr, data as d, evaluate as e, infer
from evaluation.phase4.student import StudentRanker
from evaluation.phase3 import metrics3
from evaluation.coverage_metrics import bootstrap_ci
from rank_vocabulary import GUARANTEED_CORE

BUNDLE = HERE / 'I-05_split_first_v2'
out = {}
cfg, _ = tr.load_config()
rows = tr.gather_rows()
splits = json.loads((BUNDLE / 'splits.json').read_text())
hist_splits = json.loads(d.SPLITS_PATH.read_text())
out['bundle_splits_equal_historical'] = splits == hist_splits
out['fingerprint_matches_regathered_rows'] = splits['fingerprint'] == d.rows_fingerprint(rows)
out['rows_json_equals_regathered'] = json.loads((BUNDLE / 'rows.json').read_text()) == rows
out['split_sizes'] = [len(splits[k]) for k in ('train', 'val', 'test')]
out['partitions_disjoint_and_complete'] = (
    len(set(splits['train']) | set(splits['val']) | set(splits['test'])) == len(rows) == 559
    and not (set(splits['train']) & set(splits['val'])) and not (set(splits['train']) & set(splits['test']))
    and not (set(splits['val']) & set(splits['test'])))

# 1. Does the historical 95-entry vocab regenerate from ALL rows? (confirms the leak diagnosis)
all_raw, _ = d.build_vocab(rows, cfg['vocab']['min_freq'], cfg['vocab']['max_size'])
all_vocab, all_canon = d.dedup_vocab(all_raw, cfg['vocab']['dedup_tau'])
hist_vocab = json.loads(d.VOCAB_PATH.read_text())
hist_canon = d.load_canonical()
out['all_rows_fit'] = dict(raw=len(all_raw), vocab=len(all_vocab), merges=len(all_canon),
                           equals_historical_vocab=all_vocab == hist_vocab,
                           equals_historical_canonical=all_canon == hist_canon,
                           historical_sizes=[len(hist_vocab), len(hist_canon)],
                           historical_vocab_sha=d.vocab_sha(hist_vocab), all_rows_vocab_sha=d.vocab_sha(all_vocab))

# 2. Independent train-only fit written without the implementer's fit(): own frequency count.
train_rows = [rows[i] for i in splits['train']]
freq = Counter(w.strip().lower() for r in train_rows for w in r['board']
               if w.strip().lower() and w.strip().lower() not in d.CORE_SET)
my_raw = sorted((w for w in freq if freq[w] >= cfg['vocab']['min_freq']), key=lambda w: (-freq[w], w))[:cfg['vocab']['max_size']]
my_vocab, my_canon = d.dedup_vocab(my_raw, cfg['vocab']['dedup_tau'])
b_vocab = json.loads((BUNDLE / 'vocab.json').read_text())
b_canon = json.loads((BUNDLE / 'canonical.json').read_text())
pre = json.loads((BUNDLE / 'preprocessing.json').read_text())
out['train_only_fit'] = dict(raw=len(my_raw), vocab=len(my_vocab), merges=len(my_canon),
                             equals_bundle_vocab=my_vocab == b_vocab, equals_bundle_canonical=my_canon == b_canon,
                             raw_equals_bundle_raw=my_raw == pre['raw_vocab'],
                             vocab_sha=d.vocab_sha(my_vocab), fit_indices_equal_train=pre['fit_indices'] == splits['train'])
# label mass: positive counts / pos_weight / log_priors from training labels only, own formula
idx = {w: i for i, w in enumerate(my_vocab)}
Ytr = np.stack([d.multihot(r['board'], idx, my_canon) for r in train_rows])
npos = Ytr.sum(0)
out['train_only_fit']['positive_counts_equal'] = npos.tolist() == pre['positive_counts']
out['train_only_fit']['pos_weight_equal'] = np.allclose(np.clip((len(Ytr) - npos) / npos, 1, 20), pre['pos_weight'], atol=1e-6)
out['train_only_fit']['log_priors_equal'] = np.allclose(np.log(npos / len(Ytr)), pre['log_priors'], atol=1e-6)
out['train_only_fit']['min_positive_count'] = int(npos.min())

# 3. Quantify what the leak did: held-out board words representable under the all-row vocab but not the train vocab
def rep(board, vocab, canon):
    ws = [w.strip().lower() for w in board if w.strip().lower() not in d.CORE_SET]
    return ws, [w for w in ws if canon.get(w, w) in set(vocab)]
leak = {}
for split in ('val', 'test'):
    tot = hit_all = hit_train = zero_all = zero_train = 0
    only_all = Counter()
    for i in splits[split]:
        ws, ha = rep(rows[i]['board'], all_vocab, all_canon)
        _, ht = rep(rows[i]['board'], my_vocab, my_canon)
        tot += len(ws); hit_all += len(ha); hit_train += len(ht)
        zero_all += not ha; zero_train += not ht
        only_all.update(set(ha) - set(ht))
    leak[split] = dict(slots=tot, canonical_hits_all_row_vocab=hit_all, canonical_hits_train_vocab=hit_train,
                       zero_hit_rows_all_row_vocab=zero_all, zero_hit_rows_train_vocab=zero_train,
                       top_words_lost=only_all.most_common(8))
out['leak_effect'] = leak
# entries of the all-row vocab whose training-only frequency is below min_freq (i.e. they owed their place to held-out rows)
out['all_row_vocab_entries_below_min_freq_on_train'] = sorted(w for w in all_raw if freq.get(w, 0) < cfg['vocab']['min_freq'])
# validation rows with no positive target under the frozen vocab (affects checkpoint selection)
Yval = np.stack([d.multihot(rows[i]['board'], idx, my_canon) for i in splits['val']])
out['val_rows_with_zero_targets'] = int((Yval.sum(1) == 0).sum())
Yte = np.stack([d.multihot(rows[i]['board'], idx, my_canon) for i in splits['test']])
out['test_rows_with_zero_targets'] = int((Yte.sum(1) == 0).sum())

# 4. Reload the bundle checkpoint, regenerate boards with the production inference path, compare to boards.json
ck = torch.load(BUNDLE / 'student.pt', map_location='cpu', weights_only=False)
model = StudentRanker(ck['context_dim'], ck['hidden_dim'], ck['vocab_size'], ck['dropout'])
model.load_state_dict(ck['state_dict']); model.eval()
out['checkpoint'] = dict(vocab_size=ck['vocab_size'], vocab_sha_matches_bundle=ck['vocab_sha'] == d.vocab_sha(b_vocab),
                         out_features=model.state_dict()[[k for k in model.state_dict() if k.endswith('weight')][-1]].shape[0])
boards = json.loads((BUNDLE / 'boards.json').read_text())
out['boards_rows_equal_test_split'] = [b['row_index'] for b in boards] == splits['test']
mism = 0
regen = {}
for b in boards:
    r = rows[b['row_index']]
    sb = infer.student_board(model, b_vocab, r['scene'], r['transcript'], r['is_question'], 10, 6, GUARANTEED_CORE)
    regen[(r['scene'], r['transcript'])] = sb
    mism += sb != b['student_board']
    if b['teacher_board'] != r['board']: mism += 1000
out['regenerated_board_mismatches'] = mism
out['student_boards_all_len10_core_first'] = all(len(b['student_board']) == 10 and b['student_board'][:4] == GUARANTEED_CORE for b in boards)
out['student_dynamic_entries_in_vocab_or_open'] = sum(
    1 for b in boards for w in b['student_board'][4:] if w.lower() not in set(b_vocab))

# 5. Run held_out_eval WITHOUT the board_fn shortcut (model directly), compare to metrics.json
bm = json.loads((BUNDLE / 'metrics.json').read_text())
direct = {}
for name, k, dyn in [('full10', 10, False), ('dyn6', 6, True), ('full7', 7, False)]:
    m = e.held_out_eval(model, b_vocab, 10, 6, GUARANTEED_CORE, splits_path=BUNDLE / 'splits.json', metric_k=k, dynamic_only=dyn)
    direct[name] = m == bm['metrics'][name]
    if not direct[name]:
        direct[name + '_diff'] = {sec: [kk for kk in m[sec] if m[sec][kk] != bm['metrics'][name][sec].get(kk)] for sec in m if isinstance(m[sec], dict)}
out['direct_model_eval_equals_bundle_metrics'] = direct

# 6. Own coverage + CI derivation from boards.json (no evaluate.py), including a second bootstrap seed
lookup = e._acceptable_moves_lookup()
core = {w.lower() for w in GUARANTEED_CORE}
cov = {}
for name, k, dyn in [('full10', 10, False), ('dyn6', 6, True), ('full7', 7, False)]:
    s, t = [], []
    for b in boards:
        moves = lookup.get((b['context']['scene'].strip(), b['context']['transcript'].strip()), [])
        honest = [m for m in moves if not metrics3.is_hedge(m)]
        if not moves or not honest: continue
        def win(board):
            w = [x for x in board if x.lower() not in core][:k] if dyn else board[:k]
            return [x for x in w if x.lower() not in core]
        s.append(float(metrics3.coverage_of_intent(win(b['student_board']), honest, k)))
        t.append(float(metrics3.coverage_of_intent(win(b['teacher_board']), honest, k)))
    s, t = np.array(s), np.array(t)
    deltas = s - 0.9 * t
    lo, hi = bootstrap_ci(deltas.tolist())
    rng = np.random.default_rng(12345)
    boot = [np.mean(rng.choice(deltas, len(deltas), replace=True)) for _ in range(10000)]
    cov[name] = dict(n=len(s), student_hits=int(s.sum()), teacher_hits=int(t.sum()), ratio=float(s.sum() / t.sum()),
                     margin=float(deltas.mean()), ci_repo_bootstrap=[round(lo, 4), round(hi, 4)],
                     ci_seed12345=[round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
                     bundle_margin=bm['metrics'][name]['coverage_of_intent']['margin'])
out['coverage_rederived'] = cov

# 7. Representability recount, test split
tot = hits = zero = 0
for i in splits['test']:
    ws, h = rep(rows[i]['board'], b_vocab, b_canon)
    tot += len(ws); hits += len(h); zero += not h
out['representability_test'] = dict(slots=tot, canonical_hits=hits, zero_hit_rows=zero,
                                    matches_bundle=bm['representability']['test']['canonical_hits'] == hits and bm['representability']['test']['slots'] == tot)

# 8. Historical model on the same windows for a same-harness comparison (already in I-03; re-derive coverage only)
hck = torch.load(d.SPLITS_PATH.parent / 'student.pt', map_location='cpu', weights_only=False)
hm = StudentRanker(hck['context_dim'], hck['hidden_dim'], hck['vocab_size'], hck['dropout']); hm.load_state_dict(hck['state_dict']); hm.eval()
hist = e.held_out_eval(hm, hist_vocab, 10, 6, GUARANTEED_CORE, metric_k=10, dynamic_only=False)
out['historical_full10_coverage'] = dict(paired=hist['coverage_of_intent']['paired_counts_student_teacher'],
                                         student_dynamic=hist['coverage_of_intent']['student_dynamic'],
                                         df=hist['diversity_retention']['student_distinct_functions'])
out['inputs_unchanged_vs_manifest'] = all(
    hashlib.sha256((MAC.parent / p).read_bytes()).hexdigest() == h
    for p, h in json.loads((BUNDLE / 'manifest.json').read_text())['input_sha256'].items())
(HERE / 'I-05_independent_check.json').write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(out, indent=2, default=str))
