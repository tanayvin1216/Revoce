"""Recount frozen teacher occurrences; run from mac/, output must be new.

Assumption: I-05's checked rows.json is the authoritative gathered corpus;
the raw 600-board corpus is a separate denominator. No model is invoked.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
MAC = ROOT / 'mac'
sys.path.insert(0, str(MAC))
from evaluation.phase4.data import CORE_SET, load_teacher_rows, rows_fingerprint


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def count(rows, vocab, canonical):
    per = []
    for i, row in enumerate(rows):
        words = [w.strip().lower() for w in row['board']
                 if w.strip().lower() not in CORE_SET]
        assert len(words) == 6
        per.append(dict(index=i, slots=len(words),
                        exact=sum(w in vocab for w in words),
                        canonical=sum(canonical.get(w, w) in vocab for w in words)))
    out = dict(rows=len(rows), slots=sum(r['slots'] for r in per))
    for mode in ('exact', 'canonical'):
        hits = sum(r[mode] for r in per)
        zero = sum(r[mode] == 0 for r in per)
        out[mode] = dict(hits=hits, percent=100*hits/out['slots'],
                         zero_hit_rows=zero, zero_hit_percent=100*zero/len(rows))
    out['per_row'] = per
    return out


def main():
    assert Path.cwd() == MAC and os.environ.get('VOICELINK_PERCEPTION') == '0'
    out = Path(sys.argv[1]).resolve()
    out.mkdir(exist_ok=False)
    start = time.perf_counter()
    began = datetime.datetime.now(datetime.timezone.utc).isoformat()
    old = MAC / 'evaluation/phase4'
    new = ROOT / 'paper/odi/revision/analyses/I-05_split_first_v2'
    manifest = json.loads((new/'manifest.json').read_text())
    for name, expected in manifest['artifact_sha256'].items():
        assert sha(new/name) == expected, name
    paths = [old/n for n in ('teacher_data.jsonl', 'vocab.json', 'canonical.json', 'splits.json', 'student.pt', 'config.yaml', 'data.py')]
    paths += list(new.glob('*.json')) + [new/'student.pt', Path(__file__), ROOT/'paper/odi/main.tex']
    paths += list(MAC.rglob('*.py'))
    before = {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}
    rows = json.loads((new/'rows.json').read_text())
    splits = json.loads((new/'splits.json').read_text())
    assert splits == json.loads((old/'splits.json').read_text())
    assert rows_fingerprint(rows) == splits['fingerprint']
    assert sorted(splits['train']+splits['val']+splits['test']) == list(range(len(rows)))
    corpora = {'raw600': load_teacher_rows(old/'teacher_data.jsonl'), 'gathered559': rows}
    corpora.update({s: [rows[i] for i in splits[s]] for s in ('train', 'val', 'test')})
    results = {}
    for label, bundle in [('historical95', old), ('I05_train_only45', new)]:
        vocab = json.loads((bundle/'vocab.json').read_text())
        canonical = json.loads((bundle/'canonical.json').read_text())
        assert len(vocab) == len(set(vocab))
        assert not set(canonical).intersection(vocab)
        assert set(canonical.values()) <= set(vocab)
        results[label] = dict(vocab_size=len(vocab), canonical_mappings=len(canonical),
                             corpora={k: count(v, set(vocab), canonical) for k, v in corpora.items()})
    h = results['historical95']['corpora']['raw600']
    assert (h['slots'], h['exact']['hits'], h['canonical']['hits'], h['exact']['zero_hit_rows'], h['canonical']['zero_hit_rows']) == (3600,843,1166,141,70)
    n = results['I05_train_only45']['corpora']['test']
    assert (n['canonical']['hits'], n['slots'], n['canonical']['zero_hit_rows']) == (135,504,16)
    test_start = time.perf_counter()
    command = [sys.executable, '-m', 'pytest', '-q']
    with (out/'pytest.log').open('x') as log:
        run = subprocess.run(command, cwd=MAC, stdout=log, stderr=subprocess.STDOUT)
    test_wall = time.perf_counter()-test_start
    drift = [p for p, h in before.items() if not (ROOT/p).exists() or sha(ROOT/p) != h]
    payload = dict(issue='I-16', started_utc=began, wall_seconds=time.perf_counter()-start,
                   seed=None, split_fingerprint=splits['fingerprint'],
                   historical_config_hash='d6ce2e56d50b', I05_config_hash=manifest['config_hash'],
                   results=results, input_sha256=before, changed_inputs=drift,
                   pytest=dict(command=command, exit_code=run.returncode, wall_seconds=test_wall,
                               log_sha256=sha(out/'pytest.log')))
    with (out/'recount.json').open('x') as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({k:v for k,v in payload.items() if k not in ('input_sha256', 'results')}, indent=2))
    for label, result in results.items():
        print(label, json.dumps({k:{x:y for x,y in v.items() if x!='per_row'} for k,v in result['corpora'].items()}))


if __name__ == '__main__':
    main()
