"""Archive explicitly selected pipelines; never infer a final bundle from newest mtime.
Run from mac/. --extra accepts another issue's standard input/artifact manifest.
No inference, model loading, network, or production writes are performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[4]
ANALYSES = 'paper/odi/revision/analyses/'
sys.path.insert(0, str(ROOT / 'mac'))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def verify(out):
    manifest = read_json(out / 'manifest.json')
    assert digest((out / 'manifest.json').read_bytes()) == (out / 'manifest.sha256').read_text().strip()
    for entry in manifest['files'].values():
        content = (out / entry['snapshot']).read_bytes()
        assert digest(content) == entry['sha256'], entry
        assert len(content) == entry['bytes'], entry
    return len(manifest['files'])


def build(out, extra):
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'objects').mkdir()
    files, drift, pipelines = {}, [], {}

    def capture(path, expected=None):
        p = (ROOT / path).resolve()
        key = p.relative_to(ROOT).as_posix()
        data = p.read_bytes()
        sha = digest(data)
        if key in files:
            assert files[key]['sha256'] == sha, f'Concurrent mutation: {key}'
        else:
            obj = out / 'objects' / sha
            if not obj.exists():
                with obj.open('xb') as f:
                    f.write(data)
            files[key] = dict(sha256=sha, bytes=len(data), snapshot='objects/' + sha)
        if expected is not None and sha != expected:
            drift.append(dict(path=key, recorded_sha256=expected, current_sha256=sha))
        return key

    selected = {
        'I-05_torch': ANALYSES + 'I-05_split_first_v2/manifest.json',
        'I-02_controls': ANALYSES + 'I-02_pool_v1/manifest.json',
        'I-07_export': ANALYSES + 'I-07_export_v1/manifest.json',
    }
    for path in extra:
        selected[path] = path
    for name, path in selected.items():
        capture(path)
        m = read_json(ROOT / path)
        inputs, artifacts = {}, {}
        for p, sha in m.get('input_sha256', {}).items():
            inputs[p] = dict(path=capture(p, sha), recorded_sha256=sha)
        for p, sha in m.get('artifact_sha256', {}).items():
            key = capture(str(Path(path).parent / p), sha)
            artifacts[p] = key
        pipelines[name] = dict(source_manifest=path, inputs=inputs, artifacts=artifacts,
                              provenance='Run-time declarations in the archived source manifest; no new inference.')
    # A packaging snapshot cannot silently replace source bytes that drifted since a run.
    assert not drift, f'Recorded input/artifact drift; use a fresh output after resolution: {drift}'
    i05 = read_json(ROOT / selected['I-05_torch'])
    i07 = read_json(ROOT / selected['I-07_export'])
    assert i07['steps']['bundle']['student_pt_sha256'] == i05['artifact_sha256']['student.pt']
    assert i07['steps']['bundle']['manifest_sha256'] == digest((ROOT / selected['I-05_torch']).read_bytes())
    assert i07['steps']['bundle']['vocab_sha'] == i05['vocab_sha']
    effective = read_json(ROOT / ANALYSES / 'I-05_split_first_v2/config.json')
    effective_hash = digest(json.dumps(effective, sort_keys=True).encode())[:12]
    assert effective_hash == i05['config_hash']
    from evaluation.phase4.train import load_config
    current_config, current_hash = load_config()
    historical_result = 'mac/evaluation/phase4/phase4_results.json'
    historical_hash = read_json(ROOT / historical_result)['meta']['config_hash']
    assert historical_hash == 'd6ce2e56d50b'
    historical = [historical_result, 'mac/evaluation/phase3/selftest_results.json']
    historical += ['mac/intel/artifacts/' + n for n in (
        'e2e_mac.json', 'e2e_fp32.json', 'int8_accuracy.json', 'student_fp32.xml',
        'student_fp32.bin', 'student_fp32.vocab.sha', 'student_int8.xml',
        'student_int8.bin', 'student_int8.vocab.sha')]
    for p in historical:
        capture(p)
    for name in ('I-05_result.md', 'I-05_check.md', 'I-21_manifest_implement.py'):
        capture(ANALYSES + name)
    pipelines['historical'] = dict(
        checkpoint='mac/evaluation/phase4/student.pt',
        results=historical, config_hash=historical_hash,
        config_hash_provenance=historical_result + ':meta.config_hash',
        exact_training_config='unavailable; current YAML is not its reconstruction',
        provenance='Bytes captured now; historical result-to-source/checkpoint binding was not recorded fully at run time.',
        backends='See archived e2e_mac.json per-result backend, embed_backend_used and student_runtime; historical quality uses torch.')
    pipelines['I-05_torch']['backends'] = dict(head='torch', embeddings='torch', device='cpu', scoring='torch')
    pipelines['I-02_controls']['backends'] = dict(head='torch for learned; head-free controls', embeddings='torch', scoring='torch')
    pipelines['I-07_export']['backends'] = dict(head='OpenVINO CPU FP32 / mixed INT8 with output head excluded', embeddings='torch', quality_latency='not established by export alone')
    manifest = dict(
        schema_version=1, issue='I-21', scope='Shared packaging snapshot, mandatory I-05 pipeline plus available controls/exports and separate historical artifacts',
        files=files, pipelines=pipelines,
        configuration=dict(historical_recorded_id=historical_hash,
            current_yaml_parsed_id=current_hash, current_parsed_config=current_config,
            i05_effective_id=effective_hash,
            id_algorithm='SHA-256(json.dumps(parsed_object, sort_keys=True).encode())[:12]; file entries use SHA-256 of raw bytes',
            caveat='Different IDs do not prove all settings changed. Current YAML includes v3 settings.'),
        shared_roles=dict(teacher_boards=['mac/evaluation/phase3/boards.jsonl', 'mac/evaluation/phase4/teacher_data.jsonl'],
            selectors=['mac/evaluation/phase4/infer.py', 'mac/selection.py', 'mac/evaluation/phase4/answers.py'],
            metrics=['mac/evaluation/phase4/evaluate.py', 'mac/evaluation/phase3/metrics3.py', 'mac/evaluation/coverage_metrics.py', 'mac/evaluation/phase3/taxonomy.py']),
        pending=dict(I_04='No revision quality result available at packaging; attach via --extra when final.',
                     I_10='No revision latency result available at packaging; attach via --extra when final.'),
        exclusions=['I-05b and min_freq=2 diagnostic bundles are not the authoritative I-05 handoff; no automatic promotion.',
                    'External MiniLM/teacher model weight snapshots are not archived; embedding identity is the run declaration and source, not full weight provenance.'],
        seed=i05['seed'], bootstrap=i05['bootstrap'])
    # Reject changes during the capture, including files shared by concurrent issues.
    for p, entry in files.items():
        assert digest((ROOT / p).read_bytes()) == entry['sha256'], f'Concurrent mutation: {p}'
    with (out / 'manifest.json').open('x') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write('\n')
    (out / 'manifest.sha256').write_text(digest((out / 'manifest.json').read_bytes()) + '\n')
    return dict(files_verified=verify(out), source_manifests=len(selected), drift_count=len(drift),
                wall_seconds=time.perf_counter()-start, manifest_sha256=digest((out/'manifest.json').read_bytes()),
                config=manifest['configuration'])


def selftest():
    # Exercise integrity failure without touching any real evidence.
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p/'object').write_bytes(b'original')
        m = dict(files={'fixture': dict(snapshot='object', sha256=digest(b'original'), bytes=8)})
        (p/'manifest.json').write_text(json.dumps(m))
        (p/'manifest.sha256').write_text(digest((p/'manifest.json').read_bytes()))
        assert verify(p) == 1
        (p/'object').write_bytes(b'tampered')
        try:
            verify(p)
        except AssertionError:
            pass
        else:
            raise AssertionError('tampering accepted')
        (p/'object').write_bytes(b'original')
        (p/'manifest.json').write_text('{}')
        try:
            verify(p)
        except AssertionError:
            pass
        else:
            raise AssertionError('manifest tampering accepted')
    return dict(passed=3, checks=['intact snapshot', 'artifact tampering rejected', 'manifest tampering rejected'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--extra', action='append', default=[])
    ap.add_argument('--verify', type=Path)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        result = selftest()
    elif args.verify:
        result = dict(files_verified=verify(args.verify))
    elif args.out:
        result = build(args.out, args.extra)
    else:
        ap.error('choose --out, --verify, or --selftest')
    print(json.dumps(result, indent=2))
