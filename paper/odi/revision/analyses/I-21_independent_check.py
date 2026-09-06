"""I-21 checker: independent re-derivation of the shared manifest's integrity and identities.
Does not import the implementer's script. Run from mac/ with VOICELINK_PERCEPTION=0.
"""
import hashlib, json, subprocess, sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'paper/odi/revision/analyses/I-21_manifest_v1'
sys.path.insert(0, str(ROOT / 'mac'))
H = lambda b: hashlib.sha256(b).hexdigest()
R = {}

m = json.loads((OUT / 'manifest.json').read_bytes())
R['manifest_sha256_recomputed'] = H((OUT / 'manifest.json').read_bytes())
R['manifest_sha256_sidecar_matches'] = R['manifest_sha256_recomputed'] == (OUT / 'manifest.sha256').read_text().strip()

# 1. every entry: object bytes hash + size; snapshot path == objects/<sha>
bad = []
for k, e in m['files'].items():
    b = (OUT / e['snapshot']).read_bytes()
    if H(b) != e['sha256'] or len(b) != e['bytes'] or e['snapshot'] != 'objects/' + e['sha256']:
        bad.append(k)
R['entries'] = len(m['files']); R['entries_bad'] = bad
objs = {p.name for p in (OUT / 'objects').iterdir()}
refd = {e['sha256'] for e in m['files'].values()}
R['objects_on_disk'] = len(objs); R['objects_referenced'] = len(refd)
R['objects_unreferenced'] = sorted(objs - refd); R['objects_missing'] = sorted(refd - objs)

# 2. drift of live repo vs archived bytes (concurrent edits since packaging)
drift, missing = [], []
for k, e in m['files'].items():
    p = ROOT / k
    if not p.exists(): missing.append(k); continue
    if H(p.read_bytes()) != e['sha256']: drift.append(k)
R['live_repo_drift_since_packaging'] = drift; R['live_repo_missing'] = missing

arch = lambda k: (OUT / m['files'][k]['snapshot']).read_bytes()
cid = lambda obj: H(json.dumps(obj, sort_keys=True).encode())[:12]

# 3. config IDs from archived bytes
R['i05_effective_id_from_archived_config'] = cid(json.loads(arch('paper/odi/revision/analyses/I-05_split_first_v2/config.json')))
R['current_yaml_id_from_archived_yaml'] = cid(yaml.safe_load(arch('mac/evaluation/phase4/config.yaml').decode()))
from evaluation.phase4.train import load_config
R['current_yaml_id_live_load_config'] = load_config()[1]
R['historical_id_from_archived_results'] = json.loads(arch('mac/evaluation/phase4/phase4_results.json'))['meta']['config_hash']
R['config_ids_manifest'] = {k: m['configuration'][k] for k in ('historical_recorded_id', 'current_yaml_parsed_id', 'i05_effective_id')}
i05m = json.loads(arch('paper/odi/revision/analyses/I-05_split_first_v2/manifest.json'))
i05met = json.loads(arch('paper/odi/revision/analyses/I-05_split_first_v2/metrics.json'))
R['i05_manifest_config_hash'] = i05m['config_hash']; R['i05_metrics_base_config_hash'] = i05met['base_config_hash']
# Which keys differ between the current YAML and the I-05 effective config?
cur = yaml.safe_load(arch('mac/evaluation/phase4/config.yaml').decode())
eff = json.loads(arch('paper/odi/revision/analyses/I-05_split_first_v2/config.json'))
def flat(d, pre=''):
    o = {}
    for k, v in d.items():
        o.update(flat(v, pre + k + '.') if isinstance(v, dict) else {pre + k: v})
    return o
fc, fe = flat(cur), flat(eff)
R['current_vs_i05_config_diff'] = {k: [fc.get(k), fe.get(k)] for k in sorted(set(fc) | set(fe)) if fc.get(k) != fe.get(k)}

# 4. vocab identities
from evaluation.phase4.data import vocab_sha
hv = json.loads(arch('mac/evaluation/phase4/vocab.json')); nv = json.loads(arch('paper/odi/revision/analyses/I-05_split_first_v2/vocab.json'))
R['historical_vocab'] = dict(size=len(hv), sha=vocab_sha(hv), sidecar=arch('mac/intel/artifacts/student_int8.vocab.sha').decode().strip(),
                             merges=len(json.loads(arch('mac/evaluation/phase4/canonical.json'))))
R['i05_vocab'] = dict(size=len(nv), sha=vocab_sha(nv), sidecar=arch('paper/odi/revision/analyses/I-07_export_v1/student_int8.vocab.sha').decode().strip(),
                      merges=len(json.loads(arch('paper/odi/revision/analyses/I-05_split_first_v2/canonical.json'))),
                      metrics_json_merges=i05met['canonical_merges'], manifest_vocab_sha=i05m['vocab_sha'])

# 5. I-07 -> I-05 linkage from archived bytes only
i07m = json.loads(arch('paper/odi/revision/analyses/I-07_export_v1/manifest.json'))
b = i07m['steps']['bundle']
R['i07_links_i05'] = dict(
    student_pt=b['student_pt_sha256'] == H(arch('paper/odi/revision/analyses/I-05_split_first_v2/student.pt')) == i05m['artifact_sha256']['student.pt'],
    manifest=b['manifest_sha256'] == H(arch('paper/odi/revision/analyses/I-05_split_first_v2/manifest.json')),
    vocab=b['vocab_sha'] == i05m['vocab_sha'] == vocab_sha(nv), split=b['split_fingerprint'] == i05m['split_fingerprint'],
    config=b['i05_config_hash'] == i05m['config_hash'])

# 6. every source manifest's own declared input/artifact hashes vs the archived bytes
src = {'I-05_torch': 'paper/odi/revision/analyses/I-05_split_first_v2/manifest.json',
       'I-02_controls': 'paper/odi/revision/analyses/I-02_pool_v1/manifest.json',
       'I-07_export': 'paper/odi/revision/analyses/I-07_export_v1/manifest.json'}
decl = {}
for name, path in src.items():
    sm = json.loads(arch(path)); n_ok = n_bad = 0; missing_entries = []
    for p, sha in sm.get('input_sha256', {}).items():
        if p not in m['files']: missing_entries.append(p); continue
        n_ok += m['files'][p]['sha256'] == sha; n_bad += m['files'][p]['sha256'] != sha
    for p, sha in sm.get('artifact_sha256', {}).items():
        key = (Path(path).parent / p).as_posix()
        if key not in m['files']: missing_entries.append(key); continue
        n_ok += m['files'][key]['sha256'] == sha; n_bad += m['files'][key]['sha256'] != sha
    decl[name] = dict(declared_ok=n_ok, declared_mismatch=n_bad, not_archived=missing_entries,
                      pipeline_entry_present=name in m['pipelines'])
R['source_manifest_declarations'] = decl

# 7. split fingerprint from archived splits/rows
sp = json.loads(arch('paper/odi/revision/analyses/I-05_split_first_v2/splits.json'))
R['i05_splits_fingerprint_field'] = sp.get('fingerprint')
R['historical_splits_same_bytes_as_i05'] = m['files']['mac/evaluation/phase4/splits.json']['sha256'] == m['files']['paper/odi/revision/analyses/I-05_split_first_v2/splits.json']['sha256']
try:
    from evaluation.phase4.data import rows_fingerprint
    rows = json.loads(arch('paper/odi/revision/analyses/I-05_split_first_v2/rows.json'))
    rows = rows['rows'] if isinstance(rows, dict) and 'rows' in rows else rows
    R['i05_rows_fingerprint_recomputed'] = rows_fingerprint(rows)
except Exception as ex:
    R['i05_rows_fingerprint_recomputed'] = f'not recomputed: {type(ex).__name__}: {ex}'[:200]

# 8. VCS state: manifest carries no revision; record what the checker sees
g = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
R['git_head'] = g('rev-parse', 'HEAD')
tracked_dirty = [k for k in m['files'] if g('status', '--short', '--', k)]
R['archived_paths_dirty_or_untracked_in_git'] = tracked_dirty
R['manifest_records_git_revision'] = any('git' in k.lower() for k in m) or 'git' in json.dumps(m['configuration']).lower()

# 9. required roles present as hashed entries
roles = ['mac/evaluation/phase4/config.yaml', 'paper/odi/revision/analyses/I-05_split_first_v2/config.json',
         'paper/odi/revision/analyses/I-05_split_first_v2/student.pt', 'mac/evaluation/phase4/student.pt',
         'mac/evaluation/phase3/boards.jsonl', 'mac/evaluation/phase4/teacher_data.jsonl',
         'mac/evaluation/phase4/infer.py', 'mac/selection.py', 'mac/evaluation/phase4/evaluate.py',
         'mac/evaluation/phase3/metrics3.py', 'mac/evaluation/coverage_metrics.py',
         'paper/odi/revision/analyses/I-05_split_first_v2/metrics.json']
R['required_roles_missing'] = [r for r in roles if r not in m['files']]
R['backends_declared'] = {k: v.get('backends') for k, v in m['pipelines'].items()}
print(json.dumps(R, indent=2))
