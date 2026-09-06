"""Verify the refreshed recount and checker-requested provenance; no models.
Assumption: reuse checked I-05, not separate I-05b or diagnostic configurations.
Run from mac/ with VOICELINK_PERCEPTION=0; all outputs are exclusive-create.
"""
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[4]
A = ROOT / 'paper/odi/revision/analyses'
assert Path.cwd() == ROOT / 'mac' and os.environ['VOICELINK_PERCEPTION'] == '0'
start = time.perf_counter()
r = json.loads((A/'I-16_recount_v2/recount.json').read_text())
c = json.loads((A/'I-16_recount_check.json').read_text())
for label, result in r['results'].items():
    assert result['vocab_size'] == c['results'][label]['vocab_size']
    assert result['canonical_mappings'] == c['results'][label]['canonical_mappings']
    for scope, v in result['corpora'].items():
        other = c['results'][label]['corpora'][scope]
        assert v['slots'] == other['slots'] and v['rows'] == other['rows']
        for mode, key in [('exact', 'exact'), ('canonical', 'canonical')]:
            assert v[mode]['hits'] == other[key+'_hits']
            assert v[mode]['zero_hit_rows'] == other['zero_'+key]
old = ROOT/'mac/evaluation/phase4'
new = A/'I-05_split_first_v2'
raw = [json.loads(x) for x in (old/'teacher_data.jsonl').read_text().splitlines() if x.strip()]
rows = json.loads((new/'rows.json').read_text())
raw_boards = {(x['scene'],x['transcript']):x['board'] for x in raw}
assert all(raw_boards[(x['scene'],x['transcript'])] == x['board'] for x in rows)
assert set(json.loads((new/'vocab.json').read_text())) < set(json.loads((old/'vocab.json').read_text()))
oc = json.loads((old/'canonical.json').read_text())
nc = json.loads((new/'canonical.json').read_text())
assert all(oc[k] == v for k,v in nc.items())
assert not r['changed_inputs']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(ROOT/p) == h for p,h in r['input_sha256'].items())
log = (A/'I-16_recount_v2/pytest.log').read_text()
assert '18 failed, 222 passed, 2 skipped' in log
old_r = json.loads((A/'I-16_recount_v1/recount.json').read_text())
changed = [p for p,h in old_r['input_sha256'].items() if r['input_sha256'].get(p) != h]
added = sorted(set(r['input_sha256']) - set(old_r['input_sha256']))
out = dict(issue='I-16', checker_counts_match=True, gathered_identical_teacher_boards=len(rows),
           vocab_strict_subset=True, canonical_map_subset=True, tests_passed=222, tests_failed=18,
           tests_skipped=2, tests_total=242, changed_since_v1=changed, added_since_v1=added,
           wall_seconds=time.perf_counter()-start,
           sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), A/'I-16_result.md', A/'I-16_check.md', A/'I-16_recount_v2/recount.json', A/'I-16_recount_check.json']})
with (A/'I-16_verification_v2.json').open('x') as f:
    json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
