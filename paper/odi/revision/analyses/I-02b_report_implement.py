"""Render the frozen three-student comparison; refuse existing output files."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / 'I-02b_pool_v2/numbers.json').read_text())
manifest = json.loads((HERE / 'I-02b_pool_v2/manifest.json').read_text())
repeat = json.loads((HERE / 'I-02b_pool_repeat_v1/manifest.json').read_text())
validation = json.loads((HERE / 'I-02b_validation_v1.json').read_text())
assert validation['harness_free_row_windows_checked'] == 4536
labels = ('historical', 'I-05', 'I-05b')
windows = ('full7', 'full10', 'dyn6')
arms = ('learned', 'cosine', 'frequency', 'vocab_order', 'full', 'random')

def contrast(label, window):
    # Negate the stored cosine-minus-learned effect, reversing CI endpoints.
    v = data[label]['arms']['cosine']['windows'][window]['baseline_minus_learned']['dynamic_coverage']
    return dict(difference=-v['margin'], ci95=[-v['ci'][1], -v['ci'][0]])

allowed = ('Contribution (2): an evaluation of learned and no-task-trained-head candidate filters under a shared constrained selector, with vocabulary- and window-dependent tradeoffs. On I-05b, learned-minus-cosine dynamic coverage is +0.0141 [-0.0845, 0.1127] at full@10/dyn@6 and +0.0282 [-0.0986, 0.1549] at full@7; coverage superiority or equivalence is not established. Learned improves the I-05b dynamic-only DF and SFP proxies relative to cosine, while cosine produces fewer hedges.')
forbidden = ('A general learned-head advantage or dispensability; a persistent cosine coverage advantage on the clean 95-entry vocabulary; coverage equivalence; superiority across all proxies/windows; constraints-alone causation; a causal estimate of leakage removal; clinical benefit; or exported/head-free latency inherited from historical INT8 measurements.')
files = [
    'I-02b_result.md', 'I-02b_pool_ablation_implement.py', 'I-02b_report_implement.py',
    'I-02b_pool_v2/numbers.json', 'I-02b_pool_v2/manifest.json',
    'I-02b_pool_repeat_v1/numbers.json', 'I-02b_pool_repeat_v1/manifest.json',
    'I-02b_validation_v1.json', 'I-02b_validation_v1.log',
    'I-02b_run_v1.log', 'I-02b_run_v2.log', 'I-02b_repeat_v1.log', 'I-02b_tests_v1.log',
]
payload = dict(issue='I-02b', status='done', numbers=dict(
    learned_minus_cosine_coverage={label:{w:contrast(label,w) for w in windows} for label in labels},
    I05b_coverage_counts={w:{a:round(data['I-05b']['arms'][a]['windows'][w]['means']['dynamic_coverage']*71) for a in arms} for w in windows},
    coverage_denominator=71, test_rows=84, tests_passed=43,
    verified_row_windows=4536, repeat_identical=True,
    historical_and_I05_exactly_match_prior=True),
    files=['paper/odi/revision/analyses/'+f for f in files],
    claim_allowed=allowed, claim_not_allowed=forbidden)

text = f'''# I-02b — IMPLEMENTER result: candidate pools on the clean matched-vocabulary student

Status: **done**. This follows the user-authorized I-02b extension of frozen issue I-02. Read stage3_implement.md, orchestrator_notes.md, the I-02 frozen plan and relevant audit/main.tex sections, I-02_result.md and I-05b_result.md. Neither analyses/I-02_check.md nor analyses/I-05b_check.md existed when dependencies were read; no independent checker approval is implied.

## 1. What was run

`I-02b_pool_ablation_implement.py` is a new copy of the I-02 six-arm runner, extended only to load the I-05b bundle, validate its training counts and reference boards, and assert exact historical/I-05 result continuity. It reruns all three students, all six arms and all three windows. Only the process-local candidate-pool adapter changes; production selector, answer extraction/reservation, MMR, taxonomy, hedge caps, backfill and pinning are unchanged. Non-learned arms receive model=None. No teacher calls, retraining, production-source edits, manuscript edits, export or latency benchmark.

Exact successful commands, from `mac/`:

```sh
VOICELINK_PERCEPTION=0 /usr/local/bin/python3 ../paper/odi/revision/analyses/I-02b_pool_ablation_implement.py --out ../paper/odi/revision/analyses/I-02b_pool_v2 > ../paper/odi/revision/analyses/I-02b_run_v2.log 2>&1
VOICELINK_PERCEPTION=0 /usr/local/bin/python3 ../paper/odi/revision/analyses/I-02b_pool_ablation_implement.py --out ../paper/odi/revision/analyses/I-02b_pool_repeat_v1 > ../paper/odi/revision/analyses/I-02b_repeat_v1.log 2>&1
VOICELINK_PERCEPTION=0 HF_HUB_OFFLINE=1 /usr/local/bin/python3 -m pytest -q tests/test_metric_windows_i03.py tests/test_phase3.py tests/test_phase4.py > ../paper/odi/revision/analyses/I-02b_tests_v1.log 2>&1
VOICELINK_PERCEPTION=0 /usr/local/bin/python3 ../paper/odi/revision/analyses/I-02_verify_implement.py ../paper/odi/revision/analyses/I-02b_pool_v2 ../paper/odi/revision/analyses/I-02b_pool_repeat_v1 ../paper/odi/revision/analyses/I-02b_validation_v1.json > ../paper/odi/revision/analyses/I-02b_validation_v1.log 2>&1
VOICELINK_PERCEPTION=0 /usr/local/bin/python3 ../paper/odi/revision/analyses/I-02b_report_implement.py
```

Main experiment **{manifest['wall_seconds']:.3f} s**; independent process repeat **{repeat['wall_seconds']:.3f} s**; harness-free verifier **{validation['wall_seconds']:.3f} s**. Focused tests: **43 passed**, 5 warnings, 4.38 s. The repeat overlapped with pytest; these are execution durations, not board latency. An initial script-creation attempt used a repository-relative path from mac/ and failed before creating the script; the resulting missing-script invocation used `--out ../paper/odi/revision/analyses/I-02b_pool_v1` and is retained in I-02b_run_v1.log. The successful run uses fresh v2 artifacts; no result was overwritten.

Seed 0 for torch and NumPy; CPU, one torch thread; offline cached torch MiniLM and torch head. Random uses a fresh default_rng(0) for each arm/bundle, samples 18 without replacement in saved test-index order, preserves draw order, and reuses those boards across windows. Bootstrap is the exact I-02 paired row percentile bootstrap, 10,000 resamples, seed 0, pointwise 95% intervals. The unchanged analysis-config SHA-256 is `{manifest['analysis_config_sha256']}`. Versions: {json.dumps(manifest['versions'], sort_keys=True)}.

Bundle identities: historical config `d6ce2e56d50b`, vocabulary identity `865b8083da276824` (95); I-05 config `b691659cecf3`, vocabulary identity `30e9becb9a556cd8` (45); I-05b config `b72bb3ec7254`, vocabulary identity `0f3bb094cd439111` (95). All use split 391/84/84, fingerprint `ab51f7fba4c9c39e`. Full checkpoint/vocab/canonical/data/source/dependency hashes and numbers.json hash are saved in each manifest and checked unchanged at run completion. I-05b checkpoint SHA-256 is `c3ccc2fe4a92f6e21638f073a90ce58a9397e41d6e7f1d61ff23c0742c71697c`, vocab file SHA-256 `31fee92d89a82fc8cb51d5802579a7c6189bd8fe32e91f6558f8873986b44f5a`, canonical SHA-256 `a7b25e0cf5a15d7382e5265c74d7e0e933beee87c819f3d81be27c93546734cd`.

Checks passed: every learned board equals its dependency's saved board; training canonical multi-hot counts equal preprocessing.json for both clean bundles; every board has four pins and six dynamic entries; historical/I-05 complete outputs equal I-02_pool_v1/numbers.json exactly; repeat complete numbers.json is identical; the reused verifier directly re-scores **4,536 row/window/arm/student combinations** with metrics3, bypassing held_out_eval, and re-derives every aggregate and baseline-minus-learned bootstrap interval. This is implementer validation, not an independent checker report. No mac source changed, so the conditional full-suite requirement does not apply.

## 2. Results — single side-by-side comparison

Each ordinary cell is **DF / ND / SFP / mean hedges / dynamic coverage count out of 71**. DF, near-duplicate pair rate (ND), same-function-pair rate (SFP), and hedges use all 84 rows. Coverage excludes the 13 rows without non-hedge authored targets; hedge predictions can still earn coverage. All counts can be converted to rates by dividing by 71. The Δ rows give **learned minus cosine coverage proportion [95% CI]**, so positive favors learned; stored JSON retains I-02's opposite baseline-minus-learned sign for every arm and metric. The sign reversal reverses CI endpoints as well.

Learned is top-18 logits; cosine is top-18 embed(vocab) @ build_query_vector(embed, scene, transcript, None); frequency is top-18 summed canonical-aware multi-hot labels on the 391 training rows; vocab_order is vocab[:18], labelled historical pre-split vocabulary-order control for historical and training-fitted vocabulary-order control for I-05/I-05b. Frequency/order pool overlap is 13/18, 17/18 and 15/18 respectively: both definitions remain necessary. Full uses all 95/45/95 vocabulary entries in frozen order; random follows the policy above. Score ties use explicit NumPy quicksort on negative scores. Open-answer spans enter after pool selection as before.

Full@7 takes the first seven entries (four student pins plus three dynamic); full@10 takes all ten; dyn@6 removes core before taking six. Full@10 and dyn@6 dynamic coverage are **one measurement**, not independent confirmations. All other I-02 metrics, including harness variety, all-target coverage, directional agreement and teacher-relative margins, plus ordered pools, boards, per-row metrics and all five baseline-minus-learned metric CIs, are retained in numbers.json.

| Window / arm | Historical 95 (pre-split label fitting) | I-05 clean 45 | I-05b clean 95 |
|---|---|---|---|
'''
for w in windows:
    for a in arms:
        cells=[]
        for label in labels:
            m=data[label]['arms'][a]['windows'][w]['means']
            cells.append(' / '.join(f"{m[k]:.4f}" for k in ('df','near_dup','sfp','hedges'))+f" / {round(m['dynamic_coverage']*71)}/71")
        text+='| '+w+' / '+a+' | '+' | '.join(cells)+' |\n'
    cells=[]
    for label in labels:
        c=contrast(label,w)
        cells.append(f"{c['difference']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]")
    text+='| '+w+' / **Δ coverage learned − cosine** | '+' | '.join(cells)+' |\n'

text+='''
Current paper Table 2 reports only historical learned/teacher full@7: DF 4.095/4.298, ND 0.0000/0.0006, SFP 0.1893/0.1735, dynamic coverage 0.789 (56/71)/0.873 (62/71). Its historical student cells reproduce above; no paper counterpart exists for the other arms or clean students. The paper's 0.012 student hedge count is stale; the evaluated historical full@7 mean is 0.1905. Cached teacher full@10/dyn@6 coverage is 68/71 for all bundles.

For I-05b, cosine-minus-learned DF is +0.2738 [−0.0119, +0.5476] at full@10, +0.1548 [−0.0476, +0.3452] at full@7, and **−0.2500 [−0.4762, −0.0119] at dyn@6**. SFP differences are 0.0000 [−0.0167, +0.0172], −0.0051 [−0.0227, +0.0130], and **+0.0278 [+0.0040, +0.0524]**, respectively; both dynamic-only comparisons favor learned. Cosine has fewer hedges at all windows, including −0.4167 [−0.5238, −0.3095] at full@10/dyn@6. Thus the original I-02 across-bundle wording cannot simply be extended to I-05b.

## 3. Interpretation

Contribution (2) can be stated as **“an evaluation of learned and no-task-trained-head candidate filters under a shared constrained selector, showing vocabulary- and window-dependent tradeoffs.”** On the clean matched-size student, the I-05 cosine coverage advantage disappears (learned 61/71 versus cosine 60/71 at full@10/dyn@6, difference +0.0141 [−0.0845, +0.1127]; full@7 50/71 versus 48/71, +0.0282 [−0.0986, +0.1549]), while the learned pool improves dynamic-only DF/SFP proxies and cosine reduces hedges. This supports those bounded proxy findings, but establishes neither coverage superiority nor equivalence, a general benefit or dispensability of the head, nor that constraints alone cause the quality; the stronger head-free deployment contribution remains future work pending its own quality/latency evaluation.

## 4. Assumptions and known weaknesses

- “Clean” refers to training-only label vocabulary/canonical/statistics fitting, not a pristine evaluation: global context deduplication/category caps, overlapping authored template families and repeated inspection of this test partition remain. No fresh intended-user data or human semantic validation was collected.
- I-05b matches historical vocabulary size, not content: 67/95 entries overlap. Its min_freq=2/max_size=116 cap selects within an alphabetically ordered frequency-2 tier; this is a training-size control, not a validated hyperparameter setting. Across-student differences conflate vocabulary content, size and single-seed model fitting and are not a causal estimate of leakage removal.
- Paired row intervals condition on fixed checkpoints and one random-pool realization. They omit training/random-seed variability and correlated template families; all intervals are pointwise with no multiplicity correction. Intervals containing zero are not equivalence tests. The dyn@6 learned DF/SFP advantages are bounded authored-proxy findings, not global superiority.
- Selection reuses the evaluation taxonomy and pretrained MiniLM; “no-task-trained-head” does not mean no learned components. Polarity-insensitive matching and hedge prediction credit limit semantic interpretation. More functions/fewer repeated functions are not validated communication outcomes.
- Full-vocabulary changes pool size as well as membership; every arm retains contextual downstream selection and optional open answers. Frequency controls are not context-sensitive pool filters, though their final boards are. The adapter is process-local and not thread-safe.
- Torch quality only: no new OpenVINO/INT8 artifacts, no deployment timing, and no inherited historical head-free latency claim. Orchestrator's external OpenVINO execution requirement was read and is not needed here. Other concurrent issue files were left untouched; later source drift must be checked against these manifests.

'''
text+='```json\n'+json.dumps(payload,indent=2)+'\n```\n'
with (HERE/'I-02b_result.md').open('x') as f:
    f.write(text)
print(json.dumps(payload,indent=2))
