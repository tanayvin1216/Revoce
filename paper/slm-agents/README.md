# SLM-Agents 2026 paper — "Function Survives Distillation, Lexis Does Not"

NeurIPS 2026 Workshop **SLM-Agents: 1st Workshop on SLMs for Agentic Systems** (Paris).
Long-paper track: **up to 6 content pages** (NeurIPS workshop template; references and
appendix after the body), double-blind, non-archival.

- Submission: OpenReview — https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/SLM-Agents
- **Deadline: Sun 6 Sep 2026, AoE** (= Mon 7 Sep 7:59 AM ET). Notification 22 Sep 2026.
- Workshop policy: not under review at the NeurIPS 2026 main program; not previously
  published at a major ML/AI venue. Both hold. The same system appears, with different
  claims, in the ODI / AXIOM / RTCA / NEmo / GenAI4Health submissions — send the chairs
  the one-line "different papers, different claims" note before the deadline.

## Files

| file | purpose |
|---|---|
| `main.tex` | the paper (single file; style file embedded; bibliography inline) |
| `main.pdf` | last local build (see the build note below for the current page layout) |

## Build

```
cd paper/slm-agents
tectonic main.tex
```

Body (everything before References) currently ends on page 5 of 6; check the page count after every edit.

## Where the numbers come from

Every number is from `docs/VERIFIED_NUMBERS.md` §0a (gates for the served student), **§I**
(the v2 per-function breakdown) and **§I.2** (the v3 student, the prior and random baselines,
the gate run over all four, and the slot confusion), all built 2026-09-02 by

```
cd mac && VOICELINK_PERCEPTION=0 python3 evaluation/phase4/function_transfer.py --student {v2|v3|prior|random}
cd mac && VOICELINK_PERCEPTION=0 python3 evaluation/phase4/function_transfer.py --student {v2|v3|prior|random} --gates
cd mac && VOICELINK_PERCEPTION=0 python3 evaluation/phase4/function_transfer.py --words
```

which write `mac/evaluation/phase4/function_transfer{,_v3,_prior,_random}.json` and
`function_transfer_gates_{v2,v3,prior,random}.json`. The v2 run reproduces the committed gate
values (retention 0.953, dynamic_overlap 0.067, full overlap 0.594) exactly. Unit tests:
`mac/tests/test_function_transfer.py` (synthetic boards, no model).

| Paper number | VERIFIED_NUMBERS row |
|---|---|
| Table 1 (J_F, J_W, retention, sfp, overlap, diagonal share, variety, coverage × v2/v3/prior/random) | §I.2 |
| Table 2 (presence / CI / lexical overlap, v2 and v3) | §I per-function rows (v2), §I.2 (v3) |
| Table 3 (slot confusion for emotion and I-dont-know, v2 and v3) | §I.2 confusion rows |
| Prior board words; random board definition | §I.2 preamble |
| 35 of 40 I-dont-know items representable; 38 distinct emotion phrases, 14 representable; "I'm not hungry" ×5, "I'm feeling a bit anxious" ×4 | §I "where the two lost functions' teacher words live" (`function_transfer.py --words`) |
| Appendix gate table (0.953, 0.0/0.0006, 0.1893/0.1735, 0.952, 0.789/0.873, 0.067, 10.1 ms / 7.05 s) | §0a |

Do not quote numbers from any other document.

## Pre-submission checklist

- [ ] No `\todo{}` markers remain (the helper macro has been removed).
- [ ] Bibliography: all 17 entries verified against arXiv / Crossref on 2026-09-02 (ids and DOIs in the entries).
- [ ] Anonymity sweep: `grep -i voicelink main.tex` returns nothing (it does); no repo URL, no
      names, no identifying file paths — appendix included.
- [ ] Body ≤ 6 pages with the **unmodified** `neurips_2026.sty` (no `\vspace` hacks).
- [ ] Abstract numbers match Tables 1–3 (0.678 / 0.718 / 0.953 / 1.091 / 0.032 / 0.067 / 0.716 / 0.657 / 1.003 / 0.507 vs 0.789).
- [ ] "Use of LLMs" paragraph present (end of Limitations).
- [ ] Limitations state: synthetic scenarios, SLM teacher and LLM-assisted acceptable moves,
      embedding labeller not human-validated, rank-paired confusion is a convention, n=84,
      two tiny cells, no human data.
- [ ] "Anonymised companion" appears at most twice (currently: §3 hedge wall, §6 representability).
- [ ] Email the chairs the "different papers, different claims" note.
- [ ] Upload PDF to OpenReview before **6 Sep 23:59 AoE**; re-check the page count on the upload.

## Relationship to the other submissions

Shares only bibliography entries (MMR, MiniLM, Qwen2.5-VL, Hinton) and the appendix gate
table with `paper/axiom/` and `paper/odi/`; no body text is shared. The AXIOM paper's ceiling
result is referred to once as "a companion representability analysis (anonymised)". The NEmo
paper (`paper/nemo/`, on `pivot/neuro-symbolic`) owns the policy-ablation claim; this paper
does not ablate the policy — it compares two students under the SAME policy and reads the
I-dont-know loss as a student × policy interaction.
