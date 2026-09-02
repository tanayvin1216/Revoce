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
| `main.pdf` | last local build (6 pages: body ends on p.4, Limitations + refs + appendix from p.5) |

## Build

```
cd paper/slm-agents
tectonic main.tex
```

Body currently ends on page 4 of 6, so there is about a page of slack under the 6-page limit.

## Where the numbers come from

Every number is from `docs/VERIFIED_NUMBERS.md` §0a (gates for the committed student) and
**§I** (the per-function breakdown, built 2026-09-02). §I is produced by

```
cd mac && VOICELINK_PERCEPTION=0 python3 evaluation/phase4/function_transfer.py
```

which writes `mac/evaluation/phase4/function_transfer.json` and reproduces the committed gate
values (retention 0.953, dynamic_overlap 0.067, full overlap 0.594) exactly. Unit test:
`mac/tests/test_function_transfer.py` (synthetic boards, no model).

| Paper number | VERIFIED_NUMBERS row |
|---|---|
| J_F 0.678 [0.650, 0.707]; J_W 0.032 [0.015, 0.052] | §I Jaccard rows |
| Table 1 (presence / retention / CI / slots / lexical overlap) | §I per-function rows |
| 35 of 40 I-dont-know items representable; 38 distinct emotion phrases, 14 representable | §I "where the two lost functions' teacher words live" |
| Appendix gate table (0.953, 0.0/0.0006, 0.1893/0.1735, 0.952, 0.789/0.873, 0.067, 10.1 ms / 7.05 s) | §0a |

Do not quote numbers from any other document.

## Pre-submission checklist

- [ ] Resolve or delete every `\todo{}` (none as of this draft) and the `\newcommand{\todo}` line.
- [ ] Resolve the `% VERIFY` in the bibliography (Qwen2.5-VL arXiv id / authors).
- [ ] Anonymity sweep: `grep -i voicelink main.tex` returns nothing (it does); no repo URL, no
      names, no identifying file paths — appendix included.
- [ ] Body ≤ 6 pages with the **unmodified** `neurips_2026.sty` (no `\vspace` hacks).
- [ ] Abstract numbers match Tables 1–2 (0.678 / 0.032 / 0.953 / 6.7 % / 38 distinct in 42 / 14).
- [ ] "Use of LLMs" paragraph present (end of Limitations).
- [ ] Limitations state: synthetic scenarios, LLM teacher and LLM judge, embedding labeller not
      human-validated, n=84, two tiny cells, no human data.
- [ ] Email the chairs the "different papers, different claims" note.
- [ ] Upload PDF to OpenReview before **6 Sep 23:59 AoE**; re-check the page count on the upload.

## Relationship to the other submissions

Shares only bibliography entries (MMR, MiniLM, Qwen2.5-VL, Hinton) and the appendix gate
table with `paper/axiom/` and `paper/odi/`; no body text is shared. The AXIOM paper's ceiling
result is referred to as "a companion analysis (anonymised)". The NEmo paper
(`paper/nemo/`, on `pivot/neuro-symbolic`) owns the policy-ablation claim; this paper only
observes that the hedge cap explains the I-dont-know loss and does not ablate it.
