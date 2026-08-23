# RTCA 2026 short-paper draft — build notes

## Files
- `voicelink_rtca.tex` — the draft (short-paper track, 4pp)

## To build
1. Download the **NeurIPS 2026 author kit** and put `neurips_2026.sty`
   in this directory.
2. `pdflatex voicelink_rtca.tex` (run twice)

Without the `.sty` it still compiles in a fallback 1-inch-margin layout so you
can draft — but **the page count will be wrong**. Only check the 4-page limit
after the real style file is in place.

## Before submitting — checklist
- [ ] Delete every `\todo{}` (they render in red)
- [ ] Write the **Use of LLMs** section — RTCA adopts the NeurIPS Main Track LLM
      policy, and this project has *two* separate LLM roles to disclose:
      AI-assisted development, and LLMs as system components (teacher, judge)
- [ ] Verify every reference marked `% VERIFY` — four of the eight are carried
      from project notes and have not been checked
- [ ] Confirm no author names, no repo URLs, no identifying acknowledgements
      (double-blind, and the appendix must comply too)
- [ ] Re-check every number against a freshly-run eval, not against
      `COMPETITION_EVIDENCE.md` — see `docs/SUBMISSION_SPRINT.md` §7
- [ ] Submit at
      https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/RTCA
- [ ] **Deadline: Aug 29, 2026 AoE** = Sun Aug 30, 7:59 AM ET

## Numbers used, and where they came from
| Number | Source file |
|---|---|
| 4.68 / 3.88 distinct fn, 0.0004 / 0.0274 near-dup, 1.00 / 0.81 coverage, 0.567 / 0.342 exact | `mac/evaluation/phase3/selftest_results.json` |
| 47.081 ms, 6.71 s, 142.6× | `mac/evaluation/phase4/phase4_results.json` |
| 4.083 / 4.333 (0.942), 0.0006, 0.1865 / 0.1684, 0.917 | same |
| 0.761 / 0.845 dynamic coverage, 0.056 agreement | same |
| IR parity 1.4e-6, INT8 2.7× / 0.950 mean / 0.667 min | `mac/intel/BENCHMARKS.md` |
| whisper INT8 0.35 s vs 0.42 s | `phase_status.json` phase 5 |
| 32.5% representable, 49.1% filler, 0.232 → 0.165 | `docs/claude/working-notes/rc3-open-answers/NOTES.md` |

**Not used, deliberately:** the `~7 ms / ~900×` figure from
`COMPETITION_EVIDENCE.md`. It is the bare MLP forward pass, not an end-to-end
board, and it contradicts the current measured value.
