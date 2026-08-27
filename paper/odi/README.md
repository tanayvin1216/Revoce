# ODI 2026 paper draft — build notes

NeurIPS 2026 Workshop on On-Device Intelligence: Foundation Models under
Real-World Constraints (Sydney). **Papers track.**

## Files
- `main.tex` — the draft (5 pages main text + references + appendix, one pdf)
- `neurips_2026.sty` — copied from `paper/common/neurips_2026.sty`

## To build
```
cd paper/odi
tectonic main.tex   # or pdflatex twice (footnote / table refs)
```
Figures are pulled by relative path from `../../mac/evaluation/phase3/figures/`,
so build from inside `paper/odi/` with the repo checkout intact.

No TeX toolchain was on PATH when this draft was written (no `tectonic`,
`pdflatex`, `xelatex`, `lualatex`, MacTeX) — **the draft has NOT been compiled
and the page count is unchecked.** Install MacTeX/BasicTeX or `tectonic`, build,
and trim to 5 pages before the references if needed.

## Before submitting — checklist
- [ ] Compile; confirm main text ≤ 5 pages, references + appendix after it in the same pdf
- [ ] Delete every `\todo{}` (they render in red). Current markers:
      1. §Limitations — replace the "zero Intel-hardware measurements" paragraph if Tiber / Core Ultra numbers land
      2. Appendix B — anonymise or package the code (appendix names scripts, not a repo)
      3. Appendix C — no bill of materials exists; do not quote a device cost
- [ ] Verify every reference marked `% VERIFY` (Qwen2.5-VL, Whisper, OpenVINO/NNCF, KWickChat, PrAACT)
- [ ] Double-blind: no author names, no repo URLs, no identifying acknowledgements — appendix included
- [ ] Re-check numbers against `docs/VERIFIED_NUMBERS.md` (single source of truth) — if the
      Phase-3 `selftest_results.json` gets re-committed, keep 0.78; if not, the footnote explains 0.81
- [ ] If the teacher speedup is re-measured after memoisation
      (`cd mac && VOICELINK_PERCEPTION=0 python3 evaluation/phase4/evaluate.py --teacher-samples 8`),
      update §5 — until then the paper quotes 142.6× only as the committed pre-memoisation figure
- [ ] Keep the "Use of LLMs" section (both roles disclosed)
- [ ] Submit at https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/ODI
- [ ] **Deadline: Aug 29, 2026 23:59 AoE**

## Numbers used, and where they came from
| Number | Source |
|---|---|
| 4.68 / 3.88 distinct fn, 0.0004 / 0.0274 near-dup (69×), 1.00 / 0.78 coverage (0.81 June scorer), 0.525 / 0.333 exact | `docs/VERIFIED_NUMBERS.md` §A (`mac/evaluation/phase3/selftest_results.json`, re-scored 2026-08-23) |
| 600 rows, 559 after dedup, 391/84/84, 94-word vocab, 22 merges, 769→192→94 MLP | `docs/VERIFIED_NUMBERS.md` §B |
| 4.083 / 4.333 (0.942), 0.0006, 0.1865 / 0.1684 FAIL, 0.917, 0.761 / 0.845, 0.845 / 0.944, 0.056 | `docs/VERIFIED_NUMBERS.md` §B, `mac/evaluation/phase4/PHASE4_REPORT.md` |
| 47.081 ms (IQR 45.06–49.29), 6.71 s (IQR 6.63–6.97), 142.6× (committed, pre-memoisation) | `docs/VERIFIED_NUMBERS.md` §B |
| hedges 2.107 → 0.012 (teacher 0.321), 72.6% → 0% | `docs/VERIFIED_NUMBERS.md` §B (file-only) |
| 2 of 84 / 1 of 71 open-answer rows; 32.5%, 49.1%, 12.2%, 8.5× → 56.6% | `docs/VERIFIED_NUMBERS.md` §B/§C |
| IR parity 1.43e-6, 360 calib rows, INT8 0.9505 / 0.6667, logit delta 1.053 / 0.228, 657 / 661 / 235 KB (2.8×) | `docs/VERIFIED_NUMBERS.md` §E, `mac/intel/BENCHMARKS.md` |
| head p50 0.015 / 0.025 / 0.031 ms; benchmark_app 48,354 / 36,969 FPS | `mac/intel/BENCHMARKS.md` |
| end-to-end 16.8 / 15.7 / 15.3 ms p50, p90 56.6 / 19.9 / 27.4, Jaccard 0.914 | `mac/intel/BENCHMARKS.md` "End-to-end board latency — Apple M4" (uncommitted section) |
| "71 of 73 ms re-embedding the vocab" | commit 55306d8 message + `mac/intel/bench_e2e.py` docstring; `docs/mentor/voicelink_technical_overview.tex` §7.3 |
| MiniLM OV cosine 1.000, 5.6 vs 16.0 ms | `docs/mentor/voicelink_technical_overview.tex` §7.3 |
| Whisper n=1 0.35 vs 0.42 s; n=10 real speech 681/795 vs 1162/1157 ms | `docs/VERIFIED_NUMBERS.md` §E; `mac/intel/BENCHMARKS.md` "Whisper — Apple M4" |
| VLM 0.93 s @256 px, 2.05 s @512 px; change gate 0.214 ms | `docs/VERIFIED_NUMBERS.md` §E (file-only) |
| 144 tests, 121 tap rows, config hash, split fingerprint | `docs/VERIFIED_NUMBERS.md` §B/§F/§G |

**Deliberately not used:** the `~7 ms / ~900×` bare-MLP figure, the "$15 wearable"
cost (no BOM), and any Phase-1 figure numbers (not in VERIFIED_NUMBERS).
Note: the task brief quoted board Jaccard as 0.92; the current BENCHMARKS.md table
says 0.914 and the paper uses 0.914.
