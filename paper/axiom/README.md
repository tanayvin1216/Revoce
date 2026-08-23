# AXIOM 2026 paper — "The closed-vocabulary ceiling in distilling an open-vocabulary teacher"

NeurIPS 2026 Workshop **AXIOM: Foundations of Efficient Deep Learning** (Paris).
Research-paper track: **4 pages max excluding references and appendix**, double-blind.

- Submission: OpenReview — https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/AXIOM
- **Deadline: Sat 29 Aug 2026, 23:59 UTC = Sat 29 Aug 7:59 PM ET.** This is UTC, **not AoE**
  (AoE would be 10 hours later — do not rely on it).

## Files

| file | purpose |
|---|---|
| `voicelink_axiom.tex` | the paper (single file, bibliography inline) |
| `neurips_2026.sty` | copied from `paper/common/neurips_2026.sty`; anonymous by default |
| `voicelink_axiom.pdf` | last local build (5 pages: body ends on p.4, refs + appendix on p.5) |

## Build

```
cd paper/axiom
tectonic voicelink_axiom.tex          # single pass is enough with tectonic
# or: pdflatex voicelink_axiom.tex && pdflatex voicelink_axiom.tex
```

Check the page count after every edit: the body (everything before `References`)
must end on or before page 4. As of 2026-08-23 it ends on the last lines of page 4,
so there is essentially **zero slack** — any added sentence must be paid for.

## Where the numbers come from

Every number is from `docs/VERIFIED_NUMBERS.md` §C (representability / vocab sweep) and
§D (0.232 -> 0.165 data-volume result), plus `docs/claude/working-notes/rc3-open-answers/NOTES.md`
§6 for the "2 of 84 rows" caveat. Do not quote numbers from any other document.

Derived in the paper (not in VERIFIED_NUMBERS): the log fit `C(V) = 0.121 ln V - 0.252`
(OLS over the five finite sweep rows; max residual 0.012) and the "each doubling buys ~0.08"
line. Reproduce with:

```
python3 -c "
import numpy as np
V=np.array([94,124,292,752,800]); c=np.array([.307,.332,.425,.553,.566])
A=np.vstack([np.ones(5),np.log(V)]).T; b=np.linalg.lstsq(A,c,rcond=None)[0]
print(b, np.abs(A@b-c).max(), b[1]*np.log(2))"
```

## Pre-submission checklist

- [ ] Resolve or delete every `\todo{}` (grep `\\todo` — four as of this draft; list below).
- [ ] Delete the `\newcommand{\todo}` line once none remain.
- [ ] Resolve every `% VERIFY` in the bibliography (Heaps 1978 citation form; Qwen2.5-VL arXiv id/authors).
- [ ] Confirm the "5x data" multiplier against the committed corpora (360 authored vs 580 merged rows) and fix the footnote in §3.
- [ ] Confirm the exact teacher model id and replace the footnote in §2.
- [ ] Anonymity sweep: no project name, no repo URL, no author names, no identifying file paths
      in the body. `grep -i voicelink voicelink_axiom.tex` must return only comment lines (or nothing).
- [ ] Body <= 4 pages with the **unmodified** `neurips_2026.sty` (no `\vspace` hacks, no font changes).
- [ ] Abstract numbers match §4 tables (32.5 / 49.1 / 12.2 / 0.325 / 51 % / 55 % at 8x).
- [ ] "Use of LLMs" paragraph present (it is, end of §8) — also fill in the NeurIPS checklist item if the workshop asks for it.
- [ ] Limitations paragraph states: synthetic scenarios, LLM teacher and LLM judge, single domain / teacher, n=600, no human data.
- [ ] Optional: simulate the toy model (§5) and add a figure to the appendix (appendix does not count toward the limit).
- [ ] Optional: attach the anonymised counting script (`representability.py`) as supplementary material.
- [ ] Upload PDF to OpenReview before **29 Aug 23:59 UTC**; re-check the page count on the uploaded PDF.

## Open `\todo{}` markers in this draft

1. §2 footnote — confirm exact teacher model id.
2. §3 footnote — reconcile the logged "5x" data multiplier with the 360 -> 580 row counts.
3. §5 end — optional toy-model simulation figure for the appendix.
4. Appendix B — attach the anonymised counting script as supplementary material.

## Relationship to the other submission

`paper/rtca/` is the systems/turn-taking paper for RTCA (Sydney, AoE deadline). This paper
reuses only the bibliography entries for MMR, MiniLM and Qwen2.5-VL, and shares no text.
Both papers must state the same limitations; if one changes, update the other.
