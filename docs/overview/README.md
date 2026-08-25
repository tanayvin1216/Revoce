# Project overview

`voicelink_overview.tex` — the full write-up of VoiceLink: architecture, the seam,
how a board is produced, the models, distillation, the evaluation gates (including
the two that do not pass), the closed-vocabulary ceiling, the Intel/OpenVINO results,
privacy, status, and how to run it.

Build:

    tectonic voicelink_overview.tex      # or: pdflatex twice, for the ToC

Every number in it was checked against a committed artifact on 2026-08-25, not
against a summary document. If a figure here ever disagrees with the JSON it cites,
the JSON wins — see `docs/VERIFIED_NUMBERS.md`.
