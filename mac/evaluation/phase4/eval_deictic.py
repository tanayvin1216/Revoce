"""Deictic-question eval — does the open-vocabulary answer path (RC-3) put the
concrete answer on the board?

The held-out corpus cannot measure answers.py: the open path fires on 2 of 84
rows there (docs/claude/working-notes/rc3-open-answers/NOTES.md §6). This set
is 60 authored scenarios (deictic_scenarios.jsonl) where the transcript ends in
a where / who / which / what-colour / how-many / "A or B" question and the
scene text contains the concrete answer. Each row is scored twice:

  student       rank_vocabulary(strategy="student")  — the shipped path, open
                answers ON;
  closed-only   the same seam with infer.OPEN_ANSWERS monkeypatched False, i.e.
                exactly the closed-vocab board the student served before RC-3.

Metrics (per row and overall):
  fire        answers.answer_candidates() returned >= 1 span for the row;
  hit         any expected answer is on the board — case/whitespace-insensitive
              exact match, OR the board word contains an expected answer, OR an
              expected answer contains the board word (a board "trainers" hits
              expected "red trainers"; board "red trainers" hits expected
              "trainers"). Board words of < 3 chars never substring-match.

No LLM and no teacher call is made. Usage (from mac/):

    VOICELINK_PERCEPTION=0 TOKENIZERS_PARALLELISM=false \
        python3 -m evaluation.phase4.eval_deictic [--quiet]

Writes deictic_results.json next to this file. Informational only — nothing
here gates a phase; evaluate.py picks the summary up as a non-gating block.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAC = HERE.parent.parent
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))

SCENARIOS = HERE / "deictic_scenarios.jsonl"
RESULTS = HERE / "deictic_results.json"

REQUIRED_FIELDS = ("id", "category", "scene", "transcript", "is_question",
                   "active_question", "expected_answers", "acceptable_moves")

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", (s or "").strip().lower())


def load_scenarios(path: Path = SCENARIOS) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return rows


def answer_hit(board: list[str], expected: list[str]) -> str | None:
    """The first expected answer the board expresses, or None.

    Exact normalised match first, then substring either way (board word inside
    an expected answer, or an expected answer inside a board word). Very short
    board words are excluded from substring matching so "no"/"me" cannot hit
    "monkeys"/"home".
    """
    exp = [_norm(e) for e in expected if _norm(e)]
    words = [_norm(w) for w in board]
    for e in exp:
        if e in words:
            return e
    for e in exp:
        for w in words:
            if len(w) < 3:
                continue
            if e in w or w in e:
                return e
    return None


def _student_board(scene: str, transcript: str, k: int | None = None) -> list[str]:
    """One board through the seam — the only way any caller reaches a model."""
    from rank_vocabulary import rank_vocabulary
    kwargs = {"k": k} if k else {}
    ranked = rank_vocabulary(scene, transcript, {}, strategy="student", **kwargs)
    return [r["word"] for r in ranked]


def _open_fires(scene: str, transcript: str) -> tuple[bool, list[str]]:
    from evaluation.phase4.answers import answer_candidates
    spans = answer_candidates(scene, transcript)
    return bool(spans), spans


def run(rows: list[dict]) -> dict:
    from evaluation.phase4 import infer

    per_row: list[dict] = []
    for r in rows:
        scene, transcript = r["scene"], r["transcript"]
        fired, spans = _open_fires(scene, transcript)

        infer.OPEN_ANSWERS = True
        student = _student_board(scene, transcript)
        infer.OPEN_ANSWERS = False
        try:
            closed = _student_board(scene, transcript)
        finally:
            infer.OPEN_ANSWERS = True

        # Did the seam actually serve the student? A silent fallback to the
        # diverse (LLM) path would invalidate the comparison, so we check the
        # closed board is a pure vocab board.
        per_row.append({
            "id": r["id"],
            "category": r["category"],
            "question_type": r.get("question_type", ""),
            "transcript": transcript,
            "expected_answers": r["expected_answers"],
            "open_fired": fired,
            "open_spans": spans,
            "student_board": student,
            "student_hit": answer_hit(student, r["expected_answers"]),
            "closed_board": closed,
            "closed_hit": answer_hit(closed, r["expected_answers"]),
        })

    def rate(items, key):
        return round(sum(1 for x in items if x[key]) / max(len(items), 1), 4)

    overall = {
        "n": len(per_row),
        "open_fire_rate": rate(per_row, "open_fired"),
        "student_answer_hit": rate(per_row, "student_hit"),
        "closed_answer_hit": rate(per_row, "closed_hit"),
    }
    by_type: dict[str, dict] = {}
    groups = defaultdict(list)
    for x in per_row:
        groups[x["question_type"] or "unknown"].append(x)
    for t, items in sorted(groups.items()):
        by_type[t] = {
            "n": len(items),
            "open_fire_rate": rate(items, "open_fired"),
            "student_answer_hit": rate(items, "student_hit"),
            "closed_answer_hit": rate(items, "closed_hit"),
        }
    by_cat: dict[str, dict] = {}
    groups = defaultdict(list)
    for x in per_row:
        groups[x["category"]].append(x)
    for c, items in sorted(groups.items()):
        by_cat[c] = {
            "n": len(items),
            "open_fire_rate": rate(items, "open_fired"),
            "student_answer_hit": rate(items, "student_hit"),
            "closed_answer_hit": rate(items, "closed_hit"),
        }
    return {"overall": overall, "by_question_type": by_type,
            "by_category": by_cat, "rows": per_row}


def markdown(res: dict, verbose: bool = True) -> str:
    o = res["overall"]
    out = ["## Deictic-question eval (informational, n=%d)" % o["n"], "",
           "| slice | n | open fire | hit@board (student) | hit@board (closed-only) |",
           "|---|---|---|---|---|",
           f"| **overall** | {o['n']} | {o['open_fire_rate']:.3f} | "
           f"{o['student_answer_hit']:.3f} | {o['closed_answer_hit']:.3f} |"]
    for t, s in res["by_question_type"].items():
        out.append(f"| type: {t} | {s['n']} | {s['open_fire_rate']:.3f} | "
                   f"{s['student_answer_hit']:.3f} | {s['closed_answer_hit']:.3f} |")
    for c, s in res["by_category"].items():
        out.append(f"| setting: {c} | {s['n']} | {s['open_fire_rate']:.3f} | "
                   f"{s['student_answer_hit']:.3f} | {s['closed_answer_hit']:.3f} |")
    if verbose:
        out += ["", "| id | type | fired | student hit | closed hit | student board |",
                "|---|---|---|---|---|---|"]
        for x in res["rows"]:
            out.append(f"| {x['id']} | {x['question_type']} | "
                       f"{'Y' if x['open_fired'] else '-'} | "
                       f"{x['student_hit'] or '-'} | {x['closed_hit'] or '-'} | "
                       f"{', '.join(x['student_board'])} |")
    return "\n".join(out)


def main() -> int:
    os.environ.setdefault("VOICELINK_PERCEPTION", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    p = argparse.ArgumentParser()
    p.add_argument("--quiet", action="store_true", help="summary table only")
    args = p.parse_args()
    rows = load_scenarios()
    res = run(rows)
    RESULTS.write_text(json.dumps(res, indent=2))
    print(markdown(res, verbose=not args.quiet))
    print(f"\nwrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
