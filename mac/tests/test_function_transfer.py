"""Unit tests for evaluation/phase4/function_transfer.py — pure arithmetic on
hand-built boards with a dictionary labeller, so no model is loaded."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

MAC = Path(__file__).resolve().parent.parent
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))

from evaluation.phase4 import function_transfer as ft  # noqa: E402

CORE = ["yes", "no", "I need help", "wait"]
FUNCS = ["A", "B", "C"]
LABELS = {
    "yes": "A", "no": "B", "i need help": "C", "wait": "A",
    "apple": "A", "pear": "A", "run": "B", "walk": "B", "sad": "C", "glad": "C",
    "blue": "A",
}


def label(words):
    return [LABELS[w.lower()] for w in words]


def _res(pairs):
    return ft.transfer_table(pairs, label, CORE, k=7, functions=FUNCS, n_boot=200)


def test_identical_boards_give_full_retention_and_full_overlap():
    b = ["yes", "no", "I need help", "wait", "apple", "run", "sad", "pear", "walk", "glad"]
    r = _res([(b, b), (b, b)])
    o = r["overall"]
    assert o["retention_ratio"] == 1.0
    assert o["dynamic_overlap"] == 1.0
    assert o["full_overlap_top7"] == 1.0
    assert o["function_jaccard"] == 1.0
    assert o["word_jaccard"] == 1.0
    for f in FUNCS:
        assert r["per_function"][f]["retention"] == 1.0
        assert r["per_function"][f]["lexical_overlap"] == 1.0
    assert o["functions_with_zero_lexical_overlap"] == []


def test_function_kept_but_words_changed():
    # Same functions in the top-7 window, different dynamic words -> function
    # agreement stays 1.0 while every lexical number drops to 0.
    teacher = ["yes", "no", "I need help", "wait", "apple", "run", "sad", "blue", "walk", "glad"]
    student = ["yes", "no", "I need help", "wait", "pear", "walk", "glad", "blue", "run", "apple"]
    r = _res([(student, teacher)])
    o = r["overall"]
    assert o["retention_ratio"] == 1.0
    assert o["function_jaccard"] == 1.0
    assert o["word_jaccard"] == 0.0
    # dynamic_overlap looks the teacher's top-7 dynamic words up in the student's
    # WHOLE board (gate semantics): apple and run sit in slots 9-10 of the student.
    assert o["dynamic_overlap"] == pytest.approx(2 / 3, abs=1e-3)
    pf = r["per_function"]
    assert pf["A"]["lexical_overlap"] == 1.0           # apple found (slot 10)
    assert pf["B"]["lexical_overlap"] == 1.0           # run found (slot 9)
    assert pf["C"]["lexical_overlap"] == 0.0           # sad missing
    assert o["functions_with_zero_lexical_overlap"] == ["C"]


def test_dropped_function_shows_in_retention_and_ci():
    teacher = ["yes", "no", "I need help", "wait", "sad", "glad", "apple"]
    student = ["yes", "no", "I need help", "wait", "apple", "pear", "blue"]
    pairs = [(student, teacher)] * 5
    r = _res(pairs)
    pf = r["per_function"]
    # C present in every teacher board (sad/glad) and no student board apart from
    # the core "I need help" -> presence retention 1.0 for C via the core word.
    assert pf["C"]["teacher_presence"] == 1.0
    assert pf["C"]["student_presence"] == 1.0
    # slots per board differ though: teacher 3 (help, sad, glad) vs student 1.
    assert pf["C"]["teacher_slots_per_board"] == 3.0
    assert pf["C"]["student_slots_per_board"] == 1.0
    # B: teacher has "no" (core) only; student the same -> retention 1, no dyn items.
    assert pf["B"]["teacher_dynamic_items"] == 0
    assert pf["B"]["lexical_overlap"] is None
    # distinct functions equal (3 vs 3) so overall retention is 1.0
    assert r["overall"]["retention_ratio"] == 1.0
    ci = pf["C"]["presence_margin_ci95"]
    assert ci == [0.0, 0.0]


def test_schema_and_markdown(tmp_path):
    teacher = ["yes", "no", "I need help", "wait", "sad", "glad", "apple"]
    student = ["yes", "no", "I need help", "wait", "apple", "pear", "blue"]
    r = _res([(student, teacher)])
    assert r["n_heldout"] == 1 and r["k"] == 7
    assert set(r["per_function"]) == set(FUNCS)
    for row in r["per_function"].values():
        assert set(row) >= {"teacher_presence", "student_presence", "retention",
                            "presence_margin_ci95", "teacher_slots_per_board",
                            "student_slots_per_board", "teacher_dynamic_items",
                            "lexical_overlap"}
    md = ft.markdown(r)
    assert "| function |" in md and "distinct-functions@7" in md
    (tmp_path / "x.json").write_text(json.dumps(r))   # JSON-serialisable


def test_empty_input():
    r = _res([])
    assert r["n_heldout"] == 0
    assert r["overall"]["retention_ratio"] is None


def test_word_breakdown_representability():
    teacher = ["yes", "no", "I need help", "wait", "sad", "glad", "apple"]
    student = ["yes", "no", "I need help", "wait", "apple", "pear", "blue"]
    vocab = ["apple", "glad"]
    canonical = {"sad": "glad"}          # sad merged into glad at dedup time
    wb = ft.word_breakdown([(student, teacher)], label, CORE, vocab, canonical,
                           k=7, functions=FUNCS)
    assert wb["C"]["teacher_items"] == 2 and wb["C"]["teacher_distinct"] == 2
    assert wb["C"]["representable_direct"] == 1        # glad
    assert wb["C"]["representable_via_canonical"] == 1  # sad -> glad
    assert wb["C"]["student_top"] == []
    assert wb["A"]["teacher_top"] == [("apple", 1)]
    assert wb["A"]["student_top"][0] == ("apple", 1)
    assert wb["B"]["teacher_items"] == 0


def test_confusion_over_paired_dynamic_slots():
    # core occupies slots 1-4; dynamic slots 5-7 are paired by position.
    teacher = ["yes", "no", "I need help", "wait", "sad", "run", "apple"]   # C, B, A
    student = ["yes", "no", "I need help", "wait", "pear", "walk", "glad"]  # A, B, C
    r = _res([(student, teacher)])
    c = r["confusion_dynamic_slots"]
    assert c["C"]["A"] == 1 and c["B"]["B"] == 1 and c["A"]["C"] == 1
    assert sum(sum(row.values()) for row in c.values()) == 3
    assert "confusion over paired dynamic slots" in ft.markdown(r)


def test_prior_board_is_most_frequent_training_words():
    rows = [{"board": ["yes", "no", "I need help", "wait", "apple", "run", "sad"]},
            {"board": ["yes", "no", "I need help", "wait", "apple", "run", "glad"]},
            {"board": ["yes", "no", "I need help", "wait", "apple", "blue", "pear"]}]
    b = ft._prior_board(rows, CORE, k=7, k_dyn=3)
    assert b[:4] == CORE
    assert b[4] == "apple" and b[5] == "run"
    assert len(b) == 7
