"""Deictic-question eval set (RC-3 measurability).

Schema checks on evaluation/phase4/deictic_scenarios.jsonl, plus one live
assertion that the open answer path (answers.py) fires on enough of the set to
be measurable — the held-out corpus fires on 2/84, which is why this set exists.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

MAC = Path(__file__).resolve().parent.parent
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))

from evaluation.phase4 import eval_deictic as E  # noqa: E402

SETTINGS = {"classroom", "home", "clinic", "playground", "store", "transport"}


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return E.load_scenarios()


def test_sixty_rows_ten_per_setting(rows):
    assert len(rows) == 60
    assert Counter(r["category"] for r in rows) == {s: 10 for s in SETTINGS}
    assert len({r["id"] for r in rows}) == 60


def test_every_row_has_every_field(rows):
    for r in rows:
        for f in E.REQUIRED_FIELDS:
            assert f in r, (r.get("id"), f)
        assert r["is_question"] is True
        assert r["transcript"].rstrip().endswith("?"), r["id"]
        assert r["active_question"].strip() == r["transcript"].strip()
        assert isinstance(r["expected_answers"], list) and r["expected_answers"]
        assert isinstance(r["acceptable_moves"], list) and len(r["acceptable_moves"]) >= 3


def test_every_expected_answer_is_in_the_context(rows):
    """The whole point: the concrete answer must be readable from the scene or
    the question — otherwise the open path has nothing to extract."""
    for r in rows:
        text = (r["scene"] + " " + r["transcript"]).lower()
        for a in r["expected_answers"]:
            assert a.lower() in text, (r["id"], a)


def test_answer_hit_matching_rules():
    assert E.answer_hit(["Red Trainers"], ["red trainers"]) == "red trainers"
    assert E.answer_hit(["trainers"], ["red trainers"]) == "red trainers"   # board inside expected
    assert E.answer_hit(["red trainers"], ["trainers"]) == "trainers"       # expected inside board
    assert E.answer_hit(["no", "me"], ["monkeys", "home"]) is None          # short words never substring-hit
    assert E.answer_hit(["i need help"], ["mum"]) is None


def test_open_path_fires_on_most_of_the_set(rows):
    """Measured 2026-08-23: fires on 54/60 (the 6 how-many rows stand down by
    design — 'how' is not a deictic wh-word). Margin left for lexicon tweaks."""
    from evaluation.phase4.answers import answer_candidates
    fired = sum(1 for r in rows if answer_candidates(r["scene"], r["transcript"]))
    assert fired >= 45, fired
