"""Open-vocabulary answer extraction (RC-3) — evaluation/phase4/answers.py.

The interesting behaviour is not "does it find nouns" but WHEN it declines to
answer: offering scene props to a question whose answer is not in the room is
what cost coverage in the measured ablation, so the stand-down cases are tested
as carefully as the extraction cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

MAC = Path(__file__).resolve().parent.parent
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))

from evaluation.phase4 import answers as A  # noqa: E402


# --- question typing ----------------------------------------------------------

@pytest.mark.parametrize("transcript,kind", [
    ("Do you want the window seat or the aisle seat?", "alternative"),
    ("Tea or coffee?", "alternative"),
    ("Where would you like to sit today?", "wh"),
    ("Who do you want to sit next to?", "wh"),
    ("Are you okay?", "other"),
    ("", "other"),
])
def test_question_kind(transcript, kind):
    assert A.question_kind(transcript) == kind


@pytest.mark.parametrize("transcript,expected", [
    ("Where would you like to sit?", True),
    ("Who do you want to sit next to?", True),
    ("How was your day?", True),                      # user-directed but not deictic
    ("What do plants take in from the air?", False),
    ("Who can tell me the answer to question three?", False),
])
def test_user_directed(transcript, expected):
    assert A.is_user_directed(transcript) is expected


# --- alternative questions: the answers are stated in the question ------------

def test_alternative_yields_both_conjuncts_as_multiword_spans():
    got = A.answer_candidates(
        "A train carriage with pairs of seats.",
        "Do you want the window seat or the aisle seat?")
    assert "window seat" in got and "aisle seat" in got
    # multi-word, not the bare tokens that sank the single-token CONTEXT_UNION
    assert "seat" not in got


def test_alternative_does_not_need_the_scene():
    assert A.answer_candidates("", "Tea or coffee?") == ["tea", "coffee"]


# --- deictic wh-questions: the answer is in the scene -------------------------

def test_deictic_who_surfaces_the_person_not_the_furniture():
    got = A.answer_candidates(
        "Crowded holiday dinner table, aunt pulling out an empty chair between "
        "two relatives.",
        "Who do you want to sit next to?")
    assert any("aunt" in g or "relative" in g for g in got[:2]), got


def test_deictic_where_surfaces_a_seating_answer():
    got = A.answer_candidates(
        "The teacher gestures around a room with several open desks as the user "
        "comes in.",
        "Where would you like to sit today?")
    assert any("desk" in g for g in got), got


def test_which_uses_the_question_head_noun_as_the_type():
    """"Which movie" ranks "dvd" over "living room"; the generic object
    prototype had that backwards."""
    got = A.answer_candidates(
        "Saturday night living room, older sister fanning out three DVD cases "
        "on the rug.",
        "Which movie do you want to watch tonight?")
    assert got and "dvd" in got[0]


# --- stand-down: the cases that cost coverage when over-eager -----------------

@pytest.mark.parametrize("scene,transcript", [
    # knowledge question — "carbon dioxide" is nowhere in the room
    ("A classroom with a whiteboard showing a plant diagram.",
     "What do plants take in from the air?"),
    # a state, not a thing
    ("A dinner table at home. dad is passing dishes around.",
     "How was your day?"),
    # a quantity
    ("User on the phone, a restaurant host's voice over clatter.",
     "How many people will be in your party?"),
    # an abstraction
    ("A cake glowing with candles is carried toward you.",
     "Make a wish! What are you going to wish for?"),
    # not a question at all
    ("A quiet library reading room.", "It's nice in here."),
])
def test_stands_down_when_the_answer_is_not_in_the_room(scene, transcript):
    assert A.answer_candidates(scene, transcript) == []


def test_no_vague_span_is_ever_offered():
    """A contentful head does not rescue a vague span ("something in support")."""
    got = A.answer_candidates(
        "A delivery person chats near a counter about careers.",
        "Which job do you want, something in support work or delivery?")
    assert not any("something" in g for g in got), got


# --- span hygiene -------------------------------------------------------------

def test_spans_never_straddle_punctuation_or_end_in_function_words():
    got = A.answer_candidates(
        "Crowded holiday dinner table, aunt pulling out an empty chair.",
        "Where would you like to sit?")
    for span in got:
        assert not span.split()[-1] in A._FUNCTION_WORDS
        assert "table aunt" not in span


def test_nested_spans_are_dropped_so_one_referent_cannot_take_both_slots():
    """"relatives" and "two relatives" are the same referent; keeping both
    starves the second answer."""
    kept = A._drop_nested(["relatives", "two relatives", "aunt"], 8)
    assert kept == ["relatives", "aunt"]


def test_returns_nothing_rather_than_garbage_on_empty_input():
    assert A.answer_candidates("", "") == []
    assert A.answer_candidates(None, None) == []
