"""Tests for the paired-margin confidence intervals on the phase-4 gates.

_margin_ci decides whether a gate verdict is trustworthy, so it has to be right
about the boring cases: a clear pass, a clear fail, and — the one that matters —
a margin so small the interval straddles zero, which is what v2's and v3's
coverage gates both turned out to be.
"""

from __future__ import annotations

import pytest

from evaluation.phase4.evaluate import _margin_ci


def test_a_wide_consistent_win_is_a_pass():
    student = [1.0] * 40
    teacher = [1.0] * 40                       # bar = 0.90 * 1.0

    m = _margin_ci(student, teacher, factor=0.90)

    assert m["verdict"] == "PASS"
    assert m["ci"][0] > 0


def test_a_wide_consistent_loss_is_a_fail():
    student = [0.0] * 40
    teacher = [1.0] * 40

    m = _margin_ci(student, teacher, factor=0.90)

    assert m["verdict"] == "FAIL"
    assert m["ci"][1] < 0


def test_a_hairline_margin_is_inconclusive_not_a_pass():
    """The case that motivated all of this.

    Student beats the bar on average by a hair, but scenario-to-scenario the
    margin swings wildly. The point estimate says PASS; the interval says we
    cannot tell. INCONCLUSIVE is the honest answer.
    """
    # 21 wins, 20 losses -> mean margin barely positive, huge spread.
    student = [1.0] * 21 + [0.0] * 20
    teacher = [1.0] * 41

    m = _margin_ci(student, teacher, factor=0.51)

    assert m["margin"] > 0, "point estimate should look like a pass"
    assert m["ci"][0] < 0 < m["ci"][1], "interval should straddle zero"
    assert m["verdict"] == "INCONCLUSIVE"


def test_lower_is_better_flips_the_comparison():
    """same-function-pair passes when the student is BELOW the teacher."""
    student = [0.10] * 30
    teacher = [0.20] * 30

    better = _margin_ci(student, teacher, higher_is_better=False)
    worse = _margin_ci(teacher, student, higher_is_better=False)

    assert better["verdict"] == "PASS"
    assert worse["verdict"] == "FAIL"


def test_unscored_and_mismatched_inputs_do_not_raise():
    for s, t in (([], []), ([1.0], []), ([1.0, 2.0], [1.0])):
        m = _margin_ci(s, t)
        assert m["verdict"] == "not-scored"
        assert m["ci"] is None


def test_margin_is_deterministic():
    """bootstrap_ci is seeded; two runs must agree or the reported interval is
    a lottery ticket."""
    s = [1.0, 0.0] * 20
    t = [1.0] * 40

    assert _margin_ci(s, t, factor=0.9) == _margin_ci(s, t, factor=0.9)


def test_more_scenarios_narrow_the_interval():
    """The interval must shrink as n grows, or it is not measuring sampling
    error at all. This is the property that makes 'grow n=71' the right fix for
    the inconclusive coverage gate rather than 'train a better student'."""
    pattern = [1.0, 1.0, 1.0, 0.0]          # same underlying rate at both sizes

    def width(reps: int) -> float:
        s = pattern * reps
        m = _margin_ci(s, [1.0] * len(s), factor=0.5)
        return m["ci"][1] - m["ci"][0]

    small, large = width(3), width(50)      # n=12 vs n=200

    assert large < small, f"interval did not narrow: n=12 {small} vs n=200 {large}"
    assert large < small / 2, "should narrow roughly with sqrt(n)"
