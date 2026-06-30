"""Phase-3 unit tests — model-free, like test_seam.py.

Embeddings are monkeypatched (deterministic crc32 vectors, no MiniLM / network)
and no LLM is ever loaded. These pin the math of the reframed metrics, the
taxonomy classification, the >=100 scenario guarantee, and the selftest's
--from-boards re-score path on a tiny fixture.
"""

from __future__ import annotations

import json
import zlib

import numpy as np
import pytest

import evaluation.embed as embed_mod
from evaluation.phase3 import metrics3, taxonomy, scengen, selftest


@pytest.fixture
def fast_embed(monkeypatch):
    """Offline deterministic embedder (crc32 seed -> stable unit vectors)."""

    def _embed(texts):
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        vecs = []
        for t in texts:
            seed = zlib.crc32(t.encode("utf-8"))
            v = np.random.default_rng(seed).standard_normal(384).astype(np.float32)
            v /= np.linalg.norm(v) or 1.0
            vecs.append(v)
        return np.asarray(vecs, dtype=np.float32)

    monkeypatch.setattr(embed_mod, "embed", _embed)
    taxonomy._proto_vecs = None  # rebuild prototypes under the fake embedder
    return _embed


# --- near_dup_rate math ------------------------------------------------------

def test_near_dup_rate_no_pairs_is_zero():
    assert metrics3.near_dup_rate(np.zeros((0, 4), dtype=np.float32)) == 0.0
    assert metrics3.near_dup_rate(np.eye(1, 4, dtype=np.float32)) == 0.0


def test_near_dup_rate_orthogonal_is_zero():
    orth = np.eye(3, dtype=np.float32)        # 3 orthonormal vectors, all cos 0
    assert metrics3.near_dup_rate(orth) == 0.0


def test_near_dup_rate_one_duplicate_pair():
    v = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    # pairs: (0,1)=1.0 dup, (0,2)=0, (1,2)=0  -> 1/3
    assert metrics3.near_dup_rate(v) == pytest.approx(1 / 3)


def test_near_dup_rate_all_identical_is_one():
    v = np.array([[1, 0], [1, 0], [1, 0]], dtype=np.float32)
    assert metrics3.near_dup_rate(v) == pytest.approx(1.0)


# --- coverage_of_intent hit / miss ------------------------------------------

def test_coverage_of_intent_lexical_hit(fast_embed):
    # Shares the content word "water" -> lexical leg returns True (no embed needed).
    assert metrics3.coverage_of_intent(["I want water", "no"], ["more water please"])


def test_coverage_of_intent_miss(fast_embed):
    # No shared content word and random embeddings never reach cosine 0.50.
    assert not metrics3.coverage_of_intent(["zzz", "qqq"], ["entirely unrelated phrase"])


def test_coverage_of_intent_semantic_hit(monkeypatch):
    # Constant-vector embedder -> every pair cosine 1.0 -> semantic leg fires even
    # with zero lexical overlap.
    const = np.ones((1, 8), dtype=np.float32) / np.sqrt(8)
    monkeypatch.setattr(embed_mod, "embed",
                        lambda texts: np.repeat(const, len(texts), axis=0))
    assert metrics3.coverage_of_intent(["alpha"], ["omega"])


def test_coverage_of_intent_empty_board_or_moves(fast_embed):
    assert not metrics3.coverage_of_intent([], ["water"])
    assert not metrics3.coverage_of_intent(["water"], [])


# --- taxonomy classification -------------------------------------------------

def test_classify_functions_counts_distinct(fast_embed):
    vecs, funcs = taxonomy._exemplars()              # (M, 384) under fake embed
    # One exemplar from each of three DIFFERENT functions -> 3 distinct.
    first_of: dict[str, int] = {}
    for i, f in enumerate(funcs):
        first_of.setdefault(f, i)
    idxs = [first_of[name] for name in list(first_of)[:3]]
    assert taxonomy.classify_functions(vecs[idxs])[0] == 3
    # A board of one exemplar repeated -> 1 distinct function.
    repeated = np.repeat(vecs[idxs[0]:idxs[0] + 1], 4, axis=0)
    assert taxonomy.classify_functions(repeated)[0] == 1


def test_classify_functions_empty():
    assert taxonomy.classify_functions(np.zeros((0, 384), dtype=np.float32)) == (0, [])


def test_distinct_functions_at_respects_k(fast_embed):
    protos = taxonomy._prototypes()[:10]
    board = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    # Feed 10 exemplar vectors; @7 must only see the first 7 -> at most 7 distinct.
    n = metrics3.distinct_functions_at(board, k=7, board_vecs=protos)
    assert n <= 7


# --- scengen template fallback ----------------------------------------------

def test_template_fallback_yields_at_least_100():
    rows = scengen._template_scenarios()
    assert len(rows) >= scengen.MIN_SCENARIOS


def test_template_rows_have_required_schema():
    rows = scengen._template_scenarios()
    required = {"id", "category", "scene", "transcript", "profile",
               "is_question", "active_question", "acceptable_moves", "gold_utterance"}
    ids = set()
    for r in rows:
        assert required <= set(r), f"missing keys in {r.get('id')}"
        assert len(r["acceptable_moves"]) >= 3
        assert r["category"] in scengen.CATEGORIES
        ids.add(r["id"])
    assert len(ids) == len(rows)                       # ids are unique
    assert set(scengen.CATEGORIES) <= {r["category"] for r in rows}


def test_question_rows_carry_active_question():
    for r in scengen._template_scenarios():
        if r["is_question"]:
            assert r["active_question"], f"{r['id']} is a question but has no active_question"


# --- selftest build_context + --from-boards re-score ------------------------

def test_build_context_holds_question():
    sc = {"scene": "room", "transcript": "What is your name?",
          "is_question": True, "active_question": "What is your name?",
          "profile": {"identity": {"name": "Sam"}}}
    scene, transcript, ctx = selftest.build_context(sc)
    assert scene == "room" and transcript == "What is your name?"
    assert ctx["active_question"] == "What is your name?"
    assert ctx["addressed"] is True
    assert ctx["personal_context"]["identity"]["name"] == "Sam"


def _fixture_board_rows():
    return [
        {"id": "s1", "category": "classroom", "strategy": "diverse",
         "board": ["yes", "no", "I need help", "I don't know", "carbon dioxide",
                   "can you repeat that", "I want to answer"],
         "acceptable_moves": ["carbon dioxide", "I don't know"], "gold_utterance": "carbon dioxide"},
        {"id": "s1", "category": "classroom", "strategy": "topk",
         "board": ["carbon dioxide", "carbon dioxide gas", "co2", "the gas carbon dioxide",
                   "carbon", "dioxide", "air"],
         "acceptable_moves": ["carbon dioxide", "I don't know"], "gold_utterance": "carbon dioxide"},
        {"id": "s2", "category": "doctor", "strategy": "diverse",
         "board": ["my head hurts", "I feel sick", "I'm scared", "no pain", "I don't know",
                   "help me", "where is it"],
         "acceptable_moves": ["my head", "it doesn't hurt"], "gold_utterance": "my head"},
        {"id": "s2", "category": "doctor", "strategy": "topk",
         "board": ["my head hurts", "head hurts", "my head", "head pain", "the head",
                   "hurts", "ouch"],
         "acceptable_moves": ["my head", "it doesn't hurt"], "gold_utterance": "my head"},
    ]


def test_score_boards_shape(fast_embed):
    results = selftest.score_boards(_fixture_board_rows())
    assert results["n_scenarios"] == 2
    for strat in ("diverse", "topk"):
        agg = results["overall"][strat]
        assert agg["n"] == 2
        assert 0.0 <= agg["coverage_of_intent"] <= 1.0
        assert 0.0 <= agg["near_dup"] <= 1.0
        assert 0.0 <= agg["distinct_functions"] <= 9.0
    assert "classroom" in results["per_category"] and "doctor" in results["per_category"]


def test_evaluate_gates_runs_and_is_structured(fast_embed):
    results = selftest.evaluate_gates(selftest.score_boards(_fixture_board_rows()))
    assert set(results["gates"]) == {
        "scenarios", "distinct_functions", "near_dup", "judge_proxy_labeled"}
    assert isinstance(results["pass"], bool)
    # Gate 4 is structural and must hold: judge is import-able and PROXY-labeled.
    assert results["gates"]["judge_proxy_labeled"]["pass"] is True


def test_from_boards_roundtrip(fast_embed, tmp_path):
    path = tmp_path / "boards.jsonl"
    with open(path, "w") as f:
        for row in _fixture_board_rows():
            f.write(json.dumps(row) + "\n")
    loaded = selftest.load_boards(path)
    assert len(loaded) == 4
    results = selftest.score_boards(loaded)
    assert results["n_scenarios"] == 2
