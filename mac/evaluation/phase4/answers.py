"""Open-vocabulary answer extraction (RC-3).

WHY THIS EXISTS
---------------
The student's output space is a closed vocabulary built by frequency: a word
must appear >= min_freq times across teacher boards to be learnable. Measured on
the 600-row teacher corpus, that floor is structurally fatal for concrete
answers:

    teacher dynamic word slots        3600
    unique teacher words              2366   (94.5% appear < 3 times)
    representable in the 94-word vocab  32.5%
    of what IS representable, hedges    49.1%
    rows with ZERO representable words  12.2%

Concrete answers are inherently long-tail — "right arm" is correct for exactly
one scene — so no frequency floor keeps them, and no amount of extra teacher
data helps (growing the vocab to 800 words still only reaches 56.6% coverage).
That is the real root cause of RC-3, and it also explains the Phase-4 negative
result: 5x data made agreement worse because new rows add new one-off tail words
rather than repeats.

The fix is architectural, not statistical. The closed-vocab student keeps doing
what it is genuinely good at (fast, diverse, general communicative functions);
concrete answers come from an OPEN path that reads candidates out of the actual
context at inference time.

WHY THE PREVIOUS ATTEMPT FAILED
-------------------------------
infer.CONTEXT_UNION (RC-3-lite) unioned single content TOKENS into the pool and
scored them with the standard query vector. Two defects, both fatal:

  1. single tokens — "window"/"seat"/"aisle" instead of "window seat";
  2. it scored answers by cosine to a query that is 0.7 * the question
     (selection._Q_WEIGHTS), so question-echo nouns had inflated similarity and
     displaced real answers (ablation: disabling it recovered dynamic coverage
     0.563 -> 0.803).

Defect 2 is the deep one: an answer to "Where would you like to sit?" is not
similar to that question — it is similar to the ANSWER SPACE the question opens.
So this module never scores candidates against the question vector.

THE APPROACH
------------
Question type decides where the answer lives and how to score it:

  * ALTERNATIVE ("A or B?") — the answers are literally in the question. Split
    the coordination and take both conjuncts as multi-word spans. High
    precision, no embedding needed to find them.
  * OPEN WH ("Where/Who/What ...?") — the answer is NOT in the question, it is
    in the scene. Generate multi-word spans from the scene, drop any span that
    echoes a question token, and rank by similarity to a wh-type prototype
    ("who" -> a person, "where" -> a place) blended with scene salience.

Span generation is dependency-free (no POS tagger — spaCy's model is not a
dependency of the Intel runtime): n-grams of 1..3 content tokens with
determiner stripping and function-word boundary rejection. The MiniLM embedder
already in the pipeline does the semantic filtering, which is the same division
of labour the rest of the selection code uses.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

# --- lexicon -----------------------------------------------------------------
# Function words that may not START or END a candidate span. A span like
# "pulling out an" or "with several" is not an answer.
_FUNCTION_WORDS = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "each",
    "every", "several", "many", "much", "few", "both", "either", "neither",
    "and", "or", "but", "so", "then", "than", "as", "if", "while", "because",
    "of", "in", "on", "at", "to", "for", "with", "from", "by", "about", "into",
    "over", "under", "between", "through", "during", "before", "after", "out",
    "up", "down", "off", "near", "next", "around", "across", "behind",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does",
    "did", "has", "have", "had", "will", "would", "can", "could", "should",
    "may", "might", "must", "shall",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "mine", "yours",
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "there", "here", "very", "just", "really", "also", "too", "only",
    "not", "no", "yes", "now", "today", "want", "like", "need", "get", "got",
})

# Determiners/quantifiers stripped from the FRONT of a span: "an empty chair"
# -> "empty chair". Kept out of _FUNCTION_WORDS-only handling because a span may
# legitimately contain them internally.
_LEADING_STRIP = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "some", "any",
    "several", "many", "much", "few", "both", "his", "her", "their", "my",
    "your", "our", "its",
})

# Verb-ish participles that show up in scene descriptions ("pulling", "gesturing")
# and are never answers on their own.
_PARTICIPLE_RE = re.compile(r"^(?:[a-z]+ing|[a-z]+ed)$")

# Words that carry no answer content even as head nouns.
_ANSWER_STOPWORDS = frozenset({
    "user", "scene", "person", "people", "someone", "something", "anything",
    "thing", "things", "way", "time", "moment", "place",
    "camera", "photo", "image", "picture", "view", "background", "foreground",
})

# Verbs and bare adjectives that scene descriptions use constantly and that are
# never answers as a span HEAD. Plural nouns ("desks", "relatives") also end in
# -s, so this must be an explicit list rather than a suffix rule.
_NON_HEAD = frozenset({
    "open", "closed", "empty", "full", "busy", "quiet", "loud", "crowded",
    "ready", "nearby", "visible", "seated", "other", "another", "same",
    "comes", "goes", "gets", "takes", "makes", "looks", "seems", "stands",
    "sits", "holds", "gestures", "points", "waits", "walks", "talks", "asks",
    "says", "shows", "puts", "pulls", "turns", "keeps", "leaves", "starts",
})

_MAX_SPAN_TOKENS = 3
_MAX_CANDIDATES = 8

# wh-word -> a natural-language prototype of the ANSWER TYPE it opens. Candidates
# are ranked by similarity to this, never to the question itself.
_WH_PROTOTYPES = {
    "who": "a person, a relative, a friend, a family member",
    "whom": "a person, a relative, a friend, a family member",
    "where": "a place, a location, a seat, a spot to sit",
    "what": "an object, a thing, a food, an activity",
    "which": "an object, a thing, one of the choices",
    "when": "a time, a moment, later, soon",
    "how": "a manner, a feeling, a condition",
}
_DEFAULT_PROTOTYPE = "an object, a person, or a place in the situation"

# Blend of wh-type fit and scene salience. Type fit dominates: a "who" question
# must surface the person even when the table is the most salient scene object.
_TYPE_W, _SALIENCE_W = 0.75, 0.25

# Not every wh-question is answerable from the environment, and offering scene
# props as answers to one that is not costs real coverage:
#
#   SITUATIONAL  "Where would you like to sit?"        answer = a desk in the room
#   KNOWLEDGE    "What do plants take in from the air?" answer = carbon dioxide,
#                                                       which is nowhere in the scene
#
# The discriminator is grammatical, not topical: a situational question asks the
# USER to choose something here and now, so it carries second-person volition
# ("do you want", "would you like", "your"). A knowledge question asks about the
# world and does not. This was tried first as a cosine(span, question) floor, and
# that failed on exactly the case RC-3 is about — cosine("aunt", "who do you want
# to sit next to?") is low, so the floor that protected coverage also deleted the
# person answer. Grammar separates the two cleanly where topical similarity cannot.
_USER_DIRECTED_RE = re.compile(
    r"\b(?:do|did|does|would|will|can|could|should|are|is)\s+you\b"
    r"|\byou\s+(?:want|like|prefer|need|feel|choose|pick|rather|wanna)\b"
    r"|\byour\b|\bwould\s+you\b|\bfor\s+you\b",
    re.IGNORECASE)

# A weak topical floor still runs on top of the grammatical test, to drop spans
# that are merely scene furniture. Scoring against the question is safe HERE
# because question tokens are already excluded from the spans, so the lexical
# self-overlap that inflated CONTEXT_UNION's scores cannot occur. Ranking remains
# by answer-type prototype, never by question similarity.
MIN_QUESTION_RELEVANCE = 0.10


def is_user_directed(transcript: str | None) -> bool:
    """Does this question ask the USER to choose something in the here and now?"""
    return bool(_USER_DIRECTED_RE.search(transcript or ""))


# Second-person volition alone is not enough — it admits questions that are
# about the user but whose answer is nowhere in the room. Measured on the
# held-out split, every coverage loss from an over-eager open path was one of
# these:
#
#   "How was your day?"            -> a state       ("it was good")
#   "How many in your party?"      -> a quantity    ("four people")
#   "What are you going to wish for?" -> an abstraction
#
# What separates them is the wh-word. where / who / which are DEICTIC: they ask
# the user to select among referents physically present, which is exactly when
# the scene contains the answer. what / how / when / why are open-ended, and the
# scene holds props rather than answers.
_SCENE_ANSWERABLE_WH = frozenset({"where", "who", "whom", "which"})

# "What colour cup do you want?" is deictic despite the wh-word: the user picks
# among colours physically present, and the head noun ("colour") is already the
# prototype _wh_head_noun hands to the ranker. Measured on deictic_scenarios:
# 0/6 -> 5/6 colour rows answered, no change on the other 54.
_COLOUR_Q_RE = re.compile(r"\bwhat\s+colou?r\b", re.IGNORECASE)

_ALT_RE = re.compile(r"\bor\b", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _tokens(text: str | None) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def question_kind(transcript: str | None) -> str:
    """'alternative' | 'wh' | 'other'.

    Alternative is checked first: "Do you want the window seat or the aisle
    seat?" is a wh-less choice, and "Which do you want, tea or coffee?" should
    still be handled by conjunct extraction rather than the scene path.
    """
    text = (transcript or "").strip()
    if not text:
        return "other"
    if _ALT_RE.search(text):
        return "alternative"
    for tok in _tokens(text):
        if tok in _WH_PROTOTYPES:
            return "wh"
    return "other"


def _wh_word(transcript: str | None) -> str:
    for tok in _tokens(transcript):
        if tok in _WH_PROTOTYPES:
            return tok
    return ""


def _wh_head_noun(transcript: str | None) -> str:
    """The noun a 'which/what X' question names as its answer type.

    "Which movie do you want to watch?" states the answer type outright, and it
    is a far sharper prototype than the generic "an object, a thing": ranking
    scene spans against "movie" puts "dvd" above "living room", where the
    generic prototype had it backwards.
    """
    toks = _tokens(transcript)
    for i, tok in enumerate(toks[:-1]):
        if tok in ("which", "what"):
            nxt = toks[i + 1]
            if nxt not in _FUNCTION_WORDS and nxt not in _ANSWER_STOPWORDS \
                    and len(nxt) >= 3:
                return nxt
    return ""


def _clean_span(tokens: list[str]) -> str | None:
    """Strip leading determiners, then reject spans with function-word edges."""
    toks = list(tokens)
    while toks and toks[0] in _LEADING_STRIP:
        toks = toks[1:]
    if not toks or len(toks) > _MAX_SPAN_TOKENS:
        return None
    if toks[0] in _FUNCTION_WORDS or toks[-1] in _FUNCTION_WORDS:
        return None
    # a span is only an answer if its HEAD (last token) is contentful
    head = toks[-1]
    if len(head) < 3 or head in _ANSWER_STOPWORDS or head in _NON_HEAD \
            or _PARTICIPLE_RE.match(head):
        return None
    if all(t in _FUNCTION_WORDS for t in toks):
        return None
    # A vague quantifier anywhere in the span makes the whole span vague:
    # "something in support" is not an answer even though its head ("support")
    # is contentful. Checking only the head let this through.
    if any(t in _ANSWER_STOPWORDS for t in toks):
        return None
    return " ".join(toks)


def _spans(text: str | None, exclude: frozenset[str] = frozenset()) -> list[str]:
    """Multi-word candidate spans, longest first, deduped.

    Clauses are split on punctuation so a span never straddles a comma — that is
    what produced garbage like "table aunt" from "...dinner table, aunt...".
    """
    out: list[str] = []
    seen: set[str] = set()
    for clause in re.split(r"[,;.!?()]", text or ""):
        toks = _tokens(clause)
        # longest spans first so "empty chair" is preferred over "chair"
        for n in range(_MAX_SPAN_TOKENS, 0, -1):
            for i in range(len(toks) - n + 1):
                window = toks[i:i + n]
                if any(t in exclude for t in window):
                    continue
                span = _clean_span(window)
                if span and span not in seen:
                    seen.add(span)
                    out.append(span)
    return out


def _conjuncts(transcript: str | None) -> list[str]:
    """Both sides of an "A or B" question, as spans.

    Takes the trailing span of the left conjunct and the leading span of the
    right one, which is where the coordinated nouns sit in a spoken question:
    "Do you want [the window seat] or [the aisle seat]?"
    """
    text = (transcript or "").strip().rstrip("?.! ")
    parts = _ALT_RE.split(text)
    if len(parts) < 2:
        return []
    out: list[str] = []
    for idx, part in enumerate(parts):
        toks = _tokens(part)
        if not toks:
            continue
        # left conjuncts contribute their TAIL, right conjuncts their HEAD
        window = toks[-_MAX_SPAN_TOKENS:] if idx < len(parts) - 1 \
            else toks[:_MAX_SPAN_TOKENS]
        # shrink until it cleans up ("do you want the window seat" -> "window seat")
        for size in range(len(window), 0, -1):
            cand = window[-size:] if idx < len(parts) - 1 else window[:size]
            span = _clean_span(cand)
            if span and span not in out:
                out.append(span)
                break
    return out


@lru_cache(maxsize=64)
def _prototype_vec(prototype: str) -> tuple:
    from evaluation.embed import embed
    return tuple(embed([prototype])[0].tolist())


def answer_candidates(scene: str | None, transcript: str | None,
                      max_n: int = _MAX_CANDIDATES,
                      min_relevance: float | None = None) -> list[str]:
    """Open-vocabulary answer candidates for a question, best first.

    Returns [] for non-questions and when nothing survives filtering — callers
    treat it as "no open answers available" and fall back to the closed vocab.
    """
    kind = question_kind(transcript)
    if kind == "alternative":
        # The answers are in the question. No embedding scoring: they are the
        # only two things being offered.
        return _conjuncts(transcript)[:max_n]

    if kind != "wh":
        return []

    # Stand down unless this is a deictic question the user answers by picking
    # something present: both tests must hold (see _SCENE_ANSWERABLE_WH).
    if not is_user_directed(transcript):
        return []
    if _wh_word(transcript) not in _SCENE_ANSWERABLE_WH \
            and not _COLOUR_Q_RE.search(transcript or ""):
        return []

    # Open wh-question: the answer is in the SCENE, and any span echoing the
    # question is the failure mode that killed RC-3-lite. Drop those outright.
    echo = frozenset(t for t in _tokens(transcript) if len(t) >= 3)
    cands = _spans(scene, exclude=echo)
    if not cands:
        return []

    from evaluation.embed import embed
    head = _wh_head_noun(transcript)
    proto = head or _WH_PROTOTYPES.get(_wh_word(transcript), _DEFAULT_PROTOTYPE)
    proto_vec = np.asarray(_prototype_vec(proto), dtype=np.float32)
    scene_vec = embed([scene or ""])[0]
    vecs = embed(cands)
    score = _TYPE_W * (vecs @ proto_vec) + _SALIENCE_W * (vecs @ scene_vec)

    # Situational vs knowledge question: drop spans too unrelated to the question
    # to be plausible answers to it (see MIN_QUESTION_RELEVANCE).
    floor = MIN_QUESTION_RELEVANCE if min_relevance is None else min_relevance
    if floor > 0:
        q_rel = vecs @ embed([transcript or ""])[0]
        admitted = [i for i in range(len(cands)) if q_rel[i] >= floor]
        if not admitted:
            return []
        cands = [cands[i] for i in admitted]
        score = score[admitted]

    order = np.argsort(-score)
    return _drop_nested([cands[i] for i in order], max_n)


def _drop_nested(ranked: list[str], max_n: int) -> list[str]:
    """Drop spans nested inside an already-kept span, keeping the better-scored one.

    The n-gram generator emits "relatives" and "two relatives" for the same
    entity; both surviving burns the whole answer quota on one referent and
    starves the second answer ("aunt"). Token-subset containment catches this
    exactly, where cosine does not — MiniLM puts "relatives"/"two relatives" at
    ~0.83, just under the near-duplicate guard.
    """
    kept: list[str] = []
    kept_tokens: list[set[str]] = []
    for span in ranked:
        toks = set(span.split())
        if any(toks <= prev or prev <= toks for prev in kept_tokens):
            continue
        kept.append(span)
        kept_tokens.append(toks)
        if len(kept) >= max_n:
            break
    return kept
