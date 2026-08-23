"""
Ranker — vocabulary ranking backend.

Mirrors the transcriber.py pattern exactly:
  Ranker ABC  ->  LocalLLMRanker (mlx-lm, lazy load)
                  HeuristicRanker (no model, deterministic)
              ->  make_ranker(cfg) factory (graceful degrade)

The seam (rank_vocabulary.py) calls ranker.rank(system, user) -> list[dict].
On ANY load or inference error the factory returns HeuristicRanker so the
server never crashes because of model availability.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Shared JSON parse helper
# ---------------------------------------------------------------------------

_OBJ_RE = re.compile(r"\{[^{}]*\}")


def _salvage_objects(text: str) -> list[dict]:
    """Extract every complete {...} object, tolerating truncation and junk
    tokens between objects (the 4-bit model occasionally emits stray glyphs)."""
    out = []
    for m in _OBJ_RE.findall(text):
        try:
            obj = json.loads(m)
            if isinstance(obj, dict):
                out.append(obj)
        except json.JSONDecodeError:
            continue
    return out


def _parse_json_array(text: str) -> list[dict]:
    """Extract a JSON array from LLM output.

    Handles: code fences, leading prose, trailing prose, mid-array truncation.
    Raises ValueError/JSONDecodeError on unrecoverable parse — caller catches.
    """
    t = text.strip()

    # Strip code fences if present
    if "```" in t:
        after = t[t.find("```") + 3:]
        # drop optional language tag line (e.g. "json\n")
        if after.startswith("json"):
            after = after[4:]
        after = after.lstrip("\n")
        close = after.find("```")
        t = after[:close] if close != -1 else after

    start = t.find("[")
    end = t.rfind("]")
    if start != -1 and end > start:
        try:
            data = json.loads(t[start : end + 1])
        except json.JSONDecodeError:
            data = _salvage_objects(t)
    else:
        # Truncated (no closing ]) or junk tokens between objects — salvage
        # every complete {...} object individually.
        data = _salvage_objects(t)
    if not isinstance(data, list):
        raise ValueError("parsed JSON is not a list")

    result = []
    for item in data:
        if not isinstance(item, dict):
            continue
        word = item.get("word") or item.get("phrase") or ""
        word = str(word).strip()
        if not word:
            continue
        # accept "symbol" or "symbol_keyword" as the pictogram key
        sym_key = item.get("symbol_keyword") or item.get("symbol") or word
        entry = {"word": word, "symbol_keyword": str(sym_key).strip().lower()}
        # diverse-pool tags (additive, optional — old top-k prompt omits them)
        if item.get("function"):
            entry["function"] = str(item["function"]).strip().lower()
        if item.get("topic"):
            entry["topic"] = str(item["topic"]).strip().lower()
        result.append(entry)

    if not result:
        raise ValueError("LLM returned an empty array after filtering")
    return result


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------

class Ranker(ABC):
    @abstractmethod
    def rank(self, system_prompt: str, user_prompt: str) -> list[dict]:
        """Return list of {"word": str, "symbol_keyword": str}.  Raises on failure."""
        ...


# ---------------------------------------------------------------------------
# LocalLLMRanker
# ---------------------------------------------------------------------------

class LocalLLMRanker(Ranker):
    def __init__(self, model_id: str, max_tokens: int, temp: float) -> None:
        # Heavy import deferred to __init__ so the module is importable without mlx-lm.
        from mlx_lm import load, generate  # noqa: PLC0415
        from mlx_lm.sample_utils import make_sampler  # noqa: PLC0415

        self._model, self._tokenizer = load(model_id)
        self._sampler = make_sampler(temp=temp)
        self._generate = generate
        self._max_tokens = max_tokens

    def rank(self, system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> list[dict]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_tokens = self._tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        text = self._generate(
            self._model,
            self._tokenizer,
            prompt_tokens,
            max_tokens=max_tokens or self._max_tokens,
            sampler=self._sampler,
        )
        return _parse_json_array(text)


# ---------------------------------------------------------------------------
# HfRanker — the same teacher on ANY PyTorch device (the non-Mac path)
# ---------------------------------------------------------------------------

def torch_device(preferred: str | None = None) -> str:
    """Pick the accelerator: VOICELINK_DEVICE wins, else hpu (Intel Gaudi, via
    habana_frameworks) > cuda > mps > cpu. Kept here so train.py and the HF
    ranker agree on one rule."""
    import os
    want = (preferred or os.environ.get("VOICELINK_DEVICE") or "").lower()
    import torch
    if want:
        if want == "hpu":
            import habana_frameworks.torch.core  # noqa: F401  registers the device
        return want
    try:
        import habana_frameworks.torch.core  # noqa: F401
        if torch.hpu.is_available():
            return "hpu"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HfRanker(Ranker):
    """Qwen2.5-Instruct through plain transformers, so the teacher runs on Intel
    Gaudi (device=hpu), an NVIDIA box (cuda), a Xeon (cpu) or the Mac (mps).
    LocalLLMRanker is the MLX fast path on Apple silicon; this one is the
    portable path. Prompting, parsing and temperature are identical, so boards
    are comparable across backends (greedy decoding is deterministic on all)."""

    DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

    def __init__(self, model_id: str | None = None, max_tokens: int = 300,
                 temp: float = 0.0, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = torch_device(device)
        model_id = model_id or self.DEFAULT_MODEL
        # mlx-community ids are 4-bit MLX weights; map them back to the HF original.
        if model_id.startswith("mlx-community/"):
            model_id = "Qwen/" + model_id.split("/", 1)[1].replace("-4bit", "")
        dtype = torch.bfloat16 if self.device in ("hpu", "cuda") else torch.float32
        self._tok = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        self._model.to(self.device).eval()
        self._max_tokens, self._temp = max_tokens, temp
        self.model_id = model_id

    def raw(self, system: str, user: str, temp: float | None = None,
            max_tokens: int | None = None) -> str:
        import torch
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ids = self._tok.apply_chat_template(msgs, add_generation_prompt=True,
                                            return_tensors="pt").to(self.device)
        t = self._temp if temp is None else temp
        with torch.no_grad():
            out = self._model.generate(
                ids, max_new_tokens=max_tokens or self._max_tokens,
                do_sample=t > 0, temperature=t if t > 0 else None,
                pad_token_id=self._tok.eos_token_id)
        return self._tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    def rank(self, system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> list[dict]:
        return _parse_json_array(self.raw(system_prompt, user_prompt, None, max_tokens))


# ---------------------------------------------------------------------------
# HeuristicRanker (no model; deterministic context-aware fallback)
# ---------------------------------------------------------------------------

# Core words always included; ordered best-first for an addressed situation.
_HEURISTIC_CORE = [
    {"word": "yes",         "symbol_keyword": "yes"},
    {"word": "no",          "symbol_keyword": "no"},
    {"word": "I need help", "symbol_keyword": "help"},
    {"word": "more",        "symbol_keyword": "more"},
    {"word": "stop",        "symbol_keyword": "stop"},
    {"word": "wait",        "symbol_keyword": "wait"},
    {"word": "go",          "symbol_keyword": "go"},
    {"word": "thank you",   "symbol_keyword": "thank you"},
]

# Extra words drawn in when specific signals appear in the prompt.
_NOISE_WORDS = [
    {"word": "too loud",        "symbol_keyword": "loud"},
    {"word": "I need a break",  "symbol_keyword": "break"},
]
_HUNGER_WORDS = [
    {"word": "I'm hungry",  "symbol_keyword": "hungry"},
    {"word": "water",       "symbol_keyword": "water"},
]
_TRAVEL_WORDS = [
    {"word": "where is the bathroom", "symbol_keyword": "bathroom"},
    {"word": "I need my seat",        "symbol_keyword": "seat"},
]


class HeuristicRanker(Ranker):
    """Deterministic keyword-driven fallback.  No model required."""

    def rank(self, system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> list[dict]:
        lp = (system_prompt + " " + user_prompt).lower()
        words: list[dict] = []

        # addressed -> yes/no/wait at top
        if "talking to the user right now: yes" in lp or "addressed: yes" in lp:
            words = [
                {"word": "yes",  "symbol_keyword": "yes"},
                {"word": "no",   "symbol_keyword": "no"},
                {"word": "wait", "symbol_keyword": "wait"},
            ]

        words.extend(_HEURISTIC_CORE)

        if any(kw in lp for kw in ("loud", "crowd", "noise", "terminal", "airport", "busy")):
            words.extend(_NOISE_WORDS)
        if any(kw in lp for kw in ("hungry", "eat", "food", "water", "drink", "thirst")):
            words.extend(_HUNGER_WORDS)
        if any(kw in lp for kw in ("airport", "terminal", "flight", "gate", "bathroom", "toilet")):
            words.extend(_TRAVEL_WORDS)

        # Extract vocab_hints lines from the rendered prompt
        hint_match = re.search(r"Suggested useful words[^\n]*:\s*([^\n]+)", user_prompt)
        if hint_match:
            raw_hints = hint_match.group(1)
            for hint in re.split(r",\s*", raw_hints):
                hint = hint.strip()
                if hint and hint.lower() != "none":
                    words.append({"word": hint, "symbol_keyword": hint.lower()})

        # Dedup preserving order, clamp to 12
        seen: set[str] = set()
        unique: list[dict] = []
        for w in words:
            key = w["word"].lower()
            if key not in seen:
                seen.add(key)
                unique.append(w)
            if len(unique) == 12:
                break

        return unique or _HEURISTIC_CORE[:8]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_ranker(cfg) -> Ranker:
    """Factory — mirrors make_transcriber.  Degrades to HeuristicRanker on failure."""
    name = cfg.RANKER
    if name == "local-llm":
        try:
            ranker = LocalLLMRanker(cfg.LLM_MODEL, cfg.LLM_MAX_TOKENS, cfg.LLM_TEMP)
            print(f"[ranker] local-llm loaded: {cfg.LLM_MODEL}")
            return ranker
        except Exception as exc:
            print(f"[ranker] local-llm unavailable ({exc}); using HeuristicRanker")
            return HeuristicRanker()
    if name == "hf":
        try:
            ranker = HfRanker(cfg.LLM_MODEL, cfg.LLM_MAX_TOKENS, cfg.LLM_TEMP)
            print(f"[ranker] hf loaded: {ranker.model_id} on {ranker.device}")
            return ranker
        except Exception as exc:
            print(f"[ranker] hf unavailable ({exc}); using HeuristicRanker")
            return HeuristicRanker()
    if name == "heuristic":
        return HeuristicRanker()
    raise ValueError(f"unknown RANKER={name!r}")
