# Vocabulary-prediction evaluation harness

Measures the **Task 5 prompt baseline** (`rank_vocabulary`) on held-out data and
reports **top-k vocabulary recall**. One command:

```bash
# Synthetic dataset (the headline baseline) — GATED, needs HF auth (see below)
python3 mac/evaluation/harness.py

# Bundled synthetic fixture — no download/auth, for a quick demo / plumbing check
python3 mac/evaluation/harness.py --source fixture

# REAL consented interaction logs (Task 3 taps) — the number that actually matters
python3 mac/evaluation/harness.py --source real
```

Useful flags: `--limit N` (default 200; the full test split is ~17k rows ≈ ~14 h
of LLM calls), `--ks 1,5,10`, `--ranker local-llm|heuristic`, `--verbose`.

## ⚠️ Honesty constraints (read this before quoting any number)

- The synthetic source (`willwade/AACConversations`) is **GPT-generated, text-only,
  and simulated**. It is not real AAC use.
- The baseline is **itself an LLM prompt**. So a high score on the synthetic data
  **partly measures imitation of GPT, not real communication value.** The harness
  prints this caveat with every synthetic result and labels results
  `SYNTHETIC-BENCHMARK RESULT`.
- Treat the synthetic number as a **cold-start sanity signal** for project
  direction, nothing more.
- The number that decides real value comes from **`--source real`**: the persisted
  taps + context snapshots from Task 3, once a **consented pilot** produces them.
  That path is implemented and ready (it reads `mac/taps.jsonl`); it just needs
  real logs. Real results are labeled `REAL-DATA RESULT` with **no** caveat.

## Getting the real (gated) dataset

It is gated (CC-BY-4.0). One-time setup:

1. Accept the terms at <https://huggingface.co/datasets/willwade/AACConversations>.
2. Authenticate: `huggingface-cli login`  (or `export HF_TOKEN=hf_...`).
3. Run `python3 mac/evaluation/harness.py` (optionally `--limit 500`).

If you skip steps 1–2 the harness prints exactly what to do and exits.

## How it works

- **Input construction** (`dataset.py`): for each AAC-user turn (`speaker`
  contains "(AAC)"), build the model input from `scene` + `context_utterances`
  (with `context_speakers` roles); the AAC user is treated as *addressed*. No
  personal profile is available for this dataset, so `personal_context` is empty
  — an honest cold-start baseline.
- **Ground truth**: the AAC user's actual `utterance`. Content words (stop-words
  removed; high-value AAC words like yes/no/go/help/want/need kept) form the
  target.
- **Matching** (`metrics.py`, deliberately transparent): a predicted board phrase
  is *relevant* to the utterance if they share a content word or one is a
  substring of the other.
- **Reported**: `recall@k` (fraction of examples with ≥1 relevant phrase in the
  top-k board), `token_recall@k` (content-word coverage), `MRR` and
  `mean rank of first usable word` (the "time-to-relevant-word" proxy), and mean
  generation latency.

The same code runs over all three sources, so synthetic and real numbers are
directly comparable once real logs exist.
