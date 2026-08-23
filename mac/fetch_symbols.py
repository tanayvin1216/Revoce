"""
fetch_symbols.py — one-time online tool to populate mac/symbols/.

Run ONCE while online:
    python3 mac/fetch_symbols.py

What it does:
  1. Builds a word list from the seed set + personal_context.sample.json hints.
  2. For each word: queries the ARASAAC search API, picks the best pictogram id,
     downloads the 500px PNG into mac/symbols/{id}.png (skips if already cached).
  3. Writes/updates mac/symbols/index.json  (word -> id mapping).

After this runs, Wi-Fi can be turned OFF and the server serves pictograms from disk.

ARASAAC pictograms are licensed CC BY-NC-SA 4.0.
Attribution: Pictograms © ARASAAC (arasaac.org), Government of Aragon, Spain.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Known-good seed IDs for words the ARASAAC search API ranks poorly.
# These are used as a direct fallback — no search needed.
# ---------------------------------------------------------------------------
_SEED_IDS: dict[str, int] = {
    "yes":        5584,
    "no":         5526,
    "more":       5508,
    "stop":       7196,
    "help":       32648,
    "go":         8142,
    "like":       37826,
    "don't like": 37825,
    "i want":     5441,
    "thank you":  8129,
    "hello":      6522,
    "all done":   32814,
    "finish":     32814,
    "i need":     39109,
    "wait":       6527,
    "water":      3474,
    "bathroom":   2281,
    "toilet":     2281,
    "hungry":     37837,
    "eat":        37837,
    "scared":     38212,
    "break":      7195,
    "pain":       37768,
    "seat":       2597,
    "tired":      37883,
    "loud":       3562,
    "too loud":   3562,
    "no thank you": 5526,
}

_COMMON_NEEDS = [
    "water", "bathroom", "wait", "loud", "scared", "seat",
    "hungry", "tired", "break", "pain", "no", "yes",
    "help", "stop", "more", "go", "all done", "i need", "i want",
    "thank you", "hello", "like", "don't like",
]

SYMBOLS_DIR = Path(__file__).parent / "symbols"
SAMPLE_PROFILE = Path(__file__).parent / "personal_context.sample.json"
SEARCH_BASE = "https://api.arasaac.org/api/pictograms/en/search"
IMAGE_BASE = "https://static.arasaac.org/pictograms"
SLEEP_S = 0.2
IMAGE_SIZE = 500
REQUEST_TIMEOUT = 10


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "VoiceLink-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return resp.read()


def _search_id(word: str) -> int | None:
    """Query ARASAAC search API and return the best pictogram id, or None."""
    encoded = urllib.parse.quote(word)
    url = f"{SEARCH_BASE}/{encoded}"
    try:
        data = json.loads(_get(url))
    except (urllib.error.URLError, json.JSONDecodeError, Exception) as exc:
        print(f"  [search error] {word!r}: {exc}", file=sys.stderr)
        return None

    if not isinstance(data, list) or not data:
        return None

    # Prefer entries where aac is True (specifically designed for AAC)
    for entry in data:
        if entry.get("aac", False):
            return int(entry["_id"])
    # Fall back to the first result
    return int(data[0]["_id"]) if data else None


# ---------------------------------------------------------------------------
# OpenSymbols (opensymbols.org) — optional SECOND source, searched only for
# words ARASAAC misses. Aggregates ~10 open-licensed sets (ARASAAC, Mulberry,
# Tawasol, Sclera, OpenMoji...) and supports a locale parameter, which is the
# multilingual path. Needs a shared secret from their application form:
#     OPENSYMBOLS_SECRET=... python3 fetch_symbols.py [--locale es]
# Files are saved as symbols/os_<id>.png and the licence of every symbol is
# recorded in index.json["_licenses"] (they vary per set).
# ---------------------------------------------------------------------------
OPENSYMBOLS_API = "https://www.opensymbols.org/api/v2"
_os_token: str | None = None


def _opensymbols_token() -> str | None:
    global _os_token
    import os
    secret = os.environ.get("OPENSYMBOLS_SECRET")
    if not secret:
        return None
    if _os_token is None:
        req = urllib.request.Request(
            f"{OPENSYMBOLS_API}/token", data=urllib.parse.urlencode({"secret": secret}).encode(),
            headers={"User-Agent": "VoiceLink-fetch-symbols/1.0"})
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            _os_token = json.loads(resp.read())["access_token"]
    return _os_token


def _opensymbols_search(word: str, locale: str = "en") -> dict | None:
    token = _opensymbols_token()
    if token is None:
        return None
    q = urllib.parse.urlencode({"q": word, "locale": locale, "access_token": token})
    try:
        data = json.loads(_get(f"{OPENSYMBOLS_API}/symbols?{q}"))
    except Exception as exc:
        print(f"  [opensymbols error] {word!r}: {exc}", file=sys.stderr)
        return None
    for entry in data if isinstance(data, list) else []:
        if not entry.get("unsafe_result") and entry.get("image_url"):
            return entry
    return None


def _download_png(symbol_id: int, dest: Path) -> bool:
    """Download a 500px PNG from ARASAAC static CDN.  Returns True on success."""
    if dest.exists():
        return True  # idempotent
    url = f"{IMAGE_BASE}/{symbol_id}/{symbol_id}_{IMAGE_SIZE}.png"
    try:
        data = _get(url)
        dest.write_bytes(data)
        return True
    except Exception as exc:
        print(f"  [download error] id={symbol_id}: {exc}", file=sys.stderr)
        return False


def _board_words() -> list[str]:
    """Every word the board can actually show: the pinned core, the 94-word
    student vocabulary, and the concrete answers of the deictic set. Until
    2026-08-23 only 27 seed words were cached, so most dynamic words rendered
    as an emoji or a blank box."""
    words: list[str] = []
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from rank_vocabulary import GUARANTEED_CORE, _CORE_POOL, _STUB_WORDS
        words += GUARANTEED_CORE + [c["word"] for c in _CORE_POOL] + [w["word"] for w in _STUB_WORDS]
    except Exception:
        pass
    p4 = Path(__file__).resolve().parent / "evaluation" / "phase4"
    vocab = p4 / "vocab.json"
    if vocab.exists():
        v = json.loads(vocab.read_text())
        words += list(v) if isinstance(v, list) else list(v.keys())
    deictic = p4 / "deictic_scenarios.jsonl"
    if deictic.exists():
        for line in deictic.read_text().splitlines():
            if line.strip():
                words += json.loads(line).get("expected_answers", [])
    return words


def _collect_words(extra: list[str] | None = None) -> list[str]:
    """Union of seed words + board vocabulary + personal_context.sample.json
    hints + common needs (+ anything passed on the command line)."""
    words: list[str] = list(_COMMON_NEEDS) + _board_words() + list(extra or [])

    if SAMPLE_PROFILE.exists():
        try:
            profile = json.loads(SAMPLE_PROFILE.read_text())
            for event in profile.get("todays_events", []):
                words.extend(event.get("vocab_hints", []))
            for routine in profile.get("routines", []):
                words.extend(routine.get("vocab_hints", []))
            sensory = profile.get("sensory_notes", {})
            words.extend(sensory.get("comfort_items", []))
            words.extend(sensory.get("dietary", []))
        except (json.JSONDecodeError, AttributeError):
            pass

    # Deduplicate preserving order, lowercase-normalise
    seen: set[str] = set()
    unique: list[str] = []
    for w in words:
        key = w.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def run(extra: list[str] | None = None, locale: str = "en") -> None:
    SYMBOLS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing index
    index_path = SYMBOLS_DIR / "index.json"
    if index_path.exists():
        try:
            existing = json.loads(index_path.read_text())
        except json.JSONDecodeError:
            existing = {}
    else:
        existing = {"_meta": {}}

    index: dict = {
        k: v for k, v in existing.items()
        if k in ("_meta", "_licenses") or isinstance(v, (int, str))
    }

    words = _collect_words(extra)
    print(f"[fetch_symbols] {len(words)} words to process")
    ok = 0
    miss = 0

    for word in words:
        key = word.strip().lower()

        # Already in index and PNG on disk -> skip
        existing_id = index.get(key)
        if isinstance(existing_id, int) and (SYMBOLS_DIR / f"{existing_id}.png").exists():
            ok += 1
            continue
        if isinstance(existing_id, str) and (SYMBOLS_DIR / existing_id).exists():
            ok += 1
            continue

        # Use seed ID if available (no API call needed)
        symbol_id = _SEED_IDS.get(key)

        if symbol_id is None:
            print(f"  searching: {key!r}")
            symbol_id = _search_id(key)
            if symbol_id is None:
                # Retry with first token of a multi-word phrase
                first_token = key.split()[0]
                if first_token != key:
                    print(f"    retrying first token: {first_token!r}")
                    symbol_id = _search_id(first_token)
            time.sleep(SLEEP_S)

        if symbol_id is None:
            entry = _opensymbols_search(key, locale)
            if entry is not None:
                fname = f"os_{entry['id']}.png"
                dest = SYMBOLS_DIR / fname
                try:
                    if not dest.exists():
                        dest.write_bytes(_get(entry["image_url"]))
                    index[key] = fname
                    index.setdefault("_licenses", {})[fname] = {
                        "repo": entry.get("repo_key"), "license": entry.get("license"),
                        "license_url": entry.get("license_url"), "author": entry.get("author")}
                    print(f"  [ok/opensymbols] {key!r} -> {fname} ({entry.get('repo_key')})")
                    ok += 1
                    continue
                except Exception as exc:
                    print(f"  [opensymbols download error] {key!r}: {exc}", file=sys.stderr)
            print(f"  [miss] no pictogram found for {key!r}", file=sys.stderr)
            miss += 1
            continue

        dest = SYMBOLS_DIR / f"{symbol_id}.png"
        if _download_png(symbol_id, dest):
            index[key] = symbol_id
            print(f"  [ok] {key!r} -> id={symbol_id}")
            ok += 1
        else:
            miss += 1

    index["_meta"] = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "size": IMAGE_SIZE,
        "api_version": "v1",
    }

    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))
    print(f"\n[fetch_symbols] done: {ok} ok, {miss} misses -> {index_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--locale", default="en", help="OpenSymbols locale (es, ar, ...)")
    ap.add_argument("words", nargs="*", help="extra words to fetch")
    a = ap.parse_args()
    run(a.words, a.locale)
