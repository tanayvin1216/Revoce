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


def _collect_words() -> list[str]:
    """Union of seed words + personal_context.sample.json hints + common needs."""
    words: list[str] = list(_COMMON_NEEDS)

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


def run() -> None:
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

    index: dict[str, int | dict] = {
        k: v for k, v in existing.items() if k == "_meta" or isinstance(v, int)
    }

    words = _collect_words()
    print(f"[fetch_symbols] {len(words)} words to process")
    ok = 0
    miss = 0

    for word in words:
        key = word.strip().lower()

        # Already in index and PNG on disk -> skip
        existing_id = index.get(key)
        if isinstance(existing_id, int):
            if (SYMBOLS_DIR / f"{existing_id}.png").exists():
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
    run()
