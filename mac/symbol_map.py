"""
Symbol map — resolves a keyword to a locally-cached ARASAAC pictogram path.

Usage:
    from symbol_map import resolve_symbol, EMOJI_FALLBACK
    symbol = resolve_symbol("help")           # "/symbols/32648.png" or "🆘"
    symbol = resolve_symbol("help", "🆘")    # explicit emoji fallback

The index.json is loaded ONCE at import time.  Missing file -> always returns
the emoji fallback (board still renders, never crashes).

ARASAAC pictograms are licensed CC BY-NC-SA 4.0 — non-commercial use only.
Attribution: Pictograms © ARASAAC (arasaac.org), Government of Aragon, Spain.
"""

from __future__ import annotations

import json
from pathlib import Path

SYMBOLS_DIR = Path(__file__).parent / "symbols"

# Emoji fallback for the most common core words so the board is never blank.
EMOJI_FALLBACK: dict[str, str] = {
    "yes":        "✅",
    "no":         "❌",
    "more":       "➕",
    "stop":       "✋",
    "help":       "🆘",
    "i want":     "🙋",
    "i need":     "🙋",
    "go":         "➡️",
    "like":       "👍",
    "don't like": "👎",
    "hello":      "👋",
    "thank you":  "🙏",
    "all done":   "🏁",
    "wait":       "⏳",
    "water":      "💧",
    "bathroom":   "🚻",
    "loud":       "🔊",
    "too loud":   "🔊",
    "hungry":     "🍽️",
    "scared":     "😨",
    "break":      "⏸️",
    "pain":       "😣",
    "seat":       "💺",
    "tired":      "😴",
}


def _load_index() -> dict[str, int]:
    """Load word->id mapping from symbols/index.json.  Returns {} on any error."""
    index_path = SYMBOLS_DIR / "index.json"
    try:
        raw = json.loads(index_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return {k: v for k, v in raw.items()
            if not k.startswith("_") and isinstance(v, (int, str))}


# Module-level singleton — loaded once, never reloaded.
_INDEX: dict[str, int | str] = _load_index()


def resolve_symbol(keyword: str, emoji_fallback: str = "▢") -> str:
    """Return '/symbols/{id}.png' if cached, else emoji_fallback.

    keyword is lowercased and stripped before lookup so callers don't need to
    normalise.  The PNG file must exist on disk (not just in the index) for the
    path to be returned — guards against a partially-completed fetch run.
    """
    key = keyword.strip().lower()
    symbol_id = _INDEX.get(key)
    if isinstance(symbol_id, int):          # ARASAAC id
        png = SYMBOLS_DIR / f"{symbol_id}.png"
        if png.exists():
            return f"/symbols/{symbol_id}.png"
    elif isinstance(symbol_id, str):        # OpenSymbols file name (os_<id>.png)
        if (SYMBOLS_DIR / symbol_id).exists():
            return f"/symbols/{symbol_id}"
    return emoji_fallback
