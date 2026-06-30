"""Phase-3 FIGURES — grouped diverse-vs-topk bars, one panel per metric.

    render(results)   # called by selftest.py with the in-memory results dict
    python3 evaluation/phase3/plots3.py   # render from selftest_results.json

matplotlib is imported lazily inside render() so importing this module never
pulls a plotting backend (keeps the test suite light and offline)."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
RESULTS_PATH = HERE / "selftest_results.json"

INK = "#1a1a1a"
BASE_C = "#b0b7c3"   # muted grey-blue = topk baseline
NEW_C = "#c2410c"    # burnt orange = diverse method


def render(results: dict, out_dir: Path = FIG) -> Path:
    """Write one grouped-bar figure per metric to figures/. Returns the dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir.mkdir(exist_ok=True)
    d = results["overall"]["diverse"]
    t = results["overall"]["topk"]

    # (key, title, higher_is_better, ymax)
    panels = [
        ("distinct_functions", "distinct-functions@7 (of 9)", True, 9.0),
        ("near_dup", "near-duplicate rate", False, max(0.1, t["near_dup"] * 1.3)),
        ("coverage_of_intent", "coverage-of-intent@7", True, 1.0),
        ("exact_match", "exact-match (diagnostic only)", True, 1.0),
    ]
    plt.rcParams.update({"font.size": 11, "figure.dpi": 130, "savefig.bbox": "tight"})

    for key, title, higher, ymax in panels:
        fig, ax = plt.subplots(figsize=(5, 4.2))
        vals = [t[key], d[key]]
        ax.bar([0, 1], vals, 0.6, color=[BASE_C, NEW_C])
        for x, v in zip((0, 1), vals):
            ax.text(x, v + ymax * 0.01, f"{v:.3f}", ha="center", fontsize=10, color=INK)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["topk", "diverse"])
        ax.set_ylim(0, ymax)
        arrow = "higher is better" if higher else "lower is better"
        ax.set_title(f"{title}\n({arrow})", fontweight="bold")
        name = f"phase3_{key}.png"
        fig.savefig(out_dir / name)
        plt.close(fig)
        print("wrote", name)
    return out_dir


def main() -> None:
    results = json.loads(RESULTS_PATH.read_text())
    render(results)
    print("\nfigures ->", FIG)


if __name__ == "__main__":
    main()
