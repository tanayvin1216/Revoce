"""Regenerate every README figure from committed artifacts.

    /usr/local/bin/python3 docs/figures/make_figures.py

Plain matplotlib + numpy, standard axes, white background. Every number is
read from the JSON that produced it, except two that aren't in a JSON file:
the vocabulary sweep (transcribed from the AXIOM paper's Table 2, whose
counting script isn't committed) and the M4 head timings (from
mac/intel/BENCHMARKS.md).
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
P4 = ROOT / "mac" / "evaluation" / "phase4"
P3 = ROOT / "mac" / "evaluation" / "phase3"
ART = ROOT / "mac" / "intel" / "artifacts"
OUT = pathlib.Path(__file__).resolve().parent

BASE = "#9e9e9e"     # whatever Revoce is compared against
OURS = "tab:blue"    # Revoce

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.grid": True, "grid.alpha": .3, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", facecolor="white")
    plt.close(fig)


def fig_diversity():
    d = json.load(open(P3 / "selftest_results.json"))["overall"]
    tk, dv = d["topk"], d["diverse"]
    panels = [
        ("Distinct functions (↑)", "distinct_functions", "%.2f"),
        ("Near-duplicate rate (↓)", "near_dup", "%.4f"),
        ("Coverage of intent (↑)", "coverage_of_intent", "%.2f"),
        ("Exact match (↑)", "exact_match", "%.2f"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax, (name, key, fmt) in zip(axes, panels):
        vals = np.array([tk[key], dv[key]])
        bars = ax.bar(["Top-k", "Diverse"], vals, color=[BASE, OURS], width=.6)
        ax.bar_label(bars, fmt=fmt, padding=2)
        ax.set_ylim(0, vals.max() * 1.2)
        ax.set_title(name)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Top-k ranking vs. diverse selection (same candidates, 120 scenarios)",
                 y=1.03)
    fig.tight_layout()
    save(fig, "diversity")


def fig_latency():
    e = json.load(open(ART / "e2e_mac.json"))["results"]
    lat = {r["backend"]: r["p50_ms"] for r in e}
    labels = ["PyTorch", "OpenVINO head,\nPyTorch encoder", "OpenVINO"]
    vals = np.array([lat["torch"], lat["ov-head"], lat["ov-full"]])
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bars = ax.bar(labels, vals, color=[BASE, BASE, OURS], width=.55)
    ax.bar_label(bars, fmt="%.2f ms", padding=2)
    ax.set_ylim(0, vals.max() * 1.2)
    ax.set_ylabel("Median board latency (ms)")
    ax.set_title("Board latency on Apple M4 (84 held-out scenarios)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "latency")


def fig_ceiling():
    # AXIOM Table 2, frequency-rebuilt vocabularies.
    V = np.array([94, 129, 305, 752, 800], float)
    cov = np.array([0.309, 0.333, 0.431, 0.555, 0.569])
    b1, b0 = np.polyfit(np.log(V), cov, 1)
    xs = np.logspace(np.log10(80), np.log10(2500), 200)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(xs, b0 + b1 * np.log(xs), "--", color=BASE, label="Log fit")
    ax.plot(V, cov, "o", color=OURS, label="Measured")
    ax.plot([2330], [1.0], "s", color="black", label="All teacher words (2,330)")
    ax.annotate("Shipped list: 33%", (129, .333), xytext=(150, .17),
                arrowprops=dict(arrowstyle="->", lw=.8))
    ax.annotate("800 words: 57%", (800, .569), xytext=(330, .74),
                arrowprops=dict(arrowstyle="->", lw=.8))
    ax.set_xscale("log")
    ax.set_xticks([100, 200, 500, 1000, 2000], ["100", "200", "500", "1,000", "2,000"])
    ax.minorticks_off()
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("Vocabulary size (words, log scale)")
    ax.set_ylabel("Teacher words covered")
    ax.set_title("How much of the teacher's vocabulary a fixed list can cover")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "ceiling")


def fig_inversion():
    hb = json.load(open(ART / "head_bench_tanishlaptop.json"))
    intel = {r["name"]: r["p50_ms"] for r in hb["results"]}
    i_ratio = intel["torch CPU (student.pt)"] / intel["OpenVINO FP32 IR"]
    m_ratio = 0.015 / 0.025          # mac/intel/BENCHMARKS.md, quiet run

    labels = ["Apple M4", "Intel Core Ultra 9 185H"]
    vals = np.array([m_ratio, i_ratio])
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    bars = ax.bar(labels, vals, color=["tab:orange", "tab:green"], width=.5)
    ax.bar_label(bars, fmt="%.2f×", padding=2)
    ax.axhline(1, color="black", lw=1, ls="--", label="Same speed as PyTorch")
    ax.set_ylim(0, 3.1)
    ax.set_ylabel("Speedup over PyTorch")
    ax.set_title("OpenVINO speedup for the student model, same machine")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "inversion")


def fig_deictic():
    d = json.load(open(P4 / "phase4_results.json"))["deictic"]
    by = d["by_question_type"]
    names = {"alternative": "This or that", "who": "Who", "where": "Where",
             "colour": "Colour", "which": "Which", "count": "How many"}
    order = sorted(by, key=lambda k: (-by[k]["student_answer_hit"], k))
    ours = np.array([by[k]["student_answer_hit"] for k in order]) * 100
    fixed = np.array([by[k]["closed_answer_hit"] for k in order]) * 100
    x = np.arange(len(order))
    w = .38

    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    b1 = ax.bar(x - w / 2, fixed, w, color=BASE, label="Fixed word list only")
    b2 = ax.bar(x + w / 2, ours, w, color=OURS, label="Revoce")
    ax.bar_label(b1, fmt="%.0f%%", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.0f%%", padding=2, fontsize=8)
    ax.set_xticks(x, [f"{names.get(k, k)}\n(n={by[k]['n']})" for k in order])
    ax.set_ylim(0, 118)
    ax.set_ylabel("Correct answer on the board (%)")
    ax.set_title("Answers taken from the scene or conversation (60 questions)")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "deictic")


def make_logo():
    """The board itself: 4 pinned slots, 6 dynamic, one surfaced. Transparent
    background so it sits on GitHub's light or dark README."""
    from matplotlib.patches import FancyBboxPatch
    with plt.rc_context({"axes.grid": False}):
        fig, ax = plt.subplots(figsize=(2.8, 2.8))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        PIN, DYN, HIT = "#8a8984", "#2a78d6", "#1baf7a"
        for i in range(10):
            r, c = divmod(i, 5)
            col = PIN if i < 4 else (HIT if i == 7 else DYN)
            alpha = .45 if i < 4 else (1.0 if i == 7 else .72)
            ax.add_patch(FancyBboxPatch(
                (c * 1.06, -r * 1.06), .92, .92,
                boxstyle="round,pad=0,rounding_size=.20",
                linewidth=0, facecolor=col, alpha=alpha))
        ax.set_xlim(-.14, 5.44); ax.set_ylim(-1.34, 1.06)
        ax.set_aspect("equal"); ax.axis("off")
        fig.savefig(OUT / "logo.png", transparent=True, dpi=200,
                    bbox_inches="tight", pad_inches=.06)
        plt.close(fig)


if __name__ == "__main__":
    fig_diversity()
    fig_latency()
    fig_ceiling()
    fig_inversion()
    fig_deictic()
    make_logo()
    print("wrote:", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))
