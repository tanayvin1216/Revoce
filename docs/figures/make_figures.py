"""Regenerate every README figure from committed artifacts.

    /usr/local/bin/python3 docs/figures/make_figures.py

Each figure is written twice, <name>.png (light) and <name>-dark.png, so the
README can pick the right one for the viewer's theme with a <picture> element.

The charts are deliberately plain: no titles or commentary baked into the
image (the README text does that), one accent colour for Revoce, grey for
everything it is compared against, and the value printed on every bar.

Every number is read from the JSON that produced it, except the vocabulary
sweep, which is transcribed from the AXIOM paper's Table 2 (its counting
script is not committed) and the M4 head timings, which come from
mac/intel/BENCHMARKS.md.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
P4 = ROOT / "mac" / "evaluation" / "phase4"
P3 = ROOT / "mac" / "evaluation" / "phase3"
ART = ROOT / "mac" / "intel" / "artifacts"
OUT = pathlib.Path(__file__).resolve().parent

# One accent, one neutral. Dark variants are lifted slightly so they hold
# contrast on GitHub's dark background.
LIGHT = dict(bg="#ffffff", ink="#111111", sub="#6e6e73", rule="#d2d2d7",
             accent="#0a66d6", base="#c7c7cc")
DARK = dict(bg="#0d1117", ink="#f5f5f7", sub="#a1a1a6", rule="#3a3a3c",
            accent="#4c9bff", base="#48484a")

FONT = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def style(t):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT, "font.size": 11,
        "figure.facecolor": t["bg"], "axes.facecolor": t["bg"],
        "savefig.facecolor": t["bg"],
        "text.color": t["ink"], "axes.labelcolor": t["sub"],
        "xtick.color": t["sub"], "ytick.color": t["sub"],
        "axes.edgecolor": t["rule"], "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False,
        "figure.dpi": 220, "savefig.bbox": "tight", "savefig.pad_inches": .2,
    })


def save(fig, name, dark):
    fig.savefig(OUT / f"{name}{'-dark' if dark else ''}.png")
    plt.close(fig)


def bar_label(ax, x, y, text, t, bold=True, color=None, dy=0.0, **kw):
    ax.text(x, y + dy, text, ha="center", va="bottom", fontsize=12,
            fontweight="bold" if bold else "normal", color=color or t["ink"], **kw)


# -----------------------------------------------------------------------------
# Selection policy: top-k vs diverse selection over the same candidate pool
# -----------------------------------------------------------------------------
def fig_diversity(t, dark):
    d = json.load(open(P3 / "selftest_results.json"))["overall"]
    tk, dv = d["topk"], d["diverse"]
    panels = [
        ("Distinct functions", "higher is better", "distinct_functions", "{:.2f}"),
        ("Near-duplicate rate", "lower is better", "near_dup", "{:.4f}"),
        ("Coverage of intent", "higher is better", "coverage_of_intent", "{:.2f}"),
        ("Exact match", "higher is better", "exact_match", "{:.2f}"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1))
    fig.subplots_adjust(wspace=.35)
    for ax, (name, hint, key, fmt) in zip(axes, panels):
        vals = [tk[key], dv[key]]
        top = max(vals)
        ax.bar([0, 1], vals, width=.62, color=[t["base"], t["accent"]])
        for x, v in enumerate(vals):
            bar_label(ax, x, v, fmt.format(v), t, dy=top * .03)
        ax.set_ylim(0, top * 1.25)
        ax.set_yticks([])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Top-k", "Revoce"], fontsize=11)
        ax.set_xlim(-.6, 1.6)
        ax.set_title(f"{name}\n", fontsize=12, fontweight="bold", color=t["ink"],
                     loc="center", pad=4)
        ax.text(.5, 1.02, hint, transform=ax.transAxes, ha="center",
                fontsize=9.5, color=t["sub"])
    save(fig, "diversity", dark)


# -----------------------------------------------------------------------------
# Board latency on the M4 by backend
# -----------------------------------------------------------------------------
def fig_latency(t, dark):
    e = json.load(open(ART / "e2e_mac.json"))["results"]
    lat = {r["backend"]: r["p50_ms"] for r in e}
    labels = ["PyTorch", "OpenVINO head\nPyTorch encoder", "OpenVINO"]
    vals = [lat["torch"], lat["ov-head"], lat["ov-full"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(range(3), vals, width=.56,
           color=[t["base"], t["base"], t["accent"]])
    for i, v in enumerate(vals):
        bar_label(ax, i, v, f"{v:.1f} ms", t, dy=.25)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_yticks([])
    ax.set_ylim(0, max(vals) * 1.2)
    ax.set_xlim(-.6, 2.6)
    ax.text(-.6, max(vals) * 1.2, "Median board latency, Apple M4",
            fontsize=10, color=t["sub"], va="top")
    save(fig, "latency", dark)


# -----------------------------------------------------------------------------
# Closed-vocabulary ceiling
# -----------------------------------------------------------------------------
def fig_ceiling(t, dark):
    # AXIOM Table 2, frequency-rebuilt vocabularies.
    V = np.array([94, 129, 305, 752, 800], float)
    cov = np.array([0.309, 0.333, 0.431, 0.555, 0.569])
    A = np.vstack([np.ones(5), np.log(V)]).T
    b = np.linalg.lstsq(A, cov, rcond=None)[0]

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.spines["left"].set_visible(True)
    xs = np.logspace(np.log10(80), np.log10(2600), 200)
    ax.plot(xs, b[0] + b[1] * np.log(xs), color=t["base"], lw=1.4, ls="--", zorder=1)
    ax.plot(V, cov, "o", ms=7, color=t["base"], zorder=2)
    ax.plot([129], [.333], "o", ms=9, color=t["accent"], zorder=3)
    ax.plot([2330], [1.0], "o", ms=7, color=t["ink"], mfc=t["bg"], mew=1.5, zorder=3)
    ax.text(129, .333 - .075, "Shipped list\n33%", ha="center", va="top",
            fontsize=10, color=t["accent"], fontweight="bold")
    ax.text(800, .569 - .075, "8.5× larger\n57%", ha="center", va="top",
            fontsize=10, color=t["sub"])
    ax.text(2330, 1.0 - .07, "All teacher\nwords", ha="center", va="top",
            fontsize=10, color=t["sub"])

    ax.set_xscale("log"); ax.set_xlim(70, 3600); ax.set_ylim(0, 1.08)
    ax.set_xticks([100, 300, 1000, 2330])
    ax.set_xticklabels(["100", "300", "1,000", "2,330"])
    ax.minorticks_off()
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.yaxis.grid(True, color=t["rule"], lw=.6)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Vocabulary size (words)", fontsize=10.5)
    ax.set_ylabel("Teacher words covered", fontsize=10.5)
    save(fig, "ceiling", dark)


# -----------------------------------------------------------------------------
# OpenVINO vs PyTorch for the student head, per machine
# -----------------------------------------------------------------------------
def fig_inversion(t, dark):
    hb = json.load(open(ART / "head_bench_tanishlaptop.json"))
    intel = {r["name"]: r["p50_ms"] for r in hb["results"]}
    i_ratio = intel["torch CPU (student.pt)"] / intel["OpenVINO FP32 IR"]
    m_ratio = 0.015 / 0.025          # mac/intel/BENCHMARKS.md, quiet run

    labels = ["Apple M4", "Intel Core Ultra 9 185H"]
    vals = [m_ratio, i_ratio]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.bar([0, 1], vals, width=.5, color=[t["base"], t["accent"]], zorder=2)
    ax.axhline(1, color=t["sub"], lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(1.52, 1, "PyTorch", va="center", ha="left", fontsize=10, color=t["sub"])
    for x, v in enumerate(vals):
        bar_label(ax, x, v, f"{v:.2f}×", t, dy=.06)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_yticks([])
    ax.set_ylim(0, 3.1); ax.set_xlim(-.55, 1.5)
    ax.text(-.55, 3.1, "OpenVINO speedup over PyTorch, same machine",
            fontsize=10, color=t["sub"], va="top")
    save(fig, "inversion", dark)


# -----------------------------------------------------------------------------
# Open-vocabulary extractor by question type
# -----------------------------------------------------------------------------
def fig_deictic(t, dark):
    d = json.load(open(P4 / "phase4_results.json"))["deictic"]
    by = d["by_question_type"]
    order = sorted(by, key=lambda k: (-by[k]["student_answer_hit"], k))
    open_v = [by[k]["student_answer_hit"] for k in order]
    closed = [by[k]["closed_answer_hit"] for k in order]
    x = np.arange(len(order))
    w = .38
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.bar(x - w / 2, closed, width=w, color=t["base"], label="Fixed vocabulary")
    ax.bar(x + w / 2, open_v, width=w, color=t["accent"], label="Revoce")
    for xx, v in zip(x, open_v):
        ax.text(xx + w / 2, v + .025, f"{v:.0%}", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color=t["ink"])
    for xx, v in zip(x, closed):
        ax.text(xx - w / 2, v + .025, f"{v:.0%}", ha="center", va="bottom",
                fontsize=9.5, color=t["sub"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k.capitalize()}\nn = {by[k]['n']}" for k in order],
                       fontsize=10.5)
    ax.set_yticks([]); ax.set_ylim(0, 1.18)
    ax.legend(loc="upper right", fontsize=10, handlelength=1, ncol=2,
              bbox_to_anchor=(1, 1.06))
    ax.text(-.6, 1.18, "Correct answer on the board, by question type",
            fontsize=10, color=t["sub"], va="top")
    ax.set_xlim(-.6, len(order) - .4)
    save(fig, "deictic", dark)


# -----------------------------------------------------------------------------
# Logo: the board itself, 4 pinned slots and 6 dynamic, one surfaced.
# Transparent background so it sits on GitHub's light or dark README.
# -----------------------------------------------------------------------------
def make_logo():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(2.8, 2.8))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    PIN, DYN, HIT = "#8a8984", "#2a78d6", "#1baf7a"
    for i in range(10):
        r, c = divmod(i, 5)
        col = PIN if i < 4 else (HIT if i == 7 else DYN)
        alpha = .45 if i < 4 else (1.0 if i == 7 else .72)
        ax.add_patch(FancyBboxPatch(
            (c * 1.06, -r * 1.06), .92, .92, boxstyle="round,pad=0,rounding_size=.20",
            linewidth=0, facecolor=col, alpha=alpha))
    ax.set_xlim(-.14, 5.44); ax.set_ylim(-1.34, 1.06)
    ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(OUT / "logo.png", transparent=True, dpi=200,
                bbox_inches="tight", pad_inches=.06)
    plt.close(fig)


if __name__ == "__main__":
    for theme, dark in ((LIGHT, False), (DARK, True)):
        style(theme)
        fig_diversity(theme, dark)
        fig_latency(theme, dark)
        fig_ceiling(theme, dark)
        fig_inversion(theme, dark)
        fig_deictic(theme, dark)
    make_logo()
    print("wrote:", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))
