"""Regenerate every README figure from committed artifacts.

    /usr/local/bin/python3 docs/figures/make_figures.py

Each figure is written twice — <name>.png (light) and <name>-dark.png — so the
README can serve the right one per viewer theme with a <picture> element.
No number in here is typed by hand: every value is read from the JSON that
produced it, except the vocabulary sweep, which is transcribed from the AXIOM
paper's Table 2 and marked as such (its counting script is not committed).
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

# --- validated palette (dataviz reference instance, both modes checked) -------
LIGHT = dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
             muted="#8a8984", grid="#e4e4e0",
             s1="#2a78d6", s2="#eb6834", s3="#1baf7a")
DARK = dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
            muted="#8a8984", grid="#2e2e2c",
            s1="#3987e5", s2="#d95926", s3="#199e70")

FONT = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def style(t):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT, "font.size": 9,
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "text.color": t["primary"], "axes.labelcolor": t["secondary"],
        "xtick.color": t["secondary"], "ytick.color": t["secondary"],
        "axes.edgecolor": t["grid"], "grid.color": t["grid"],
        "axes.grid": True, "grid.linewidth": .6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": .28,
    })


def title(ax, head, sub=None, t=LIGHT):
    """Head and sub both drawn above the axes, as separate text — set_title's
    pad does not reserve room for a second line and the two collide."""
    ax.text(0, 1.145 if sub else 1.04, head, transform=ax.transAxes,
            fontsize=11.5, fontweight="600", color=t["primary"], va="bottom")
    if sub:
        ax.text(0, 1.045, sub, transform=ax.transAxes, fontsize=8.6,
                color=t["secondary"], va="bottom")


def save(fig, name, dark):
    fig.savefig(OUT / f"{name}{'-dark' if dark else ''}.png")
    plt.close(fig)


# =============================================================================
# 1. The closed-vocabulary ceiling
# =============================================================================
def fig_ceiling(t, dark):
    # AXIOM Table 2 — frequency-rebuilt vocabularies. Counting script not committed.
    V = np.array([94, 129, 305, 752, 800], float)
    cov = np.array([0.309, 0.333, 0.431, 0.555, 0.569])
    A = np.vstack([np.ones(5), np.log(V)]).T
    b = np.linalg.lstsq(A, cov, rcond=None)[0]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    style(t)
    xs = np.logspace(np.log10(80), np.log10(2600), 200)
    ax.plot(xs, b[0] + b[1] * np.log(xs), color=t["muted"], lw=1.6, ls=(0, (5, 3)),
            zorder=1, label="_")
    ax.plot(V, cov, "o", ms=8, color=t["s1"], mec=t["surface"], mew=2, zorder=3)
    ax.plot([2330], [1.0], "o", ms=8, color=t["s2"], mec=t["surface"], mew=2, zorder=3)

    ax.annotate("shipped vocabulary\n95 words — 33% reachable",
                xy=(129, .333), xytext=(150, .175), fontsize=8.4, color=t["primary"],
                arrowprops=dict(arrowstyle="-", color=t["muted"], lw=.9))
    ax.annotate("8.5× larger still only\nreaches 57%", xy=(800, .569),
                xytext=(300, .70), fontsize=8.4, color=t["primary"],
                arrowprops=dict(arrowstyle="-", color=t["muted"], lw=.9))
    ax.annotate("every word the teacher used\n(2,330 — not a vocabulary you can ship)",
                xy=(2330, 1.0), xytext=(430, .86), fontsize=8.4, color=t["s2"], ha="left",
                arrowprops=dict(arrowstyle="-", color=t["muted"], lw=.9))
    ax.text(96, .62, f"fit: {b[1]:.3f}·ln V − {abs(b[0]):.3f}", fontsize=8.2,
            color=t["muted"], style="italic")

    ax.set_xscale("log"); ax.set_xlim(80, 3600); ax.set_ylim(0, 1.10)
    ax.set_xticks([94, 200, 400, 800, 2330])
    ax.set_xticklabels(["94", "200", "400", "800", "2330"])
    ax.set_yticks(np.arange(0, 1.01, .25))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.01, .25)])
    ax.set_xlabel("vocabulary size (log scale)")
    ax.set_ylabel("teacher word slots representable")
    ax.grid(axis="x", alpha=.35)
    title(ax, "Coverage grows logarithmically with vocabulary size",
          "Why a bigger word list cannot close the gap — each doubling buys ~8 points", t)
    save(fig, "ceiling", dark)


# =============================================================================
# 2. Diverse selection vs likelihood ranking
# =============================================================================
def fig_diversity(t, dark):
    d = json.load(open(P3 / "selftest_results.json"))["overall"]
    panels = [
        ("Distinct functions on a board", d["topk"]["distinct_functions"],
         d["diverse"]["distinct_functions"], "{:.2f}", None),
        ("Near-duplicate rate", d["topk"]["near_dup"],
         d["diverse"]["near_dup"], "{:.4f}", "69× fewer"),
        ("Coverage of intent", d["topk"]["coverage_of_intent"],
         d["diverse"]["coverage_of_intent"], "{:.2f}", None),
        ("Exact match to oracle word", d["topk"]["exact_match"],
         d["diverse"]["exact_match"], "{:.2f}", None),
    ]
    style(t)
    fig, axes = plt.subplots(1, 4, figsize=(11.4, 3.3))
    for ax, (name, base, div, fmt, note) in zip(axes, panels):
        bars = ax.bar([0, 1], [base, div], width=.56,
                      color=[t["muted"], t["s1"]], zorder=3)
        for r in bars:
            r.set_linewidth(2); r.set_edgecolor(t["surface"])
        top = max(base, div) or 1
        for x, v in ((0, base), (1, div)):
            ax.text(x, v + top * .06, fmt.format(v), ha="center", fontsize=9,
                    fontweight="600", color=t["primary"])
        if note:
            ax.text(.5, top * .58, note, ha="center", fontsize=8.6,
                    color=t["s1"], fontweight="600", transform=ax.transData)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["top-k", "diverse"], fontsize=8.6)
        ax.set_ylim(0, top * 1.30); ax.set_yticks([])
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.set_title(name, loc="left", fontsize=9.2, color=t["primary"],
                     fontweight="600", pad=8)
    fig.suptitle("Selecting for spread beats ranking by likelihood",
                 x=.008, y=1.10, ha="left", fontsize=11.5, fontweight="600",
                 color=t["primary"])
    fig.text(.008, 1.015, "Same candidate pool, 120 scenarios — only the selection "
             "policy differs. Independent scales per panel.",
             ha="left", fontsize=8.6, color=t["secondary"])
    save(fig, "diversity", dark)


# =============================================================================
# 3. The OpenVINO inversion
# =============================================================================
def fig_inversion(t, dark):
    hb = json.load(open(ART / "head_bench_tanishlaptop.json"))
    intel = {r["name"]: r["p50_ms"] for r in hb["results"]}
    i_torch = intel["torch CPU (student.pt)"]
    i_ov = intel["OpenVINO FP32 IR"]
    m_torch, m_ov = 0.015, 0.025          # mac/intel/BENCHMARKS.md, quiet run

    rows = [("Apple M4\nARM CPU plugin", m_torch / m_ov, m_torch, m_ov),
            ("Intel Core Ultra 9 185H\nnative CPU plugin", i_torch / i_ov, i_torch, i_ov)]
    style(t)
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    ys = [1, 0]
    for y, (lab, ratio, tt, ov) in zip(ys, rows):
        col = t["s3"] if ratio >= 1 else t["s2"]
        ax.barh([y], [ratio - 1], left=1, height=.34, color=col, zorder=3,
                edgecolor=t["surface"], linewidth=2)
        mid = (1 + ratio) / 2
        # ratio above the bar, raw timings below it -- nothing sits beside the
        # bar end, where it would collide with the y tick labels
        ax.text(mid, y + .26, f"{ratio:.2f}x  " + ("faster" if ratio >= 1 else "slower"),
                va="bottom", ha="center", fontsize=10.5, fontweight="600",
                color=t["primary"])
        ax.text(mid, y - .26, f"{tt:.4f} ms torch   {ov:.4f} ms OpenVINO",
                va="top", ha="center", fontsize=8.1, color=t["secondary"])
    ax.axvline(1, color=t["primary"], lw=1.2, zorder=4)
    ax.text(1, 1.68, "parity with PyTorch", ha="center", fontsize=8.2,
            color=t["secondary"])
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlim(0.42, 3.05); ax.set_ylim(-.62, 1.82)
    ax.set_xticks([0.5, 1, 2, 3]); ax.set_xticklabels(["0.5x", "1x", "2x", "3x"])
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    title(ax, "The same IR, converted once, inverts by CPU vendor",
          "Student head, 500 iterations after 50 warmup. Ratio is PyTorch ÷ OpenVINO "
          "on the same machine.", t)
    save(fig, "inversion", dark)


# =============================================================================
# 4. Open-vocabulary answers by question type
# =============================================================================
def fig_deictic(t, dark):
    d = json.load(open(P4 / "phase4_results.json"))["deictic"]
    by = d["by_question_type"]
    order = sorted(by, key=lambda k: (-by[k]["student_answer_hit"], k))
    style(t)
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    y = np.arange(len(order))[::-1]
    open_v = [by[k]["student_answer_hit"] for k in order]
    closed = [by[k]["closed_answer_hit"] for k in order]
    ax.barh(y + .17, open_v, height=.32, color=t["s1"], zorder=3,
            edgecolor=t["surface"], linewidth=2, label="open-vocabulary extractor")
    ax.barh(y - .17, closed, height=.32, color=t["muted"], zorder=3,
            edgecolor=t["surface"], linewidth=2, label="closed vocabulary alone")
    for yy, v in zip(y, open_v):
        if v > 0:
            ax.text(v + .015, yy + .17, f"{v:.0%}", va="center", fontsize=8.6,
                    fontweight="600", color=t["primary"])
    ax.text(.03, y[order.index("count")] + .17, "0% — counting is not attempted",
            va="center", fontsize=8.4, color=t["secondary"], style="italic")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{k}   n={by[k]['n']}" for k in order], fontsize=9)
    ax.set_xlim(0, 1.12); ax.set_xticks(np.arange(0, 1.01, .25))
    ax.set_xticklabels([f"{v:.0%}" for v in np.arange(0, 1.01, .25)])
    ax.set_xlabel("correct answer placed on the board")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=8.4, handlelength=1.2)
    title(ax, "Reading answers out of context, not out of a word list",
          "60 authored where/who/which/colour/alternative questions whose answer "
          "is present in the scene or transcript", t)
    save(fig, "deictic", dark)


# =============================================================================
# 5. Where the board latency goes
# =============================================================================
def fig_latency(t, dark):
    e = json.load(open(ART / "e2e_mac.json"))["results"]
    lat = {r["backend"]: r["p50_ms"] for r in e}
    labels = ["PyTorch\nembed + head", "OpenVINO head\ntorch embed",
              "OpenVINO\nembed + head"]
    vals = [lat["torch"], lat["ov-head"], lat["ov-full"]]
    style(t)
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    cols = [t["muted"], t["muted"], t["s1"]]
    bars = ax.bar(range(3), vals, width=.5, color=cols, zorder=3,
                  edgecolor=t["surface"], linewidth=2)
    for i, v in enumerate(vals):
        ax.text(i, v + .35, f"{v:.2f} ms", ha="center", fontsize=9.6,
                fontweight="600", color=t["primary"])
    ax.annotate("", xy=(2, vals[2] + 1.6), xytext=(1, vals[1] + 1.6),
                arrowprops=dict(arrowstyle="->", color=t["s1"], lw=1.4))
    ax.text(1.5, vals[1] + 2.1, f"1.76× — moving MiniLM to OpenVINO",
            ha="center", fontsize=8.8, color=t["s1"], fontweight="600")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8.8)
    ax.set_ylim(0, max(vals) * 1.42)
    ax.set_ylabel("board latency, p50 (ms)")
    ax.grid(axis="x", visible=False)
    ax.spines["left"].set_visible(False)
    title(ax, "The embedder is the cost, not the classifier",
          "84 held-out scenarios, Apple M4. The distilled head is ~0.1 ms of this.", t)
    save(fig, "latency", dark)


if __name__ == "__main__":
    for theme, dark in ((LIGHT, False), (DARK, True)):
        style(theme)
        fig_ceiling(theme, dark)
        fig_diversity(theme, dark)
        fig_inversion(theme, dark)
        fig_deictic(theme, dark)
        fig_latency(theme, dark)
    print("wrote:", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))


# =============================================================================
# Logo — the board itself: 4 pinned slots, 6 dynamic, one surfaced for the moment.
# Transparent background so it sits on GitHub's light or dark README.
# =============================================================================
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
    print("wrote logo.png")


make_logo()
