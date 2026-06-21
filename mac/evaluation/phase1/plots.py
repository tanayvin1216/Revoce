"""
Phase-1 FIGURES — render the write-up graphs from results.json / sim_results.json.

    python3 evaluation/phase1/plots.py

Writes PNGs to evaluation/phase1/figures/. No model calls; pure plotting.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

INK = "#1a1a1a"
BASE_C = "#b0b7c3"      # muted grey-blue = baseline
NEW_C = "#c2410c"       # burnt orange = diverse method
GRID = "#e6e6e6"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": INK,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK,
    "ytick.color": INK, "axes.grid": True, "grid.color": GRID,
    "axes.axisbelow": True, "figure.dpi": 130, "savefig.bbox": "tight",
})


def _save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", name)


def headline_bars(r):
    b, p = r["baseline"]["agg"], r["sweep"][f"{r['prod_lambda']:.1f}"]["agg"]
    s = r["stats"]
    labels = ["Coverage@7", "ILD@7", "Function\ncoverage (/9)"]
    base = [b["covered"], b["ild"], b["fcov"] / 9]
    new = [p["covered"], p["ild"], p["fcov"] / 9]
    err = [abs(s["cov_ci"][1] - s["cov_ci"][0]) / 2,
           abs(s["ild_ci"][1] - s["ild_ci"][0]) / 2, 0]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - 0.2, base, 0.38, label="Baseline (mode-seeking)", color=BASE_C)
    ax.bar(x + 0.2, new, 0.38, yerr=err, capsize=4, label=f"Diverse (MMR λ={r['prod_lambda']})", color=NEW_C)
    for xi, (bv, nv) in enumerate(zip(base, new)):
        ax.text(xi - 0.2, bv + 0.01, f"{bv:.2f}", ha="center", fontsize=9, color=INK)
        ax.text(xi + 0.2, nv + 0.01, f"{nv:.2f}", ha="center", fontsize=9, color=NEW_C)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0); ax.set_ylabel("score")
    ax.set_title("Diverse coverage vs mode-seeking baseline", fontweight="bold")
    ax.legend(frameon=False)
    _save(fig, "01_headline_bars.png")


def lambda_sweep(r):
    lams = sorted(float(k) for k in r["sweep"])
    cov = [r["sweep"][f"{l:.1f}"]["agg"]["covered"] for l in lams]
    setc = [r["sweep"][f"{l:.1f}"]["agg"]["setcov"] for l in lams]
    ild = [r["sweep"][f"{l:.1f}"]["agg"]["ild"] for l in lams]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(lams, cov, "o-", color="#0f766e", label="Coverage@7")
    ax.plot(lams, setc, "s-", color=NEW_C, label="SetCov@7")
    ax.plot(lams, ild, "^-", color="#6d28d9", label="ILD@7")
    ax.axvline(r["prod_lambda"], ls="--", color=INK, alpha=0.5)
    ax.text(r["prod_lambda"] + 0.01, 0.05, f"chosen λ={r['prod_lambda']}", fontsize=9)
    ax.set_xlabel("λ  (1.0 = pure relevance / mode-seeking,  →0 = pure diversity)")
    ax.set_ylabel("score"); ax.set_ylim(0, 1.0)
    ax.set_title("λ sweep: the diversity/coverage trade-off", fontweight="bold")
    ax.legend(frameon=False)
    _save(fig, "02_lambda_sweep.png")


def ild_hist(r):
    base = [t["ild"] for t in r["baseline"]["per_turn"]]
    new = [t["ild"] for t in r["sweep"][f"{r['prod_lambda']:.1f}"]["per_turn"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 26)
    ax.hist(base, bins, alpha=0.7, color=BASE_C, label=f"Baseline (mean {np.mean(base):.2f})")
    ax.hist(new, bins, alpha=0.7, color=NEW_C, label=f"Diverse (mean {np.mean(new):.2f})")
    ax.set_xlabel("Intra-List Diversity@7"); ax.set_ylabel("turns")
    ax.set_title("Board diversity distribution", fontweight="bold")
    ax.legend(frameon=False)
    _save(fig, "03_ild_distribution.png")


def setcov_vs_ild_scatter(r):
    bp = r["baseline"]["per_turn"]
    np_ = r["sweep"][f"{r['prod_lambda']:.1f}"]["per_turn"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter([t["ild"] for t in bp], [t["setcov"] for t in bp], s=28,
               color=BASE_C, alpha=0.7, label="Baseline")
    ax.scatter([t["ild"] for t in np_], [t["setcov"] for t in np_], s=28,
               color=NEW_C, alpha=0.7, label="Diverse")
    ax.set_xlabel("ILD@7 (board diversity)"); ax.set_ylabel("SetCov@7 (space covered)")
    ax.set_title("Coverage/diversity frontier (ideal = top-right)", fontweight="bold")
    ax.legend(frameon=False)
    _save(fig, "04_setcov_vs_ild.png")


def delta_setcov(r):
    bp = r["baseline"]["per_turn"]
    np_ = r["sweep"][f"{r['prod_lambda']:.1f}"]["per_turn"]
    d = sorted(p["setcov"] - b["setcov"] for p, b in zip(np_, bp))
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [NEW_C if x >= 0 else "#9ca3af" for x in d]
    ax.bar(range(len(d)), d, color=colors, width=1.0)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("turn (sorted by Δ)"); ax.set_ylabel("Δ SetCov@7 (diverse − baseline)")
    win = sum(1 for x in d if x > 0)
    ax.set_title(f"Per-turn SetCov gain  ({win}/{len(d)} turns improve)", fontweight="bold")
    _save(fig, "05_delta_setcov.png")


def func_coverage(r):
    base = [t["fcov"] for t in r["baseline"]["per_turn"]]
    new = [t["fcov"] for t in r["sweep"][f"{r['prod_lambda']:.1f}"]["per_turn"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.arange(0.5, 9.6, 1)
    ax.hist(base, bins, alpha=0.7, color=BASE_C, label=f"Baseline (mean {np.mean(base):.1f}/9)")
    ax.hist(new, bins, alpha=0.7, color=NEW_C, label=f"Diverse (mean {np.mean(new):.1f}/9)")
    ax.set_xlabel("distinct communication functions on the 7-board (of 9)")
    ax.set_ylabel("turns"); ax.set_xticks(range(1, 10))
    ax.set_title("How many DIFFERENT things the board lets you say", fontweight="bold")
    ax.legend(frameon=False)
    _save(fig, "09_function_coverage.png")


def sim_timeline(sim):
    convs = sim["conversations"]
    maxt = max(len(c["turns"]) for c in convs)
    grid = np.full((len(convs), maxt), np.nan)
    for ci, c in enumerate(convs):
        for ti, t in enumerate(c["turns"]):
            grid[ci, ti] = 1.0 if t["covered"] else 0.0
    fig, ax = plt.subplots(figsize=(9, 3.6))
    cmap = matplotlib.colors.ListedColormap(["#dc2626", "#16a34a"])
    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks(range(len(convs)))
    ax.set_yticklabels([f"conv {c['conversation_id']}" for c in convs])
    ax.set_xlabel("turn index"); ax.set_xticks(range(maxt))
    ax.set_title("Production sim: covered (green) vs stranded (red) per turn", fontweight="bold")
    _save(fig, "06_sim_timeline.png")


def sim_latency(sim):
    flat = [t for c in sim["conversations"] for t in c["turns"]]
    pool = np.sort([t["pool_lat_s"] for t in flat])
    sel = np.sort([t["select_lat_s"] * 1000 for t in flat])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(pool, np.linspace(0, 1, len(pool)), color=NEW_C, lw=2)
    a1.axvline(4.0, ls="--", color=INK, alpha=0.6); a1.text(4.05, 0.1, "4s deadline", fontsize=9)
    a1.set_xlabel("pool generation latency (s)"); a1.set_ylabel("CDF")
    a1.set_title("LLM pool call (the only slow part)")
    a2.plot(sel, np.linspace(0, 1, len(sel)), color="#0f766e", lw=2)
    a2.set_xlabel("MMR select latency (ms)"); a2.set_ylabel("CDF")
    a2.set_title("Board selection (re-runs free, non-blocking)")
    fig.suptitle("Latency split — audio path never blocks on the model", fontweight="bold")
    _save(fig, "07_sim_latency.png")


def sim_divergence(sim):
    convs = sim["conversations"]
    labels = [f"conv {c['conversation_id']}" for c in convs]
    new = [np.mean([t["covered"] for t in c["turns"]]) for c in convs]
    base = [np.mean([t["base_covered"] for t in c["turns"]]) for c in convs]
    x = np.arange(len(convs))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - 0.2, base, 0.38, color=BASE_C, label="Baseline coverage")
    ax.bar(x + 0.2, new, 0.38, color=NEW_C, label="Diverse coverage")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("turn coverage"); ax.set_ylim(0, 1.0)
    ax.set_title("Per-conversation coverage (production sim)", fontweight="bold")
    ax.legend(frameon=False)
    _save(fig, "08_sim_divergence.png")


def main():
    r = json.loads((HERE / "results.json").read_text())
    headline_bars(r); lambda_sweep(r); ild_hist(r)
    setcov_vs_ild_scatter(r); delta_setcov(r); func_coverage(r)
    sim_path = HERE / "sim_results.json"
    if sim_path.exists():
        sim = json.loads(sim_path.read_text())
        sim_timeline(sim); sim_latency(sim); sim_divergence(sim)
    print("\nall figures ->", FIG)


if __name__ == "__main__":
    main()
