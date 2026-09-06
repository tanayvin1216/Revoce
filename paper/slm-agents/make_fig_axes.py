"""Figure 1: both axes of agreement (pins included / dynamic slots only), word
agreement, and coverage of intent under the shipped and corrected rule, for both
students and both context-free baselines on the 84 test boards.
Reads the CLEAN (train-only inventory) results. Usage: python3 make_fig_axes.py fig_axes.pdf
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path("/Users/tanayvinaykya/VoiceLink-wt/function-vs-lexis/mac/evaluation/phase4/clean")
arms = [("v2", "v2 (closed)", "#2a78d6", "o"),
        ("v3", "v3 (larger bank)", "#eb6834", "s"),
        ("prior", "Prior board", "#1baf7a", "D"),
        ("random", "Random board", "#eda100", "^")]
rows = ["Function agreement $J_F$\n(top-7, pins included)", "Function agreement $J_F$\n(dynamic slots only)",
        "Word agreement $J_W$\n(dynamic slots only)", "Coverage of intent\n(shipped rule)", "Coverage of intent\n(patched rule)"]
DYN = json.loads((D / "function_transfer_dynamic.json").read_text())
COV = json.loads((D / "coverage_rule_check.json").read_text())["test"]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.size": 8, "axes.linewidth": 0.5})
fig, ax = plt.subplots(figsize=(5.5, 2.3))
off = [-0.27, -0.09, 0.09, 0.27]
for i, (k, name, col, mk) in enumerate(arms):
    o = json.loads((D / f"function_transfer_test_{k}.json").read_text())["overall"]
    g = json.loads((D / f"function_transfer_gates_test_{k}.json").read_text())
    vals = [(o["function_jaccard"], o["function_jaccard_ci95"]), (DYN[k]["JF_dynamic"], DYN[k]["JF_dynamic_ci"]),
            (o["word_jaccard"], o["word_jaccard_ci95"]), (g["coverage_dynamic"], None), (COV["corrected"][k]["coverage"], None)]
    for r, (v, ci) in enumerate(vals):
        y = r + off[i]
        if ci: ax.plot(ci, [y, y], color=col, lw=1.2, solid_capstyle="butt", zorder=2)
        ax.plot([v], [y], marker=mk, color=col, ms=5, mec="white", mew=0.6, zorder=3, label=name if r == 0 else None)
for row, tc in [(3, COV["shipped"]["teacher"]["coverage"]), (4, COV["corrected"]["teacher"]["coverage"])]:
    ax.plot([tc, tc], [row - 0.4, row + 0.4], color="#0b0b0b", lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.text(tc + 0.012, row - 0.3, f"teacher {tc:.2f}", ha="left", va="center", fontsize=6.5, color="#52514e")
ax.set_yticks(range(5)); ax.set_yticklabels(rows); ax.invert_yaxis()
ax.set_xlim(0, 1); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.grid(axis="x", color="#e5e4e0", lw=0.5, zorder=0); ax.set_axisbelow(True)
for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", length=2, width=0.5, colors="#52514e")
ax.set_xlabel("score on the 84 held-out boards (95% bootstrap CI where paired)", fontsize=7, color="#52514e")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False, handletextpad=0.4, borderaxespad=0.0, labelspacing=0.4, markerscale=1.1)
fig.tight_layout(pad=0.3)
out = Path(sys.argv[1]); fig.savefig(out, format="pdf"); fig.savefig(out.with_suffix(".png"), dpi=150)
print("wrote", out)
