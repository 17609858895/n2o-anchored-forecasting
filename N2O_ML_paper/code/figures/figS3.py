import sys, os, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from figstyle import *
import matplotlib.pyplot as plt
configure()
ROOT = os.path.join(HERE, "..", "analysis"); OUTDIR = os.path.join(HERE, "..", "..", "figures", "supplementary", "FigS03_gate_sensitivity")
D = pickle.load(open(os.path.join(ROOT, "figdata", "si", "S14.pkl"), "rb"))
lam = D["lam"].copy(); lam["x"] = np.log10(lam.lam.clip(1, 1e5))


def p_lam(fig, spec):
    ax = fig.add_subplot(spec); a2 = ax.twinx()
    ax.plot(lam.x, lam.RMSE, color=C["red"], lw=2.2, marker="o", ms=6, mfc="white"); a2.plot(lam.x, lam.absCum, color=C["blue"], lw=2.2, marker="s", ms=6, mfc="white")
    ax.axvline(np.log10(500), color=MUTED, lw=1.1, ls="--")
    ax.set_xticks([0, 1, 2, 3, 4, 5]); ax.set_xticklabels(["0", "10", "10²", "10³", "10⁴", "∞"])
    style_axis(ax, "Shrinkage strength λ", "Test RMSE (gN h$^{-1}$)", ny=4); ax.yaxis.label.set_color(C["red"])
    style_axis(a2, None, None, spines=("right",), ny=4); a2.tick_params(axis="y", right=True, labelright=True); bold_ticks(a2)
    a2.set_ylabel("|Cumulative error| (%)", fontsize=15.5, fontweight="bold", color=C["blue"])


def p_folds(fig, spec):
    ax = fig.add_subplot(spec); f = pd.DataFrame([{k: v for k, v in d.items() if k != "W"} for d in D["folds"]]); x = np.arange(len(f))
    ax.scatter(x, f.RMSE, s=150, marker="D", color=[C["red"] if n == 4 else C["pink"] for n in f.folds], edgecolor=INK, lw=0.8, zorder=3)
    for xi, r, c in zip(x, f.RMSE, f.absCum): ax.text(xi, r + 0.12, f"{r:.2f}\n|CumErr| {c:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(f.folds); lo, hi = f.RMSE.min(), f.RMSE.max(); ax.set_ylim(lo - 0.6, hi + 0.7); ax.set_xlim(-0.5, len(f) - 0.5)
    style_axis(ax, "Forward-chained OOF folds", "Test RMSE (gN h$^{-1}$)", ny=4)


panels = {"a": p_lam, "b": p_folds}
sizes = {"a": (6.4, 4.8), "b": (5.6, 4.8)}


def layout(fig):
    gs = fig.add_gridspec(1, 2); return {"a": gs[0], "b": gs[1]}


ALIGN = {"top": [["a", "b"]], "bottom": [["a", "b"]]}

if __name__ == "__main__":
    save("FigS03_gate_sensitivity", OUTDIR, layout, panels, sizes, figsize=(13, 5.4), engine="subfig", align=ALIGN)
