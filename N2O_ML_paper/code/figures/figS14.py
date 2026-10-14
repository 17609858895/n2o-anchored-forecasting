import sys, os, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from figstyle import *
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
configure()
ROOT = os.path.join(HERE, "..", "analysis"); OUTDIR = os.path.join(HERE, "..", "..", "figures", "supplementary", "FigS14_zone_level")
D = pickle.load(open(os.path.join(ROOT, "figdata", "si", "S13.pkl"), "rb"))
r2 = lambda y, p: r2_score(y, np.clip(p, 0, None))


def p_bar(fig, spec):
    ax = fig.add_subplot(spec); zs = ["N2O1", "N2O2", "N2O3"]; x = np.arange(3)
    for k, (m, c, lb) in enumerate([("EGMoE", C["red"], "EG-MoE"), ("LightGBM", C["blue"], "LightGBM"), ("Persistence", C["gold"], "Persistence")]):
        v = [r2(D[z]["y"], D[z][m]) for z in zs]; ax.bar(x + (k - 1) * 0.26, v, 0.26, color=c, edgecolor=INK, lw=0.6, label=lb)
        for xi, vi in zip(x + (k - 1) * 0.26, v): ax.text(xi, vi + 0.01, f"{vi:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold", rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(["Z1 (swing)", "Z2", "Z3"]); ax.set_ylim(0, 1.12)
    style_axis(ax, "Zone", "Test R², 2022", ny=5); ax.legend(loc="upper left", ncol=3, fontsize=11)


def scat(z):
    def draw(fig, spec):
        ax = fig.add_subplot(spec); f = lambda v: np.log10(np.clip(v, 0, None) + 1); y = D[z]["y"]; p = D[z]["EGMoE"]
        hb = ax.hexbin(f(y), f(p), gridsize=35, cmap=SEQ_RED, mincnt=1, bins="log", linewidths=0)
        hi = max(f(y).max(), f(p).max()); ax.plot([0, hi], [0, hi], color=INK, lw=1.1, ls="--")
        ax.text(0.04, 0.96, f"R² = {r2(y, p):.3f}\nRMSE = {mean_squared_error(y, np.clip(p, 0, None)) ** .5:.1f}", transform=ax.transAxes, va="top", fontsize=12, fontweight="bold")
        style_axis(ax, f"Observed, Z{z[-1]} log$_{{10}}$(N$_2$O + 1)", "EG-MoE, log$_{10}$(N$_2$O + 1)" if z == "N2O1" else None, nx=4, ny=4)
    return draw


panels = {"a": p_bar, "b": scat("N2O1"), "c": scat("N2O2"), "d": scat("N2O3")}
sizes = {"a": (5.8, 4.8), "b": (4.8, 4.6), "c": (4.8, 4.6), "d": (4.8, 4.6)}


def layout(fig):
    gs = fig.add_gridspec(1, 4, width_ratios=[1.3, 1, 1, 1]); return {"a": gs[0], "b": gs[1], "c": gs[2], "d": gs[3]}


ALIGN = {"top": [["a", "b", "c", "d"]], "bottom": [["a", "b", "c", "d"]]}

if __name__ == "__main__":
    save("FigS14_zone_level", OUTDIR, layout, panels, sizes, figsize=(19, 5.4), engine="subfig", align=ALIGN)
