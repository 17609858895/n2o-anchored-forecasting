import sys, os, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from figstyle import *
import matplotlib.pyplot as plt
configure()
ROOT = os.path.join(HERE, "..", "analysis"); OUTDIR = os.path.join(HERE, "..", "..", "figures", "supplementary", "FigS15_threshold_sensitivity")
D = pickle.load(open(os.path.join(ROOT, "figdata", "si", "S8.pkl"), "rb"))
QC = {0.7: C["sky"], 0.8: C["red"], 0.9: C["indigo"]}
EN = {"anchor": "Anchor", "resid": "Residual", "raw": "Raw-scale", "log": "Log-scale", "tweedie": "Tweedie"}


def p_metrics(fig, spec):
    gs = spec.subgridspec(1, 4); qs = sorted(D)
    for k, (m, yl) in enumerate([("R2", "R²"), ("RMSE", "RMSE (gN h$^{-1}$)"), ("F1", "Episode F1"), ("AUC", "Gate ROC-AUC")]):
        ax = fig.add_subplot(gs[k]); v = [D[q][m] for q in qs]
        ax.scatter(range(3), v, s=140, marker="D", color=[QC[q] for q in qs], edgecolor=INK, lw=0.8, zorder=3)
        lo = min(v); hi = max(v); pad = max(hi - lo, 0.01 * hi)
        for i, vi in enumerate(v): ax.text(i, vi + pad * 0.25, f"{vi:.3f}" if m != "RMSE" else f"{vi:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_ylim(lo - pad * 0.8, hi + pad * 1.0); ax.set_xlim(-0.6, 2.6)
        ax.set_xticks(range(3)); ax.set_xticklabels([f"P{int(100 * q)}" for q in qs]); style_axis(ax, "τ", yl, ny=4)


def p_weights(fig, spec):
    gs = spec.subgridspec(1, 3); qs = sorted(D)
    for b, bn in enumerate(["p < 0.15", "0.15–0.5", "p ≥ 0.5"]):
        ax = fig.add_subplot(gs[b]); names = D[qs[0]]["names"]; x = np.arange(len(names))
        for k, q in enumerate(qs):
            w = D[q]["W"][b].mean(0); ax.bar(x + (k - 1) * 0.26, w, 0.26, color=QC[q], edgecolor=INK, lw=0.5, label=f"τ = P{int(100 * q)}")
        ax.set_xticks(x); ax.set_xticklabels([EN[n] for n in names], rotation=35, ha="right"); ax.set_ylim(0, 0.85)
        style_axis(ax, None, "Mean expert weight" if b == 0 else None, ny=4)
        ax.text(0.97, 0.95, bn, transform=ax.transAxes, ha="right", va="top", fontsize=12.5, fontweight="bold")
        if b == 0: ax.legend(loc="upper left")


panels = {"a": p_metrics, "b": p_weights}
sizes = {"a": (14, 4.2), "b": (14, 4.6)}


def layout(fig):
    gs = fig.add_gridspec(2, 1); return {"a": gs[0], "b": gs[1]}


ALIGN = {"left": [["a", "b"]], "right": [["a", "b"]]}

if __name__ == "__main__":
    save("FigS15_threshold_sensitivity", OUTDIR, layout, panels, sizes, figsize=(16, 10), engine="subfig", align=ALIGN)
