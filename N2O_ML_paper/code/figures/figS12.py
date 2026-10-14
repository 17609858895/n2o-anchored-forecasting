import sys, os, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from figstyle import *
import matplotlib.pyplot as plt
configure()
ROOT = os.path.join(HERE, "..", "analysis"); OUTDIR = os.path.join(HERE, "..", "..", "figures", "supplementary", "FigS12_feature_ablation")
D = pickle.load(open(os.path.join(ROOT, "figdata", "si", "S9.pkl"), "rb"))
ROWS = list(D.index)


def bars(metric, xl, absval=False):
    def draw(fig, spec):
        ax = fig.add_subplot(spec); y = np.arange(len(ROWS))
        for k, (m, c) in enumerate([("EG-MoE", C["red"]), ("LightGBM", C["blue"])]):
            v = D[f"{m} {metric}"].astype(float).values; v = np.abs(v) if absval else v
            if metric == "R2":
                ax.scatter(v, y + (k - 0.5) * 0.3, s=110, marker="D", color=c, edgecolor=INK, lw=0.7, label=m, zorder=3)
            else:
                ax.barh(y + (k - 0.5) * 0.38, v, 0.38, color=c, edgecolor=INK, lw=0.6, label=m)
            for yi, vi in zip(y + (k - 0.5) * (0.3 if metric == "R2" else 0.38), v): ax.text(vi, yi, f"  {vi:.3f}" if metric == "R2" else f" {vi:.1f}", va="center", fontsize=10.5, fontweight="bold")
        ax.set_yticks(y); ax.set_yticklabels(ROWS); ax.invert_yaxis()
        if metric == "R2": vals = D[["EG-MoE R2", "LightGBM R2"]].astype(float).values; ax.set_xlim(vals.min() - 0.01, vals.max() + 0.025)
        else: ax.set_xlim(0, ax.get_xlim()[1] * 1.18)
        style_axis(ax, xl, None, nx=4)
        if metric == "R2": ax.legend(loc="lower right")
    return draw


panels = {"a": bars("R2", "Test R², 2022"), "b": bars("RMSE", "Test RMSE (gN h$^{-1}$)"), "c": bars("CumErr", "|Cumulative error| (%)", absval=True)}
sizes = {"a": (6.2, 4.8), "b": (6.2, 4.8), "c": (6.2, 4.8)}


def layout(fig):
    gs = fig.add_gridspec(1, 3); return {"a": gs[0], "b": gs[1], "c": gs[2]}


ALIGN = {"top": [["a", "b", "c"]], "bottom": [["a", "b", "c"]]}

if __name__ == "__main__":
    save("FigS12_feature_ablation", OUTDIR, layout, panels, sizes, figsize=(19, 5.6), engine="subfig", align=ALIGN)
