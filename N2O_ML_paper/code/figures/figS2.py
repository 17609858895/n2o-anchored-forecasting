import sys, os, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from figstyle import *
import matplotlib.pyplot as plt
configure()
ROOT = os.path.join(HERE, "..", "analysis"); OUTDIR = os.path.join(HERE, "..", "..", "figures", "supplementary", "FigS02_missingness")
D = pickle.load(open(os.path.join(ROOT, "figdata", "si", "S2.pkl"), "rb"))
COLS = D["cols"]


def heat(s):
    def draw(fig, spec):
        gs = spec.subgridspec(1, 2, width_ratios=[1, 0.025]); ax = fig.add_subplot(gs[0]); cax = fig.add_subplot(gs[1])
        M = D["miss"][s][COLS].T
        im = ax.imshow(M.values, aspect="auto", cmap=SEQ, vmin=0, vmax=100, interpolation="nearest")
        ax.set_yticks(range(len(COLS))); ax.set_yticklabels([VARNAME.get(c, c) for c in COLS])
        idx = [i for i, t in enumerate(M.columns) if t.month in (1, 7)]
        ax.set_xticks(idx); ax.set_xticklabels([M.columns[i].strftime("%y-%m") for i in idx])
        style_axis(ax, f"Month (street {s})", None); ax.tick_params(length=5)
        for sp in ax.spines.values(): sp.set_visible(False)
        cb = fig.colorbar(im, cax=cax); cb.set_label("Missing (%)", fontsize=12.5, fontweight="bold"); bold_ticks(cb.ax, 10.5)
    return draw


def p_tot(fig, spec):
    ax = fig.add_subplot(spec); y = np.arange(len(COLS))
    for k, s in enumerate([1, 2]):
        ax.barh(y + (k - 0.5) * 0.38, D["tot"][s][COLS].values, 0.38, color=STREET_COLOR[s], edgecolor=INK, lw=0.6, label=f"Street {s}")
    ax.set_yticks(y); ax.set_yticklabels([VARNAME.get(c, c) for c in COLS]); ax.invert_yaxis()
    style_axis(ax, "Missing, 2020–2022 (%)", None, nx=4); ax.legend(loc="center right")


panels = {"a": heat(1), "b": heat(2), "c": p_tot}
sizes = {"a": (10, 6.4), "b": (10, 6.4), "c": (5.2, 6.4)}


def layout(fig):
    gs = fig.add_gridspec(2, 2, width_ratios=[2.3, 1])
    return {"a": gs[0, 0], "b": gs[1, 0], "c": gs[:, 1]}


ALIGN = {"top": [["a", "c"]], "bottom": [["b", "c"]], "left": [["a", "b"]], "right": [["a", "b"]]}

if __name__ == "__main__":
    save("FigS02_missingness", OUTDIR, layout, panels, sizes, figsize=(16, 13), engine="subfig", align=ALIGN)
