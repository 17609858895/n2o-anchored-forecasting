import sys, os, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from figstyle import *
import matplotlib.pyplot as plt
configure()
ROOT = os.path.join(HERE, "..", "analysis"); OUTDIR = os.path.join(HERE, "..", "..", "figures", "supplementary", "FigS09_deep_training")
D = pickle.load(open(os.path.join(ROOT, "figdata", "si", "S6.pkl"), "rb"))
NC = {"LSTM": C["indigo"], "GRU": C["lilac"], "TCN": C["sky"], "Transformer": C["blue"]}


def p_loss(fig, spec):
    gs = spec.subgridspec(1, 2)
    for k, resid in enumerate([False, True]):
        ax = fig.add_subplot(gs[k])
        for (n, r), d in D.items():
            if r != resid: continue
            ax.plot(d.epoch, d.train_loss, color=NC[n], lw=2.0, label=n)
        style_axis(ax, "Epoch", "Training loss (scaled MSE)" if k == 0 else None, nx=5, ny=4)
        ax.text(0.97, 0.95, "Residual-anchored" if resid else "Direct target", transform=ax.transAxes, ha="right", va="top", fontsize=12, fontweight="bold")
        if k == 0: ax.legend(loc="center right")


def p_val(fig, spec):
    gs = spec.subgridspec(1, 2)
    for k, resid in enumerate([False, True]):
        ax = fig.add_subplot(gs[k])
        for (n, r), d in D.items():
            if r != resid: continue
            ax.plot(d.epoch, d.R2_fit, color=NC[n], lw=1.8); ax.plot(d.epoch, d.R2_val, color=NC[n], lw=1.8, ls="--")
        ax.plot([], [], color=INK, lw=1.8, label="Training"); ax.plot([], [], color=INK, lw=1.8, ls="--", label="Validation")
        style_axis(ax, "Epoch", "R²" if k == 0 else None, nx=5, ny=5)
        ax.text(0.97, 0.05, "Residual-anchored" if resid else "Direct target", transform=ax.transAxes, ha="right", va="bottom", fontsize=12, fontweight="bold")
        if k == 0: ax.legend(loc="lower left")


def p_test(fig, spec):
    ax = fig.add_subplot(spec); ns = list(NC); x = np.arange(len(ns))
    for k, (resid, c, lb) in enumerate([(False, C["lilac"], "Direct target"), (True, C["red"], "Residual-anchored")]):
        v = [D[(n, resid)].R2_test.iloc[-1] for n in ns]
        ax.bar(x + (k - 0.5) * 0.38, v, 0.38, color=c, edgecolor=INK, lw=0.6, label=lb)
        for xi, vi in zip(x + (k - 0.5) * 0.38, v): ax.text(xi, max(vi, 0) + 0.01, f"{vi:.2f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(ns); ax.set_ylim(min(0, ax.get_ylim()[0]), 1.0)
    style_axis(ax, None, "Test R², 2022 (epoch 20)", ny=5); ax.legend(loc="upper left", ncol=2)


panels = {"a": p_loss, "b": p_val, "c": p_test}
sizes = {"a": (10, 4.4), "b": (10, 4.4), "c": (6.4, 4.6)}


def layout(fig):
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.9])
    return {"a": gs[0, 0:2], "b": gs[1, 0:2], "c": gs[:, 2]}


ALIGN = {"top": [["a", "c"]], "bottom": [["b", "c"]], "left": [["a", "b"]], "right": [["a", "b"]]}

if __name__ == "__main__":
    save("FigS09_deep_training", OUTDIR, layout, panels, sizes, figsize=(18, 10), engine="subfig", align=ALIGN)
