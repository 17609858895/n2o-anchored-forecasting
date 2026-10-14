import sys, os, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from figstyle import *
import matplotlib.pyplot as plt
configure()
ROOT = os.path.join(HERE, "..", "analysis"); OUTDIR = os.path.join(HERE, "..", "..", "figures", "supplementary", "FigS19_ice_pdp_ale")
D = pickle.load(open(os.path.join(ROOT, "figdata", "si", "S7.pkl"), "rb"))
FS = ["T_in", "DO2", "NO3", "Supern", "Cl_m72", "Q_out"]
XL = {"T_in": "T (°C)", "DO2": "DO$_2$ (mg L$^{-1}$)", "NO3": "NO$_3^-$ (mgN L$^{-1}$)", "Supern": "Supernatant valve (%)",
      "Cl_m72": "Cl$^-$ 72 h mean (mg L$^{-1}$)", "Q_out": "Q$_{out}$ (dataset units)"}


def p_ice(fig, spec):
    gs = spec.subgridspec(1, 6)
    for k, f in enumerate(FS):
        ax = fig.add_subplot(gs[k]); d = D[f]
        for line in d["ice"]: ax.plot(d["grid"], line, color=C["lilac"], lw=0.6, alpha=0.5)
        ax.plot(d["grid"], d["pdp"] - np.mean(d["pdp"]), color=C["blue"], lw=2.4, ls="--", label="PDP")
        style_axis(ax, XL[f], "Centred effect (gN h$^{-1}$)" if k == 0 else None, nx=3, ny=4)
        if k == 0: ax.legend(loc="lower right")


def p_ale(fig, spec):
    gs = spec.subgridspec(1, 6)
    for k, f in enumerate(FS):
        ax = fig.add_subplot(gs[k]); d = D[f]
        ax.fill_between(d["grid"], d["lo"], d["hi"], color=C["red"], alpha=0.2, lw=0)
        ax.plot(d["grid"], d["ale"], color=C["red"], lw=2.4, label="ALE")
        ax.axhline(0, color=INK, lw=0.8)
        q = np.nanpercentile(d["x"], np.arange(10, 100, 10)); ax.plot(q, np.full(len(q), 0.04), "|", color=INK, ms=10, mew=1.6, transform=ax.get_xaxis_transform(), clip_on=False)
        style_axis(ax, XL[f], "ALE (gN h$^{-1}$)" if k == 0 else None, nx=3, ny=4)
        if k == 0: ax.legend(loc="lower right")


def corr(xc, xl):
    def draw(fig, spec):
        ax = fig.add_subplot(spec); d = D["_corr"]
        hb = ax.hexbin(d[xc], d["DO2"], gridsize=40, cmap=SEQ, mincnt=1, bins="log", linewidths=0)
        from scipy.stats import spearmanr; r = spearmanr(d[xc], d["DO2"])[0]
        ax.text(0.04, 0.96, f"ρ = {r:.2f}", transform=ax.transAxes, va="top", fontsize=12.5, fontweight="bold")
        style_axis(ax, xl, "DO$_2$ (mg L$^{-1}$)", nx=4, ny=4)
    return draw


panels = {"a": p_ice, "b": p_ale, "c": corr("T_in", "T (°C)"), "d": corr("Air2", "Air$_2$ (dataset units)"), "e": corr("Air3", "Air$_3$ (dataset units)")}
sizes = {"a": (18, 3.8), "b": (18, 3.8), "c": (5.4, 4.4), "d": (5.4, 4.4), "e": (5.4, 4.4)}


def layout(fig):
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.15])
    return {"a": gs[0, :], "b": gs[1, :], "c": gs[2, 0], "d": gs[2, 1], "e": gs[2, 2]}


ALIGN = {"top": [["c", "d", "e"]], "bottom": [["c", "d", "e"]], "left": [["a", "b", "c"]], "right": [["a", "b", "e"]]}

if __name__ == "__main__":
    save("FigS19_ice_pdp_ale", OUTDIR, layout, panels, sizes, figsize=(19, 13), engine="subfig", align=ALIGN)
