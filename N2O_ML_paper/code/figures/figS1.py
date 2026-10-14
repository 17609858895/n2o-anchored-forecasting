import sys, os, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from figstyle import *
import matplotlib.pyplot as plt
configure()
ROOT = os.path.join(HERE, "..", "analysis"); OUTDIR = os.path.join(HERE, "..", "..", "figures", "supplementary", "FigS01_data_cleaning")
D = pickle.load(open(os.path.join(ROOT, "figdata", "si", "S1.pkl"), "rb"))


def p_flag(fig, spec):
    ax = fig.add_subplot(spec); f = D["flag"]
    vs = [v for v in f["var"].unique()]; y = np.arange(len(vs))
    for k, s in enumerate([1, 2]):
        v = f[f.street == s].set_index("var").loc[vs, "pct"].values
        ax.barh(y + (k - 0.5) * 0.38, v, 0.38, color=STREET_COLOR[s], edgecolor=INK, lw=0.6, label=f"Street {s}")
        for yi, vi in zip(y + (k - 0.5) * 0.38, v): ax.text(vi + 0.8, yi, f"{vi:.1f}" if vi >= 0.05 else ("<0.05" if vi > 0 else "0"), va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([VARNAME.get(v, v) for v in vs]); ax.invert_yaxis()
    ax.set_xlim(0, 95); style_axis(ax, "Values filtered or clipped (%)", None, nx=5); ax.legend(loc="center right")


def p_hood(fig, spec):
    ax = fig.add_subplot(spec); bins = np.linspace(-20, 20, 81)
    for i, c in enumerate(["N2O1", "N2O2", "N2O3"]):
        v = D["hood"][c]; ax.hist(v[(v > -20) & (v < 20)], bins=bins, histtype="step", lw=1.8, color=ZONE_COLOR[f"Z{i + 1}"], label=f"Z{i + 1}")
    ax.axvline(0, color=INK, lw=1.0, ls="--"); ax.set_yscale("log")
    style_axis(ax, "Raw hood N$_2$O (gN h$^{-1}$)", "Hours", nx=5); ax.legend(loc="upper right")


def p_negmonth(fig, spec):
    ax = fig.add_subplot(spec); nm = D["neg_month"]
    d = (nm.loc[1] + nm.loc[2].reindex(nm.loc[1].index)) / 2
    for i in (1, 2, 3):
        ax.plot(d.index, d[f"Z{i}"], lw=1.9, marker="o", ms=4, mfc="white", color=ZONE_COLOR[f"Z{i}"], label=f"Z{i}")
    style_axis(ax, "Month", "Negative readings (%)", ny=5)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m")); bold_ticks(ax)
    ax.set_ylim(0, 100); ax.legend(ncol=3, loc="upper left")


def p_ann(fig, spec):
    ax = fig.add_subplot(spec); a = D["ann"].reset_index(); a.columns = ["year", "street", "raw", "clip", "diff_pct"]
    x = np.arange(a.year.nunique()); yrs = sorted(a.year.unique())
    for k, s in enumerate([1, 2]):
        v = a[a.street == s].set_index("year").loc[yrs, "diff_pct"].values
        ax.bar(x + (k - 0.5) * 0.36, v, 0.36, color=STREET_COLOR[s], edgecolor=INK, lw=0.6, label=f"Street {s}")
        for xi, vi in zip(x + (k - 0.5) * 0.36, v): ax.text(xi, vi, f"{vi:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(yrs); style_axis(ax, "Year", "Change in annual N$_2$O (%)", ny=4)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18); ax.legend(loc="upper left", ncol=2)


panels = {"a": p_flag, "b": p_hood, "c": p_negmonth, "d": p_ann}
sizes = {"a": (6.0, 6.0), "b": (6.0, 4.6), "c": (7.2, 4.6), "d": (5.6, 4.6)}


def layout(fig):
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15])
    return {"a": gs[0, 0], "b": gs[0, 1], "c": gs[1, 0], "d": gs[1, 1]}


ALIGN = {"top": [["a", "b"], ["c", "d"]], "bottom": [["a", "b"], ["c", "d"]], "left": [["a", "c"]], "right": [["b", "d"]]}

if __name__ == "__main__":
    save("FigS01_data_cleaning", OUTDIR, layout, panels, sizes, figsize=(14, 11), engine="subfig", align=ALIGN)
