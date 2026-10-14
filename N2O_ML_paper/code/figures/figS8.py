import sys, os, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from figstyle import *
import matplotlib.pyplot as plt
configure()
ROOT = os.path.join(HERE, "..", "analysis"); OUTDIR = os.path.join(HERE, "..", "..", "figures", "supplementary", "FigS08_campaign_audit")
E = pd.read_csv(os.path.join(ROOT, "figdata", "campaign_errors.csv"))
WK = sorted(E.weeks.unique())


def p_lines(fig, spec):
    ax = fig.add_subplot(spec); cols = [C["teal"], C["orange"], C["red"], C["blue"], C["lilac"], C["gold"]]
    for (k, d), c in zip(E.groupby(["year", "street"]), cols):
        med = d.groupby("weeks").err.apply(lambda v: np.median(np.abs(v)))
        ax.plot(med.index, med.values, marker="o", lw=1.9, ms=6, mfc="white", color=c, label=f"{k[0]}, street {k[1]}")
    ax.set_xscale("log"); ax.set_xticks(WK); ax.set_xticklabels(WK)
    style_axis(ax, "Monitoring campaign (weeks)", "Median |error| in annual mean (%)", ny=5); bold_ticks(ax); ax.legend(loc="lower left", fontsize=11)


def p_season(fig, spec):
    ax = fig.add_subplot(spec)
    E["season"] = np.where(E.start_month.isin([11, 12, 1, 2, 3]), "Winter start (Nov–Mar)", np.where(E.start_month.isin([5, 6, 7, 8, 9]), "Summer start (May–Sep)", "Other"))
    sel = [w for w in WK if w in (2, 4, 8, 13)]; x = np.arange(len(sel))
    for k, (s, c) in enumerate([("Winter start (Nov–Mar)", C["blue"]), ("Summer start (May–Sep)", C["orange"])]):
        data = [E[(E.weeks == w) & (E.season == s)].err.clip(-100, 300).values for w in sel]
        bp = ax.boxplot(data, positions=x + (k - 0.5) * 0.36, widths=0.32, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.6))
        for b in bp["boxes"]: b.set_facecolor(c); b.set_alpha(0.75); b.set_edgecolor(INK)
        ax.plot([], [], "s", color=c, ms=10, label=s)
    ax.axhline(0, color=C["red"], lw=1.1, ls="--"); ax.set_xticks(x); ax.set_xticklabels(sel)
    style_axis(ax, "Monitoring campaign (weeks)", "Error in annual mean (%)", ny=5); ax.legend(loc="upper right")


import fig2 as F2
panels = {"a": F2.p_campaign, "b": p_lines, "c": p_season}
sizes = {"a": (5.6, 5.0), "b": (6.6, 5.0), "c": (6.6, 5.0)}


def layout(fig):
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1, 1]); return {"a": gs[0], "b": gs[1], "c": gs[2]}


ALIGN = {"top": [["a", "b", "c"]], "bottom": [["a", "b", "c"]]}

if __name__ == "__main__":
    save("FigS08_campaign_audit", OUTDIR, layout, panels, sizes, figsize=(19, 5.8), engine="subfig", align=ALIGN)
