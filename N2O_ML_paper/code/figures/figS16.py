import sys, os, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from figstyle import *
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from sklearn.metrics import r2_score, mean_squared_error
configure()
ROOT = os.path.join(HERE, "..", "analysis"); OUTDIR = os.path.join(HERE, "..", "..", "figures", "supplementary", "FigS16_operating_data_only")
D = pickle.load(open(os.path.join(ROOT, "figdata", "si", "S10.pkl"), "rb"))
y = D["y"]; P = {k: np.clip(v, 0, None) for k, v in D["preds"].items()}
f = lambda v: np.log10(v + 1)


def p_scatter(fig, spec):
    ax = fig.add_subplot(spec)
    hb = ax.hexbin(f(y), f(P["LightGBM"]), gridsize=40, cmap=SEQ, mincnt=1, bins="log", linewidths=0, extent=(0, 3.2, 0, 3.2))
    ax.plot([0, 3.2], [0, 3.2], color=INK, lw=1.1, ls="--"); ax.set_xlim(0, 3.2); ax.set_ylim(0, 3.2)
    ax.text(0.04, 0.96, f"R² = {r2_score(y, P['LightGBM']):.2f}\nR² (log) = {r2_score(f(y), f(P['LightGBM'])):.2f}", transform=ax.transAxes, va="top",
            fontsize=12, fontweight="bold", bbox=dict(boxstyle="round", fc="white", ec=MUTED, alpha=0.9))
    style_axis(ax, "Observed log$_{10}$(N$_2$O$_{6h}$ + 1)", "Predicted log$_{10}$(N$_2$O$_{6h}$ + 1)", nx=4, ny=4)


def p_bars(fig, spec):
    ax = fig.add_subplot(spec); ms = ["EG-MoE", "LightGBM", "ExtraTrees", "Ridge", "ClimMean"]
    v1 = [r2_score(y, P[m]) for m in ms]; v2 = [r2_score(f(y), f(P[m])) for m in ms]; x = np.arange(len(ms))
    ax.bar(x - 0.19, v1, 0.38, color=C["blue"], edgecolor=INK, lw=0.6, label="R²")
    ax.bar(x + 0.19, v2, 0.38, color=C["sky"], edgecolor=INK, lw=0.6, label="R² (log scale)")
    ax.axhline(0, color=INK, lw=1); ax.set_xticks(x); ax.set_xticklabels([lab(m) for m in ms], rotation=30, ha="right")
    style_axis(ax, None, "Test R², 2022", ny=5); ax.legend(loc="lower left")


def p_roc(fig, spec):
    ax = fig.add_subplot(spec)
    for key, c, nm in [("roc_gate", C["red"], f"With N$_2$O history ({D['auc_gate']:.3f})"), ("roc_ops", C["teal"], f"Operating data only ({D['auc_ops']:.3f})")]:
        fpr, tpr, _ = D[key]; ax.plot(fpr, tpr, color=c, lw=2.2, label=nm)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--"); style_axis(ax, "False-positive rate", "True-positive rate", nx=3, ny=3)
    ax.legend(loc="lower right", title="Episode ROC-AUC", title_fontsize=11)


def p_ts(fig, spec):
    ax = fig.add_subplot(spec); t = pd.to_datetime(D["time"]); m = (D["street"] == 1) & (t < "2022-02-01")
    o = np.argsort(t[m]); tt = t[m][o]
    ax.plot(tt, y[m][o], color=INK, lw=1.6, label="Observed")
    ax.plot(tt, P["LightGBM"][m][o], color=C["teal"], lw=1.4, label="Operating data only (LightGBM)")
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 8, 15, 22, 29])); ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    style_axis(ax, "January 2022 (street 1)", "N$_2$O$_{6h}$ (gN h$^{-1}$)", ny=5); bold_ticks(ax); ax.legend(loc="upper right")


panels = {"a": p_scatter, "b": p_bars, "c": p_roc, "d": p_ts}
sizes = {"a": (5.4, 5.0), "b": (5.4, 5.0), "c": (5.4, 5.0), "d": (13, 4.2)}


def layout(fig):
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.85])
    return {"a": gs[0, 0], "b": gs[0, 1], "c": gs[0, 2], "d": gs[1, :]}


ALIGN = {"top": [["a", "b", "c"]], "bottom": [["a", "b", "c"]], "left": [["a", "d"]], "right": [["c", "d"]]}

if __name__ == "__main__":
    save("FigS16_operating_data_only", OUTDIR, layout, panels, sizes, figsize=(17, 11), engine="subfig", align=ALIGN)
