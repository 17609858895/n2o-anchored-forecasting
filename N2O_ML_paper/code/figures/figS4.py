import sys, os, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from figstyle import *
import matplotlib.pyplot as plt
configure()
ROOT = os.path.join(HERE, "..", "analysis"); OUTDIR = os.path.join(HERE, "..", "..", "figures", "supplementary", "FigS04_hyperparameter_tuning")
D = pickle.load(open(os.path.join(ROOT, "figdata", "si", "S4.pkl"), "rb"))
MC = {"LightGBM": C["blue"], "XGBoost": C["sky"], "CatBoost": C["indigo"], "Extra trees": C["teal"]}


def p_trace(fig, spec):
    ax = fig.add_subplot(spec)
    for m, d in D.items():
        v = np.array(d["values"]); best = np.minimum.accumulate(v)
        ax.scatter(np.arange(1, len(v) + 1), v, s=18, color=MC[m], alpha=0.45, lw=0)
        ax.plot(np.arange(1, len(v) + 1), best, color=MC[m], lw=2.2, label=m)
        ax.axhline(d["cv_default"], color=MC[m], lw=1.1, ls=":")
    style_axis(ax, "Trial", "Forward-CV RMSE (gN h$^{-1}$)", nx=5, ny=5); ax.legend(loc="upper right")


def p_imp(fig, spec):
    ax = fig.add_subplot(spec); imp = pd.Series(D["LightGBM"]["importance"]).sort_values()
    ax.barh(range(len(imp)), imp.values, color=C["blue"], edgecolor=INK, lw=0.6, height=0.65)
    ax.set_yticks(range(len(imp))); ax.set_yticklabels(imp.index); style_axis(ax, "Importance (fANOVA, LightGBM)", None, nx=4)


def p_test(fig, spec):
    ax = fig.add_subplot(spec); ms = list(D); x = np.arange(len(ms))
    ax.bar(x - 0.19, [D[m]["test_default"]["R2"] for m in ms], 0.38, color=C["lilac"], edgecolor=INK, lw=0.6, label="Default setting")
    ax.bar(x + 0.19, [D[m]["test_tuned"]["R2"] for m in ms], 0.38, color=[MC[m] for m in ms], edgecolor=INK, lw=0.6, label="Tuned")
    for i, m in enumerate(ms):
        for dx, k in [(-0.19, "test_default"), (0.19, "test_tuned")]:
            ax.text(i + dx, D[m][k]["R2"] + 0.005, f"{D[m][k]['R2']:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold", rotation=90)
    lo = min(min(D[m]["test_default"]["R2"], D[m]["test_tuned"]["R2"]) for m in ms)
    ax.set_ylim(0, 1.08); ax.set_xticks(x); ax.set_xticklabels(ms, rotation=20, ha="right")
    style_axis(ax, None, "Test R², 2022", ny=4); ax.legend(loc="upper right", ncol=2)


panels = {"a": p_trace, "b": p_imp, "c": p_test}
sizes = {"a": (6.6, 5.0), "b": (5.4, 5.0), "c": (5.6, 5.0)}


def layout(fig):
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1]); return {"a": gs[0], "b": gs[1], "c": gs[2]}


ALIGN = {"top": [["a", "b", "c"]], "bottom": [["a", "b", "c"]]}

if __name__ == "__main__":
    save("FigS04_hyperparameter_tuning", OUTDIR, layout, panels, sizes, figsize=(18, 5.8), engine="subfig", align=ALIGN)
