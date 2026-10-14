import sys, os, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from figstyle import *
import matplotlib.pyplot as plt
configure()
ROOT = os.path.join(HERE, "..", "analysis"); OUTDIR = os.path.join(HERE, "..", "..", "figures", "supplementary", "FigS11_learning_curves")
D = pickle.load(open(os.path.join(ROOT, "figdata", "si", "S5.pkl"), "rb"))
MS = ["EG-MoE", "LightGBM", "Extra trees", "Ridge", "Persistence"]
MC = {"EG-MoE": C["red"], "LightGBM": C["blue"], "Extra trees": C["teal"], "Ridge": C["pink"], "Persistence": C["gold"]}


def curve(models):
    def draw(fig, spec):
        ax = fig.add_subplot(spec); fr = sorted(D); n = [D[f]["n"] for f in fr]
        for m in models:
            tr_ = np.array([D[f][m][0] for f in fr]); va = np.array([D[f][m][1] for f in fr])
            ax.plot(n, tr_, color=MC[m], lw=2.0, marker="o", ms=6, label=m)
            ax.plot(n, np.clip(va, -1, None), color=MC[m], lw=2.0, ls="--", marker="s", ms=6, mfc="white")
            for xi, v in zip(n, va):
                if v < -1: ax.text(xi, -0.93, f"{v:.1f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=INK)
        ax.plot([], [], color=INK, lw=2, label="Training"); ax.plot([], [], color=INK, lw=2, ls="--", label="Validation (Jul–Dec 2021)")
        clipped = ax.get_ylim()[0] < -0.5
        if clipped: ax.set_ylim(-1.05, 1.08)
        style_axis(ax, "Training street-hours", "R²", nx=5, ny=5)
        ax.legend(loc="center right" if clipped else "lower right", bbox_to_anchor=(1.0, 0.42) if clipped else None, fontsize=11, ncol=2)
    return draw


A = pd.read_csv(os.path.join(ROOT, "tables", "ml_audit_H6.csv")).sort_values("R2_test")
NM = {"SeasonalNaive": "Seasonal naive", "ClimMean": "Climatology", "DT": "Decision tree", "RF": "Random forest", "ExtraTrees": "Extra trees"}


def p_audit(fig, spec):
    ax = fig.add_subplot(spec); A2 = A[A.model != "ClimMean"].copy(); y = np.arange(len(A2)); raw_min = A2.CV_R2_min.values.copy()
    A2[["CV_R2_min", "CV_R2_mean", "R2_test"]] = A2[["CV_R2_min", "CV_R2_mean", "R2_test"]].clip(lower=-1)
    for yi, (_, r) in zip(y, A2.iterrows()):
        ax.plot([r.CV_R2_min, r.R2_train], [yi, yi], color=MUTED, lw=1.2, zorder=1)
    ax.scatter(A2.R2_train, y, marker="^", s=60, color=C["blue"], zorder=3, label="Train")
    ax.scatter(A2.CV_R2_mean, y, marker="o", s=55, facecolor="white", edgecolor=C["teal"], lw=1.8, zorder=3, label="Forward CV, mean")
    ax.scatter(A2.CV_R2_min, y, marker="|", s=120, color=C["teal"], lw=2.2, zorder=3, label="Forward CV, worst fold")
    ax.scatter(A2.R2_test, y, marker="D", s=50, color=C["red"], zorder=4, label="Test 2022")
    ax.set_yticks(y); ax.set_yticklabels([NM.get(m, m) for m in A2.model]); ax.set_xlim(-1.0, 1.05)
    for t in ax.get_yticklabels():
        if t.get_text() == "EG-MoE": t.set_color(C["red"])
    style_axis(ax, "R² (values below −1 shown at −1)", None, nx=5); ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.68), fontsize=11)
    lo = raw_min
    for yi, v in zip(y, lo):
        if v < -1: ax.text(-0.95, yi + 0.32, f"{v:.1f}", va="center", ha="left", fontsize=9.5, color=C["teal"], fontweight="bold")


panels = {"a": curve(["EG-MoE", "LightGBM", "Persistence"]), "b": curve(["Extra trees", "Ridge", "Persistence"]), "c": p_audit}
sizes = {"a": (6.6, 5.0), "b": (6.6, 5.0), "c": (6.8, 8.6)}


def layout(fig):
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.05]); return {"a": gs[0, 0], "b": gs[1, 0], "c": gs[:, 1]}


ALIGN = {"top": [["a", "c"]], "bottom": [["b", "c"]], "left": [["a", "b"]], "right": [["a", "b"]]}

if __name__ == "__main__":
    save("FigS11_learning_curves", OUTDIR, layout, panels, sizes, figsize=(15, 11), engine="subfig", align=ALIGN)
