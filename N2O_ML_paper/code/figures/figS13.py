import sys, os, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from figstyle import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
configure()
ROOT = os.path.join(HERE, "..", "analysis"); OUTDIR = os.path.join(HERE, "..", "..", "figures", "supplementary", "FigS13_rolling_retraining")
R = pickle.load(open(os.path.join(ROOT, "figdata", "si", "S11.pkl"), "rb"))
st = pickle.load(open(os.path.join(ROOT, "pred_H6_chrono_h1_2022.pkl"), "rb"))
mse = lambda y, p: mean_squared_error(y, np.clip(p, 0, None))
rows = []
tm = pd.to_datetime(st["time"])
for r in R:
    m = tm.month.isin([r["month"], r["month"] + 1]); ys = st["y"][m]
    rows.append(dict(month=r["month"], roll_eg=mse(r["y"], r["EGMoE"]) ** .5, roll_lg=mse(r["y"], r["LightGBM"]) ** .5, pers=mse(r["y"], r["Persistence"]) ** .5,
                     stat_eg=mse(ys, st["preds"]["EG-MoE"][m]) ** .5, stat_lg=mse(ys, st["preds"]["LightGBM"][m]) ** .5,
                     sk_roll=1 - mse(r["y"], r["EGMoE"]) / mse(r["y"], r["Persistence"]), sk_stat=1 - mse(ys, st["preds"]["EG-MoE"][m]) / mse(ys, st["preds"]["Persistence"][m]),
                     sk_lg=1 - mse(r["y"], r["LightGBM"]) / mse(r["y"], r["Persistence"])))
T = pd.DataFrame(rows)
MN = {1: "Jan–Feb", 3: "Mar–Apr", 5: "May–Jun", 7: "Jul–Aug", 9: "Sep–Oct", 11: "Nov–Dec"}


def p_rmse(fig, spec):
    ax = fig.add_subplot(spec); x = T.month.values
    for col, c, ls, lb in [("roll_eg", C["red"], "-", "EG-MoE, retrained"), ("stat_eg", C["red"], "--", "EG-MoE, static (V1)"),
                           ("roll_lg", C["blue"], "-", "LightGBM, retrained"), ("stat_lg", C["blue"], "--", "LightGBM, static (V1)"), ("pers", C["gold"], ":", "Persistence")]:
        ax.plot(x, T[col], color=c, ls=ls, lw=2.0, marker="o", ms=5, mfc="white", label=lb)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([MN[i] for i in x], rotation=30, ha="right")
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ax.yaxis.set_major_locator(FixedLocator([5, 10, 20, 50, 100])); ax.yaxis.set_major_formatter(FixedFormatter(["5", "10", "20", "50", "100"])); ax.yaxis.set_minor_locator(NullLocator()); ax.set_ylim(5, 130)
    style_axis(ax, "Forecast window, 2022", "RMSE (gN h$^{-1}$)"); bold_ticks(ax); ax.legend(loc="upper right", fontsize=11)


def p_skill(fig, spec):
    ax = fig.add_subplot(spec); x = T.month.values; w = 0.27
    for k, (col, c, lb) in enumerate([("sk_roll", C["red"], "EG-MoE, retrained"), ("sk_stat", C["pink"], "EG-MoE, static"), ("sk_lg", C["blue"], "LightGBM, retrained")]):
        ax.bar(x + (k - 1) * w * 2, T[col], w * 2, color=c, edgecolor=INK, lw=0.5, label=lb)
    ax.axhline(0, color=INK, lw=1); ax.set_xticks(x); ax.set_xticklabels([MN[i] for i in x], rotation=30, ha="right")
    style_axis(ax, "Forecast window, 2022", "MSE skill vs persistence", ny=5); ax.set_ylim(top=0.85); ax.legend(loc="upper right", fontsize=11)


panels = {"a": p_rmse, "b": p_skill}
sizes = {"a": (7.2, 5.0), "b": (7.2, 5.0)}


def layout(fig):
    gs = fig.add_gridspec(1, 2); return {"a": gs[0], "b": gs[1]}


ALIGN = {"top": [["a", "b"]], "bottom": [["a", "b"]]}

if __name__ == "__main__":
    T.round(4).to_csv(os.path.join(ROOT, "tables", "FigS13_bimonthly_rmse.csv"), index=False)
    save("FigS13_rolling_retraining", OUTDIR, layout, panels, sizes, figsize=(15, 5.8), engine="subfig", align=ALIGN)
