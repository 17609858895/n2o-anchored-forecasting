import sys, pickle, numpy as np, pandas as pd
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import *
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
configure()
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "analysis"); OUTDIR = os.path.join(HERE, "..", "..", "figures", "main", "Fig4_horizon_extrapolation_transfer")
D = pickle.load(open(f"{ROOT}/figdata/fig2to6.pkl", "rb"))["fig5"]
HS = [1, 3, 6, 12, 24]
MC = {"EG-MoE": MODEL_COLOR["EG-MoE"], "LightGBM": MODEL_COLOR["LightGBM"], "[expert] resid": MODEL_COLOR["Residual expert"],
      "SeasonalNaive": MODEL_COLOR["Seasonal naive"], "Persistence": MODEL_COLOR["Persistence"], "ExtraTrees": MODEL_COLOR["Extra trees"], "GRU-A": MODEL_COLOR["GRU-A"]}


def p_skill(fig, spec):
    ax = fig.add_subplot(spec); x = np.arange(len(HS))
    for m, mk in [("EG-MoE", "o"), ("GRU-A", "v"), ("[expert] resid", "^"), ("LightGBM", "s"), ("SeasonalNaive", "D")]:
        s = [D["hor"][h][m]["skill"] for h in HS]; lo = [D["hor"][h][m]["lo"] for h in HS]; hi = [D["hor"][h][m]["hi"] for h in HS]
        ax.fill_between(x, lo, hi, color=MC[m], alpha=0.18, lw=0)
        ax.plot(x, s, color=MC[m], marker=mk, lw=2, ms=6.5, mfc="white", mew=1.6, label=lab(m))
    ax.axhline(0, color=MODEL_COLOR["Persistence"], lw=1.6, ls="--", label="Persistence")
    ax.set_xticks(x); ax.set_xticklabels(HS); ax.set_ylim(-1.0, 0.55)
    style_axis(ax, "Forecast window H (h)", "MSE skill vs persistence", tick=12.5, label=15.5, ny=5)
    ax.legend(loc="lower left", fontsize=11, ncol=2, columnspacing=0.8, handlelength=1.6)


def p_heat(fig, spec):
    gs = spec.subgridspec(1, 2, width_ratios=[1, 0.05]); ax = fig.add_subplot(gs[0]); cax = fig.add_subplot(gs[1])
    ms = ["EG-MoE", "GRU-A", "[expert] resid", "LightGBM", "ExtraTrees", "CatBoost", "XGBoost", "Persistence", "SeasonalNaive", "Ridge"]
    R = np.array([[D["hor"][h][m]["R2"] for h in HS] for m in ms])
    im = ax.imshow(R, cmap=SEQ, vmin=0.3, vmax=1.0, aspect="auto")
    for i in range(len(ms)):
        for j in range(len(HS)):
            best = R[i, j] == R[:, j].max()
            ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=11, fontweight="bold" if best else "normal",
                    color="white" if R[i, j] > 0.85 else INK)
    ax.set_xticks(range(len(HS))); ax.set_xticklabels(HS); ax.set_yticks(range(len(ms))); ax.set_yticklabels([lab(m) for m in ms])
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=3); bold_ticks(ax, 12); ax.get_yticklabels()[0].set_color(C["red"])
    ax.set_xlabel("Forecast window H (h)", fontsize=13, fontweight="bold")
    cb = fig.colorbar(im, cax=cax); cb.set_label("R²", fontsize=12, fontweight="bold"); bold_ticks(cb.ax, 10.5)


def p_scen(fig, spec):
    gs = spec.subgridspec(1, 2); axs = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    ms = ["EG-MoE", "GRU-A", "[expert] resid", "LightGBM", "Persistence"]; sc = ["V1", "V2", "V3"]; w = 0.16
    for k, met in enumerate(["R2", "RMSE"]):
        ax = axs[k]
        for i, m in enumerate(ms):
            v = [D["scen"][s][m][met] for s in sc]
            ax.bar(np.arange(3) + (i - 2) * w, v, w, color=MC[m], edgecolor=INK, lw=0.7, label=lab(m))
        ax.set_xticks(range(3)); ax.set_xticklabels(["2022", "2021\n(fwd)", "Cross-\nstreet"])
        style_axis(ax, None, "R²" if met == "R2" else "RMSE (gN h$^{-1}$)", tick=12, label=15, ny=5)
        if met == "R2": ax.set_ylim(0, 1.0)
        else: ax.set_ylim(0, 100)
    axs[1].legend(loc="upper left", fontsize=11, ncol=2, handlelength=1.0, columnspacing=0.8, borderaxespad=0.2)


def p_scatter(fig, spec):
    gs = spec.subgridspec(1, 2, width_ratios=[1, 0.4])
    ax = fig.add_subplot(gs[0]); ar = fig.add_subplot(gs[1])
    f = lambda v: np.log10(np.clip(v, 0, None) + 1)
    yt, pt = D["tr_y"], D["tr_p"]; ye, pe = D["te"]["y"], np.clip(D["te"]["preds"]["EG-MoE"], 0, None)
    rng = np.random.default_rng(0); it = rng.choice(len(yt), 2500, replace=False)
    ax.scatter(f(ye), f(pe), marker="o", s=14, facecolor="none", edgecolor=C["red"], lw=0.6, alpha=0.55, label="Test 2022", rasterized=True)
    ax.scatter(f(yt[it]), f(pt[it]), marker="^", s=13, color=C["blue"], alpha=0.75, lw=0, label="Train 2020–21", rasterized=True)
    lim = [0, 3.15]; ax.plot(lim, lim, color=INK, lw=1.1, ls="--"); ax.set_xlim(lim); ax.set_ylim(lim)
    style_axis(ax, "Observed log$_{10}$(N$_2$O$_{6h}$ + 1)", "Predicted log$_{10}$(N$_2$O$_{6h}$ + 1)", tick=12.5, label=15, nx=4, ny=4)
    txt = (f"Train R² = {r2_score(yt, pt):.3f}\nTest R² = {r2_score(ye, pe):.3f}\nTest RMSE = {mean_squared_error(ye, pe) ** .5:.1f} gN h$^{{-1}}$")
    ax.text(0.04, 0.97, txt, transform=ax.transAxes, va="top", fontsize=11.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=MUTED, lw=0.6, alpha=0.9))
    ax.legend(loc="lower right", fontsize=11, markerscale=1.8, frameon=True, framealpha=0.95, edgecolor="white")
    res = f(pe) - f(ye)
    ar.hist(res, bins=np.linspace(-1.2, 1.2, 41), color=C["pink"], edgecolor=INK, lw=0.3)
    ar.axvline(0, color=INK, lw=1)
    style_axis(ar, "Residual (log$_{10}$)", "Hours", tick=11.5, label=14, nx=3, ny=3)


def p_jan(fig, spec):
    ax = fig.add_subplot(spec)
    B = D["te"]; t = pd.to_datetime(B["time"]); s = B["street"] == 1; m = s & (t < pd.Timestamp("2022-02-01"))
    tt = t[m]; o = np.argsort(tt.values); tt = tt[o]
    y = B["y"][m][o]; pe = np.clip(B["preds"]["EG-MoE"], 0, None)[m][o]
    lo = D["conf"][0.1]["lo"][m][o]; hi = D["conf"][0.1]["hi"][m][o]
    ax.fill_between(tt, lo, hi, color=C["pink"], alpha=0.55, lw=0, label="EG-MoE 90% interval")
    ax.plot(tt, y, color=INK, lw=1.6, label="Observed")
    ax.plot(tt, np.clip(B["preds"]["Persistence"], 0, None)[m][o], color=MODEL_COLOR["Persistence"], lw=1.1, ls="--", label="Persistence")
    ax.plot(tt, np.clip(B["preds"]["LightGBM"], 0, None)[m][o], color=MODEL_COLOR["LightGBM"], lw=1.1, label="LightGBM")
    ax.plot(tt, pe, color=MODEL_COLOR["EG-MoE"], lw=1.6, label="EG-MoE")
    ax.axhline(B["tau"], color=MUTED, lw=1, ls=":")
    pw = B["p"][m][o] >= 0.5; top = max(hi.max(), y.max()) * 1.05
    ax.scatter(tt[pw], np.full(pw.sum(), top), marker="|", s=40, color=C["red"], lw=1.2)
    ax.text(tt[0], top * 1.03, "Episode warnings (p ≥ 0.5)", fontsize=11, fontweight="bold", color=C["red"], va="bottom")
    ax.set_ylim(0, top * 1.12)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d Jan")); ax.xaxis.set_major_locator(plt.matplotlib.dates.DayLocator([1, 8, 15, 22, 29]))
    style_axis(ax, "January 2022 (street 1)", "N$_2$O$_{6h}$ (gN h$^{-1}$)", tick=12.5, label=15, ny=5)
    ax.legend(loc="center right", fontsize=11, ncol=1, bbox_to_anchor=(1.0, 0.5))
    ax.set_xlim(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-01-31 23:00"))


def p_extrap(fig, spec):
    ax = fig.add_subplot(spec)
    v = D["v2"]; y = v["y"]
    for mname, mk, col, lbl in [("LightGBM", "s", MODEL_COLOR["LightGBM"], "LightGBM"), ("EG-MoE", "o", MODEL_COLOR["EG-MoE"], "EG-MoE")]:
        p = np.clip(v["preds"][mname], 0, None)
        ax.scatter(y, p, s=10 if mname == "LightGBM" else 12, marker=mk, facecolor=col if mname == "LightGBM" else "none", edgecolor=col,
                   lw=0.6, alpha=0.5, label=f"{lbl} (R² = {r2_score(y, p):.2f})", rasterized=True)
    mx = y.max() * 1.05; ax.plot([0, mx], [0, mx], color=INK, lw=1, ls="--")
    ax.axhline(D["v2_trainmax"], color=MUTED, lw=1.3, ls="-.")
    ax.text(mx * 0.98, D["v2_trainmax"] * 1.04, f"2020 training max = {D['v2_trainmax']:.0f}", ha="right", va="bottom", fontsize=11, fontweight="bold", color=MUTED)
    ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    style_axis(ax, "Observed N$_2$O$_{6h}$, 2021 (gN h$^{-1}$)", "Predicted (gN h$^{-1}$)", tick=12.5, label=15, nx=4, ny=4)
    ax.legend(loc="upper left", fontsize=11, markerscale=2)


def p_monthly(fig, spec):
    ax = fig.add_subplot(spec)
    Mo = D["monthly"]; ms = ["EG-MoE", "GRU-A", "LightGBM", "ExtraTrees", "XGBoost", "CatBoost", "Persistence"]
    cols = [MODEL_COLOR["EG-MoE"], MODEL_COLOR["GRU-A"], MODEL_COLOR["LightGBM"], MODEL_COLOR["Extra trees"], FAMILY_COLOR["Boosting"], FAMILY_COLOR["Boosting"], MODEL_COLOR["Persistence"]]
    rng = np.random.default_rng(1)
    for i, (m, c) in enumerate(zip(ms, cols)):
        v = Mo[m].values
        bp = ax.boxplot([v], positions=[i], widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.6))
        bp["boxes"][0].set_facecolor(c); bp["boxes"][0].set_alpha(0.55); bp["boxes"][0].set_edgecolor(INK)
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=14, color=c, edgecolor=INK, lw=0.4, zorder=3)
        ax.text(i, 1.06, f"{(v < 0).sum()}", ha="center", fontsize=11.5, fontweight="bold", color=C["red"] if (v < 0).sum() else INK)
    ax.axhline(0, color=INK, lw=1, ls="--"); ax.set_ylim(-0.8, 1.13)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels([lab(m) for m in ms], rotation=30, ha="right")
    style_axis(ax, None, "R² per street-month (n = 22)", tick=12, label=15, ny=5)
    ax.set_xlim(-0.9, len(ms) - 0.4)


panels = {"a": p_skill, "b": p_heat, "c": p_scen, "d": p_scatter, "e": p_jan, "f": p_extrap, "g": p_monthly}
sizes = {"a": (5.6, 4.6), "b": (5.4, 4.8), "c": (6.6, 4.4), "d": (5.8, 4.8), "e": (11, 4.2), "f": (5.4, 4.8), "g": (5.6, 4.8)}


def layout(fig):
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1])
    return {"a": gs[0, 0:2], "b": gs[0, 2:4], "c": gs[0, 4:6], "d": gs[1, 0:2], "e": gs[1, 2:6], "f": gs[2, 0:3], "g": gs[2, 3:6]}


ALIGN = {"top": [["a", "b", "c"], ["d", "e"], ["f", "g"]], "bottom": [["a", "b", "c"], ["d", "e"], ["f", "g"]], "left": [["a", "d", "f"]], "right": [["c", "e", "g"]]}

if __name__ == "__main__":
    save("Fig4_horizon_extrapolation_transfer", OUTDIR, layout, panels, sizes, figsize=(18, 15), engine="subfig", align=ALIGN)
    print({a: (D["conf"][a]["cover"].mean(), D["stat"][a]["cover"].mean()) for a in D["conf"]})
