import sys, pickle, numpy as np, pandas as pd
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
configure()
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "analysis"); OUTDIR = os.path.join(HERE, "..", "..", "figures", "main", "Fig3_model_benchmark")
D = pickle.load(open(f"{ROOT}/figdata/fig2to6.pkl", "rb"))["fig4"]
M = D["M"].copy(); M["fam"] = M.model.map(FAMILY); M["name"] = M.model.map(lab)
M = M.sort_values("RMSE").reset_index(drop=True)
stars = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
FAMS = ["Proposed", "Neural, anchored", "Boosting", "Tree/bagging", "Neural", "Kernel/NN", "Linear", "Naive"]


def fam_legend(ax, **kw):
    h = [Line2D([], [], marker="o", ls="", ms=8, color=FAMILY_COLOR[f], mfc=FAMILY_COLOR[f], mec=INK, label=f) for f in FAMS]
    ax.legend(handles=h, **kw)


def p_heat(fig, spec):
    gs = spec.subgridspec(1, 2, width_ratios=[0.05, 1], wspace=0.01)
    a0 = fig.add_subplot(gs[0]); ax = fig.add_subplot(gs[1], sharey=a0)
    cols = [("R2", "R²", 1), ("R2_jan", "R² Jan", 1), ("R2_nonjan", "R² non-Jan", 1), ("RMSE", "RMSE", -1), ("MAE", "MAE", -1),
            ("R2_log", "R² log", 1), ("F1", "F1", 1), ("absCum", "|CumErr| %", -1), ("Skill", "Skill", 1)]
    V = np.zeros((len(M), len(cols)))
    for j, (c, _, d) in enumerate(cols):
        v = M[c].values.astype(float); r = pd.Series(d * v).rank(pct=True).values; V[:, j] = r
    ax.imshow(V, cmap=SEQ, vmin=0, vmax=1.15, aspect="auto")
    for i in range(len(M)):
        for j, (c, _, _) in enumerate(cols):
            v = M[c].iloc[i]; txt = f"{v:.1f}" if c in ("RMSE", "MAE", "absCum") else f"{v:.2f}"
            ax.text(j, i, txt.replace("-", "−"), ha="center", va="center", fontsize=10, fontweight="bold" if M.model.iloc[i] == "EG-MoE" else "normal",
                    color="white" if V[i, j] > 0.8 else INK)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[1] for c in cols], rotation=40, ha="right")
    ax.xaxis.tick_top(); ax.set_xticklabels([c[1] for c in cols], rotation=40, ha="left")
    plt.setp(ax.get_yticklabels(), visible=False); ax.tick_params(left=False, length=3)
    for s in ax.spines.values(): s.set_visible(False)
    bold_ticks(ax, 12)
    a0.imshow(np.arange(len(M))[:, None], cmap=plt.matplotlib.colors.ListedColormap([FAMILY_COLOR[f] for f in M.fam]), aspect="auto")
    a0.set_xticks([]); a0.set_yticks(range(len(M))); a0.set_yticklabels(M.name)
    for s in a0.spines.values(): s.set_visible(False)
    a0.tick_params(length=5, width=1.4); bold_ticks(a0, 12)
    [t.set_color(C["red"]) for t in a0.get_yticklabels() if t.get_text() == "EG-MoE"]


def p_skill(fig, spec):
    ax = fig.add_subplot(spec)
    d = M[M.model != "Persistence"].sort_values("Skill")
    yv = np.arange(len(d)); xs = np.clip(d.Skill.values, -3.4, None)
    ax.hlines(yv, 0, xs, color=[FAMILY_COLOR[f] for f in d.fam], lw=2.2)
    ax.scatter(xs, yv, s=55, color=[FAMILY_COLOR[f] for f in d.fam], edgecolor=INK, zorder=3, lw=0.8)
    for i, (x, sk, p) in enumerate(zip(xs, d.Skill, d.DM_p)):
        lbl = stars(p) + (f" ({sk:.1f})" if sk < -3.4 else "")
        ax.text(x + (0.06 if x >= 0 else -0.06), i, lbl, va="center", ha="left" if x >= 0 else "right", fontsize=10.5, fontweight="bold")
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_yticks(yv); ax.set_yticklabels(d.name); ax.set_xlim(-3.9, 0.55)
    style_axis(ax, "MSE skill vs persistence", None, tick=12, label=15.5, nx=5)
    [t.set_color(C["red"]) for t in ax.get_yticklabels() if t.get_text() == "EG-MoE"]
    fam_legend(ax, loc="upper left", fontsize=11, ncol=1, handletextpad=0.2, labelspacing=0.35, title="Model family", title_fontsize=11)


def p_taylor(fig, spec):
    ax = fig.add_subplot(spec, projection="polar")
    ax.set_thetamin(0); ax.set_thetamax(90)
    ysd = D["ysd"]; smax = 1.4
    rs = np.linspace(0, smax, 100)
    for r0 in [0.5, 1.0]:
        th = np.linspace(0, np.pi / 2, 100)
        # centred-RMS contours around reference point (1, 0)
    for crmse in [0.25, 0.5, 0.75, 1.0]:
        th = np.linspace(0, np.pi / 2, 300); x0 = 1.0
        rr = []
        for t in th:
            # solve r^2 - 2 r x0 cos t + x0^2 = crmse^2
            b = -2 * x0 * np.cos(t); c_ = x0 ** 2 - crmse ** 2; disc = b * b - 4 * c_
            rr.append((-b + np.sqrt(disc)) / 2 if disc >= 0 else np.nan)
        ax.plot(th, rr, color=MUTED, lw=0.7, ls=":")
    ax.plot(np.linspace(0, np.pi / 2, 100), np.ones(100), color=INK, lw=0.9, ls="--")
    for _, r in M[M.model != "ClimMean"].iterrows():
        th = np.arccos(np.clip(r["corr"], 0, 1)); rad = r["sd"] / ysd
        mk = "*" if r.model == "EG-MoE" else "o"; ms = 260 if r.model == "EG-MoE" else 55
        ax.scatter(th, rad, marker=mk, s=ms, color=FAMILY_COLOR[r.fam], edgecolor=INK, lw=0.8, zorder=4)
    ax.scatter(0, 1, marker="D", s=50, color=INK, zorder=5)
    ax.set_rmax(smax)
    cors = [0, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]
    ax.set_thetagrids(np.degrees(np.arccos(cors)), [str(c) for c in cors])
    ax.set_rgrids([0.4, 0.8, 1.2], ["0.4", "0.8", "1.2"])
    ax.tick_params(labelsize=11); bold_ticks(ax, 11)
    ax.text(np.radians(45), smax * 1.22, "Correlation", fontsize=12.5, fontweight="bold", rotation=-45, ha="center", va="center")
    ax.annotate("Normalised SD", xy=(0, smax / 2), xycoords="data", xytext=(0, -26), textcoords="offset points", ha="center", va="top", fontsize=12.5, fontweight="bold")
    ax.grid(color="#D9DCE1", lw=0.6)


def p_bubble(fig, spec):
    ax = fig.add_subplot(spec)
    d = M[M.model != "ClimMean"]
    ax.scatter(d.absCum, d.R2, s=(d.F1 * 22) ** 2 / 2, color=[FAMILY_COLOR[f] for f in d.fam], edgecolor=INK, lw=0.8, alpha=0.9)
    offs = {"GRU-A": (2, 16), "EG-MoE": (8, 6), "Persistence": (6, -14), "LightGBM": (10, 6), "Transformer": (6, 4), "GRU": (6, 4), "KNN": (-6, 6), "Ridge": (6, 4), "SVR": (6, -12)}
    for _, r in d.iterrows():
        if r.model in offs:
            ax.annotate(r["name"], (r.absCum, r.R2), xytext=offs[r.model], textcoords="offset points", fontsize=11, fontweight="bold",
                        color=C["red"] if r.model == "EG-MoE" else INK, ha="right" if offs[r.model][0] < 0 else "left",
                        arrowprops=dict(arrowstyle="-", lw=0.6, color=MUTED) if r.model in ("LightGBM", "Persistence") else None)
    ax.axvspan(0, 8, ymin=0, ymax=1, color=C["pink"], alpha=0.18, lw=0); ax.set_ylim(top=0.94)
    ax.axhline(0.8, color=MUTED, lw=0.8, ls=":")
    style_axis(ax, "|Cumulative emission error| (%)", "R²", tick=12.5, label=15.5, nx=5, ny=5)
    for f1 in [0.6, 0.7, 0.8]:
        ax.scatter([], [], s=(f1 * 22) ** 2 / 2, color="white", edgecolor=INK, label=f"F1 = {f1}")
    ax.legend(loc="lower left", fontsize=11, labelspacing=1.1, borderpad=0.8)


def p_topsis(fig, spec):
    ax = fig.add_subplot(spec)
    d = M.dropna(subset=["TOPSIS"]).sort_values("TOPSIS", ascending=False).head(10)[::-1]
    ax.barh(range(len(d)), d.TOPSIS, color=[FAMILY_COLOR[f] for f in d.fam], edgecolor=INK, lw=0.8, height=0.65)
    for i, v in enumerate(d.TOPSIS): ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d.name); ax.set_xlim(0, 1.08)
    style_axis(ax, "Entropy-weighted TOPSIS score", None, tick=12, label=15.5, nx=5)
    [t.set_color(C["red"]) for t in ax.get_yticklabels() if t.get_text() == "EG-MoE"]


PAIRS = [("LSTM", "LSTM-A"), ("GRU", "GRU-A"), ("TCN", "TCN-A"), ("Transformer", "Transformer-A"), ("LightGBM", "Residual expert")]


def _pair_metrics():
    """R2 over the test year and outside January for unanchored models and their anchored counterparts."""
    from sklearn.metrics import r2_score
    b = pickle.load(open(f"{ROOT}/pred_H6_chrono_h1_2022.pkl", "rb")); y = b["y"]; nj = np.asarray(pd.to_datetime(b["time"]).month != 1)
    pr = np.clip(b["preds"]["[expert] resid"], 0, None)
    r = M.set_index("model")[["R2", "R2_nonjan"]].to_dict("index")
    r["Residual expert"] = dict(R2=r2_score(y, pr), R2_nonjan=r2_score(y[nj], pr[nj]))
    return r


def p_pair(fig, spec):
    R = _pair_metrics(); gs = spec.subgridspec(1, 2, wspace=0.08)
    pers = M.set_index("model").loc["Persistence"]; eg = M.set_index("model").loc["EG-MoE"]
    for j, (col, lab_) in enumerate([("R2", "R² (test year)"), ("R2_nonjan", "R² (months other than January)")]):
        ax = fig.add_subplot(gs[j]); yy = np.arange(len(PAIRS))[::-1]
        for yv, (u, a) in zip(yy, PAIRS):
            xu, xa = R[u][col], R[a][col]
            ax.plot([xu, xa], [yv, yv], color="#9a9a9a", lw=2.2, zorder=1)
            ax.scatter(xu, yv, s=90, color=FAMILY_COLOR[FAMILY.get(u, "Neural")], edgecolor=INK, lw=0.8, zorder=3)
            ax.scatter(xa, yv, s=90, color=C["coral"] if a == "Residual expert" else FAMILY_COLOR["Neural, anchored"], edgecolor=INK, lw=0.8, zorder=3, marker="D")
        ax.axvline(pers[col], color=C["gold"], ls="--", lw=1.4); ax.axvline(eg[col], color=C["red"], ls=":", lw=1.6)
        ax.set_yticks(yy); ax.set_yticklabels([f"{u} → {a.replace("Residual expert", "residual expert")}" for u, a in PAIRS] if j == 0 else [])
        lo = min(min(R[u][col], R[a][col]) for u, a in PAIRS); ax.set_xlim(min(lo, pers[col]) - 0.05, 0.95 if col == "R2" else 0.78)
        style_axis(ax, lab_, None, tick=12, label=14.5, nx=4)
    h = [Line2D([], [], marker="o", ls="", ms=9, color=FAMILY_COLOR["Neural"], mec=INK, label="Unanchored"),
         Line2D([], [], marker="D", ls="", ms=8, color=FAMILY_COLOR["Neural, anchored"], mec=INK, label="Anchored"),
         Line2D([], [], color=C["gold"], ls="--", lw=1.4, label="Persistence"), Line2D([], [], color=C["red"], ls=":", lw=1.6, label="EG-MoE")]
    ax.legend(handles=h, loc="upper right", fontsize=11, frameon=False)


def p_boot(fig, spec):
    ax = fig.add_subplot(spec)
    top = M.head(10).model.tolist()[::-1]
    bp = ax.boxplot([D["boot"][m] for m in top], vert=False, widths=0.6, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.6))
    for i, b in enumerate(bp["boxes"]): b.set_facecolor(FAMILY_COLOR[FAMILY[top[i]]]); b.set_edgecolor(INK)
    for w in bp["whiskers"] + bp["caps"]: w.set_color(INK)
    ax.set_yticks(range(1, len(top) + 1)); ax.set_yticklabels([lab(m) for m in top])
    style_axis(ax, "RMSE, calendar-week bootstrap (gN h$^{-1}$)", None, tick=12, label=15.5, nx=5)
    [t.set_color(C["red"]) for t in ax.get_yticklabels() if t.get_text() == "EG-MoE"]


panels = {"a": p_heat, "b": p_skill, "e": p_pair, "f": p_boot}   # TOPSIS, Taylor diagram and accuracy-bias plot -> Fig. S17   # Taylor diagram and accuracy-bias plot -> Fig. S17
sizes = {"a": (7.6, 8.0), "b": (6.0, 7.4), "c": (5.8, 5.4), "d": (5.8, 5.0), "e": (9.0, 4.8), "f": (6.4, 4.8)}


def layout(fig):
    gs = fig.add_gridspec(2, 6, height_ratios=[1.5, 0.85])
    return {"a": gs[0, 0:3], "b": gs[0, 3:6], "e": gs[1, 0:3], "f": gs[1, 3:6]}


ALIGN = {"top": [["a", "b"], ["e", "f"]], "bottom": [["a", "b"], ["e", "f"]], "left": [["a", "e"]], "right": [["b", "f"]]}

if __name__ == "__main__":
    save("Fig3_model_benchmark", OUTDIR, layout, panels, sizes, figsize=(18, 14), align=ALIGN)
    print(M[["model", "RMSE", "Skill", "DM_p", "TOPSIS"]].round(3).to_string())
