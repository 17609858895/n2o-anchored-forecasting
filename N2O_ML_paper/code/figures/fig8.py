import sys, pickle, numpy as np, pandas as pd
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import *
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import to_rgba
configure()
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "analysis"); OUTDIR = os.path.join(HERE, "..", "..", "figures", "main", "Fig8_causal_levers")
D = pickle.load(open(f"{ROOT}/figdata/dml_fig8.pkl", "rb"))
ATE = pd.DataFrame(D["ate"]); REF = pd.read_csv(f"{ROOT}/dml_pilot_v2.csv")
B = pickle.load(open(f"{ROOT}/pred_H6_chrono_h1_2022.pkl", "rb"))
pct = lambda v: 100 * (np.exp(v) - 1)
TN = {"swing": "Swing-zone aeration", "supern": "Supernatant valve open"}
TC = {"swing": C["blue"], "supern": C["orange"]}


def p_dag(fig, spec):
    ax = fig.add_subplot(spec); ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")
    def box(x, y, w, h, txt, fc, ec=INK, fs=12):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.08,rounding_size=0.18", fc=fc, ec=ec, lw=1.3))
        ax.text(x, y, txt, ha="center", va="center", fontsize=fs, fontweight="bold", color=INK)
    def arr(x1, y1, x2, y2, col=INK, ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=col, lw=1.5, ls=ls, connectionstyle=f"arc3,rad={rad}"))
    box(5, 5.3, 7.8, 1.2, "Confounders at t − 1\nT, Q$_{out}$, NH$_4^+$, NO$_3^-$, Cl$^-$, RAS, pH, calendar, N$_2$O history", to_rgba(C["lilac"], 0.15), fs=11.5)
    box(2.0, 2.9, 3.2, 0.95, "Swing-zone aeration (t)", to_rgba(C["blue"], 0.22))
    box(2.0, 1.5, 3.2, 0.95, "Supernatant valve (t)", to_rgba(C["orange"], 0.22))
    box(8.0, 2.2, 3.0, 1.1, "N$_2$O, next 6 h\n(log scale)", to_rgba(C["red"], 0.18))
    box(5.0, 0.45, 3.4, 0.7, "DO$_2$ (control loop, SI)", "white", ec=MUTED, fs=11)
    arr(2.0, 4.7, 2.0, 3.4); arr(1.1, 4.7, 0.6, 1.95, rad=0.25)
    for y0 in [2.9, 1.5]:
        arr(3.6, y0, 6.5, 2.2 + (0.25 if y0 > 2 else -0.25), col=C["red"])
    arr(7.6, 4.7, 8.0, 2.75)
    arr(6.7, 0.45, 7.4, 1.65, col=MUTED, ls="--")
    ax.text(5.0, 3.35, "ATE / CATE\n(DML)", ha="center", fontsize=11.5, fontweight="bold", color=C["red"])
    ax.text(8.6, 0.95, "Unmeasured: NO$_2^-$, COD", ha="center", fontsize=11, color=MUTED, style="italic")


def p_forest(fig, spec):
    ax = fig.add_subplot(spec); rows = []
    for t in ["swing", "supern"]:
        for e in ["LinearDML-GBM", "LinearDML-RF", "CausalForestDML"]:
            r = ATE[(ATE.treatment == t) & (ATE.estimator == e)].iloc[0]
            lo, hi = (r.lo_block, r.hi_block) if np.isfinite(r.lo_block) else (r.lo_naive, r.hi_naive)
            rows.append((t, e, pct(r.ate), pct(lo), pct(hi), np.isfinite(r.lo_block)))
    yv = np.arange(len(rows))[::-1] + np.repeat([0.6, 0], 3)
    for (t, e, m, lo, hi, blk), y in zip(rows, yv):
        ax.hlines(y, lo, hi, color=TC[t], lw=2.4); ax.scatter(m, y, s=70, marker="o" if blk else "D", color=TC[t], edgecolor=INK, zorder=3)
        ax.text(hi + 2, y, f"{m:+.0f}% [{lo:.0f}, {hi:.0f}]", va="center", fontsize=11, fontweight="bold")
    ax.axvline(0, color=INK, lw=1.1)
    ax.set_yticks(yv); ax.set_yticklabels([e.replace("LinearDML-", "DML-") for _, e, *_ in rows])
    for y, t in [(yv[0], "swing"), (yv[3], "supern")]:
        ax.text(3, y + 0.62, TN[t], ha="left", va="center", fontsize=12, fontweight="bold", color=TC[t])
    ax.set_xlim(-5, 165); ax.set_ylim(yv.min() - 0.6, yv.max() + 1.0)
    style_axis(ax, "Effect on next-6 h N$_2$O (%)", None, tick=12, label=15, nx=5)


def p_refute(fig, spec):
    ax = fig.add_subplot(spec)
    labs_ = [("LinearDML-GBM", "Estimate"), ("refute:placebo", "Block\nplacebo"), ("refute:random_common_cause", "Random\ncommon cause"), ("refute:subset50", "50%\nsubsample")]
    tmap = {"swing": "Swing-zone aeration (on/off)", "supern": "Supernatant valve open (>5 %)"}
    x = np.arange(len(labs_)); w = 0.18
    for k, t in enumerate(["swing", "supern"]):
        v = [REF[(REF.treatment == tmap[t]) & (REF.estimator == e)].pct.iloc[0] for e, _ in labs_]
        xs = x + (k - 0.5) * 2 * w
        ax.vlines(xs, 0, v, color=TC[t], lw=2.4); ax.scatter(xs, v, s=60, color=TC[t], edgecolor=INK, zorder=3, label=TN[t])
        for xx, vv in zip(xs, v): ax.text(xx, vv + (3 if vv >= 0 else -6), f"{abs(vv):.0f}" if abs(vv) < 0.5 else f"{vv:.0f}", ha="center", fontsize=10.5, fontweight="bold")
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in labs_])
    style_axis(ax, None, "Estimated effect (%)", tick=12, label=15, ny=5); ax.set_ylim(-10, 72)
    ax.legend(loc="upper left", fontsize=11, bbox_to_anchor=(0.0, 1.12), ncol=2, columnspacing=0.8)


def p_cate_curve(fig, spec):
    gs = spec.subgridspec(2, 1, height_ratios=[1, 0.25], hspace=0.05)
    ax = fig.add_subplot(gs[0]); ah = fig.add_subplot(gs[1], sharex=ax)
    for t in ["swing", "supern"]:
        c = D["cate_curve"][t]
        ax.fill_between(c["T"], pct(c["lo"]), pct(c["hi"]), color=TC[t], alpha=0.18, lw=0)
        ax.plot(c["T"], pct(c["eff"]), color=TC[t], lw=2.2, label=TN[t])
    ax.axhline(0, color=INK, lw=1)
    style_axis(ax, None, "CATE (%)", tick=12, label=15, ny=5); plt.setp(ax.get_xticklabels(), visible=False)
    ax.legend(loc="upper left", fontsize=11)
    ah.hist(D["cate_curve"]["swing"]["Tobs"], bins=40, color=C["lilac"], edgecolor=INK, lw=0.3)
    style_axis(ah, "Influent T at t − 1 (°C)", "Hours", tick=11, label=13.5, ny=2)


def p_surface(fig, spec):
    ax = fig.add_subplot(spec); cax = ax.inset_axes([1.03, 0.0, 0.045, 0.7])
    s = D["cate_surface"]["supern"]; Z = pct(s["eff"])
    from matplotlib.colors import TwoSlopeNorm
    from scipy.stats import gaussian_kde
    xo0 = s["Xobs"]; kd = gaussian_kde(xo0[np.random.default_rng(1).choice(len(xo0), 4000, replace=False)].T)
    dens = kd(np.vstack([s["T"].ravel(), s["N"].ravel()])).reshape(s["T"].shape)
    Z = np.where(dens > np.percentile(dens, 30), Z, np.nan)
    vmax = 150; Z = np.clip(Z, 0, vmax)
    cs = ax.contourf(s["T"], s["N"], Z, levels=np.linspace(0, vmax, 11), cmap=SEQ_RED)
    ax.contour(s["T"], s["N"], Z, levels=8, colors=INK, linewidths=0.5)
    xo = s["Xobs"]; ii = np.random.default_rng(0).choice(len(xo), 2500, replace=False)
    ax.scatter(xo[ii, 0], xo[ii, 1], s=2, color=INK, alpha=0.25, lw=0, rasterized=True)
    ax.set_xlim(s["T"].min(), s["T"].max()); ax.set_ylim(s["N"].min(), s["N"].max())
    style_axis(ax, "Influent T at t − 1 (°C)", "NH$_4^+$ Z1 at t − 1 (mgN L$^{-1}$)", tick=12, label=14.5, nx=5, ny=5)
    cb = fig.colorbar(cs, cax=cax); cb.set_label("CATE, valve open (%)", fontsize=12, fontweight="bold"); bold_ticks(cb.ax, 10.5)


def scenario():
    Dd = D["D_supern"].copy(); Dd["time"] = pd.to_datetime(Dd["time"])
    Dd = Dd[(Dd.time >= "2022-01-01")]
    pser = pd.DataFrame({"time": pd.to_datetime(B["time"]), "street": B["street"], "p": B["p"]})
    Dd = Dd.merge(pser, on=["time", "street"], how="inner")
    L = pd.read_pickle(f"{ROOT}/long.pkl")[["time", "street", "y"]]
    L["y_next"] = L.groupby("street").y.shift(-1)
    Dd = Dd.merge(L[["time", "street", "y_next"]], on=["time", "street"], how="left").dropna(subset=["y_next"])
    base = Dd.y_next.sum()
    a = ATE[(ATE.treatment == "supern") & (ATE.estimator == "LinearDML-GBM")].iloc[0]
    out = {}
    for name, sel in [("Close valve when warned (p ≥ 0.5)", (Dd.Supern_open == 1) & (Dd.p >= 0.5)),
                      ("Close valve when warned (p ≥ 0.3)", (Dd.Supern_open == 1) & (Dd.p >= 0.3)),
                      ("Close valve at all open hours", Dd.Supern_open == 1)]:
        r = {}
        yv = Dd.loc[sel, "y_next"]
        r["mid"] = (yv * (1 - np.exp(-Dd.loc[sel, "theta"].clip(lower=0)))).sum()            # heterogeneous (CATE) effect
        r["ate"] = (yv * (1 - np.exp(-a.ate))).sum()                                       # constant ATE
        r["lo"] = (yv * (1 - np.exp(-a.lo_block))).sum(); r["hi"] = (yv * (1 - np.exp(-a.hi_block))).sum()
        r["hours"] = int(sel.sum()); out[name] = r
    return out, base


def p_scen(fig, spec):
    ax = fig.add_subplot(spec)
    out, base = scenario(); names = list(out)
    f = lambda g: g / 1000 * 44 / 28 * 273 / 1000  # gN -> t CO2-eq
    v = np.array([f(out[n]["ate"]) for n in names]); lo = np.array([f(out[n]["lo"]) for n in names]); hi = np.array([f(out[n]["hi"]) for n in names])
    vc = np.array([f(out[n]["mid"]) for n in names])
    yv = np.arange(len(names))[::-1]
    ax.barh(yv, v, color=C["orange"], edgecolor=INK, lw=0.8, height=0.5, label="Constant effect (ATE, calendar-week bootstrap 95% CI)")
    ax.errorbar(v, yv, xerr=[v - lo, hi - v], fmt="none", ecolor=INK, color=INK, capsize=4, lw=1.3)
    ax.scatter(vc, yv, marker="D", s=55, color=C["red"], edgecolor=INK, zorder=4, label="Heterogeneous effect (CATE)")
    ax.legend(loc="lower left", fontsize=11, bbox_to_anchor=(0.0, 1.0), ncol=2, columnspacing=1.0)
    for y, n, val in zip(yv, names, v):
        ax.text(hi.max() * 1.03, y, f"{100 * out[n]['ate'] / base:.0f}% · {out[n]['hours']} h closed", fontsize=11, fontweight="bold", va="center")
    ax.set_yticks(yv); ax.set_yticklabels(names)
    style_axis(ax, "Model-based avoided emission, 2022 (t CO$_2$-eq)", None, tick=12, label=14.5, nx=5)
    ax.set_xlim(0, hi.max() * 1.42)
    ax.axvline(0, color=INK, lw=1)
    return out, base


panels = {"a": p_dag, "b": p_forest, "c": p_refute, "d": p_cate_curve, "e": p_surface, "f": p_scen}
sizes = {"a": (6.8, 4.4), "b": (7.4, 4.4), "c": (5.6, 4.6), "d": (5.6, 4.8), "e": (5.8, 4.8), "f": (8.6, 3.6)}


def layout(fig):
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 0.7])
    return {"a": gs[0, 0:3], "b": gs[0, 3:6], "c": gs[1, 0:2], "d": gs[1, 2:4], "e": gs[1, 4:6], "f": gs[2, 1:5]}


ALIGN = {"top": [["a", "b"], ["c", "d", "e"]], "bottom": [["a", "b"], ["c", "d", "e"]], "left": [["a", "c"]], "right": [["b", "e"]]}

if __name__ == "__main__":
    save("Fig8_causal_levers", OUTDIR, layout, panels, sizes, figsize=(18, 14), engine="subfig", align=ALIGN)
    print(ATE.assign(pct=pct(ATE.ate), lo=pct(ATE.lo_block), hi=pct(ATE.hi_block)).round(3).to_string())
    o, b = scenario(); print(b, {k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in o.items()})
