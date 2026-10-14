import sys, pickle, numpy as np, pandas as pd
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import *
import matplotlib.pyplot as plt
configure()
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "analysis"); OUTDIR = os.path.join(HERE, "..", "..", "figures", "main", "Fig5_egmoe_mechanism")
ALL = pickle.load(open(f"{ROOT}/figdata/fig2to6.pkl", "rb")); D = ALL["fig6"]; F5 = ALL["fig5"]; F3 = ALL["fig3"]
EN = {"anchor": "Anchor", "resid": "Residual", "raw": "Raw-scale", "log": "Log-scale", "tweedie": "Tweedie"}
order = F3["reg_stats"].sort_values("mean", ascending=False).index.tolist(); RN = {g: f"R{i + 1}" for i, g in enumerate(order)}


def p_ablation(fig, spec):
    gs = spec.subgridspec(1, 2); a1 = fig.add_subplot(gs[0]); a2 = fig.add_subplot(gs[1], sharey=a1)
    A = D["ABL"].copy(); base = A.loc["EG-MoE (full)"]
    A = A.drop("EG-MoE (full)"); A["dR"] = A.RMSE - base.RMSE; A["dC"] = A.absCum - base.absCum
    A = A.sort_values("dR")
    yv = np.arange(len(A))
    cols = [EXPERT_COLOR[i.split(": ")[1]] if i.startswith("Expert") else C["lilac"] for i in A.index]
    a1.barh(yv, A.dR, color=cols, edgecolor=INK, lw=0.7, height=0.65)
    a2.barh(yv, A.dC, color=cols, edgecolor=INK, lw=0.7, height=0.65)
    for ax in (a1, a2): ax.axvline(0, color=INK, lw=1.2)
    for i, v in enumerate(A.dR): a1.text(v + 0.15, i, f"+{v:.1f}" if v >= 0 else f"{v:.1f}", va="center", fontsize=10.5, fontweight="bold")
    nm = {"Expert only: anchor": "Only anchor", "Expert only: resid": "Only residual", "Expert only: raw": "Only raw-scale",
          "Expert only: log": "Only log-scale", "Expert only: tweedie": "Only Tweedie"}
    a1.set_yticks(yv); a1.set_yticklabels([nm.get(i, i) for i in A.index])
    plt.setp(a2.get_yticklabels(), visible=False)
    style_axis(a1, "ΔRMSE vs full (gN h$^{-1}$)", None, tick=12, label=14.5, nx=4)
    style_axis(a2, "Δ|CumErr| vs full (%)", None, tick=12, label=14.5, nx=4); a2.tick_params(axis="y", left=False)
    a1.set_xlim(right=A.dR.max() * 1.25)


def p_gate(fig, spec):
    ax = fig.add_subplot(spec); cax = ax.inset_axes([1.04, 0.0, 0.05, 0.5])
    rows, lbl = [], []
    for b, bn in zip(range(3), ["p<0.15", "0.15–0.5", "p≥0.5"]):
        for k in order:
            rows.append(D["W"][b][k]); lbl.append(f"{bn} · {RN[k]}")
    Wm = np.array(rows)
    im = ax.imshow(Wm, cmap=SEQ_RED, vmin=0, vmax=1, aspect="auto")
    for i in range(Wm.shape[0]):
        for j in range(Wm.shape[1]):
            ax.text(j, i, f"{Wm[i, j]:.2f}".replace("0.", "."), ha="center", va="center", fontsize=11, color="white" if Wm[i, j] > 0.6 else INK)
    for yb in [5.5, 11.5]: ax.axhline(yb, color=INK, lw=1.4)
    ax.set_xticks(range(5)); ax.set_xticklabels([EN[n] for n in D["names"]], rotation=35, ha="right")
    ax.set_yticks(range(len(lbl))); ax.set_yticklabels(lbl)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=5); bold_ticks(ax, 11)
    cb = fig.colorbar(im, cax=cax); cb.set_label("Expert weight", fontsize=12, fontweight="bold"); bold_ticks(cb.ax, 10.5)


def p_weights(fig, spec):
    gs = spec.subgridspec(2, 1, height_ratios=[0.6, 1])
    a0 = fig.add_subplot(gs[0]); ax = fig.add_subplot(gs[1], sharex=a0)
    t = pd.to_datetime(D["tm"]); s = D["st"] == 1; m = s & (t >= "2022-01-01") & (t < "2022-02-01")
    o = np.argsort(t[m].values); tt = t[m][o]; W = D["Wte"][m][o]; y = D["y"][m][o]; p = D["p"][m][o]
    a0.plot(tt, y, color=INK, lw=1.3); a0.fill_between(tt, 0, y, color=C["sky"], alpha=0.6, lw=0)
    a0b = a0.twinx(); a0b.plot(tt, p, color=C["red"], lw=1.1); a0b.set_ylim(0, 1.05)
    style_axis(a0, None, "N$_2$O$_{6h}$\n(gN h$^{-1}$)", tick=11.5, label=13.5, ny=3); plt.setp(a0.get_xticklabels(), visible=False)
    style_axis(a0b, None, None, tick=11.5, spines=("right",)); a0b.tick_params(axis="y", right=True, labelright=True)
    a0b.set_ylabel("p$_{episode}$", fontsize=12, fontweight="bold", color=C["red"])
    a0.set_xlim(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-01-31 23:00"))
    ax.stackplot(tt, W.T, colors=[EXPERT_COLOR[n] for n in D["names"]], labels=[EN[n] for n in D["names"]], edgecolor=INK, lw=0.2)
    ax.set_ylim(0, 1); ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d Jan"))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.DayLocator([1, 8, 15, 22, 29]))
    style_axis(ax, "January 2022 (street 1)", "Expert weight", tick=12, label=14.5, ny=3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=5, fontsize=11, handlelength=1.2, columnspacing=0.9)


def p_rocpr(fig, spec):
    gs = spec.subgridspec(1, 2); a1 = fig.add_subplot(gs[0]); a2 = fig.add_subplot(gs[1])
    fg, tg, _ = D["roc_gate"]; fo, to, _ = D["roc_ops"]
    a1.plot(fg, tg, color=C["red"], lw=2, label=f"EG-MoE gate ({D['auc_gate']:.3f})")
    a1.plot(fo, to, color=C["teal"], lw=2, label=f"Operational only ({D['auc_ops']:.3f})")
    a1.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
    style_axis(a1, "False-positive rate", "True-positive rate", tick=12, label=14.5, nx=3, ny=3); 
    a1.legend(loc="lower right", bbox_to_anchor=(1.04, -0.02), fontsize=11, title="ROC-AUC", title_fontsize=11, handlelength=1.0, borderaxespad=0.1)
    pg, rg, _ = D["pr_gate"]; po, ro, _ = D["pr_ops"]
    a2.plot(rg, pg, color=C["red"], lw=2, label=f"{D['ap_gate']:.3f}"); a2.plot(ro, po, color=C["teal"], lw=2, label=f"{D['ap_ops']:.3f}")
    a2.axhline(D["base"], color=MUTED, lw=0.8, ls="--")
    style_axis(a2, "Recall", "Precision", tick=12, label=14.5, nx=3, ny=3); a2.set_xlim(0, 1); a2.set_ylim(0, 1.02)
    a2.legend(loc="lower left", fontsize=11, title="PR-AUC", title_fontsize=11, handlelength=1.2, borderaxespad=0.2)
    ins = a2.inset_axes([0.58, 0.50, 0.26, 0.25]); pt, pp = D["calib"]
    ins.plot(pp, pt, color=C["red"], marker="o", ms=3, lw=1.2); ins.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls="--")
    style_axis(ins, None, None, tick=10, nx=2, ny=2); ins.text(0.04, 0.96, "Reliability", transform=ins.transAxes, ha="left", va="top", fontsize=10.5, fontweight="bold")


def p_conformal(fig, spec):
    ax = fig.add_subplot(spec)
    pb = np.digitize(D["p"], [0.15, 0.5]); groups = [("All", np.ones(len(pb), bool)), ("p<0.15", pb == 0), ("0.15–0.5", pb == 1), ("p≥0.5", pb == 2)]
    x = np.arange(len(groups))
    for kind, col, mk, dx, lbl in [("conf", C["red"], "o", -0.12, "Rolling"), ("stat", C["indigo"], "s", 0.12, "Static")]:
        for a, fill in [(0.1, True), (0.2, False)]:
            cv = [F5[kind][a]["cover"][g].mean() * 100 for _, g in groups]
            ax.scatter(x + dx, cv, marker=mk, s=70, color=col if fill else "white", edgecolor=col, lw=1.8, zorder=3,
                       label=f"{lbl} {int((1 - a) * 100)}%")
            ax.plot(x + dx, cv, color=col, lw=0.8, alpha=0.6)
    ax.axhline(80, color=INK, lw=1, ls="--"); ax.axhline(90, color=INK, lw=1, ls=":")
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], rotation=35, ha="right"); ax.set_ylim(70, 96); ax.set_xlim(-0.5, 3.5)
    style_axis(ax, "Episode-probability bin", "Empirical coverage (%)", tick=12, label=14.5, ny=5)
    ax.legend(loc="lower left", fontsize=11, ncol=1, handletextpad=0.3, labelspacing=0.3)


def onset_stats():
    L = pd.read_pickle(f"{ROOT}/long.pkl"); B = pickle.load(open(f"{ROOT}/pred_H6_chrono_h1_2022.pkl", "rb"))
    tau = B["tau"]; pser = pd.Series(B["p"], index=pd.MultiIndex.from_arrays([B["street"], pd.to_datetime(B["time"])]))
    res = {}
    for sustain in [1, 6]:
        wins = []
        for s in [1, 2]:
            ys = L[L.street == s].set_index("time").y; ys = ys[ys.index >= "2022-01-02"]
            above = ys > tau; fwd = ys.rolling(sustain, min_periods=sustain).mean().shift(-(sustain - 1)) > tau
            prev = above.shift(1).rolling(24, min_periods=24).max(); ps = pser.loc[s]
            for u in ys.index[above & (prev == 0) & fwd]:
                w = ps[(ps.index >= u - pd.Timedelta(hours=24)) & (ps.index <= u - pd.Timedelta(hours=1))]
                if len(w) >= 12: wins.append((u, w))
        res[sustain] = wins
    return res


OS = onset_stats()


def p_lead(fig, spec):
    gs = spec.subgridspec(1, 2, width_ratios=[1, 1]); a1 = fig.add_subplot(gs[0]); a2 = fig.add_subplot(gs[1])
    thr = np.linspace(0.05, 0.9, 35)
    for sus, col, lbl in [(1, C["sky"], "Any onset"), (6, C["red"], "Sustained ≥ 6 h")]:
        hr = [100 * np.mean([(w >= t).any() for _, w in OS[sus]]) for t in thr]
        a1.plot(thr, hr, color=col, lw=2, label=f"{lbl} (n = {len(OS[sus])})")
    a1.axvline(0.3, color=MUTED, lw=0.8, ls=":")
    style_axis(a1, "Warning threshold on p", "Onsets warned ≤24 h ahead (%)", tick=11.5, label=14, nx=4, ny=5); a1.set_ylim(0, 100)
    a1.legend(loc="upper right", fontsize=11, framealpha=1, frameon=True, edgecolor="white")
    ld = [(u - w[w >= 0.3].index.min()) / pd.Timedelta(hours=1) for u, w in OS[6] if (w >= 0.3).any()]
    a2.hist(ld, bins=np.arange(0.5, 25.5, 2), color=C["coral"], edgecolor=INK, lw=0.6)
    a2.axvline(np.median(ld), color=C["red"], lw=1.6, ls="--")
    a2.text(0.95, 0.95, f"p ≥ 0.3, sustained\nwarned {len(ld)}/{len(OS[6])}\nmedian {np.median(ld):.1f} h", transform=a2.transAxes, ha="right", va="top", fontsize=10.5, fontweight="bold")
    a2.set_xlim(0, 25); a2.set_xticks([1, 6, 12, 18, 24])
    style_axis(a2, "Lead time (h)", "Onsets", tick=11.5, label=14, ny=4)


def p_dom(fig, spec):
    ax = fig.add_subplot(spec)
    t = pd.to_datetime(D["tm"]); dom = D["Wte"].argmax(1)
    df = pd.DataFrame({"m": t.month, "s": D["st"], "d": dom})
    tab = pd.crosstab([df.m, df.s], df.d, normalize="index").reindex(columns=range(5), fill_value=0)
    x = np.arange(len(tab)); bot = np.zeros(len(tab))
    for j, n in enumerate(D["names"]):
        ax.bar(x, tab[j].values, bottom=bot, color=EXPERT_COLOR[n], edgecolor=INK, lw=0.3, width=0.85, label=EN[n]); bot += tab[j].values
    ax.set_xticks(x[::2] + 0.5); ax.set_xticklabels([f"{m}" for m, s in tab.index[::2]])
    ax.set_ylim(0, 1.0)
    style_axis(ax, "Month of 2022 (street 1 | street 2 pairs)", "Share of hours as dominant expert", tick=12, label=14.5, ny=3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=11, handlelength=1.2, columnspacing=0.9)


panels = {"a": p_ablation, "b": p_gate, "c": p_weights, "d": p_rocpr, "f": p_lead}   # coverage and dominant-expert shares -> Fig. S18
sizes = {"a": (8.2, 5.0), "b": (5.0, 6.4), "c": (8.0, 5.0), "d": (8.0, 4.0), "e": (4.4, 5.6), "f": (7.6, 4.0), "g": (7.6, 4.6)}


def layout(fig):
    gs = fig.add_gridspec(3, 6, height_ratios=[1.1, 1, 0.95])
    return {"a": gs[0, 0:3], "b": gs[0:2, 3:6], "c": gs[1, 0:3], "d": gs[2, 0:3], "f": gs[2, 3:6]}


ALIGN = {"top": [["a", "b"], ["d", "f"]], "bottom": [["c", "b"], ["d", "f"]], "left": [["a", "c", "d"]], "right": [["b", "f"]]}

if __name__ == "__main__":
    save("Fig5_egmoe_mechanism", OUTDIR, layout, panels, sizes, figsize=(19, 16), engine="subfig", align=ALIGN)
    print(D["ABL"].round(3))
