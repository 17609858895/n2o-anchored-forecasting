import sys, pickle, numpy as np, pandas as pd
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import *
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.colors import ListedColormap
configure()
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "analysis"); OUTDIR = os.path.join(HERE, "..", "..", "figures", "main", "Fig6_operating_regimes")
D = pickle.load(open(f"{ROOT}/figdata/fig2to6.pkl", "rb"))["fig3"]
VN = {"T_in": "T", "Q_out": "Q$_{out}$", "Supern": "Supern.", "RS": "RAS", "DO1": "DO$_1$", "DO2": "DO$_2$", "DO3": "DO$_3$",
      "Air1": "Air$_1$", "Air2": "Air$_2$", "Air3": "Air$_3$", "NH4": "NH$_4^+$", "NO3": "NO$_3^-$", "Cl": "Cl$^-$"}
order = D["reg_stats"].sort_values("mean", ascending=False).index.tolist()     # R1 = highest mean emission
rk = {g: i for i, g in enumerate(order)}; RN = [f"R{i + 1}" for i in range(len(order))]
RCOL = REGIME_COLOR
relab = lambda a: np.vectorize(rk.get)(a)


def p_umap(fig, spec):
    ax = fig.add_subplot(spec)
    e, l = D["emb"], relab(D["emb_lab"])
    for i in range(6)[::-1]:
        m = l == i
        ax.scatter(e[m, 0], e[m, 1], s=5, color=RCOL[i], alpha=0.7, lw=0, label=RN[i], rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    style_axis(ax, "UMAP 1", "UMAP 2", label=15.5)
    h, lb = ax.get_legend_handles_labels()
    ax.legend(h[::-1], lb[::-1], ncol=1, markerscale=3.5, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=12, handletextpad=0.2)


def p_centroid(fig, spec):
    ax = fig.add_subplot(spec); cax = ax.inset_axes([1.04, 0.0, 0.05, 0.6])
    Cm = D["cent"].loc[order].T
    im = ax.imshow(Cm.values, cmap=DIV, vmin=-2, vmax=2, aspect="auto")
    for i in range(Cm.shape[0]):
        for j in range(Cm.shape[1]):
            v = Cm.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=10.5, fontweight="bold", color="white" if abs(v) > 1.3 else INK)
    ax.set_xticks(range(6)); ax.set_xticklabels(RN); ax.set_yticks(range(len(Cm))); ax.set_yticklabels([VN[v] for v in Cm.index])
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=3, width=1); bold_ticks(ax, 12)
    ax.set_xlabel("Regime (ordered by mean N$_2$O)", fontsize=13, fontweight="bold")
    cb = fig.colorbar(im, cax=cax); cb.set_label("Centroid z-score", fontsize=12, fontweight="bold"); bold_ticks(cb.ax, 10.5)


def p_dist(fig, spec):
    gs = spec.subgridspec(2, 1, height_ratios=[0.32, 1], hspace=0.05)
    a0 = fig.add_subplot(gs[0]); ax = fig.add_subplot(gs[1], sharex=a0)
    lab, y = relab(D["lab_tr"]), D["ytr"]
    er = [np.mean(y[lab == i] > D["tau"]) * 100 for i in range(6)]
    a0.bar(range(6), er, color=[RCOL[i] for i in range(6)], edgecolor=INK, lw=0.8, width=0.65)
    for i, v in enumerate(er): a0.text(i, v + 2, f"{v:.0f}", ha="center", fontsize=11, fontweight="bold")
    a0.set_ylim(0, max(er) * 1.35); style_axis(a0, None, "Episode\nhours (%)", tick=11.5, label=13.5, ny=3)
    plt.setp(a0.get_xticklabels(), visible=False)
    data = [np.log10(y[lab == i] + 1) for i in range(6)]
    bp = ax.boxplot(data, positions=range(6), widths=0.6, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.8))
    for i, b in enumerate(bp["boxes"]): b.set_facecolor(RCOL[i]); b.set_edgecolor(INK); b.set_alpha(0.9)
    for w in bp["whiskers"] + bp["caps"]: w.set_color(INK); w.set_linewidth(1.2)
    ax.set_xticks(range(6)); ax.set_xticklabels(RN)
    style_axis(ax, "Regime", "log$_{10}$(N$_2$O$_{6h}$ + 1)", tick=12.5, label=15.5, ny=5)
    ax.set_xlim(-0.6, 5.6)


def p_trans(fig, spec):
    ax = fig.add_subplot(spec); cax = ax.inset_axes([1.04, 0.0, 0.05, 0.6])
    T = D["TM"].loc[order, order].values
    im = ax.imshow(np.log10(T + 1e-4), cmap=SEQ, vmin=-3, vmax=0)
    for i in range(6):
        for j in range(6):
            v = T[i, j]; txt = f"{v:.2f}" if v >= 0.01 else ("<.01" if v > 0 else "0")
            ax.text(j, i, txt.replace("0.", "."), ha="center", va="center", fontsize=10.5, fontweight="bold", color="white" if v > 0.3 else INK)
    for j in [0, 1]:
        for i in range(6):
            if i != j: ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=C["red"], lw=1.4))
    ax.set_xticks(range(6)); ax.set_xticklabels(RN); ax.set_yticks(range(6)); ax.set_yticklabels(RN)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=3, width=1); bold_ticks(ax, 12)
    ax.set_xlabel("Regime at t + 1 h", fontsize=13, fontweight="bold"); ax.set_ylabel("Regime at t", fontsize=13, fontweight="bold")
    cb = fig.colorbar(im, cax=cax); cb.set_ticks([-3, -2, -1, 0]); cb.set_ticklabels(["10$^{-3}$", "10$^{-2}$", "10$^{-1}$", "1"])
    cb.set_label("P(transition)", fontsize=12, fontweight="bold"); bold_ticks(cb.ax, 10.5)


def p_ribbon(fig, spec):
    gs = spec.subgridspec(2, 1, height_ratios=[0.22, 1], hspace=0.04)
    a0 = fig.add_subplot(gs[0]); ax = fig.add_subplot(gs[1], sharex=a0)
    t = pd.to_datetime(D["te_time"]); s = D["te_street"] == 1
    df = pd.DataFrame({"t": t[s], "r": relab(D["lab_te"][s]), "y": D["te_y"][s]}).set_index("t")
    full = df.reindex(pd.date_range("2022-01-01", "2022-12-31 23:00", freq="h"))
    x = np.arange(len(full)); img = full.r.values[None, :]
    a0.imshow(np.ma.masked_invalid(img), aspect="auto", cmap=ListedColormap(RCOL), vmin=-0.5, vmax=5.5, interpolation="nearest",
              extent=(-0.5, len(full) - 0.5, 0, 1))
    a0.set_facecolor(to_rgba(C["lilac"], 0.25)); a0.set_yticks([]); a0.set_ylabel("Regime", fontsize=12, fontweight="bold", rotation=0, ha="right", va="center")
    for sp in a0.spines.values(): sp.set_visible(False)
    plt.setp(a0.get_xticklabels(), visible=False); a0.tick_params(bottom=False)
    ax.fill_between(x, 0, np.log10(full.y.values + 1), color=C["sky"], lw=0)
    ax.plot(x, np.log10(full.y.values + 1), color=C["blue"], lw=0.6)
    ax.axhline(np.log10(D["tau"] + 1), color=C["red"], lw=1.1, ls="--")
    ms = pd.date_range("2022-01-01", periods=12, freq="MS"); ax.set_xticks([(m - full.index[0]) / pd.Timedelta(hours=1) for m in ms])
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]); ax.set_xlim(0, len(full))
    style_axis(ax, "2022 (street 1)", "log$_{10}$(N$_2$O$_{6h}$ + 1)", tick=12.5, label=15, ny=4)


def p_k(fig, spec):
    gs = spec.subgridspec(1, 3, width_ratios=[1, 1, 1.1])
    ks = D["ks"]; bic = np.array(D["bic"]) / 1e5
    ax = fig.add_subplot(gs[0]); ax.plot(ks, bic, color=C["blue"], marker="o", lw=1.8, ms=6, mfc="white")
    ax2 = fig.add_subplot(gs[1]); ax2.plot(ks, D["sil"], color=C["orange"], marker="s", lw=1.8, ms=6, mfc="white")
    for a, yl in [(ax, "BIC (×10$^5$)"), (ax2, "Silhouette")]:
        a.axvline(6, color=C["red"], lw=1.1, ls="--"); style_axis(a, "k", yl, ny=4); a.set_xticks([2, 4, 6, 8, 10]); bold_ticks(a)
    ax3 = fig.add_subplot(gs[2])
    kk = sorted(D["ari"]); bp = ax3.boxplot([D["ari"][k] for k in kk], positions=range(len(kk)), widths=0.55, patch_artist=True, showfliers=False,
                                           medianprops=dict(color=INK, lw=1.6))
    for i, b in enumerate(bp["boxes"]): b.set_facecolor(C["red"] if kk[i] == 6 else C["lilac"]); b.set_edgecolor(INK)
    ax3.set_xticks(range(len(kk))); ax3.set_xticklabels(kk); ax3.set_ylim(0, 1.02)
    style_axis(ax3, "k", "Bootstrap ARI", ny=5)


panels = {"b": p_centroid, "c": p_dist, "d": p_trans, "e": p_k, "f": p_ribbon}   # UMAP -> Fig. S16
sizes = {"a": (5.6, 4.6), "b": (5.0, 5.0), "c": (5.0, 4.8), "d": (5.4, 4.8), "e": (11.0, 4.2), "f": (13, 3.6)}


def layout(fig):
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 0.72])
    return {"b": gs[0, 0:3], "c": gs[0, 3:6], "d": gs[1, 0:2], "e": gs[1, 2:6], "f": gs[2, :]}


ALIGN = {"top": [["b", "c"], ["d", "e"]], "bottom": [["b", "c"], ["d", "e"]], "left": [["b", "d", "f"]], "right": [["c", "e", "f"]]}

if __name__ == "__main__":
    save("Fig6_operating_regimes", OUTDIR, layout, panels, sizes, figsize=(17, 13), engine="subfig", align=ALIGN)
    print(D["reg_stats"].loc[order].round(2)); print(D["cent"].loc[order].round(2).T)
