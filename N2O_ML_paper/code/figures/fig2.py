import sys, numpy as np, pandas as pd
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import *
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.colors import Normalize
from scipy import stats
configure()
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "analysis"); OUTDIR = os.path.join(HERE, "..", "..", "figures", "main", "Fig2_emission_structure")
L = pd.read_pickle(f"{ROOT}/long.pkl")
camp = pd.read_csv(f"{ROOT}/figdata/campaign_errors.csv")
VARS = ["T_in", "pH_in", "Q_out", "Supern", "RS", "DO1", "DO2", "DO3", "Air1", "Air2", "Air3", "aOTE1", "aOTE2", "aOTE3", "NH4", "NO3", "Cl", "y"]
NAMES = ["T", "pH", "Q$_{out}$", "Supern.", "RAS", "DO$_1$", "DO$_2$", "DO$_3$", "Air$_1$", "Air$_2$", "Air$_3$",
         "αOTE$_1$", "αOTE$_2$", "αOTE$_3$", "NH$_4^+$", "NO$_3^-$", "Cl$^-$", "N$_2$O"]


def p_calendar(fig, spec):
    gs = spec.subgridspec(2, 3, height_ratios=[1, 0.06], width_ratios=[0.3, 0.4, 0.3], hspace=0.02)
    ax = fig.add_subplot(gs[0, :]); cax = fig.add_subplot(gs[1, 1])
    rows, labels = [], []
    for yr in [2020, 2021, 2022]:
        for s in [1, 2]:
            d = L[(L.street == s) & (L.time.dt.year == yr)]
            M = np.full((24, 366), np.nan)
            M[d.time.dt.hour.values, d.time.dt.dayofyear.values - 1] = np.log10(d.y.values + 1)
            rows.append(M); rows.append(np.full((3, 366), np.nan)); labels.append(f"{yr} S{s}")
    img = np.vstack(rows[:-1])
    im = ax.imshow(img, aspect="auto", cmap=SEQ_RED, vmin=0, vmax=3, interpolation="nearest")
    ax.set_facecolor(to_rgba(C["lilac"], 0.25))
    ax.set_yticks([12 + 27 * i for i in range(6)]); ax.set_yticklabels(labels)
    mt = pd.date_range("2021-01-01", periods=12, freq="MS").dayofyear - 1
    ax.set_xticks(mt); ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    style_axis(ax, None, "Year × street (24 h per band)", tick=12.5, label=15.5)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("log$_{10}$(N$_2$O + 1)  [gN h$^{-1}$]", fontsize=12.5, fontweight="bold"); cb.outline.set_linewidth(1.0)
    cb.set_ticks([0, 1, 2, 3]); bold_ticks(cb.ax, 10.5)


def p_monthly(fig, spec):
    ax = fig.add_subplot(spec)
    m = L.copy(); m["mon"] = m.time.dt.to_period("M")
    z = m.groupby("mon")[["N2O1", "N2O2", "N2O3"]].sum() / 1000  # kg N per month (both streets)
    T = m.groupby("mon").T_in.mean()
    x = np.arange(len(z)); bot = np.zeros(len(z))
    for c, zn in zip(["N2O1", "N2O2", "N2O3"], ["Z1", "Z2", "Z3"]):
        ax.bar(x, z[c].values, bottom=bot, color=ZONE_COLOR[zn], edgecolor=INK, linewidth=0.5, width=0.8, label=zn)
        bot += z[c].values
    ax2 = ax.twinx(); ax2.plot(x, T.values, color=INK, lw=1.8, marker="o", ms=3.5, mfc="white")
    ax.set_xticks(x[::3]); ax.set_xticklabels([str(p)[2:4] + "-" + str(p)[5:] for p in z.index[::3]], rotation=0)
    style_axis(ax, "Month (yy-mm)", "N$_2$O emitted (kg N month$^{-1}$)", tick=12, label=15.5, ny=5)
    style_axis(ax2, None, None, tick=12, spines=("right",)); ax2.spines["right"].set_visible(True)
    ax2.tick_params(axis="y", right=True, labelright=True); ax2.set_ylabel("Influent T (°C)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower center", ncol=3, fontsize=12, handlelength=1.2, columnspacing=0.8, bbox_to_anchor=(0.5, 1.0))
    ax.set_xlim(-0.7, len(z) - 0.3)


def p_hist(fig, spec):
    ax = fig.add_subplot(spec)
    y = L.y.dropna().values
    bins = np.concatenate([[0], np.logspace(-1, np.log10(y.max() + 1), 45)])
    ax.hist(np.clip(y, 0.05, None), bins=np.logspace(np.log10(0.05), np.log10(y.max() + 1), 50), color=C["sky"], edgecolor=INK, linewidth=0.4)
    ax.set_xscale("log")
    style_axis(ax, "Street N$_2$O (gN h$^{-1}$)", "Hours", tick=12.5, label=15.5, ny=5)
    ax.text(0.10, 0.97, f"<0.5 gN h$^{{-1}}$\n{100 * (y < 0.5).mean():.0f}% of hours", transform=ax.transAxes, va="top", fontsize=12, fontweight="bold")
    ins = ax.inset_axes([0.60, 0.50, 0.37, 0.42])
    for yr, col in zip([2020, 2021, 2022], [C["teal"], C["orange"], C["red"]]):
        v = np.sort(L[L.time.dt.year == yr].y.dropna().values)[::-1]
        ins.plot(np.arange(1, len(v) + 1) / len(v) * 100, np.cumsum(v) / v.sum() * 100, color=col, lw=1.6, label=str(yr))
    ins.plot([0, 100], [0, 100], color=MUTED, lw=0.8, ls="--")
    ins.axvline(5, color=MUTED, lw=0.8, ls=":")
    style_axis(ins, "Top hours (%)", "Emission (%)", tick=10.5, label=12, nx=3, ny=3)
    ins.legend(fontsize=10.5, loc="lower right", handlelength=1)


def p_acf(fig, spec):
    ax = fig.add_subplot(spec)
    for s in [1, 2]:
        v = np.log1p(L[L.street == s].set_index("time").y)
        ac = [v.autocorr(k) for k in range(1, 169)]
        ax.plot(range(1, 169), ac, color=STREET_COLOR[s], lw=1.8, label=f"Street {s}")
    n = L.y.notna().sum() / 2; ax.axhspan(-1.96 / np.sqrt(n), 1.96 / np.sqrt(n), color=MUTED, alpha=0.25)
    for k in [24, 48, 72, 96, 120, 144, 168]: ax.axvline(k, color=MUTED, lw=0.6, ls=":")
    ax.set_xticks([1, 24, 48, 72, 96, 120, 144, 168]); ax.set_xlim(0, 170); ax.set_ylim(0, 1)
    style_axis(ax, "Lag (h)", "ACF of log(1 + N$_2$O)", tick=12.5, label=15.5, ny=5)
    ax.legend(loc="upper right")


def p_streets(fig, spec):
    gs = spec.subgridspec(1, 2, width_ratios=[1, 1.15], wspace=0.08)
    ax = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])
    W = L.pivot_table(index="time", columns="street", values="y").dropna()
    hb = ax.hexbin(np.log1p(W[1]), np.log1p(W[2]), gridsize=35, cmap=SEQ, mincnt=1, bins="log", linewidths=0)
    mx = np.log1p(W.values.max()); ax.plot([0, mx], [0, mx], color=INK, lw=1, ls="--")
    style_axis(ax, "Street 1  log(1 + N$_2$O)", "Street 2  log(1 + N$_2$O)", tick=12.5, label=15, nx=4, ny=4)
    ax.text(0.04, 0.96, f"r = {np.corrcoef(W[1], W[2])[0, 1]:.2f}", transform=ax.transAxes, va="top", fontsize=12, fontweight="bold")
    q = W.groupby(W.index.to_period("Q")).apply(lambda d: d[1].corr(d[2]))
    ax2.plot(range(len(q)), q.values, color=C["blue"], marker="o", lw=1.8, ms=5, mfc="white")
    ax2.set_xticks(range(0, len(q), 2)); ax2.set_xticklabels([f"{str(p)[2:4]}Q{str(p)[-1]}" for p in q.index[::2]], rotation=0)
    ax2.set_ylim(0.5, 1.0)
    style_axis(ax2, "Quarter", "Between-street r", tick=12, label=15, ny=5)


def p_corr(fig, spec):
    ax = fig.add_subplot(spec); cax = ax.inset_axes([1.03, 0.25, 0.035, 0.5])
    D = L[VARS].dropna()
    R = D.corr(method="spearman").values; n = len(D)
    t = R * np.sqrt((n - 2) / np.clip(1 - R ** 2, 1e-12, None)); pv = 2 * stats.t.sf(np.abs(t), n - 2)
    k = len(VARS); Mk = np.where(np.tril(np.ones((k, k)), -1).astype(bool), R, np.nan)
    im = ax.imshow(Mk, cmap=DIV, vmin=-1, vmax=1, aspect="auto")
    for i in range(k):
        for j in range(i):
            r = R[i, j]
            if abs(r) >= 0.4:
                ax.text(j, i, f"{r:.1f}".replace("0.", ".").replace("-.", "−."), ha="center", va="center", fontsize=9.5, fontweight="bold",
                        color="white" if abs(r) > 0.6 else INK)
            elif abs(r) >= 0.1 and pv[i, j] < 0.001:
                ax.text(j, i, "·", ha="center", va="center", fontsize=10.5, color=INK)
    ax.set_xticks(range(k)); ax.set_xticklabels(NAMES, rotation=90); ax.set_yticks(range(k)); ax.set_yticklabels(NAMES)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=3, width=1, labelsize=11); bold_ticks(ax, 11)
    ax.add_patch(plt.Rectangle((-0.5, k - 1.5), k - 1, 1, fill=False, edgecolor=C["red"], lw=1.6))
    cb = fig.colorbar(im, cax=cax); cb.set_label("Spearman ρ", fontsize=12.5, fontweight="bold"); cb.outline.set_linewidth(1.0)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1]); bold_ticks(cb.ax, 10)


def p_campaign(fig, spec):
    ax = fig.add_subplot(spec)
    ws = [1, 2, 4, 8, 13, 26]
    data = [np.clip(camp[camp.weeks == w].err.values, -100, 300) for w in ws]
    vp = ax.violinplot(data, positions=range(len(ws)), widths=0.8, showextrema=False)
    for b in vp["bodies"]:
        b.set_facecolor(C["sky"]); b.set_edgecolor(INK); b.set_linewidth(0.8); b.set_alpha(0.9)
    for i, d in enumerate(data):
        q5, q50, q95 = np.percentile(d, [5, 50, 95])
        ax.vlines(i, q5, q95, color=INK, lw=1.6); ax.scatter(i, q50, s=30, color="white", edgecolor=INK, zorder=3, lw=1.2)
        ax.text(i, 312, f"{np.median(np.abs(camp[camp.weeks == ws[i]].err)):.0f}%", ha="center", fontsize=11, fontweight="bold")
    ax.axhline(0, color=C["red"], lw=1.2, ls="--")
    ax.set_xticks(range(len(ws))); ax.set_xticklabels(ws); ax.set_ylim(-110, 330)
    style_axis(ax, "Monitoring campaign (weeks)", "Error in annual mean (%)", tick=12.5, label=15.5, ny=5)


panels = {"a": p_calendar, "b": p_monthly, "c": p_hist, "d": p_acf, "e": p_streets}   # sensor correlations -> Fig. S16, campaign audit -> Fig. S15
sizes = {"a": (14, 4.6), "b": (8.5, 4.2), "c": (5.2, 4.2), "d": (5.2, 4.2), "e": (8.6, 4.0)}


def layout(fig):
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 1, 1], width_ratios=[1, 1, 1.18])
    return {"a": gs[0, :], "b": gs[1, :2], "c": gs[1, 2], "d": gs[2, 0], "e": gs[2, 1:]}


ALIGN = {"top": [["b", "c"], ["d", "e"]], "bottom": [["b", "c"], ["d", "e"]], "left": [["a", "b", "d"]], "right": [["a", "c", "e"]]}

if __name__ == "__main__":
    save("Fig2_emission_structure", OUTDIR, layout, panels, sizes, figsize=(17, 13.5), align=ALIGN)
    print("saved")
