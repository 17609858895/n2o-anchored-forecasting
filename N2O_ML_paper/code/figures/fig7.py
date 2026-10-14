import sys, pickle, numpy as np, pandas as pd
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import *
import matplotlib.pyplot as plt
configure()
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "analysis"); OUTDIR = os.path.join(HERE, "..", "..", "figures", "main", "Fig7_drivers_shap_ale")
D = pickle.load(open(f"{ROOT}/figdata/fig7.pkl", "rb")); F3 = pickle.load(open(f"{ROOT}/figdata/fig2to6.pkl", "rb"))["fig3"]
sv, Xs = D["sv"], D["Xs"]; cols = list(Xs.columns)
GROUPS = {"N$_2$O history": lambda c: c.startswith(("yl", "y_", "N2O")),
          "Side-stream": lambda c: c.startswith("Supern"),
          "Aeration / DO": lambda c: c.startswith(("DO", "Air", "aOTE", "swing")),
          "N loading": lambda c: c.startswith(("NH4", "NO3", "Nload")),
          "Hydraulics": lambda c: c.startswith(("Q_out", "RS")),
          "Temperature / snowmelt": lambda c: c.startswith(("T_in", "Cl", "T_drop", "pH")),
          "Calendar": lambda c: c.startswith(("hour", "doy", "weekend", "street")),
          "Regime posterior": lambda c: c.startswith("regime")}
GCOL = {"N$_2$O history": C["lilac"], "Side-stream": C["orange"], "Aeration / DO": C["blue"], "N loading": C["teal"],
        "Hydraulics": C["sky"], "Temperature / snowmelt": C["coral"], "Calendar": C["gold"], "Regime posterior": C["pink"]}
grp = {c: next(g for g, f in GROUPS.items() if f(c)) for c in cols}
PRETTY = {"T_in": "T", "Q_out": "Q$_{out}$", "Supern": "Supern.", "RS": "RAS", "NH4": "NH$_4^+$", "NO3": "NO$_3^-$", "Cl": "Cl$^-$",
          "N2O1": "N$_2$O Z1", "N2O2": "N$_2$O Z2", "N2O3": "N$_2$O Z3", "yl": "N$_2$O lag ", "y": "N$_2$O"}
def pretty(c):
    if c.startswith("yl"): return f"N$_2$O(t−{c[2:]} h)" if c != "yl0" else "N$_2$O(t)"
    if c.startswith("N2O") and c.endswith("_l0"): return f"N$_2$O Z{c[3]}(t)"
    if c.startswith("y_"):
        import re
        k, n = re.match(r"([a-z]+)(\d+)", c[2:]).groups()
        return "N$_2$O " + {"m": "mean", "max": "max", "sd": "SD", "d": "Δ"}[k] + f" {n} h"
    base, *suf = c.split("_")
    if base == "T" and c.startswith("T_in"): base, suf = "T_in", c.split("_")[2:]
    if base == "Q" and c.startswith("Q_out"): base, suf = "Q_out", c.split("_")[2:]
    if base == "T" and c.startswith("T_drop"): return "T drop 72 h"
    name = {"T_in": "T", "Q_out": "Q$_{out}$", "Supern": "Supern.", "RS": "RAS", "NH4": "NH$_4^+$", "NO3": "NO$_3^-$", "Cl": "Cl$^-$",
            "hour": "Hour", "doy": "Day of year", "swing": "Swing aeration", "Nload": "N load"}.get(base, base)
    name = name.replace("DO", "DO$_").replace("Air", "Air$_").replace("aOTE", "αOTE$_") + ("$" if base[:2] in ("DO", "Ai", "aO") and base[-1].isdigit() else "")
    if name.endswith("_$"): name = name[:-2]
    s = " ".join(suf).replace("m", "mean ").replace("sd", "SD ").replace("d1", "Δ1 h").replace("d24", "Δ24 h").replace("sin", "sin").replace("cos", "cos")
    s = s.replace("frac24", "share 24 h")
    if s and s[-1].isdigit(): s += " h"
    return (name + " " + s).strip()
imp = pd.Series(np.abs(sv).mean(0), index=cols).sort_values(ascending=False)


def p_bar(fig, spec):
    ax = fig.add_subplot(spec)
    top = imp.head(15)[::-1]
    ax.barh(range(15), top.values, color=[GCOL[grp[c]] for c in top.index], edgecolor=INK, lw=0.6, height=0.7)
    ax.set_yticks(range(15)); ax.set_yticklabels([pretty(c) for c in top.index])
    style_axis(ax, "Mean |SHAP| (gN h$^{-1}$)", None, tick=12, label=15, nx=4)
    g = pd.Series({k: imp[[c for c in cols if grp[c] == k]].sum() for k in GROUPS}); g = g[g > 0].sort_values(ascending=False)
    ins = ax.inset_axes([0.40, 0.18, 0.36, 0.48])
    wedges, _ = ins.pie(g.values, colors=[GCOL[k] for k in g.index], startangle=90, counterclock=False,
                        wedgeprops=dict(width=0.38, edgecolor="white", lw=1.2))
    ins.text(0, 0, f"{100 * (1 - g['N$_2$O history'] / g.sum()):.0f}%\nnon-N$_2$O\nhistory", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.legend(wedges, [f"{k} {100 * v / g.sum():.0f}%" for k, v in g.items()], loc="lower right", bbox_to_anchor=(1.0, 0.02), fontsize=10,
              handlelength=0.8, frameon=False, ncol=1, labelspacing=0.25)


def p_bee(fig, spec):
    gs = spec.subgridspec(2, 1, height_ratios=[1, 0.035])
    ax = fig.add_subplot(gs[0]); cax = fig.add_subplot(gs[1])
    ops = [c for c in imp.index if grp[c] not in ("N$_2$O history", "Regime posterior", "Calendar")][:12][::-1]
    rng = np.random.default_rng(0)
    for i, c in enumerate(ops):
        j = cols.index(c); v = sv[:, j]; x = Xs[c].values.astype(float)
        lo, hi = np.nanpercentile(x, [2, 98]); z = np.clip((x - lo) / (hi - lo + 1e-12), 0, 1)
        dens = np.histogram(v, bins=60)[0]; bi = np.clip(np.digitize(v, np.histogram(v, bins=60)[1]) - 1, 0, 59)
        jit = rng.uniform(-1, 1, len(v)) * 0.38 * (dens[bi] / dens.max()) ** 0.5
        sc = ax.scatter(v, i + jit, c=z, cmap=DIV, s=4, lw=0, vmin=0, vmax=1, rasterized=True)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(len(ops))); ax.set_yticklabels([pretty(c) for c in ops])
    lim = np.percentile(np.abs(sv[:, [cols.index(c) for c in ops]]), 99.5); ax.set_xlim(-lim, lim)
    style_axis(ax, "SHAP value (gN h$^{-1}$, change vs anchor)", None, tick=12, label=14.5, nx=5)
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal"); cb.set_ticks([0, 1]); cb.set_ticklabels(["Low", "High"])
    cb.set_label("Feature value", fontsize=12, fontweight="bold", labelpad=1); bold_ticks(cb.ax, 10.5)


def p_ale(fig, spec):
    keys = [("T_in", "T (°C)"), ("DO2", "DO$_2$ (mg L$^{-1}$)"), ("NO3", "NO$_3^-$ (mgN L$^{-1}$)"),
            ("Supern", "Supernatant valve (%)"), ("Cl_m72", "Cl$^-$ 72 h mean (mg L$^{-1}$)"), ("Q_out", "Q$_{out}$ (dataset units)")]
    gs = spec.subgridspec(2, 3)
    for k, (f, xl) in enumerate(keys):
        ax = fig.add_subplot(gs[k // 3, k % 3]); A = D["ALE"][f]
        ax.fill_between(A["q"], A["lo"], A["hi"], color=C["pink"], alpha=0.7, lw=0)
        ax.plot(A["q"], A["a"], color=C["red"], lw=2, label="ALE")
        ax.plot(A["q"], A["pdp"], color=C["blue"], lw=1.3, ls="--", label="PDP")
        ax.axhline(0, color=MUTED, lw=0.7)
        rug = np.nanpercentile(A["x"], np.arange(10, 100, 10))
        ax.plot(rug, np.full(len(rug), 0.04), "|", color=INK, ms=10, mew=1.6, transform=ax.get_xaxis_transform(), clip_on=False)
        style_axis(ax, xl, "ALE (gN h$^{-1}$)" if k % 3 == 0 else None, tick=11, label=13, nx=4, ny=4)
        if k == 0: ax.legend(loc="lower right", fontsize=11, handlelength=1.6)


def p_inter(fig, spec):
    gs = spec.subgridspec(2, 2, height_ratios=[1, 0.05])
    pairs = [("T_in", "DO2", "T (°C)", "DO$_2$ (mg L$^{-1}$)"), ("Supern", "swing_on", "Supernatant valve (%)", "Swing aeration")]
    for k, (a, b, xa, cb_lab) in enumerate(pairs):
        ax = fig.add_subplot(gs[0, k]); cax = fig.add_subplot(gs[1, k])
        x = Xs[a].values; v = sv[:, cols.index(a)]; z = Xs[b].values.astype(float)
        if b == "swing_on":
            for val, col, lb in [(0, C["sky"], "Off"), (1, C["red"], "On")]:
                m = z == val; ax.scatter(x[m], v[m], s=7, color=col, alpha=0.6, lw=0, label=lb, rasterized=True)
            ax.legend(loc="upper left", fontsize=11, markerscale=2.5, title="Swing aeration", title_fontsize=11); cax.axis("off")
        else:
            lo, hi = np.nanpercentile(z, [2, 98])
            sc = ax.scatter(x, v, c=np.clip(z, lo, hi), cmap=DIV, s=7, lw=0, alpha=0.8, rasterized=True)
            cb = fig.colorbar(sc, cax=cax, orientation="horizontal"); cb.set_label(cb_lab, fontsize=11.5, fontweight="bold", labelpad=1); bold_ticks(cb.ax, 10.5)
        ax.axhline(0, color=INK, lw=0.8)
        style_axis(ax, xa, "SHAP (gN h$^{-1}$)", tick=11.5, label=14, nx=4, ny=4)


def p_regime(fig, spec):
    gs = spec.subgridspec(1, 2, width_ratios=[1, 0.05]); ax = fig.add_subplot(gs[0]); cax = fig.add_subplot(gs[1])
    order = F3["reg_stats"].sort_values("mean", ascending=False).index.tolist()
    labs = D["lab"]; gl = [g for g in GROUPS if g != "Regime posterior"]
    M = np.zeros((len(gl), 6))
    for j, r in enumerate(order):
        m = labs == r
        for i, g in enumerate(gl):
            idx = [cols.index(c) for c in cols if grp[c] == g]
            M[i, j] = np.abs(sv[m][:, idx]).sum(1).mean() if m.sum() else np.nan
    Mn = M / np.nansum(M, axis=0, keepdims=True) * 100
    im = ax.imshow(Mn, cmap=SEQ, vmin=0, vmax=np.nanmax(Mn), aspect="auto")
    for i in range(Mn.shape[0]):
        for j in range(6):
            ax.text(j, i, f"{Mn[i, j]:.0f}", ha="center", va="center", fontsize=11, fontweight="bold", color="white" if Mn[i, j] > 0.6 * np.nanmax(Mn) else INK)
    ax.set_xticks(range(6)); ax.set_xticklabels([f"R{i + 1}" for i in range(6)])
    ax.set_yticks(range(len(gl))); ax.set_yticklabels(gl)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=3); bold_ticks(ax, 11.5)
    cb = fig.colorbar(im, cax=cax); cb.set_label("Share of |SHAP| (%)", fontsize=12, fontweight="bold"); bold_ticks(cb.ax, 10.5)


panels = {"a": p_bar, "b": p_bee, "e": p_regime, "c": p_ale}   # SHAP dependence -> Fig. S19
sizes = {"a": (7.2, 5.6), "b": (6.8, 6.0), "c": (9.6, 6.0), "d": (9.0, 4.6), "e": (6.6, 4.6)}


def layout(fig):
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.05])
    return {"a": gs[0, 0:2], "b": gs[0, 2:4], "e": gs[0, 4:6], "c": gs[1, :]}


ALIGN = {"top": [["a", "b", "e"]], "bottom": [["a", "b", "e"]], "left": [["a", "c"]], "right": [["e", "c"]]}

if __name__ == "__main__":
    save("Fig7_drivers_shap_ale", OUTDIR, layout, panels, sizes, figsize=(18, 12), engine="subfig", align=ALIGN)
    print(imp.head(15).round(3)); print([pretty(c) for c in imp.head(20).index])
