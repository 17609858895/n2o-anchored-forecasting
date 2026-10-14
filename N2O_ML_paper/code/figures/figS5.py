import sys, os, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from figstyle import *
import matplotlib.pyplot as plt
configure()
ROOT = os.path.join(HERE, "..", "analysis"); OUTDIR = os.path.join(HERE, "..", "..", "figures", "supplementary", "FigS05_dml_diagnostics")
D = pickle.load(open(os.path.join(ROOT, "figdata", "si", "S12.pkl"), "rb"))
TN = {"swing": "Swing-zone aeration", "supern": "Supernatant valve open"}


def overlap(key):
    def draw(fig, spec):
        ax = fig.add_subplot(spec); ps, T = D["overlap"][key]; bins = np.linspace(0, 1, 41)
        ax.hist(ps[T == 0], bins=bins, color=C["sky"], alpha=0.75, edgecolor=INK, lw=0.4, label="Untreated hours")
        ax.hist(ps[T == 1], bins=bins, color=C["orange"], alpha=0.65, edgecolor=INK, lw=0.4, label="Treated hours")
        ax.set_yscale("log"); style_axis(ax, f"Propensity, {TN[key].lower()}", "Hours", nx=5)
        ax.legend(loc="upper center")
    return draw


def p_nuis(fig, spec):
    ax = fig.add_subplot(spec); n = pd.DataFrame(D["nuis"]).set_index("treatment")
    keys = ["swing", "supern", "supern_dose", "do2_dose"]; lb = ["Swing aeration", "Valve open", "Valve opening (%)", "DO$_2$"]
    x = np.arange(len(keys))
    ax.bar(x - 0.19, n.loc[keys, "R2_outcome"], 0.38, color=C["blue"], edgecolor=INK, lw=0.6, label="Outcome model R²")
    tq = [n.loc[k, "AUC_treatment"] if pd.notna(n.loc[k].get("AUC_treatment", np.nan)) else n.loc[k, "R2_treatment"] for k in keys]
    ax.bar(x + 0.19, tq, 0.38, color=C["orange"], edgecolor=INK, lw=0.6, label="Treatment model AUC / R²")
    for xi, v in zip(np.r_[x - 0.19, x + 0.19], list(n.loc[keys, "R2_outcome"]) + tq): ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(lb, rotation=20, ha="right"); ax.set_ylim(0, 1.38)
    style_axis(ax, None, "Cross-fitted score", ny=5); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); bold_ticks(ax); ax.legend(loc="upper left", ncol=1, fontsize=11)


def dose(key, xl, c):
    def draw(fig, spec):
        gs = spec.subgridspec(2, 1, height_ratios=[1, 0.3]); ax = fig.add_subplot(gs[0]); ah = fig.add_subplot(gs[1], sharex=ax)
        d = D["dose"][key]; pct = lambda v: 100 * (np.exp(v) - 1)
        ax.fill_between(d["grid"], pct(d["lo"]), pct(d["hi"]), color=c, alpha=0.22, lw=0); ax.plot(d["grid"], pct(d["eff"]), color=c, lw=2.4)
        ax.axhline(0, color=INK, lw=0.9); style_axis(ax, None, "Effect vs lowest level (%)", ny=5); ax.tick_params(labelbottom=False)
        t = d["T"]; ah.hist(t[(t >= d["grid"][0]) & (t <= d["grid"][-1])], bins=40, color=C["lilac"], edgecolor=INK, lw=0.3)
        style_axis(ah, xl, "Hours", ny=2)
    return draw


def p_evalue(fig, spec):
    ax = fig.add_subplot(spec); e = pd.DataFrame(D["evalue"]); x = np.arange(len(e))
    ax.bar(x - 0.19, e.E_point, 0.38, color=C["red"], edgecolor=INK, lw=0.6, label="Point estimate")
    ax.bar(x + 0.19, e.E_ci, 0.38, color=C["pink"], edgecolor=INK, lw=0.6, label="Lower 95% bound")
    for xi, v in zip(np.r_[x - 0.19, x + 0.19], list(e.E_point) + list(e.E_ci)): ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax.axhline(1, color=INK, lw=0.9, ls="--"); ax.set_xticks(x); ax.set_xticklabels([TN[t] for t in e.treatment], rotation=12, ha="right")
    ax.set_ylim(0, max(e.E_point) * 1.3); style_axis(ax, None, "E-value (risk-ratio scale)", ny=4); ax.legend(loc="upper left", fontsize=11)


panels = {"a": overlap("swing"), "b": overlap("supern"), "c": p_nuis, "d": dose("supern_dose", "Supernatant valve opening (%)", C["orange"]),
          "e": dose("do2_dose", "DO$_2$ (mg L$^{-1}$)", C["blue"]), "f": p_evalue}
sizes = {"a": (5.6, 4.6), "b": (5.6, 4.6), "c": (5.6, 4.6), "d": (5.6, 5.0), "e": (5.6, 5.0), "f": (5.6, 5.0)}


def layout(fig):
    gs = fig.add_gridspec(2, 3); return {"a": gs[0, 0], "b": gs[0, 1], "c": gs[0, 2], "d": gs[1, 0], "e": gs[1, 1], "f": gs[1, 2]}


ALIGN = {"top": [["a", "b", "c"], ["d", "e", "f"]], "bottom": [["a", "b", "c"], ["d", "e", "f"]], "left": [["a", "d"]], "right": [["c", "f"]]}

if __name__ == "__main__":
    save("FigS05_dml_diagnostics", OUTDIR, layout, panels, sizes, figsize=(17, 11), engine="subfig", align=ALIGN)
