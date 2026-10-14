import sys, os, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from figstyle import *
import matplotlib.pyplot as plt, matplotlib.dates as mdates
configure()
ROOT = os.path.join(HERE, "..", "analysis"); OUTDIR = os.path.join(HERE, "..", "..", "figures", "supplementary", "FigS06_street_structure")
D = pickle.load(open(os.path.join(ROOT, "figdata", "si", "S3.pkl"), "rb"))


def monthly(s):
    def draw(fig, spec):
        ax = fig.add_subplot(spec); m = D["mon"].loc[s]; x = np.arange(len(m)); bot = np.zeros(len(m))
        for i in (1, 2, 3):
            v = m[f"N2O{i}"].fillna(0).values; ax.bar(x, v, 0.8, bottom=bot, color=ZONE_COLOR[f"Z{i}"], edgecolor=INK, lw=0.5, label=f"Z{i}"); bot += v
        ax2 = ax.twinx(); ax2.plot(x, D["T"].reindex(m.index).values, color=INK, lw=1.8, marker="o", ms=4, mfc="white")
        idx = [i for i, t in enumerate(m.index) if t.month in (1, 7)]
        ax.set_xticks(idx); ax.set_xticklabels([m.index[i].strftime("%y-%m") for i in idx])
        style_axis(ax, f"Month (street {s})", "N$_2$O emitted (kg N month$^{-1}$)", ny=5)
        style_axis(ax2, None, None, spines=("right",), ny=4); ax2.tick_params(axis="y", right=True, labelright=True)
        ax2.set_ylabel("Influent T (°C)", fontsize=15.5, fontweight="bold"); bold_ticks(ax2)
        tv = D["T"].values; ax2.set_ylim(np.nanmin(tv) - 0.5, np.nanmax(tv) + 4.5); ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        ax.legend(loc="upper left", ncol=3)
    return draw


def dayhour(s):
    def draw(fig, spec):
        gs = spec.subgridspec(1, 2, width_ratios=[1, 0.02]); ax = fig.add_subplot(gs[0]); cax = fig.add_subplot(gs[1])
        M = D["dh"][s]; full = pd.date_range("2020-01-01", "2022-12-31", freq="D"); M = M.reindex(columns=full)
        im = ax.imshow(np.log10(M.values + 1), aspect="auto", cmap=SEQ_RED, vmin=0, vmax=3, interpolation="nearest", origin="lower",
                       extent=[mdates.date2num(full[0]), mdates.date2num(full[-1]) + 1, 0, 24])
        ax.set_facecolor("#EEEEF4"); ax.xaxis_date(); ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
        ax.set_yticks([0, 6, 12, 18, 24]); style_axis(ax, f"Date (street {s})", "Hour of day")
        cb = fig.colorbar(im, cax=cax); cb.set_label("log$_{10}$(N$_2$O + 1)", fontsize=12.5, fontweight="bold"); bold_ticks(cb.ax, 10.5)
    return draw


def p_diur(fig, spec):
    ax = fig.add_subplot(spec); d = D["diur"]
    for s, ls in [(1, "-"), (2, "--")]:
        for yr, c in zip([2020, 2021, 2022], [C["teal"], C["orange"], C["red"]]):
            v = d.loc[(s, yr)]; ax.plot(v.index, v.values, ls=ls, lw=1.9, color=c, label=f"{yr}, street {s}")
    ax.set_xticks([0, 6, 12, 18, 23]); style_axis(ax, "Hour of day", "Median N$_2$O (gN h$^{-1}$)", ny=5)
    ax.legend(ncol=1, fontsize=11, loc="upper right", handlelength=1.8)


panels = {"a": monthly(1), "b": monthly(2), "c": dayhour(1), "d": dayhour(2), "e": p_diur}
sizes = {"a": (8.4, 4.4), "b": (8.4, 4.4), "c": (9.6, 4.0), "d": (9.6, 4.0), "e": (6.0, 4.4)}


def layout(fig):
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1])
    return {"a": gs[0, 0:3], "b": gs[0, 3:6], "c": gs[1, 0:4], "d": gs[2, 0:4], "e": gs[1:3, 4:6]}


ALIGN = {"top": [["a", "b"], ["c", "e"]], "bottom": [["a", "b"], ["d", "e"]], "left": [["a", "c", "d"]], "right": [["b", "e"], ["c", "d"]]}

if __name__ == "__main__":
    save("FigS06_street_structure", OUTDIR, layout, panels, sizes, figsize=(18, 13), engine="subfig", align=ALIGN)
