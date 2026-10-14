"""Shared style for the ARA Hofen N2O figure set (adapted from nature-research-figures ml_publication_style).

Every panel is a function draw(fig, spec). The same function renders the panel inside the combined
figure and again on its own canvas, so subfigures are independently rendered, not cropped.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from scientific_palettes import get_palette, STYLE_COLORS

INK = STYLE_COLORS["ink"]; MUTED = STYLE_COLORS["muted"]
PAL = get_palette("ptsn_warm_cool_10")  # E17878 D8292C F0BAB9 E08A2E CB9A1A 65ACA8 7EB5DC 4E85AC BEC3E1 7D89BD
C = dict(zip(["coral", "red", "pink", "orange", "gold", "teal", "sky", "blue", "lilac", "indigo"], PAL))

# ---- one semantic colour map shared by every figure ----
MODEL_COLOR = {"EG-MoE": C["red"], "Persistence": C["gold"], "LightGBM": C["blue"], "Residual expert": C["coral"],
               "Extra trees": C["teal"], "Seasonal naive": C["lilac"], "GRU-A": C["sky"]}
FAMILY_COLOR = {"Naive": C["gold"], "Linear": C["pink"], "Kernel/NN": C["orange"], "Tree/bagging": C["teal"],
                "Boosting": C["blue"], "Neural": C["indigo"], "Neural, anchored": C["sky"], "Proposed": C["red"]}
EXPERT_COLOR = {"anchor": C["gold"], "resid": C["coral"], "raw": C["blue"], "log": C["sky"], "tweedie": C["lilac"]}
STREET_COLOR = {1: C["teal"], 2: C["orange"]}
ZONE_COLOR = {"Z1": C["orange"], "Z2": C["teal"], "Z3": C["blue"]}
REGIME_COLOR = [C["red"], C["orange"], C["gold"], C["teal"], C["sky"], C["indigo"]]
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#FFFFFF", C["sky"], C["blue"]])
SEQ_RED = LinearSegmentedColormap.from_list("seq_red", ["#FFFFFF", C["pink"], C["red"]])
DIV = LinearSegmentedColormap.from_list("div", [C["blue"], C["sky"], "#FFFFFF", C["pink"], C["red"]])

FAMILY = {"Persistence": "Naive", "SeasonalNaive": "Naive", "ClimMean": "Naive", "Ridge": "Linear", "ElasticNet": "Linear",
          "SVR": "Kernel/NN", "KNN": "Kernel/NN", "DT": "Tree/bagging", "RF": "Tree/bagging", "ExtraTrees": "Tree/bagging",
          "AdaBoost": "Boosting", "HistGB": "Boosting", "XGBoost": "Boosting", "LightGBM": "Boosting", "CatBoost": "Boosting",
          "MLP": "Neural", "LSTM": "Neural", "GRU": "Neural", "TCN": "Neural", "Transformer": "Neural", "EG-MoE": "Proposed",
          "LSTM-A": "Neural, anchored", "GRU-A": "Neural, anchored", "TCN-A": "Neural, anchored", "Transformer-A": "Neural, anchored"}
LABEL = {"SeasonalNaive": "Seasonal naive", "ClimMean": "Climatology", "DT": "Decision tree", "RF": "Random forest",
         "ExtraTrees": "Extra trees", "[expert] resid": "Residual expert"}
lab = lambda m: LABEL.get(m, m)


FONT_DIR = Path(__file__).resolve().parent / "fonts"   # drop arial.ttf / arialbd.ttf here to render in Arial


def configure():
    if FONT_DIR.is_dir():
        for f in FONT_DIR.glob("*.ttf"):
            font_manager.fontManager.addfont(str(f))
    names = {f.name for f in font_manager.fontManager.ttflist}
    fam = "Arial" if "Arial" in names else "Liberation Sans"   # Liberation Sans is metric-compatible with Arial
    plt.rcParams.update({
        "font.family": fam, "font.sans-serif": [fam, "DejaVu Sans"], "axes.unicode_minus": False, "mathtext.default": "regular",
        "figure.dpi": 110, "savefig.dpi": 600, "axes.labelsize": 15.5, "axes.labelweight": "bold",
        "xtick.labelsize": 12.5, "ytick.labelsize": 12.5, "legend.fontsize": 11.5, "axes.linewidth": 1.55,
        "xtick.major.size": 7.2, "ytick.major.size": 7.2, "xtick.major.width": 1.55, "ytick.major.width": 1.55,
        "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    })
    return fam


def style_axis(ax, xlabel=None, ylabel=None, tick=12.5, label=15.5, nx=None, ny=None, spines=("left", "bottom")):
    ax.grid(False)
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(s in spines)
        ax.spines[s].set_linewidth(1.55); ax.spines[s].set_color(INK)
    ax.tick_params(axis="both", which="major", direction="out", length=7.2, width=1.55, labelsize=tick, colors=INK,
                   bottom="bottom" in spines, left="left" in spines, top=False, right=False)
    if nx: ax.xaxis.set_major_locator(MaxNLocator(nbins=nx))
    if ny: ax.yaxis.set_major_locator(MaxNLocator(nbins=ny))
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold"); t.set_fontsize(tick)
    if xlabel is not None: ax.set_xlabel(xlabel, fontsize=label, fontweight="bold", labelpad=5)
    if ylabel is not None: ax.set_ylabel(ylabel, fontsize=label, fontweight="bold", labelpad=5)


def bold_ticks(ax, size=12.5):
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold"); t.set_fontsize(size)


# ---------------------------------------------------------------- export helpers
def _all_axes_sf(sf):
    return list(sf.axes)


def _all_axes(fig):
    out = []
    def walk(a):
        out.append(a)
        for c in getattr(a, "child_axes", []): walk(c)
    def walkfig(f):
        for a in f.axes: walk(a)
        for sf in getattr(f, "subfigs", []): walkfig(sf)
    walkfig(fig)
    return list(dict.fromkeys(out))


def finalize(fig):
    """Bold every tick label (incl. insets, colorbars, polar axes); return a tick-mark audit."""
    fig.canvas.draw()
    issues = []
    for ax in _all_axes(fig):
        if not ax.get_visible() or not ax.axison: continue
        polar = ax.name == "polar"
        for axis, lbls in (("x", ax.get_xticklabels()), ("y", ax.get_yticklabels())):
            vis = [t for t in lbls if t.get_visible() and t.get_text()]
            for t in lbls:
                t.set_fontweight("bold")
            axobj = ax.xaxis if axis == "x" else ax.yaxis
            if vis and not polar and axobj.get_visible():
                ticks = (ax.xaxis if axis == "x" else ax.yaxis).get_major_ticks()
                on = any((tk.tick1line.get_visible() or tk.tick2line.get_visible()) and tk.tick1line.get_markersize() > 0 for tk in ticks)
                if not on: issues.append((ax.get_label() or ax.name, axis))
        for t in ax.get_yticklabels(minor=True) + ax.get_xticklabels(minor=True):
            t.set_fontweight("bold")
    return issues


def _pos_root(fig, a):
    return fig.transFigure.inverted().transform_bbox(a.get_window_extent())


def _set_root(fig, a, bb):
    from matplotlib.transforms import Bbox
    disp = fig.transFigure.transform_bbox(Bbox(bb))
    a.set_position(a.get_figure(root=False).transSubfigure.inverted().transform_bbox(disp))


def _foot(fig, axs, r):
    from matplotlib.transforms import Bbox
    inv = fig.transFigure.inverted()
    vis = Bbox.union([inv.transform_bbox(a.get_tightbbox(r)) for a in axs if a.get_visible()])
    pos = Bbox.union([_pos_root(fig, a) for a in axs if a.get_visible()])
    return vis, pos


def calibrate(fig, groups, align, iters=6):
    """Align the outermost visual footprint (tick labels, axis labels, legends, colorbars included) of panels.
    align = {"top": [[keys], ...], "bottom": [...], "left": [...], "right": [...]}; each edge group is moved to its
    outermost member: the group already occupies that band, so panels expand within it and never into another row/column. Returns the residual misalignment (figure fraction)."""
    from matplotlib.transforms import Bbox
    for _ in range(iters):
        fig.canvas.draw(); r = fig.canvas.get_renderer()
        F = {k: _foot(fig, a, r) for k, a in groups.items() if a}
        tgt = {k: dict() for k in F}
        for edge, lists in align.items():
            for keys in lists:
                keys = [k for k in keys if k in F]
                if len(keys) < 2: continue
                v = [getattr(F[k][0], {"top": "y1", "bottom": "y0", "left": "x0", "right": "x1"}[edge]) for k in keys]
                t = max(v) if edge in ("top", "right") else min(v)
                for k in keys: tgt[k][edge] = t
        for k, (vis, pos) in F.items():
            x0, x1, y0, y1 = pos.x0, pos.x1, pos.y0, pos.y1
            if "top" in tgt[k]: y1 = tgt[k]["top"] - (vis.y1 - pos.y1)
            if "bottom" in tgt[k]: y0 = tgt[k]["bottom"] + (pos.y0 - vis.y0)
            if "left" in tgt[k]: x0 = tgt[k]["left"] + (pos.x0 - vis.x0)
            if "right" in tgt[k]: x1 = tgt[k]["right"] - (vis.x1 - pos.x1)
            sx, sy = (x1 - x0) / pos.width, (y1 - y0) / pos.height
            for a in groups[k]:
                p = _pos_root(fig, a)
                _set_root(fig, a, [[x0 + (p.x0 - pos.x0) * sx, y0 + (p.y0 - pos.y0) * sy],
                                   [x0 + (p.x1 - pos.x0) * sx, y0 + (p.y1 - pos.y0) * sy]])
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    F = {k: _foot(fig, a, r)[0] for k, a in groups.items() if a}
    worst = 0.0
    for edge, lists in align.items():
        for keys in lists:
            v = [getattr(F[k], {"top": "y1", "bottom": "y0", "left": "x0", "right": "x1"}[edge]) for k in keys if k in F]
            if len(v) > 1: worst = max(worst, max(v) - min(v))
    return worst


def write_notebook(folder, stem, script_name, keys):
    import json
    cells = [
        ("markdown", [f"# {stem}\n", "\n", f"Regenerates the combined figure, all subfigures and this folder's exports from `code/figures/{script_name}`.\n",
                      "The notebook calls the script, so it always uses the current parameters, palette and labels.\n"]),
        ("code", ["%matplotlib inline\n", f"%run ../../../code/figures/{script_name}\n"]),
        ("code", ["from IPython.display import Image, display\n", f"display(Image(filename='{stem}.png', width=1100))\n"]),
        ("markdown", ["Render a single panel interactively (same drawing function as the export):\n"]),
        ("code", ["import sys, importlib, matplotlib.pyplot as plt\n", "sys.path.insert(0, '../../../code/figures')\n",
                  f"mod = importlib.import_module('{script_name[:-3]}')\n", f"key = '{keys[0]}'  # one of {keys}\n",
                  "fig = plt.figure(figsize=mod.sizes[key])\n", "mod.panels[key](fig, fig.add_gridspec(1, 1, left=0.14, right=0.95, bottom=0.15, top=0.93)[0])\n",
                  "plt.show()\n"]),
    ]
    nb = {"cells": [dict(cell_type=t, metadata={}, source=s, **({"outputs": [], "execution_count": None} if t == "code" else {})) for t, s in cells],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (Path(folder) / f"{stem}_jupyter_code.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


def palette_audit(fig):
    """Colours of all plotted artists (lines, bars, markers, fills, legend proxies) that are not in the palette,
    the ink/muted greys, white or tints of these. Colour-mapped artists (heatmaps, hexbin, colour-coded scatter) are skipped."""
    import numpy as np
    from matplotlib.colors import to_rgba
    from matplotlib.collections import Collection
    ok = [to_rgba(c)[:3] for c in list(C.values()) + [INK, MUTED, "#FFFFFF", "#000000", "#EEEEF4"]]
    def bad(c):
        try: r = to_rgba(c)
        except Exception: return False
        if r[3] == 0: return False
        rgb = np.array(r[:3])
        for o in ok:
            o = np.array(o)
            if np.abs(rgb - o).max() < 0.02: return False
            # alpha-blended or white-mixed tint of a palette colour
            for a in np.linspace(0.05, 0.95, 19):
                if np.abs(rgb - (a * o + (1 - a))).max() < 0.02: return False
        return True
    found = set()
    for ax in _all_axes(fig):
        arts = list(ax.lines) + list(ax.patches) + list(ax.collections)
        leg = ax.get_legend()
        if leg is not None: arts += list(leg.legend_handles)
        for a in arts:
            if isinstance(a, Collection) and a.get_array() is not None: continue
            cols = []
            for getter in ("get_color", "get_facecolor", "get_edgecolor", "get_markerfacecolor", "get_markeredgecolor"):
                if hasattr(a, getter):
                    try: v = getattr(a, getter)()
                    except Exception: continue
                    v = np.atleast_2d(np.array(to_rgba_array_safe(v)))
                    cols += [tuple(np.round(x, 3)) for x in v]
            for c in cols:
                if bad(c): found.add(c)
    return sorted(found)


def to_rgba_array_safe(v):
    from matplotlib.colors import to_rgba_array
    try: return to_rgba_array(v)
    except Exception: return [[0, 0, 0, 0]]


def panel_letters(fig, groups, size=22, pad=0.006):
    """Bold lower-case panel letters at the top-left corner of each panel's tight bounding box, assigned in reading order
    (rows from top to bottom, panels from left to right), so that letters stay consecutive when panels are moved."""
    r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted(); boxes = {}
    for k, axs in groups.items():
        bb = [a.get_tightbbox(r) for a in axs if a.get_visible()]
        bb = [b for b in bb if b is not None and b.width > 0]
        if not bb: continue
        x0 = min(b.x0 for b in bb); y1 = max(b.y1 for b in bb)
        boxes[k] = inv.transform((x0, y1))
    rows = []
    for k, (x, y) in sorted(boxes.items(), key=lambda kv: -kv[1][1]):
        for row in rows:
            if abs(row[0][1] - y) < 0.035: row[1].append((x, k)); break
        else: rows.append(((x, y), [(x, k)]))
    order = [k for _, row in rows for _, k in sorted(row)]
    for i, k in enumerate(order):
        x, y = boxes[k]
        fig.text(x, y + pad, "abcdefghijklmnop"[i], fontsize=size, fontweight="bold", ha="left", va="bottom", color="black")
    return "".join(order)


def save(stem, folder, layout, panels, sizes, figsize, engine="constrained", align=None, sub_rect=None, letters=True):
    """layout(fig) -> {key: SubplotSpec}; panels {key: draw(fig, spec)}; sizes {key: (w, h)}.
    align: edge groups for calibrate(); None for schematic figures."""
    import sys, inspect
    folder = Path(folder); sub = folder / "subfigures"; sub.mkdir(parents=True, exist_ok=True)
    for f in list(folder.glob(f"{stem}*.png")) + list(folder.glob(f"{stem}*.pdf")) + list(sub.glob("*")):
        f.unlink()
    if engine in ("constrained", "subfig"):
        fig = plt.figure(figsize=figsize, layout="constrained")
        fig.get_layout_engine().set(w_pad=0.08, h_pad=0.24, wspace=0.04, hspace=0.04)
    else:
        fig = plt.figure(figsize=figsize)
    specs = layout(fig); groups = {}
    for k, draw in panels.items():
        if engine == "subfig":   # each panel gets its own sub-figure; constrained layout solves its interior
            sfp = fig.add_subfigure(specs[k]); sfp.set_facecolor("none")
            before = set(_all_axes_sf(sfp)); draw(sfp, sfp.add_gridspec(1, 1)[0]); groups[k] = [a for a in sfp.axes if a not in before]
        else:
            before = set(fig.axes); draw(fig, specs[k]); groups[k] = [a for a in fig.axes if a not in before]
    fig.canvas.draw()
    if engine in ("constrained", "subfig"): fig.set_layout_engine("none")
    report = {"font": plt.rcParams["font.family"][0]}
    if align: report["misalignment_in"] = round(calibrate(fig, groups, align) * figsize[0], 3)
    report["tick_issues"] = finalize(fig)
    report["off_palette"] = palette_audit(fig)
    if letters: report["letters"] = panel_letters(fig, groups)
    kw = dict(bbox_inches="tight", pad_inches=0.15)
    fig.savefig(folder / f"{stem}.png", dpi=600, facecolor="white", **kw)
    fig.savefig(folder / f"{stem}.pdf", facecolor="white", **kw)
    plt.close(fig)
    lines = ["# Subfigures", "", "Each panel is rendered independently on its own canvas by the same drawing function used for the",
             "combined figure (not cropped from the combined export).", ""]
    rect = sub_rect or dict(left=0.12, right=0.95, bottom=0.14, top=0.94)
    for k, draw in panels.items():
        if engine in ("constrained", "subfig"):
            sf = plt.figure(figsize=sizes[k], layout="constrained"); sf.get_layout_engine().set(w_pad=0.06, h_pad=0.06)
            draw(sf, sf.add_gridspec(1, 1)[0])
        else:
            sf = plt.figure(figsize=sizes[k]); draw(sf, sf.add_gridspec(1, 1, **rect)[0])
        sf.canvas.draw()
        if engine in ("constrained", "subfig"): sf.set_layout_engine("none")
        iss = finalize(sf)
        if iss: report.setdefault("sub_tick_issues", {})[k] = iss
        kw = dict(bbox_inches="tight", pad_inches=0.12)
        sf.savefig(sub / f"{stem}_{k}.png", dpi=600, facecolor="white", **kw); sf.savefig(sub / f"{stem}_{k}.pdf", facecolor="white", **kw)
        plt.close(sf); lines.append(f"- ({k}) `{stem}_{k}.png`, `{stem}_{k}.pdf`")
    (sub / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    script = Path(sys.modules["__main__"].__file__).name if hasattr(sys.modules["__main__"], "__file__") else f"{stem}.py"
    write_notebook(folder, stem, script, list(panels))
    print(stem, report)
    return report


VARNAME = {"T_in": "T", "pH_in": "pH", "Q_out": "Q$_{out}$", "Supern": "Supern.", "RS": "RAS", "DO1": "DO$_1$", "DO2": "DO$_2$", "DO3": "DO$_3$",
           "Air1": "Air$_1$", "Air2": "Air$_2$", "Air3": "Air$_3$", "aOTE1": "αOTE$_1$", "aOTE2": "αOTE$_2$", "aOTE3": "αOTE$_3$",
           "NH4": "NH$_4^+$", "NO3": "NO$_3^-$", "Cl": "Cl$^-$", "N2O1": "N$_2$O Z1", "N2O2": "N$_2$O Z2", "N2O3": "N$_2$O Z3", "y": "N$_2$O (street)"}
