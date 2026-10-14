import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from figstyle import *
configure()
import fig5 as fig6
ROOT = os.path.join(HERE, "..", "analysis"); OUTDIR = os.path.join(HERE, "..", "..", "figures", "supplementary", "FigS17_coverage_dominance")
panels = {"a": fig6.p_conformal, "b": fig6.p_dom}
sizes = {"a": (4.4, 5.6), "b": (7.6, 4.6)}

def layout(fig):
    gs = fig.add_gridspec(1, 2, width_ratios=[4.4, 7.6]); return {"a": gs[0], "b": gs[1]}

if __name__ == "__main__":
    save("FigS17_coverage_dominance", OUTDIR, layout, panels, sizes, figsize=(13, 5.4), engine="subfig", align={"top": [["a", "b"]]})
