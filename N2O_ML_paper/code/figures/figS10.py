import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from figstyle import *
configure()
import fig3 as fig4
ROOT = os.path.join(HERE, "..", "analysis"); OUTDIR = os.path.join(HERE, "..", "..", "figures", "supplementary", "FigS10_taylor_bias")
panels = {"a": fig4.p_taylor, "b": fig4.p_bubble, "c": fig4.p_topsis}
sizes = {"a": (5.8, 5.4), "b": (5.8, 5.0), "c": (5.6, 4.6)}

def layout(fig):
    gs = fig.add_gridspec(1, 3, width_ratios=[5.8, 5.8, 5.6]); return {"a": gs[0], "b": gs[1], "c": gs[2]}

if __name__ == "__main__":
    save("FigS10_taylor_bias", OUTDIR, layout, panels, sizes, figsize=(18.5, 5.4), engine="subfig", align={"top": [["a", "b", "c"]]})
