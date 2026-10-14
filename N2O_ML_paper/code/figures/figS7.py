import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from figstyle import *
configure()
import fig2
import fig6 as fig3
ROOT = os.path.join(HERE, "..", "analysis"); OUTDIR = os.path.join(HERE, "..", "..", "figures", "supplementary", "FigS07_correlations_embedding")
panels = {"a": fig2.p_corr, "b": fig3.p_umap}
sizes = {"a": (7.2, 6.6), "b": (5.6, 4.6)}

def layout(fig):
    gs = fig.add_gridspec(1, 2, width_ratios=[7.2, 5.6]); return {"a": gs[0], "b": gs[1]}

if __name__ == "__main__":
    save("FigS07_correlations_embedding", OUTDIR, layout, panels, sizes, figsize=(15, 6.4), engine="subfig", align={"top": [["a", "b"]]})
