import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from figstyle import *
configure()
import fig7
ROOT = os.path.join(HERE, "..", "analysis"); OUTDIR = os.path.join(HERE, "..", "..", "figures", "supplementary", "FigS18_shap_dependence")
panels = {"a": fig7.p_inter}
sizes = {"a": (9.0, 4.6)}

def layout(fig):
    gs = fig.add_gridspec(1, 1); return {"a": gs[0]}

if __name__ == "__main__":
    save("FigS18_shap_dependence", OUTDIR, layout, panels, sizes, figsize=(14, 4.8), engine="subfig", align=None)
