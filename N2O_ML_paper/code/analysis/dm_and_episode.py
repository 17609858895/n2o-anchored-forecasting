"""Diebold-Mariano tests (squared loss, HAC lag 4H) and January-episode split. Run after driver.py."""
import pickle, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import r2_score
from stats_joint import dm_joint

def dm(e1, e2, h):
    d = e1 ** 2 - e2 ** 2; n = len(d); db = d.mean(); L = 4 * h
    g = [np.mean((d[k:] - db) * (d[:n - k] - db)) for k in range(L + 1)]
    v = g[0] + 2 * sum((1 - k / (L + 1)) * g[k] for k in range(1, L + 1)); s = db / np.sqrt(v / n)
    return s, 2 * (1 - stats.norm.cdf(abs(s)))

rows = []
for k in ["H1_chrono_h1_2022", "H3_chrono_h1_2022", "H6_chrono_h1_2022", "H12_chrono_h1_2022", "H24_chrono_h1_2022", "H6_chrono_h1_2021", "H6_xstreet_h1_2022"]:
    d = pickle.load(open(f"pred_{k}.pkl", "rb")); tm = pd.to_datetime(d["time"]); o = np.lexsort((tm.values, d["street"]))
    y = d["y"][o]; P = {m: np.clip(d["preds"][m], 0, None)[o] for m in ["EG-MoE", "LightGBM", "Persistence", "[expert] resid", "GRU-A"]}
    h = int(k.split("_")[0][1:]); jan = (tm.month == 1)[o] if k.endswith("2022") else np.zeros(len(y), bool)
    r = {"scenario": k}
    for b in ["LightGBM", "Persistence", "[expert] resid", "GRU-A"]:
        r[f"DM_vs_{b}"], r[f"p_vs_{b}"] = dm_joint(y - P["EG-MoE"], y - P[b], tm[o], h)
    if jan.any():
        for m in P: r[f"R2_exJan_{m}"] = r2_score(y[~jan], P[m][~jan]); r[f"R2_Jan_{m}"] = r2_score(y[jan], P[m][jan])
    rows.append(r)
pd.DataFrame(rows).to_csv("dm_episode_split.csv", index=False)
