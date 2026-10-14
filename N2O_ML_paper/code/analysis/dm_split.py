"""Diebold-Mariano tests of EG-MoE against the main competitors for the whole test year, January and the other months (streets averaged per hour)."""
import pickle, numpy as np, pandas as pd
from stats_joint import dm_joint
NAME = {"[expert] resid": "Residual expert"}; rows = []
for H in [1, 3, 6, 12, 24]:
    d = pickle.load(open(f"pred_H{H}_chrono_h1_2022.pkl", "rb")); y = d["y"]; t = pd.to_datetime(d["time"]); jan = np.asarray(t.month == 1)
    P = {m: np.clip(d["preds"][m], 0, None) for m in ["EG-MoE", "GRU-A", "LightGBM", "Persistence", "[expert] resid"]}
    for b in ["GRU-A", "LightGBM", "Persistence", "[expert] resid"]:
        r = dict(H=H, Comparison=f"EG-MoE vs {NAME.get(b, b)}")
        for lab, m in [("All", np.ones(len(y), bool)), ("January", jan), ("Other months", ~jan)]:
            s, p = dm_joint((y - P["EG-MoE"])[m], (y - P[b])[m], t[m], H); r[f"DM {lab}"] = round(s, 2); r[f"P {lab}"] = round(p, 4)
        rows.append(r)
pd.DataFrame(rows).to_csv("tables/Table_S7_DM_split.csv", index=False); print(pd.DataFrame(rows).to_string())
