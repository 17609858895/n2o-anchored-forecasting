"""Street-month task-level R2 (H=6) and monitoring-campaign length audit. Run after driver.py."""
import pickle, numpy as np, pandas as pd
from sklearn.metrics import r2_score
M = ["EG-MoE", "GRU-A", "LightGBM", "Persistence", "CatBoost", "XGBoost", "ExtraTrees"]
d = pickle.load(open("pred_H6_chrono_h1_2022.pkl", "rb"))
df = pd.DataFrame({"y": d["y"], "t": pd.to_datetime(d["time"]), "s": d["street"]})
for m in M: df[m] = np.clip(d["preds"][m], 0, None)
rows = [{"month": mo, "street": s, **{m: r2_score(g.y, g[m]) for m in M}}
        for (mo, s), g in df.groupby([df.t.dt.month, "s"]) if len(g) >= 200]
pd.DataFrame(rows).to_csv("monthly_r2_H6.csv", index=False)
# campaign audit: summarise the campaigns drawn in figdata (street-years with >= 80% valid hours)
E = pd.read_csv("figdata/campaign_errors.csv")
out = E.groupby(["year", "street", "weeks"]).err.agg(med_abs=lambda x: np.median(np.abs(x)), p5=lambda x: np.percentile(x, 5),
                                                    p95=lambda x: np.percentile(x, 95), n="size").reset_index()
out.to_csv("campaign_length_audit.csv", index=False)
