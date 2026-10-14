"""Seed variability of the residual-anchored GRU (H = 6 h, V1): five additional seeds, same settings as deep_anchored.py."""
import sys, pickle, numpy as np, pandas as pd, torch
import deep_anchored as DA
from sklearn.metrics import r2_score


def summary():
    """Table S6: seed variability of GRU-A and Diebold-Mariano tests against EG-MoE (whole year and months other than January)."""
    from stats_joint import dm_joint
    from sklearn.metrics import mean_squared_error
    d = pickle.load(open("pred_H6_gruA_seeds.pkl", "rb")); b = pickle.load(open("pred_H6_chrono_h1_2022.pkl", "rb"))
    y = b["y"]; t = pd.to_datetime(b["time"]); nj = np.asarray(t.month != 1); eg = np.clip(b["preds"]["EG-MoE"], 0, None)
    P = {42: np.clip(b["preds"]["GRU-A"], 0, None)}; P.update({k: np.clip(v, 0, None) for k, v in d["preds"].items()})
    rows = [dict(seed=s, R2=r2_score(y, p), R2_nonJan=r2_score(y[nj], p[nj]), R2_Jan=r2_score(y[~nj], p[~nj]), RMSE=mean_squared_error(y, p) ** .5,
                 R2_log=r2_score(np.log1p(y), np.log1p(p)), P_vs_EG=dm_joint(y - eg, y - p, t, 6)[1],
                 P_vs_EG_nonJan=dm_joint((y - eg)[nj], (y - p)[nj], t[nj], 6)[1]) for s, p in P.items()]
    pd.DataFrame(rows).to_csv("tables/Table_S6_GRU_seeds.csv", index=False)


if len(sys.argv) > 1 and sys.argv[1] == "summary":
    summary(); sys.exit()
H = 6; F, feats = DA.B.build(H); TR, TE = DA.split(F, H)
orig = torch.manual_seed; out = {}
for seed in [1, 2, 3, 4, 5]:
    torch.manual_seed = lambda s, _seed=seed: orig(_seed)
    p = DA.fit_eval(TR, [TE], ["GRU"])["GRU-A"][0]; out[seed] = p
    print(seed, round(r2_score(TE.target, np.clip(p, 0, None)), 4), flush=True)
    pickle.dump(dict(y=TE.target.values, time=TE.time.values, street=TE.street.values, preds=out), open("pred_H6_gruA_seeds.pkl", "wb"))
summary(); print("DONE")
