"""Mean EG-MoE expert weights at Avedoere (test year), overall and by episode-probability bin.
Run from code/analysis after aved_replication.py. Writes ../../results/aved_gate_weights.csv"""
import numpy as np, pandas as pd
import bench2 as B, aved_replication as A
h, _ = A.load(); Ff, feats = A.features(h); F = Ff.dropna(subset=["target"])
tr = F[F.time < A.SPLIT - pd.Timedelta(hours=A.H)]; te = F[F.time >= A.SPLIT]
med = np.nanmedian(tr.target); ytr = tr.target.values
atr, ate = tr.pers.fillna(med).values, te.pers.fillna(med).values
B.REG_VARS = A.REG
mo = B.EGMoE(np.quantile(ytr, 0.8)).fit(tr[feats], ytr, atr, tr.time.values)
out, P, pi = mo.predict(te[feats], ate, parts=True)
W = pd.DataFrame(P["W"], columns=mo.names); W["bin"] = pd.cut(P["p"], [0, .15, .5, 1.01], right=False, labels=["p<0.15", "0.15-0.5", "p>=0.5"])
S = pd.concat([W[mo.names].mean().rename("all"), W.groupby("bin", observed=True)[mo.names].mean().T], axis=1).T
S["share_of_hours"] = [1.0] + list(W.bin.value_counts(normalize=True).reindex(["p<0.15", "0.15-0.5", "p>=0.5"]).values)
S.round(3).to_csv("../../results/aved_gate_weights.csv"); print(S.round(3))
