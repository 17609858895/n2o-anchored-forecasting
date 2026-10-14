"""Pilot interpretation on the H=6 EG-MoE: SHAP of residual expert and episode gate; ALE vs PDP; analyser-free episode AUC."""
import numpy as np, pandas as pd, lightgbm as lgb, shap, pickle, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, average_precision_score
from bench2 import build, LG
mo = pickle.load(open("egmoe_H6.pkl", "rb")); feats = pickle.load(open("feats.pkl", "rb"))
te = pd.read_pickle("te_H6.pkl"); tr = pd.read_pickle("tr_H6.pkl")
X = te[feats].reset_index(drop=True); pi = mo._regime(X); Xa = X.assign(**{f"regime{k}": pi[:, k] for k in range(mo.k)})
Xs = Xa.sample(3000, random_state=0)
out = {}
for name, model in [("resid", mo.E["resid"]), ("gate_clf", mo.E["clf"])]:
    sv = shap.TreeExplainer(model).shap_values(Xs)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.Series(np.abs(sv).mean(0), index=Xs.columns).sort_values(ascending=False)
    out[name] = imp; print(name); print(imp.head(15).round(4))
pd.DataFrame(out).to_csv("shap_pilot.csv")
# ops-only share of SHAP mass in the residual expert
n2o_cols = [c for c in Xs.columns if c.startswith(("yl", "y_", "N2O"))]
for name in out:
    s = out[name]; print(name, "N2O-history share", round(s[n2o_cols].sum() / s.sum(), 3))

def ale(model, X, f, bins=20):
    q = np.unique(np.nanquantile(X[f], np.linspace(0, 1, bins + 1))); x = X[f].values
    k = np.clip(np.searchsorted(q, x, side="right") - 1, 0, len(q) - 2); eff = []
    for j in range(len(q) - 1):
        S = X[k == j]
        if len(S) == 0: eff.append(0); continue
        lo, hi = S.copy(), S.copy(); lo[f] = q[j]; hi[f] = q[j + 1]; eff.append(np.mean(model.predict(hi) - model.predict(lo)))
    a = np.concatenate([[0], np.cumsum(eff)]); return q, a - a.mean()
def pdp(model, X, f, grid):
    o = [];
    for v in grid: Z = X.copy(); Z[f] = v; o.append(model.predict(Z).mean())
    o = np.array(o); return o - o.mean()
rows = []
for f in ["T_in", "DO2", "DO3", "NH4", "NO3", "Air1", "Supern", "Cl", "Q_out"]:
    q, a = ale(mo.E["resid"], Xs, f); p = pdp(mo.E["resid"], Xs, f, q)
    rows.append(dict(feature=f, ale_range=a.ptp() if hasattr(a,'ptp') else np.ptp(a), pdp_range=np.ptp(p), ale_pdp_r=np.corrcoef(a, p)[0, 1]))
print(pd.DataFrame(rows).round(3))
# analyser-free episode classification (operational data only)
F, f0 = build(6, hist=False); cut = pd.Timestamp("2022-01-01")
a_tr = F[F.time < cut - pd.Timedelta(hours=6)]; a_te = F[F.time >= cut]; tau = np.quantile(a_tr.target, .8)
c = lgb.LGBMClassifier(**LG).fit(a_tr[f0], a_tr.target > tau); pp = c.predict_proba(a_te[f0])[:, 1]
print("ops-only episode AUC", round(roc_auc_score(a_te.target > tau, pp), 3), "AP", round(average_precision_score(a_te.target > tau, pp), 3), "base rate", round((a_te.target > tau).mean(), 3))
print("EG-MoE gate test AUC", round(roc_auc_score(te.target.values > mo.tau, mo._pred_experts(mo.E, Xa, te.pers.fillna(te.target.median()).values)["p"]), 3))
