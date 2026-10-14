"""Fig. 8 inputs: DML ATE with weekly block-bootstrap CIs, CATE curves/surfaces, mitigation scenario."""
import numpy as np, pandas as pd, pickle, warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb
from econml.dml import LinearDML, CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from bench2 import build
import os
os.makedirs("figdata", exist_ok=True)

H = 6
F, feats = build(H)
F = F.copy(); g = F.groupby("street")
lagc = ["T_in", "Q_out", "NH4", "NO3", "Cl", "RS", "pH_in", "DO2", "DO3", "Air2", "Air3", "Supern", "swing_on"]
for c in lagc:
    F[c + "_lag1"] = g[c].shift(1)
for c in ["T_in_m168", "Q_out_m72", "NH4_m168", "Cl_m72", "y_m24", "y_m168"]:
    F[c] = g[c].shift(1)
W_env = ["T_in_lag1", "T_in_m168", "Q_out_lag1", "Q_out_m72", "NH4_lag1", "NH4_m168", "NO3_lag1", "Cl_lag1", "Cl_m72",
         "RS_lag1", "pH_in_lag1", "hour_sin", "hour_cos", "doy_sin", "doy_cos", "weekend", "yl1", "y_m24", "y_m168"]
F["Y"] = np.log1p(F["target"])
F["week"] = (F.time.dt.isocalendar().week.astype(int) + 100 * F.time.dt.year).values * 10 + F.street
F["Supern_open"] = (F["Supern"] > 5).astype(float)
TREAT = {
    "swing": ("swing_on", ["DO2_lag1", "DO3_lag1", "Air2_lag1", "Air3_lag1", "Supern_lag1"]),
    "supern": ("Supern_open", ["DO2_lag1", "DO3_lag1", "Air2_lag1", "Air3_lag1", "swing_on_lag1"]),
}
lgr = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, verbose=-1, random_state=0)
rng = np.random.default_rng(7)
out = {"ate": [], "cate_curve": {}, "cate_surface": {}, "overlap": {}}
for key, (t, extra) in TREAT.items():
    D = F.dropna(subset=["Y", t] + W_env + extra).reset_index(drop=True)
    Y, T, W = D["Y"].values, D[t].values, D[W_env + extra]
    X = D[["T_in_lag1", "NH4_lag1"]]
    wk = D["week"].values; uw = np.unique(wk)
    for name, my, mt in [("LinearDML-GBM", lgb.LGBMRegressor(**lgr), lgb.LGBMClassifier(**lgr)),
                         ("LinearDML-RF", RandomForestRegressor(150, min_samples_leaf=20, n_jobs=-1, random_state=0),
                          RandomForestClassifier(150, min_samples_leaf=20, n_jobs=-1, random_state=0))]:
        est = LinearDML(model_y=my, model_t=mt, discrete_treatment=True, cv=5, random_state=0)
        est.fit(Y, T, X=X, W=W, groups=wk, cache_values=True)
        ate = float(est.ate(X))
        Yr, Tr, Xr, _ = est.residuals_
        Tr = Tr.ravel(); Yr = Yr.ravel(); Z = np.column_stack([np.ones(len(Xr)), Xr])
        # final stage refit on weekly block bootstrap of cross-fitted residuals
        idx_by_w = {w: np.where(wk == w)[0] for w in uw}
        c0, *_ = np.linalg.lstsq(Z * Tr[:, None], Yr, rcond=None); print("check full-sample", float(np.mean(Z @ c0)), ate, flush=True)
        boots = []
        for b in range(400):
            ws = rng.choice(uw, len(uw), replace=True); ii = np.concatenate([idx_by_w[w] for w in ws])
            A = Z[ii] * Tr[ii, None]; coef, *_ = np.linalg.lstsq(A, Yr[ii], rcond=None)
            boots.append(float(np.mean(Z @ coef)))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        nlo, nhi = [float(v) for v in est.ate_interval(X, alpha=0.05)]
        out["ate"].append(dict(treatment=key, estimator=name, n=len(D), ate=ate, lo_block=lo, hi_block=hi, lo_naive=nlo, hi_naive=nhi))
        print(out["ate"][-1], flush=True)
        if name == "LinearDML-GBM":
            ps = lgb.LGBMClassifier(**lgr).fit(W.assign(X1=X.iloc[:, 0], X2=X.iloc[:, 1]), T).predict_proba(W.assign(X1=X.iloc[:, 0], X2=X.iloc[:, 1]))[:, 1]
            out["overlap"][key] = (ps, T)
    cf = CausalForestDML(model_y=lgb.LGBMRegressor(**lgr), model_t=lgb.LGBMClassifier(**lgr), discrete_treatment=True,
                         cv=5, n_estimators=400, min_samples_leaf=50, random_state=0)
    cf.fit(Y, T, X=X, W=W, groups=wk)
    ate = float(cf.ate(X)); lo, hi = [float(v) for v in cf.ate_interval(X, alpha=0.05)]
    out["ate"].append(dict(treatment=key, estimator="CausalForestDML", n=len(D), ate=ate, lo_block=np.nan, hi_block=np.nan, lo_naive=lo, hi_naive=hi))
    print(out["ate"][-1], flush=True)
    tg = np.linspace(np.nanpercentile(X.iloc[:, 0], 2), np.nanpercentile(X.iloc[:, 0], 98), 40)
    ng = np.linspace(np.nanpercentile(X.iloc[:, 1], 2), np.nanpercentile(X.iloc[:, 1], 98), 40)
    Xc = pd.DataFrame({"T_in_lag1": tg, "NH4_lag1": np.full(40, np.nanmedian(X.iloc[:, 1]))})
    e = cf.effect(Xc); l, u = cf.effect_interval(Xc, alpha=0.05)
    out["cate_curve"][key] = dict(T=tg, eff=e.ravel(), lo=l.ravel(), hi=u.ravel(), Tobs=X.iloc[:, 0].values)
    TT, NN = np.meshgrid(tg, ng)
    Xs = pd.DataFrame({"T_in_lag1": TT.ravel(), "NH4_lag1": NN.ravel()})
    out["cate_surface"][key] = dict(T=TT, N=NN, eff=cf.effect(Xs).reshape(TT.shape), Xobs=X.values)
    # individual effects for scenario
    D["theta"] = cf.effect(X).ravel(); tl, tu = cf.effect_interval(X, alpha=0.05)
    D["theta_lo"], D["theta_hi"] = tl.ravel(), tu.ravel()
    out[f"D_{key}"] = D[["time", "street", "theta", "theta_lo", "theta_hi", t]]
pd.DataFrame(out["ate"]).to_csv("figdata/dml_ate_block.csv", index=False)
pickle.dump(out, open("figdata/dml_fig8.pkl", "wb"))
print("DONE")
