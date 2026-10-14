"""Pilot DML: adjusted effects of three operating levers on next-6-h N2O (log scale)."""
import numpy as np, pandas as pd, warnings, json
warnings.filterwarnings("ignore")
import lightgbm as lgb
from econml.dml import LinearDML, CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from bench2 import build

H = 6
F, feats = build(H)
F = F.copy()
g = F.groupby("street")
# confounders measured before the decision hour (t-1) to avoid post-treatment adjustment
lagc = ["T_in", "Q_out", "NH4", "NO3", "Cl", "RS", "pH_in", "DO2", "DO3", "Air2", "Air3", "Supern", "swing_on"]
for c in lagc:
    F[c + "_lag1"] = g[c].shift(1)
# rolling confounders shifted by 1 h so that no adjustment variable contains the treatment hour (avoid post-treatment adjustment)
for c in ["T_in_m168", "Q_out_m72", "NH4_m168", "Cl_m72", "y_m24", "y_m168"]:
    F[c] = g[c].shift(1)
W_env = ["T_in_lag1", "T_in_m168", "Q_out_lag1", "Q_out_m72", "NH4_lag1", "NH4_m168", "NO3_lag1", "Cl_lag1", "Cl_m72",
         "RS_lag1", "pH_in_lag1", "hour_sin", "hour_cos", "doy_sin", "doy_cos", "weekend", "yl1", "y_m24", "y_m168"]
F["Y"] = np.log1p(F["target"])
F["week"] = (F.time.dt.isocalendar().week.astype(int) + 100 * F.time.dt.year).values * 10 + F.street
F["Supern10"] = F["Supern"] / 10
F["Supern_open"] = (F["Supern"] > 5).astype(float)
TREAT = {
    "Swing-zone aeration (on/off)": ("swing_on", True, ["DO2_lag1", "DO3_lag1", "Air2_lag1", "Air3_lag1", "Supern_lag1"]),
    "Supernatant valve (+10 %)": ("Supern10", False, ["DO2_lag1", "DO3_lag1", "Air2_lag1", "Air3_lag1", "swing_on_lag1"]),
    "Supernatant valve open (>5 %)": ("Supern_open", True, ["DO2_lag1", "DO3_lag1", "Air2_lag1", "Air3_lag1", "swing_on_lag1"]),
    "DO zone 2 (+1 mg/L)": ("DO2", False, ["DO3_lag1", "Air3_lag1", "Supern_lag1", "swing_on_lag1"]),
}
lgr = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, verbose=-1, random_state=0)
rows = []
for lab, (t, disc, extra) in TREAT.items():
    D = F.dropna(subset=["Y", t] + W_env + extra).reset_index(drop=True)
    Y, T, W = D["Y"].values, D[t].values, D[W_env + extra]
    X = D[["T_in_lag1", "NH4_lag1"]]
    mt = lgb.LGBMClassifier(**lgr) if disc else lgb.LGBMRegressor(**lgr)
    for est_name, est in [
        ("LinearDML-GBM", LinearDML(model_y=lgb.LGBMRegressor(**lgr), model_t=mt, discrete_treatment=disc, cv=5, random_state=0)),
        ("LinearDML-RF", LinearDML(model_y=RandomForestRegressor(150, min_samples_leaf=20, n_jobs=-1, random_state=0),
                                   model_t=(RandomForestClassifier if disc else RandomForestRegressor)(150, min_samples_leaf=20, n_jobs=-1, random_state=0),
                                   discrete_treatment=disc, cv=5, random_state=0)),
        ("CausalForestDML", CausalForestDML(model_y=lgb.LGBMRegressor(**lgr), model_t=mt, discrete_treatment=disc, cv=5, n_estimators=300, min_samples_leaf=50, random_state=0)),
    ]:
        est.fit(Y, T, X=X, W=W, groups=D["week"].values)
        ate = float(est.ate(X)); lo, hi = [float(v) for v in est.ate_interval(X, alpha=0.05)]
        rows.append(dict(treatment=lab, estimator=est_name, n=len(D), ATE_log=ate, lo=lo, hi=hi, pct=100 * (np.exp(ate) - 1)))
        print(rows[-1], flush=True)
        if est_name == "LinearDML-GBM":
            # refutations: placebo treatment (block-permuted), random common cause
            rng = np.random.RandomState(1)
            wk = D["week"].values; uw = np.unique(wk); perm = dict(zip(uw, rng.permutation(uw)))
            # block placebo: whole weekly treatment trajectories are reassigned to other weeks (keeps autocorrelation)
            blocks = {w: T[wk == w] for w in uw}; Tp = np.empty_like(T, dtype=float)
            for w in uw:
                src = blocks[perm[w]]; idx = np.where(wk == w)[0]
                Tp[idx] = np.resize(src, len(idx))
            e2 = LinearDML(model_y=lgb.LGBMRegressor(**lgr), model_t=mt, discrete_treatment=disc, cv=5, random_state=0).fit(Y, Tp, X=X, W=W, groups=wk)
            W2 = W.assign(rcc=rng.randn(len(W)))
            e3 = LinearDML(model_y=lgb.LGBMRegressor(**lgr), model_t=mt, discrete_treatment=disc, cv=5, random_state=0).fit(Y, T, X=X, W=W2, groups=wk)
            half = np.isin(wk, uw[rng.rand(len(uw)) < 0.5])
            e4 = LinearDML(model_y=lgb.LGBMRegressor(**lgr), model_t=mt, discrete_treatment=disc, cv=5, random_state=0).fit(Y[half], T[half], X=X[half], W=W[half], groups=wk[half])
            for rn, e, XX in [("placebo", e2, X), ("random_common_cause", e3, X), ("subset50", e4, X[half])]:
                rows.append(dict(treatment=lab, estimator="refute:" + rn, n=len(D), ATE_log=float(e.ate(XX)), lo=np.nan, hi=np.nan, pct=100 * (np.exp(float(e.ate(XX))) - 1)))
                print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("dml_pilot_v2.csv", index=False)
