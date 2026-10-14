"""Smooth CATE for Fig. 8d-f: LinearDML with spline / tensor-spline featurizers (replaces the jagged forest curves)."""
import numpy as np, pandas as pd, pickle, warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb
from econml.dml import LinearDML
from sklearn.preprocessing import SplineTransformer, PolynomialFeatures
from sklearn.pipeline import make_pipeline
exec(open("figdata_dml.py").read().split("lgr = dict")[0])          # same data preparation as figdata_dml.py
lgr = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, verbose=-1, random_state=0)
out = pickle.load(open("figdata/dml_fig8.pkl", "rb"))
for key, (t, extra) in TREAT.items():
    D = F.dropna(subset=["Y", t] + W_env + extra).reset_index(drop=True)
    Y, T, W = D["Y"].values, D[t].values, D[W_env + extra]; wk = D["week"].values
    X1 = D[["T_in_lag1"]]
    e1 = LinearDML(model_y=lgb.LGBMRegressor(**lgr), model_t=lgb.LGBMClassifier(**lgr), discrete_treatment=True, cv=5, random_state=0,
                   featurizer=SplineTransformer(n_knots=5, degree=3, extrapolation="linear"))
    e1.fit(Y, T, X=X1, W=W, groups=wk)
    tg = np.linspace(np.nanpercentile(X1.values, 2), np.nanpercentile(X1.values, 98), 60)
    Xc = pd.DataFrame({"T_in_lag1": tg}); e = e1.effect(Xc); l, u = e1.effect_interval(Xc, alpha=0.05)
    out["cate_curve"][key] = dict(T=tg, eff=e.ravel(), lo=l.ravel(), hi=u.ravel(), Tobs=X1.values.ravel())
    X2 = D[["T_in_lag1", "NH4_lag1"]]
    e2 = LinearDML(model_y=lgb.LGBMRegressor(**lgr), model_t=lgb.LGBMClassifier(**lgr), discrete_treatment=True, cv=5, random_state=0,
                   featurizer=make_pipeline(SplineTransformer(n_knots=4, degree=2), PolynomialFeatures(2, interaction_only=True, include_bias=False)))
    e2.fit(Y, T, X=X2, W=W, groups=wk)
    ng = np.linspace(np.nanpercentile(X2.NH4_lag1, 2), np.nanpercentile(X2.NH4_lag1, 98), 50); tg2 = np.linspace(tg.min(), tg.max(), 50)
    TT, NN = np.meshgrid(tg2, ng)
    out["cate_surface"][key] = dict(T=TT, N=NN, eff=e2.effect(pd.DataFrame({"T_in_lag1": TT.ravel(), "NH4_lag1": NN.ravel()})).reshape(TT.shape), Xobs=X2.values)
    D["theta"] = e2.effect(X2).ravel(); tl, tu = e2.effect_interval(X2, alpha=0.05)
    D["theta_lo"], D["theta_hi"] = tl.ravel(), tu.ravel()
    out[f"D_{key}"] = D[["time", "street", "theta", "theta_lo", "theta_hi", t]]
    print(key, "ATE spline-1D", float(e1.ate(X1)), "ATE tensor", float(e2.ate(X2)), flush=True)
pickle.dump(out, open("figdata/dml_fig8.pkl", "wb")); print("DONE")
