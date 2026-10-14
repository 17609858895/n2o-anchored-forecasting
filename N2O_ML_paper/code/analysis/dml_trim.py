"""(1) Weekly block-bootstrap CIs with calendar weeks shared by both streets; (2) trimmed-overlap sensitivity (0.05 <= e <= 0.95).
Same data preparation and nuisance models as figdata_dml.py; updates lo_block/hi_block in figdata/dml_ate_block.csv, writes figdata/dml_trim.csv."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
exec(open("figdata_dml.py").read().split("lgr = dict")[0])
import lightgbm as lgb
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
lgr = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, verbose=-1, random_state=0)
cal_week = lambda t: (t.dt.isocalendar().year.astype(int) * 100 + t.dt.isocalendar().week.astype(int)).values
rng = np.random.default_rng(7)


def block_ci(Z, Tr, Yr, blocks, nb=400):
    ub = np.unique(blocks); ix = {w: np.where(blocks == w)[0] for w in ub}; bs = []
    for _ in range(nb):
        ii = np.concatenate([ix[w] for w in rng.choice(ub, len(ub), replace=True)])
        coef, *_ = np.linalg.lstsq(Z[ii] * Tr[ii, None], Yr[ii], rcond=None); bs.append(float(np.mean(Z @ coef)))
    return np.percentile(bs, [2.5, 97.5])


def fit(D, t, extra, name):
    Y, T, W, X = D["Y"].values, D[t].values, D[W_env + extra], D[["T_in_lag1", "NH4_lag1"]]
    my, mt = ((lgb.LGBMRegressor(**lgr), lgb.LGBMClassifier(**lgr)) if name.endswith("GBM") else
              (RandomForestRegressor(150, min_samples_leaf=20, n_jobs=1, random_state=0), RandomForestClassifier(150, min_samples_leaf=20, n_jobs=1, random_state=0)))
    est = LinearDML(model_y=my, model_t=mt, discrete_treatment=True, cv=5, random_state=0)
    est.fit(Y, T, X=X, W=W, groups=D["week"].values, cache_values=True)
    Yr, Tr, Xr, _ = est.residuals_; Tr, Yr = Tr.ravel(), Yr.ravel(); Z = np.column_stack([np.ones(len(Xr)), Xr])
    lo, hi = block_ci(Z, Tr, Yr, cal_week(D["time"]))
    return float(est.ate(X)), lo, hi, T - Tr      # cross-fitted propensity e = T - residual


ate = pd.read_csv("figdata/dml_ate_block.csv"); rows = []
for key, (t, extra) in TREAT.items():
    D = F.dropna(subset=["Y", t] + W_env + extra).reset_index(drop=True)
    for name in ["LinearDML-GBM", "LinearDML-RF"]:
        a, lo, hi, e = fit(D, t, extra, name)
        m = (ate.treatment == key) & (ate.estimator == name); ate.loc[m, ["lo_block", "hi_block"]] = [lo, hi]
        print(key, name, "full", round(a, 4), "orig", round(float(ate.loc[m, "ate"].iloc[0]), 4), "calweek CI", round(lo, 3), round(hi, 3), flush=True)
        if name == "LinearDML-GBM":
            keep = (e >= 0.05) & (e <= 0.95)
            at, lt, ht, _ = fit(D[keep].reset_index(drop=True), t, extra, name)
            rows.append(dict(treatment=key, estimator=name, n_full=len(D), n_trim=int(keep.sum()), share_kept=keep.mean(),
                             share_low=(e < 0.05).mean(), share_high=(e > 0.95).mean(), ate_full=a, ate_trim=at, lo_trim=lt, hi_trim=ht))
            print(rows[-1], flush=True)
ate.to_csv("figdata/dml_ate_block.csv", index=False); pd.DataFrame(rows).to_csv("figdata/dml_trim.csv", index=False); print("DONE")
