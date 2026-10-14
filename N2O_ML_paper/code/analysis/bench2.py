"""Pilot v2: window-mean N2O forecasting (next H hours), 20 baselines + proposed EG-MoE."""
import sys, time, warnings, json
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, f1_score, roc_auc_score
from scipy.optimize import nnls
import lightgbm as lgb, xgboost as xgb, catboost as cb
from prep import engineer, BASE
warnings.filterwarnings("ignore")
RS = 42
REG_VARS = ["T_in", "Q_out", "Supern", "RS", "DO1", "DO2", "DO3", "Air1", "Air2", "Air3", "NH4", "NO3", "Cl"]
LG = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8,
          subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, random_state=RS)


def metrics(y, p, tau, pers=None):
    p = np.clip(p, 0, None)
    d = dict(R2=r2_score(y, p), RMSE=mean_squared_error(y, p) ** .5, MAE=mean_absolute_error(y, p),
             R2_log=r2_score(np.log1p(y), np.log1p(p)), F1=f1_score(y > tau, p > tau),
             CumErr=100 * (p.sum() - y.sum()) / y.sum())
    if pers is not None:
        d["Skill"] = 1 - mean_squared_error(y, p) / mean_squared_error(y, pers)
    return d


def build(H, hist=True, sister=False):
    L = pd.read_pickle("long.pkl")
    F = engineer(L, 1, use_n2o_hist=hist)
    yy = L.groupby("street")["y"]
    F["target"] = yy.transform(lambda x: x[::-1].rolling(H, min_periods=int(np.ceil(.8 * H))).mean()[::-1].shift(-1))
    F["pers"] = L["y"]
    F["pers_day"] = yy.transform(lambda x: x.rolling(H, min_periods=1).mean().shift(24 - H))  # same window yesterday
    F = F.dropna(subset=["target"])
    drop = {"time", "target", "y_now", "pers", "pers_day"}
    if not sister: drop.add("y_sister")
    feats = [c for c in F.columns if c not in drop]
    return F, feats


def base_models():
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "Ridge": make_pipeline(imp(), StandardScaler(), Ridge(alpha=10)),
        "ElasticNet": make_pipeline(imp(), StandardScaler(), ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000)),
        "KNN": make_pipeline(imp(), StandardScaler(), KNeighborsRegressor(n_neighbors=25, weights="distance")),
        "SVR": "svr",
        "DT": make_pipeline(imp(), DecisionTreeRegressor(max_depth=10, min_samples_leaf=50, random_state=RS)),
        "AdaBoost": make_pipeline(imp(), AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators=60, learning_rate=0.05, random_state=RS)),
        "RF": make_pipeline(imp(), RandomForestRegressor(n_estimators=300, min_samples_leaf=10, max_features=0.4, n_jobs=-1, random_state=RS)),
        "ExtraTrees": make_pipeline(imp(), ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=RS)),
        "HistGB": HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=RS),
        "XGBoost": xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.7, min_child_weight=10, n_jobs=-1, random_state=RS),
        "LightGBM": lgb.LGBMRegressor(**LG),
        "CatBoost": cb.CatBoostRegressor(iterations=800, learning_rate=0.05, depth=6, l2_leaf_reg=3, verbose=0, random_seed=RS),
        "MLP": make_pipeline(imp(), StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, early_stopping=True, max_iter=300, random_state=RS)),
    }


class EGMoE:
    """Episode- and regime-Gated Mixture of Experts.
    Experts: persistence anchor, residual-to-persistence GBM, raw-scale GBM, log-scale GBM (smeared), Tweedie GBM.
    Gate: w_j(x) = a_j + b_j * p_episode(x) + sum_k c_jk * pi_k(x) >= 0 (NNLS on forward-chained OOF)."""
    names = ["anchor", "resid", "raw", "log", "tweedie"]

    def __init__(self, tau, k=6, n_folds=4):
        self.tau, self.k, self.n_folds = tau, k, n_folds

    def _fit_experts(self, X, y, a):
        E = {}
        E["resid"] = lgb.LGBMRegressor(**LG).fit(X, y - a)
        E["raw"] = lgb.LGBMRegressor(**LG).fit(X, y)
        m = lgb.LGBMRegressor(**LG).fit(X, np.log1p(y)); E["log"] = m
        E["smear"] = y.mean() / np.expm1(m.predict(X)).clip(0).mean()
        E["tweedie"] = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5, **LG).fit(X, y)
        E["clf"] = lgb.LGBMClassifier(**LG).fit(X, y > self.tau)
        return E

    def _pred_experts(self, E, X, a):
        return {"anchor": a, "resid": a + E["resid"].predict(X), "raw": E["raw"].predict(X),
                "log": np.expm1(E["log"].predict(X)) * E["smear"], "tweedie": E["tweedie"].predict(X),
                "p": E["clf"].predict_proba(X)[:, 1]}

    def _regime(self, X, fit=False):
        Z = X[REG_VARS]
        if fit:
            self.sc = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(Z)
            self.gmm = GaussianMixture(self.k, covariance_type="full", random_state=RS, n_init=3).fit(self.sc.transform(Z))
        return self.gmm.predict_proba(self.sc.transform(Z))

    @staticmethod
    def _simplex(Fm, y, sw=None):
        from scipy.optimize import minimize
        sw = np.ones(len(y)) if sw is None else sw
        f = lambda z: np.sum(sw * (Fm @ (np.exp(z) / np.exp(z).sum()) - y) ** 2)
        z = minimize(f, np.zeros(Fm.shape[1]), method="L-BFGS-B").x
        return np.exp(z) / np.exp(z).sum()

    BINS = [0, .15, .5, 1.01]; LAM = 500

    def _fit_gate(self, P, pi, y):
        Fm = np.column_stack([np.clip(P[n], 0, None) for n in self.names])
        self.w_global = self._simplex(Fm, y); self.W = {}
        for b in range(3):
            m = (P["p"] >= self.BINS[b]) & (P["p"] < self.BINS[b + 1]); nb = m.sum()
            wb = (nb * self._simplex(Fm[m], y[m]) + self.LAM * self.w_global) / (nb + self.LAM)
            Wk = []
            for k in range(self.k):
                sw = pi[m, k]; nk = sw.sum()
                wk = self._simplex(Fm[m], y[m], sw) if nk > 50 else wb
                Wk.append((nk * wk + self.LAM * wb) / (nk + self.LAM))
            self.W[b] = np.array(Wk)

    def gate_weights(self, P, pi):
        Wx = np.zeros((len(pi), len(self.names)))
        for b in range(3):
            m = (P["p"] >= self.BINS[b]) & (P["p"] < self.BINS[b + 1]); Wx[m] = pi[m] @ self.W[b]
        return Wx

    def fit(self, X, y, a, t):
        X = X.reset_index(drop=True); a = np.asarray(a); y = np.asarray(y)
        pi = self._regime(X, fit=True)
        Xa = X.assign(**{f"regime{k}": pi[:, k] for k in range(self.k)})
        order = np.argsort(np.asarray(t)); cuts = np.array_split(order, self.n_folds + 1)
        idx, Ps = [], []
        for i in range(1, self.n_folds + 1):
            tr = np.concatenate(cuts[:i]); va = cuts[i]
            E = self._fit_experts(Xa.iloc[tr], y[tr], a[tr]); Ps.append(self._pred_experts(E, Xa.iloc[va], a[va])); idx.append(va)
        idx = np.concatenate(idx)
        P = {n: np.concatenate([p[n] for p in Ps]) for n in self.names + ["p"]}
        self.oof_p_auc = roc_auc_score(y[idx] > self.tau, P["p"])
        self._fit_gate(P, pi[idx], y[idx])
        self.E = self._fit_experts(Xa, y, a)
        return self

    def predict(self, X, a, parts=False):
        X = X.reset_index(drop=True); pi = self._regime(X)
        Xa = X.assign(**{f"regime{k}": pi[:, k] for k in range(self.k)})
        P = self._pred_experts(self.E, Xa, np.asarray(a))
        Fm = np.column_stack([np.clip(P[n], 0, None) for n in self.names])
        Wx = self.gate_weights(P, pi); out = (Fm * Wx).sum(1)
        P["W"] = Wx
        return (out, P, pi) if parts else out


def run(H=6, scenario="chrono", hist=True, deep=True, models="all", sister=False, test_year=2022):
    F, feats = build(H, hist, sister)
    cut = pd.Timestamp(f"{test_year}-01-01")
    tr = F[F.time < cut - pd.Timedelta(hours=H)]; te = F[(F.time >= cut) & (F.time < pd.Timestamp(f"{test_year + 1}-01-01"))]
    if scenario == "xstreet":
        tr = tr[tr.street == 1]; te = te[te.street == 2]
    med = np.nanmedian(tr.target)
    Xtr, Xte, ytr, yte = tr[feats], te[feats], tr.target.values, te.target.values
    atr, ate = tr.pers.fillna(med).values, te.pers.fillna(med).values
    if not hist:   # analyser-free setting: no N2O persistence anchor available
        atr, ate = np.full(len(tr), ytr.mean() if False else np.nanmean(tr.target)), np.full(len(te), np.nanmean(tr.target))
    tau = np.quantile(ytr, 0.8)
    preds = {"Persistence": ate, "SeasonalNaive": te.pers_day.fillna(med).values, "ClimMean": np.full(len(yte), ytr.mean())}
    if not hist:
        preds = {"ClimMean": preds["ClimMean"]}
    timing = {}
    for name, m in base_models().items():
        if models != "all" and name not in models: continue
        t0 = time.time()
        if name == "SVR":
            ii = np.random.RandomState(RS).choice(len(Xtr), 8000, replace=False)
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVR(C=30, epsilon=1.0, gamma="scale"))
            m.fit(Xtr.iloc[ii], ytr[ii])
        else:
            m.fit(Xtr, ytr)
        preds[name] = m.predict(Xte); timing[name] = time.time() - t0
        print(f"  {name} {timing[name]:.0f}s", flush=True)
    if deep:
        from deep import run_deep
        for name, p in run_deep(F, tr, te, H, hist).items():
            preds[name] = p; print("  ", name, flush=True)
    t0 = time.time()
    mo = EGMoE(tau).fit(Xtr, ytr, atr, tr.time.values)
    preds["EG-MoE"], P, pi = mo.predict(Xte, ate, parts=True)
    timing["EG-MoE"] = time.time() - t0
    for n in ["resid", "raw", "log", "tweedie"]:
        preds[f"[expert] {n}"] = P[n]
    print(f"  EG-MoE {timing['EG-MoE']:.0f}s  OOF AUC={mo.oof_p_auc:.3f}", flush=True)
    pers = ate if hist else None
    R = pd.DataFrame([dict(H=H, scenario=scenario, hist=hist, model=n, **metrics(yte, p, tau, pers)) for n, p in preds.items()])
    return R, dict(F=F, te=te, tr=tr, preds=preds, model=mo, feats=feats, tau=tau, P=P, pi=pi)


if __name__ == "__main__":
    H = int(sys.argv[1]); sc = sys.argv[2]; hist = sys.argv[3] == "1"; deep = sys.argv[4] == "1"
    R, out = run(H, sc, hist, deep)
    R.to_csv(f"res2_H{H}_{sc}_hist{int(hist)}.csv", index=False)
    pd.to_pickle({k: out[k] for k in ["preds", "tau"]} | {"y": out["te"].target.values, "time": out["te"].time.values, "street": out["te"].street.values}, f"pred_H{H}_{sc}_hist{int(hist)}.pkl")
    print(R.round(3).sort_values("R2", ascending=False).to_string())
