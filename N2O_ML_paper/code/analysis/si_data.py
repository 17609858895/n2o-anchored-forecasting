"""Supporting-Information analyses (Fig. S1-S15, Table S1-S5 inputs). Usage: python si_data.py S1 S2 ...
Each section caches to figdata/si/<S>.pkl. H = 6 h, V1 split unless stated."""
import sys, os, time, pickle, warnings, json
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.metrics import r2_score, mean_squared_error, f1_score, roc_auc_score, roc_curve
import lightgbm as lgb
import bench2 as B
from bench2 import build, LG, EGMoE, metrics
from prep import RAW, SHARED, STREET, BASE
os.makedirs("figdata/si", exist_ok=True)
REGV0 = list(B.REG_VARS)
H = 6
CUT = pd.Timestamp("2022-01-01")
r2 = lambda y, p: r2_score(y, np.clip(p, 0, None))
rmse = lambda y, p: mean_squared_error(y, np.clip(p, 0, None)) ** .5


def dump(name, obj):
    pickle.dump(obj, open(f"figdata/si/{name}.pkl", "wb")); print(name, "saved", flush=True)


def split(F, test_year=2022):
    cut = pd.Timestamp(f"{test_year}-01-01")
    tr = F[F.time < cut - pd.Timedelta(hours=H)]; te = F[(F.time >= cut) & (F.time < pd.Timestamp(f"{test_year + 1}-01-01"))]
    return tr, te


def fit_egmoe(tr, te, feats, tau=None, **kw):
    med = np.nanmedian(tr.target); tau = np.quantile(tr.target, 0.8) if tau is None else tau
    mo = EGMoE(tau, **kw).fit(tr[feats], tr.target.values, tr.pers.fillna(med).values, tr.time.values)
    p, P, pi = mo.predict(te[feats], te.pers.fillna(med).values, parts=True)
    return mo, p, P, pi


# ------------------------------------------------------------------ S1 cleaning
def S1():
    df = pd.read_csv(RAW); df["time"] = pd.to_datetime(df["_time"].str[:19]).dt.round("h")
    rows, raw = [], {}
    for s, m in STREET.items():
        d = df[["time"] + list(SHARED) + list(m)].rename(columns={**SHARED, **m}); d["street"] = s; raw[s] = d
    R = pd.concat(raw.values(), ignore_index=True)
    rules = {**{f"aOTE{i}": ("outside −5–30 %", lambda v: (v > 30) | (v < -5)) for i in (1, 2, 3)},
             **{f"Air{i}": ("negative → 0", lambda v: v < 0) for i in (1, 2, 3)},
             "pH_in": ("outside 6–9.5", lambda v: (v < 6) | (v > 9.5)),
             **{f"DO{i}": ("> 8 mg L⁻¹", lambda v: v > 8) for i in (1, 2, 3)},
             **{f"N2O{i}": ("negative → 0", lambda v: v < 0) for i in (1, 2, 3)}}
    for c, (rule, f) in rules.items():
        for s in (1, 2):
            v = R.loc[R.street == s, c]; n = v.notna().sum()
            rows.append(dict(var=c, street=s, rule=rule, n=n, flagged=int(f(v).sum()), pct=100 * f(v).sum() / max(n, 1)))
    flag = pd.DataFrame(rows)
    # N2O: raw street sum vs clipped sum
    for c in ["N2O1", "N2O2", "N2O3"]: R[c] = R[c] * 1000
    R["y_raw"] = R[["N2O1", "N2O2", "N2O3"]].sum(axis=1, min_count=3)
    R["y_clip"] = R[["N2O1", "N2O2", "N2O3"]].clip(lower=0).sum(axis=1, min_count=3)
    R["ym"] = R.time.dt.to_period("M").dt.to_timestamp()
    neg_month = R.groupby(["street", "ym"]).apply(lambda d: pd.Series({f"Z{i}": 100 * (d[f"N2O{i}"] < 0).sum() / max(d[f"N2O{i}"].notna().sum(), 1) for i in (1, 2, 3)}))
    ann = R.dropna(subset=["y_raw"]).groupby([R.time.dt.year, "street"]).agg(raw=("y_raw", "sum"), clip=("y_clip", "sum"))
    ann["diff_pct"] = 100 * (ann["clip"] - ann["raw"]) / ann["raw"]
    hood = {c: R[c].dropna().values for c in ["N2O1", "N2O2", "N2O3"]}
    before = {c: R[c].dropna().values for c in ["aOTE1", "aOTE2", "aOTE3", "pH_in", "DO1", "DO2", "DO3"]}
    dump("S1", dict(flag=flag, neg_month=neg_month, ann=ann, hood=hood, before=before))


# ------------------------------------------------------------------ S2 missingness
def S2():
    L = pd.read_pickle("long.pkl"); L["ym"] = L.time.dt.to_period("M").dt.to_timestamp()
    cols = BASE + ["Cl", "N2O1", "N2O2", "N2O3", "y"]
    miss = {s: L[L.street == s].groupby("ym")[cols].apply(lambda d: 100 * d.isna().mean()) for s in (1, 2)}
    tot = {s: 100 * L.loc[L.street == s, cols].isna().mean() for s in (1, 2)}
    dump("S2", dict(miss=miss, tot=tot, cols=cols))


# ------------------------------------------------------------------ S3 per-street structure
def S3():
    L = pd.read_pickle("long.pkl"); L["ym"] = L.time.dt.to_period("M").dt.to_timestamp()
    mon = L.groupby(["street", "ym"])[["N2O1", "N2O2", "N2O3"]].sum(min_count=1) / 1000
    T = L.groupby("ym").T_in.mean()
    L["date"] = L.time.dt.normalize(); L["hour"] = L.time.dt.hour
    dh = {s: L[L.street == s].pivot_table(index="hour", columns="date", values="y") for s in (1, 2)}
    diur = L.dropna(subset=["y"]).groupby(["street", L.time.dt.year, "hour"]).y.median()
    dump("S3", dict(mon=mon, T=T, dh=dh, diur=diur))


# ------------------------------------------------------------------ S4 Optuna
def S4():
    import optuna, xgboost as xgb, catboost as cb
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    F, feats = build(H); tr, te = split(F)
    t0, t1 = pd.Timestamp("2021-01-01"), pd.Timestamp("2021-07-01")
    folds = [(tr[tr.time < t0 - pd.Timedelta(hours=H)], tr[(tr.time >= t0) & (tr.time < t1)]),
             (tr[tr.time < t1 - pd.Timedelta(hours=H)], tr[tr.time >= t1])]
    spaces = {
        "LightGBM": lambda t: lgb.LGBMRegressor(n_estimators=t.suggest_int("n_estimators", 200, 1200, step=100), learning_rate=t.suggest_float("learning_rate", 0.01, 0.1, log=True),
                                                 num_leaves=t.suggest_int("num_leaves", 15, 127, log=True), min_child_samples=t.suggest_int("min_child_samples", 10, 200, log=True),
                                                 subsample=t.suggest_float("subsample", 0.5, 1.0), subsample_freq=1, colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
                                                 reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True), verbose=-1, random_state=0, n_jobs=1),
        "XGBoost": lambda t: xgb.XGBRegressor(n_estimators=t.suggest_int("n_estimators", 200, 1000, step=100), learning_rate=t.suggest_float("learning_rate", 0.01, 0.1, log=True),
                                               max_depth=t.suggest_int("max_depth", 3, 10), min_child_weight=t.suggest_float("min_child_weight", 1, 50, log=True),
                                               subsample=t.suggest_float("subsample", 0.5, 1.0), colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
                                               reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True), n_jobs=1, random_state=0),
        "CatBoost": lambda t: cb.CatBoostRegressor(iterations=t.suggest_int("iterations", 300, 1200, step=100), learning_rate=t.suggest_float("learning_rate", 0.02, 0.15, log=True),
                                                    depth=t.suggest_int("depth", 4, 9), l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1, 20, log=True), verbose=0, random_seed=0, thread_count=1),
        "Extra trees": lambda t: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(n_estimators=200, min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 50, log=True),
                                               max_features=t.suggest_float("max_features", 0.2, 1.0), max_depth=t.suggest_categorical("max_depth", [None, 12, 20, 30]), n_jobs=1, random_state=0)),
    }
    defaults = {"LightGBM": lambda: lgb.LGBMRegressor(**LG), "XGBoost": lambda: B.base_models()["XGBoost"], "CatBoost": lambda: B.base_models()["CatBoost"],
                "Extra trees": lambda: B.base_models()["ExtraTrees"]}
    ntr = {"LightGBM": 40, "XGBoost": 30, "CatBoost": 20, "Extra trees": 15}
    out = {}
    for name, mk in spaces.items():
        def obj(t):
            s = []
            for a, v in folds:
                m = mk(t).fit(a[feats], a.target.values); s.append(rmse(v.target.values, m.predict(v[feats])))
            return float(np.mean(s))
        st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0)); t0_ = time.time()
        st.optimize(obj, n_trials=ntr[name])
        try: imp = optuna.importance.get_param_importances(st)
        except Exception: imp = {}
        best = mk(optuna.trial.FixedTrial(st.best_params)).fit(tr[feats], tr.target.values); dfl = defaults[name]().fit(tr[feats], tr.target.values)
        cvd = float(np.mean([rmse(v.target.values, defaults[name]().fit(a[feats], a.target.values).predict(v[feats])) for a, v in folds]))
        out[name] = dict(values=[t.value for t in st.trials], best_params=st.best_params, importance=imp, cv_default=cvd,
                         train_tuned=r2(tr.target, best.predict(tr[feats])), train_default=r2(tr.target, dfl.predict(tr[feats])),
                         test_tuned=dict(R2=r2(te.target, best.predict(te[feats])), RMSE=rmse(te.target, best.predict(te[feats]))),
                         test_default=dict(R2=r2(te.target, dfl.predict(te[feats])), RMSE=rmse(te.target, dfl.predict(te[feats]))), secs=time.time() - t0_)
        print(name, out[name]["best_params"], out[name]["test_tuned"], out[name]["test_default"], flush=True)
        dump("S4", out)


# ------------------------------------------------------------------ S5 learning curves
def S5():
    from sklearn.ensemble import ExtraTreesRegressor
    F, feats = build(H); tr, _ = split(F)
    v0 = pd.Timestamp("2021-07-01"); pool = tr[tr.time < v0 - pd.Timedelta(hours=H)]; val = tr[tr.time >= v0]
    tmax = pool.time.max(); span = (tmax - pool.time.min())
    fr = [0.25, 0.5, 0.75, 1.0]; out = {}
    for f in fr:
        sub = pool[pool.time >= tmax - span * f]; med = np.nanmedian(sub.target)
        res = {}
        for name, mk in [("Ridge", lambda: B.base_models()["Ridge"]), ("Extra trees", lambda: B.base_models()["ExtraTrees"]), ("LightGBM", lambda: lgb.LGBMRegressor(**LG))]:
            m = mk().fit(sub[feats], sub.target.values)
            res[name] = (r2(sub.target, m.predict(sub[feats])), r2(val.target, m.predict(val[feats])))
        mo = EGMoE(np.quantile(sub.target, 0.8)).fit(sub[feats], sub.target.values, sub.pers.fillna(med).values, sub.time.values)
        res["EG-MoE"] = (r2(sub.target, mo.predict(sub[feats], sub.pers.fillna(med).values)), r2(val.target, mo.predict(val[feats], val.pers.fillna(med).values)))
        res["Persistence"] = (r2(sub.target, sub.pers.fillna(med)), r2(val.target, val.pers.fillna(med)))
        out[f] = dict(n=len(sub), **res); print(f, out[f], flush=True)
    dump("S5", out)


# ------------------------------------------------------------------ S6 deep training curves
def S6():
    import torch, torch.nn as nn, deep as D
    torch.set_num_threads(1)
    F, feats = build(H); tr, te = split(F)
    v0 = pd.Timestamp("2021-07-01"); fit = tr[tr.time < v0 - pd.Timedelta(hours=H)]; val = tr[tr.time >= v0]
    cols = D.COLS + ["yl0"]; L = pd.read_pickle("long.pkl"); from prep import engineer
    E = engineer(L, 1, use_n2o_hist=True)[cols + ["street"]]; E[cols] = E.groupby("street")[cols].ffill(limit=6)
    m_ = E.index.isin(fit.index); mu, sd = E.loc[m_, cols].mean(), E.loc[m_, cols].std() + 1e-6
    A = ((E[cols] - mu) / sd).fillna(0).values.astype(np.float32)
    start = E.groupby("street").apply(lambda d: d.index.min()).to_dict(); st = E.street.values

    def seqs(df):
        idx = np.asarray(df.index); ok = idx - np.vectorize(start.get)(st[idx]) >= D.SEQ - 1; idx = idx[ok]
        return idx, np.stack([A[i - D.SEQ + 1:i + 1] for i in idx])
    sets = {k: seqs(d) for k, d in [("fit", fit), ("val", val), ("test", te)]}
    out = pickle.load(open("figdata/si/S6.pkl", "rb")) if os.path.exists("figdata/si/S6.pkl") else {}
    for resid in [False, True]:
        for name, mk in [("LSTM", lambda d: D.RNN(d, nn.LSTM)), ("GRU", lambda d: D.RNN(d, nn.GRU)), ("TCN", D.TCN), ("Transformer", D.TF)]:
            if (name, resid) in out: continue
            torch.manual_seed(42); m = mk(len(cols)); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
            def target(df, idx):
                y = df.target.loc[idx].values; a = df.pers.loc[idx].fillna(np.nanmedian(fit.target)).values
                return y, a
            yf, af = target(fit, sets["fit"][0]); ys = np.std(yf - af if resid else yf)
            Xt = torch.tensor(sets["fit"][1]); yt = torch.tensor(((yf - af) / ys if resid else yf / ys).astype(np.float32))
            curve = []
            for ep in range(20):
                m.train(); perm = torch.randperm(len(Xt)); tl = []
                for b in range(0, len(Xt), 256):
                    j = perm[b:b + 256]; opt.zero_grad(); loss = nn.functional.mse_loss(m(Xt[j]), yt[j]); loss.backward(); opt.step(); tl.append(loss.item())
                m.eval(); row = dict(epoch=ep + 1, train_loss=float(np.mean(tl)))
                for k, df in [("fit", fit), ("val", val), ("test", te)]:
                    idx, X = sets[k]; y, a = target(df, idx)
                    with torch.no_grad(): p = np.concatenate([m(torch.tensor(X[b:b + 2048])).numpy() for b in range(0, len(X), 2048)]) * ys
                    row[f"R2_{k}"] = r2(y, p + a if resid else p)
                curve.append(row)
            out[(name, resid)] = pd.DataFrame(curve); print(name, resid, curve[-1], flush=True)
            dump("S6", out)


# ------------------------------------------------------------------ S7 ICE / PDP / ALE
def S7():
    mo = pickle.load(open("egmoe_H6.pkl", "rb")); d = pd.read_pickle("figdata/fig7.pkl"); Xs = d["Xs"]
    rng = np.random.default_rng(0); ii = rng.choice(len(Xs), 120, replace=False); out = {}
    for f, a in d["ALE"].items():
        grid = a["q"]; ice = np.zeros((len(ii), len(grid)))
        for j, g in enumerate(grid):
            Xg = Xs.iloc[ii].copy(); Xg[f] = g; ice[:, j] = mo.E["resid"].predict(Xg)
        ice -= ice.mean(1, keepdims=True)
        out[f] = dict(grid=grid, ice=ice, pdp=a["pdp"], ale=a["a"], lo=a["lo"], hi=a["hi"], x=a["x"])
    te = pd.read_pickle("te_H6.pkl")
    out["_corr"] = te[["DO2", "T_in", "Air2", "Air3"]].dropna().sample(4000, random_state=0)
    dump("S7", out)


# ------------------------------------------------------------------ S8 episode-threshold sensitivity
def S8():
    F, feats = build(H); tr, te = split(F); out = {}
    base = pickle.load(open("egmoe_H6.pkl", "rb")); lab = pickle.load(open("lab_H6.pkl", "rb")); bp = pickle.load(open("pred_H6_chrono_h1_2022.pkl", "rb"))
    for q in [0.7, 0.8, 0.9]:
        tau = np.quantile(tr.target, q)
        if q == 0.8:
            mo, p, P = base, bp["preds"]["EG-MoE"], {"p": lab["T"]["p"]}
        else:
            mo, p, P, pi = fit_egmoe(tr, te, feats, tau=tau)
        y = te.target.values
        out[q] = dict(tau=tau, R2=r2(y, p), RMSE=rmse(y, p), F1=f1_score(y > tau, p > tau), AUC=roc_auc_score(y > tau, P["p"]),
                      CumErr=100 * (np.clip(p, 0, None).sum() - y.sum()) / y.sum(), W=mo.W, wglobal=mo.w_global, names=mo.names,
                      share=np.bincount(np.digitize(P["p"], [0.15, 0.5]), minlength=3) / len(p))
        print(q, {k: v for k, v in out[q].items() if k in ["tau", "R2", "RMSE", "F1", "AUC"]}, flush=True)
    dump("S8", out)


# ------------------------------------------------------------------ S9 feature-set ablation
def S9():
    F, feats = build(H, sister=True); tr, te = split(F)
    base = [c for c in feats if c != "y_sister"]
    sets = {"Full (92)": base, "− αOTE": [c for c in base if not c.startswith("aOTE")], "+ sister-street N$_2$O": feats,
            "− Cl$^-$": [c for c in base if not c.startswith("Cl")], "− calendar": [c for c in base if c not in ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "weekend"]]}
    y = te.target.values
    bp = pickle.load(open("pred_H6_chrono_h1_2022.pkl", "rb")); base = pickle.load(open("egmoe_H6.pkl", "rb"))
    part = "figdata/si/S9_parts.pkl"; out = pickle.load(open(part, "rb")) if os.path.exists(part) else {}
    for k, fs in sets.items():
        if k in out: continue
        B.REG_VARS[:] = [v for v in REGV0 if v in fs] if k.startswith("− Cl") else REGV0
        if k.startswith("Full"):
            mo, p, lg = base, bp["preds"]["EG-MoE"], bp["preds"]["LightGBM"]
        else:
            mo, p, P, pi = fit_egmoe(tr, te, fs); lg = lgb.LGBMRegressor(**LG).fit(tr[fs], tr.target.values).predict(te[fs])
        out[k] = dict(n=len(fs), **{f"EG-MoE {m}": v for m, v in metrics(y, p, mo.tau).items()}, **{f"LightGBM {m}": v for m, v in metrics(y, lg, mo.tau).items()})
        print(k, len(fs), round(out[k]["EG-MoE R2"], 3), round(out[k]["LightGBM R2"], 3), flush=True); pickle.dump(out, open(part, "wb"))
    B.REG_VARS[:] = REGV0
    dump("S9", pd.DataFrame(out).T)


# ------------------------------------------------------------------ S10 operating-data-only
def S10():
    d = pickle.load(open("pred_H6_chrono_h0_2022.pkl", "rb")); f6 = pd.read_pickle("figdata/fig2to6.pkl")["fig6"]
    dump("S10", dict(y=d["y"], time=d["time"], street=d["street"], preds={k: d["preds"][k] for k in ["LightGBM", "EG-MoE", "ExtraTrees", "Ridge", "ClimMean"]},
                     roc_ops=f6["roc_ops"], roc_gate=f6["roc_gate"], auc_ops=f6["auc_ops"], auc_gate=f6["auc_gate"], tau=d["tau"]))


# ------------------------------------------------------------------ S11 rolling monthly retraining
def S11():
    F, feats = build(H); out = []
    for mth in range(1, 13, 2):   # retrain every two months; each model forecasts the following two months
        m0 = pd.Timestamp(f"2022-{mth:02d}-01"); m1 = m0 + pd.offsets.MonthBegin(2)
        tr = F[F.time < m0 - pd.Timedelta(hours=H)]; te = F[(F.time >= m0) & (F.time < m1)]
        if len(te) < 100: continue
        mo, p, P, pi = fit_egmoe(tr, te, feats); lg = lgb.LGBMRegressor(**LG).fit(tr[feats], tr.target.values).predict(te[feats])
        med = np.nanmedian(tr.target); pe = te.pers.fillna(med).values; y = te.target.values
        out.append(dict(month=mth, n=len(te), y=y, EGMoE=p, LightGBM=lg, Persistence=pe, time=te.time.values, street=te.street.values))
        print(mth, len(te), rmse(y, p), rmse(y, lg), rmse(y, pe), flush=True)
        dump("S11", out)


# ------------------------------------------------------------------ S12 DML diagnostics
def S12():
    from econml.dml import LinearDML
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.preprocessing import SplineTransformer
    F, feats = build(H); F = F.copy(); g = F.groupby("street")
    lagc = ["T_in", "Q_out", "NH4", "NO3", "Cl", "RS", "pH_in", "DO2", "DO3", "Air2", "Air3", "Supern", "swing_on"]
    for c in lagc: F[c + "_lag1"] = g[c].shift(1)
    for c in ["T_in_m168", "Q_out_m72", "NH4_m168", "Cl_m72", "y_m24", "y_m168"]: F[c] = g[c].shift(1)
    W_env = ["T_in_lag1", "T_in_m168", "Q_out_lag1", "Q_out_m72", "NH4_lag1", "NH4_m168", "NO3_lag1", "Cl_lag1", "Cl_m72",
             "RS_lag1", "pH_in_lag1", "hour_sin", "hour_cos", "doy_sin", "doy_cos", "weekend", "yl1", "y_m24", "y_m168"]
    F["Y"] = np.log1p(F["target"]); F["week"] = (F.time.dt.isocalendar().week.astype(int) + 100 * F.time.dt.year).values * 10 + F.street
    F["Supern_open"] = (F["Supern"] > 5).astype(float)
    lgr = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, verbose=-1, random_state=0)
    TREAT = {"swing": ("swing_on", ["DO2_lag1", "DO3_lag1", "Air2_lag1", "Air3_lag1", "Supern_lag1"], True),
             "supern": ("Supern_open", ["DO2_lag1", "DO3_lag1", "Air2_lag1", "Air3_lag1", "swing_on_lag1"], True),
             "supern_dose": ("Supern", ["DO2_lag1", "DO3_lag1", "Air2_lag1", "Air3_lag1", "swing_on_lag1"], False),
             "do2_dose": ("DO2", ["T_in", "Air2_lag1", "Air3_lag1", "DO3_lag1", "swing_on_lag1", "Supern_lag1"], False)}
    out = {"nuis": [], "dose": {}, "evalue": []}
    ate = pd.read_csv("figdata/dml_ate_block.csv")
    for key, (t, extra, disc) in TREAT.items():
        D = F.dropna(subset=["Y", t] + W_env + extra).reset_index(drop=True)
        Y, T, W, X = D["Y"].values, D[t].values, D[W_env + extra], D[["T_in_lag1", "NH4_lag1"]]
        WX = pd.concat([W, X[[c for c in X.columns if c not in W.columns]]], axis=1); WX = WX.loc[:, ~WX.columns.duplicated()]; gk = GroupKFold(5)
        py = cross_val_predict(lgb.LGBMRegressor(**lgr), WX, Y, groups=D.week, cv=gk)
        row = dict(treatment=key, n=len(D), R2_outcome=r2_score(Y, py))
        if disc:
            pt = cross_val_predict(lgb.LGBMClassifier(**lgr), WX, T, groups=D.week, cv=gk, method="predict_proba")[:, 1]
            row.update(AUC_treatment=roc_auc_score(T, pt), share_ps_lt005=float(np.mean(pt < 0.05)), share_ps_gt095=float(np.mean(pt > 0.95)))
        else:
            pt = cross_val_predict(lgb.LGBMRegressor(**lgr), WX, T, groups=D.week, cv=gk); row.update(R2_treatment=r2_score(T, pt))
            lo, hi = np.nanpercentile(T, [5, 95]) if key == "supern_dose" else np.nanpercentile(T, [5, 95])
            sp = SplineTransformer(n_knots=4, degree=3, include_bias=False)
            from sklearn.multioutput import MultiOutputRegressor
            est = LinearDML(model_y=lgb.LGBMRegressor(**lgr), model_t=MultiOutputRegressor(lgb.LGBMRegressor(**lgr)), treatment_featurizer=sp, cv=5, random_state=0)
            est.fit(Y, T.reshape(-1, 1), X=None, W=WX, groups=D.week)
            grid = np.linspace(lo, hi, 30); t0 = np.full_like(grid, grid[0])
            eff = est.effect(X=None, T0=t0.reshape(-1, 1), T1=grid.reshape(-1, 1)).ravel()
            l, u = est.effect_interval(X=None, T0=t0.reshape(-1, 1), T1=grid.reshape(-1, 1), alpha=0.05)
            out["dose"][key] = dict(grid=grid, eff=eff, lo=np.ravel(l), hi=np.ravel(u), T=T)
        out["nuis"].append(row); print(row, flush=True)
    for _, r in ate[ate.estimator == "LinearDML-GBM"].iterrows():
        def ev(rr):
            rr = np.exp(rr); rr = rr if rr >= 1 else 1 / rr; return rr + np.sqrt(rr * (rr - 1))
        out["evalue"].append(dict(treatment=r.treatment, RR=np.exp(r.ate), E_point=ev(r.ate), E_ci=ev(r.lo_block) if r.lo_block > 0 else 1.0))
    out["overlap"] = pd.read_pickle("figdata/dml_fig8.pkl")["overlap"]
    dump("S12", out)


# ------------------------------------------------------------------ S13 zone-level forecasts
def S13():
    L = pd.read_pickle("long.pkl"); F, feats = build(H); out = {}
    for z in ["N2O1", "N2O2", "N2O3"]:
        yz = L.groupby("street")[z]
        Fz = F.copy()
        Fz["target"] = yz.transform(lambda x: x[::-1].rolling(H, min_periods=int(np.ceil(.8 * H))).mean()[::-1].shift(-1)).reindex(F.index)
        Fz["pers"] = L[z].reindex(F.index); Fz = Fz.dropna(subset=["target"])
        tr, te = split(Fz); mo, p, P, pi = fit_egmoe(tr, te, feats)
        lg = lgb.LGBMRegressor(**LG).fit(tr[feats], tr.target.values).predict(te[feats]); pe = te.pers.fillna(np.nanmedian(tr.target)).values; y = te.target.values
        out[z] = dict(y=y, EGMoE=p, LightGBM=lg, Persistence=pe, tau=mo.tau)
        print(z, r2(y, p), r2(y, lg), r2(y, pe), flush=True)
    dump("S13", out)


# ------------------------------------------------------------------ S14 gate shrinkage and fold sensitivity
def S14():
    from scipy.optimize import minimize
    lab = pickle.load(open("lab_H6.pkl", "rb")); mo = pickle.load(open("egmoe_H6.pkl", "rb"))
    O, T_ = lab["O"], lab["T"]; yte = lab["yte"]; pit = lab["pi_te"]; pio = lab["pi_tr"][O["idx"]]; yo = lab["ytr"][O["idx"]]
    names = mo.names
    def simplex(Fm_, y_, sw=None):
        sw = np.ones(len(y_)) if sw is None else sw
        f = lambda z: np.sum(sw * (Fm_ @ (np.exp(z) / np.exp(z).sum()) - y_) ** 2)
        z = minimize(f, np.zeros(Fm_.shape[1]), method="L-BFGS-B").x; return np.exp(z) / np.exp(z).sum()
    FO = np.column_stack([np.clip(O[n], 0, None) for n in names]); FT = np.column_stack([np.clip(T_[n], 0, None) for n in names])
    wg = simplex(FO, yo); res = []
    for lam in [1e-6, 10, 50, 100, 250, 500, 1000, 2500, 1e4, 1e9]:
        pred = np.zeros(len(yte)); bins = [0, .15, .5, 1.01]
        for b in range(3):
            mo_ = (O["p"] >= bins[b]) & (O["p"] < bins[b + 1]); mt = (T_["p"] >= bins[b]) & (T_["p"] < bins[b + 1])
            nb = mo_.sum(); wb = (nb * simplex(FO[mo_], yo[mo_]) + lam * wg) / (nb + lam)
            Wk = []
            for k in range(pio.shape[1]):
                sw = pio[mo_, k]; nk = sw.sum(); wk_ = simplex(FO[mo_], yo[mo_], sw) if nk > 50 else wb
                Wk.append((nk * wk_ + lam * wb) / (nk + lam))
            pred[mt] = (FT[mt] * (pit[mt] @ np.array(Wk))).sum(1)
        res.append(dict(lam=lam, RMSE=rmse(yte, pred), R2=r2(yte, pred), absCum=abs(100 * (pred.sum() - yte.sum()) / yte.sum())))
        print(res[-1], flush=True)
    dump("S14", dict(lam=pd.DataFrame(res), folds=[]))
    F, feats = build(H); tr, te = split(F); folds = []
    bp = pickle.load(open("pred_H6_chrono_h1_2022.pkl", "rb"))
    for nf in [2, 3, 4]:
        y = te.target.values
        if nf == 4: mo2, p = mo, bp["preds"]["EG-MoE"]
        else: mo2, p, P, pi = fit_egmoe(tr, te, feats, n_folds=nf)
        folds.append(dict(folds=nf, RMSE=rmse(y, p), R2=r2(y, p), absCum=abs(100 * (np.clip(p, 0, None).sum() - y.sum()) / y.sum()), W=mo2.W))
        print(folds[-1]["folds"], folds[-1]["RMSE"], flush=True)
    dump("S14", dict(lam=pd.DataFrame(res), folds=folds))


if __name__ == "__main__":
    for s in sys.argv[1:]:
        t0 = time.time(); globals()[s](); print(f"{s} done in {time.time() - t0:.0f}s", flush=True)
