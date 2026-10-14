"""Second-plant replication of the anchoring benchmark at the Avedoere WWTP (Denmark).

Data: Hansen et al. (2024) time-series dataset, Mendeley Data doi:10.17632/xmbxhscgpr.1, 2-min SCADA and sensor data of
line 3 (alternating aeration, tanks 1 and 2), 12 June 2022 - 11 June 2024. The N2O signal is the dissolved N2O
concentration in tank 1 (mg N L-1); there is no off-gas measurement, so the target differs from Hofen in kind.

Design mirrors Hofen V1: hourly means, target = mean dissolved N2O over the next 6 h (>= 5 valid hours),
persistence = current-hour value, training year 12 Jun 2022 - 11 Jun 2023, test year 12 Jun 2023 - 11 Jun 2024.
Same model settings as the Hofen benchmark (LightGBM etc. from bench2.py, networks from deep.py, 12 epochs).
Also repeats the validation-design audit (hour-level random split vs chronological split).

Run from code/analysis:  python aved_replication.py
Writes ../../results/aved_benchmark.csv, aved_split_inflation.csv, aved_summary.json and aved_predictions.pkl (test-year forecasts)
"""
import json, time, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
import lightgbm as lgb, xgboost as xgb, catboost as cb
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import bench2 as B, deep as D
from stats_joint import dm_joint

warnings.filterwarnings("ignore")
torch.set_num_threads(2)
RAW = "../../data/avedore/aved_raw.csv.gz"
H = 6; RS = 42; SEQ = 48
SPLIT = pd.Timestamp("2023-06-12", tz="Europe/Copenhagen")
REN = {"BLOWERSTATION 1.Q.AIRFLOW": "Air_blower", "LINE 3 TANK 1 VALVE 1.PCT": "Valve1", "LINE 3 TANK 1.N2O": "N2O",
       "LINE 3 TANK 1.NH4": "NH4", "LINE 3 TANK 1.NO3": "NO3", "LINE 3 TANK 1.O2": "DO1", "LINE 3 TANK 1.O2.SETPOINT": "DOsp1",
       "LINE 3 TANK 1.PROCESSPHASE": "Phase1", "LINE 3 TANK 1.Q.AIRFLOW": "Air1", "LINE 3 TANK 1.SS": "SS1",
       "LINE 3 TANK 1.TEMPERATURE": "T1", "LINE 3 TANK 2 VALVE 1.PCT": "Valve2", "LINE 3 TANK 2.O2": "DO2",
       "LINE 3 TANK 2.O2.SETPOINT": "DOsp2", "LINE 3 TANK 2.PROCESSPHASE": "Phase2", "LINE 3 TANK 2.Q.AIRFLOW": "Air2",
       "LINE 3 TANK 2.SS": "SS2", "LINE 3 TANK 2.TEMPERATURE": "T2", "LINE 3.PHASECODE.SETPOINT": "PhaseCode",
       "LINE 3.PROCESSPHASE.INLET TANK": "InletTank", "LINE 3.PROCESSPHASE.OUTLET TANK": "OutletTank",
       "LINE 3 TANK 1.PO4": "PO4", "INLET.Q": "Q_in", "INLET.STATE.SWM INLET FLOW": "StormMode"}
SENS = ["Air_blower", "Valve1", "NH4", "NO3", "DO1", "DOsp1", "Phase1", "Air1", "SS1", "T1", "Valve2", "DO2", "DOsp2",
        "Phase2", "Air2", "SS2", "T2", "InletTank", "OutletTank", "PO4", "Q_in", "StormMode"]
KEY = ["NH4", "NO3", "DO1", "Air1", "Air_blower", "Q_in", "T1", "PO4", "SS1"]
REG = ["T1", "NH4", "NO3", "DO1", "Air1", "Q_in", "SS1", "PO4", "Phase1", "Valve1"]


def load():
    df = pd.read_csv(RAW)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.rename(columns={c: REN[c.replace("BIOLOGY.", "").replace(" value", "")] for c in df.columns if c != "time"})
    raw_n = df["N2O"].notna().sum()
    sat = (df["N2O"] >= 12).sum()
    df.loc[df["N2O"] >= 12, "N2O"] = np.nan            # sensor upper range (11 Nov 2022), not a measurement
    neg = (df["N2O"] < 0).mean()
    df["N2O"] = df["N2O"].clip(lower=0)                 # negative readings = zero offset
    for c in ["NH4", "NO3", "DO1", "DO2", "PO4"]:
        df[c] = df[c].clip(lower=0)
    df.loc[df["Q_in"] < 0, "Q_in"] = np.nan
    h = df.set_index("time").resample("1h").mean()
    h.index = h.index.tz_convert("Europe/Copenhagen")
    info = dict(raw_records=int(len(df)), n2o_readings=int(raw_n), removed_saturated=int(sat),
                negative_share=float(neg), hours=int(len(h)), valid_n2o_hours=int(h["N2O"].notna().sum()))
    return h.reset_index(), info


def features(h):
    F = h[["time"] + SENS].copy()
    for c in KEY:
        F[c + "_d1"] = h[c].diff(1)
        F[c + "_m6"] = h[c].rolling(6, min_periods=3).mean()
        F[c + "_m24"] = h[c].rolling(24, min_periods=12).mean()
    F["NH4_sd24"] = h["NH4"].rolling(24, min_periods=12).std()
    F["T1_m168"] = h["T1"].rolling(168, min_periods=84).mean()
    F["T_drop72"] = h["T1"] - h["T1"].shift(72)
    F["NH4_m168"] = h["NH4"].rolling(168, min_periods=84).mean()
    F["Nload"] = h["NH4"] * h["Q_in"]
    t = F["time"]
    F["hour_sin"] = np.sin(2 * np.pi * t.dt.hour / 24); F["hour_cos"] = np.cos(2 * np.pi * t.dt.hour / 24)
    F["doy_sin"] = np.sin(2 * np.pi * t.dt.dayofyear / 365.25); F["doy_cos"] = np.cos(2 * np.pi * t.dt.dayofyear / 365.25)
    F["weekend"] = (t.dt.dayofweek >= 5).astype(float)
    ly = np.log1p(100 * h["N2O"])                         # log scale on 0.01 mg N L-1 units
    for k in [0, 1, 2, 3, 6, 12, 24]:
        F[f"yl{k}"] = ly.shift(k)
    F["y_d1"] = ly.diff(1); F["y_d3"] = ly.diff(3)
    F["y_max3"] = ly.rolling(3, min_periods=1).max()
    for w in [6, 24, 72, 168]:
        F[f"y_m{w}"] = ly.rolling(w, min_periods=w // 2).mean()
    F["y_max24"] = ly.rolling(24, min_periods=12).max(); F["y_sd24"] = ly.rolling(24, min_periods=12).std()
    y = h["N2O"]
    F["target"] = y[::-1].rolling(H, min_periods=int(np.ceil(.8 * H))).mean()[::-1].shift(-1)
    F["pers"] = y
    F["pers_day"] = y.rolling(H, min_periods=1).mean().shift(24 - H)
    F["street"] = 1                                       # single line; keeps helper code unchanged
    feats = [c for c in F.columns if c not in {"time", "target", "pers", "pers_day", "street"}]
    return F, feats


def tab_models():
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "Ridge": make_pipeline(imp(), StandardScaler(), Ridge(alpha=10)),
        "Random forest": make_pipeline(imp(), RandomForestRegressor(n_estimators=300, min_samples_leaf=10, max_features=0.4, n_jobs=-1, random_state=RS)),
        "Extra trees": make_pipeline(imp(), ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=RS)),
        "XGBoost": xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.7, min_child_weight=10, n_jobs=-1, random_state=RS),
        "LightGBM": lgb.LGBMRegressor(**B.LG),
        "CatBoost": cb.CatBoostRegressor(iterations=800, learning_rate=0.05, depth=6, l2_leaf_reg=3, verbose=0, random_seed=RS),
    }


def nets(Ffull, tr, te, anchored, epochs=12):
    # Ffull keeps every hour, so row label = row position and 48-h windows are contiguous in time
    cols = [c for c in SENS if c not in ("DOsp2",)] + ["hour_sin", "hour_cos", "yl0"]
    E = Ffull[cols].ffill(limit=6)
    trm = Ffull.index.isin(tr.index)
    mu, sd = E.loc[trm].mean(), E.loc[trm].std() + 1e-6
    A = ((E - mu) / sd).fillna(0).values.astype(np.float32)
    med = np.nanmedian(tr.target)

    def seqs(df):
        idx = np.asarray(df.index); idx = idx[idx >= SEQ - 1]
        return idx, np.stack([A[i - SEQ + 1:i + 1] for i in idx])
    itr, Xtr = seqs(tr); ite, Xte = seqs(te)
    a_tr = tr.pers.loc[itr].fillna(med).values
    r_tr = tr.target.loc[itr].values - (a_tr if anchored else 0)
    sc = r_tr.std(); yt = torch.tensor((r_tr / sc).astype(np.float32)); Xt = torch.tensor(Xtr)
    out = {}
    for name, mk in [("LSTM", lambda d: D.RNN(d, nn.LSTM)), ("GRU", lambda d: D.RNN(d, nn.GRU)), ("TCN", D.TCN), ("Transformer", D.TF)]:
        torch.manual_seed(RS); m = mk(len(cols)); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4); t0 = time.time()
        for ep in range(epochs):
            m.train(); perm = torch.randperm(len(Xt))
            for b in range(0, len(Xt), 256):
                j = perm[b:b + 256]; opt.zero_grad(); nn.functional.mse_loss(m(Xt[j]), yt[j]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p = np.concatenate([m(torch.tensor(Xte[b:b + 2048])).numpy() for b in range(0, len(Xte), 2048)]) * sc
        full = pd.Series(np.nan, index=te.index); full.loc[ite] = p
        if anchored:
            pred = te.pers.fillna(med).values + full.fillna(0).values
        else:
            pred = full.fillna(med).values
        out[name + ("-A" if anchored else "")] = pred
        print(f"   {name}{'-A' if anchored else ''} {time.time() - t0:.0f}s", flush=True)
    return out


def metrics(y, p, pers, tau, t):
    same = np.array_equal(p, pers)
    p = np.clip(p, 0, None)
    s, pv = (np.nan, np.nan) if same else dm_joint(y - p, y - pers, t, H)
    return dict(R2=r2_score(y, p), RMSE=mean_squared_error(y, p) ** .5, MAE=mean_absolute_error(y, p),
                Skill=1 - mean_squared_error(y, p) / mean_squared_error(y, pers),
                CumErr=100 * (p.sum() - y.sum()) / y.sum(), DM_vs_pers=s, P_vs_pers=pv,
                F1=float(((p > tau) & (y > tau)).sum() * 2 / ((p > tau).sum() + (y > tau).sum())))


def main():
    h, info = load()
    Ffull, feats = features(h)
    assert (Ffull.index == np.arange(len(Ffull))).all()
    F = Ffull.dropna(subset=["target"]).copy()
    tr = F[F.time < SPLIT - pd.Timedelta(hours=H)]; te = F[F.time >= SPLIT]
    med = np.nanmedian(tr.target); ytr, yte = tr.target.values, te.target.values
    atr, ate = tr.pers.fillna(med).values, te.pers.fillna(med).values
    tau = np.quantile(ytr, 0.8)
    info.update(n_train=int(len(tr)), n_test=int(len(te)), tau=float(tau), train_max=float(ytr.max()), test_max=float(yte.max()),
                ac1=float(np.log1p(100 * h.N2O).autocorr(1)), ac6=float(np.log1p(100 * h.N2O).autocorr(6)))
    preds = {"Persistence": ate, "Seasonal naive": te.pers_day.fillna(med).values, "Climatology": np.full(len(te), ytr.mean())}
    for name, m in tab_models().items():
        t0 = time.time(); preds[name] = m.fit(tr[feats], ytr).predict(te[feats]); print(f"  {name} {time.time() - t0:.0f}s", flush=True)
    preds["Anchored LightGBM"] = ate + lgb.LGBMRegressor(**B.LG).fit(tr[feats], ytr - atr).predict(te[feats])
    B.REG_VARS = REG
    t0 = time.time(); mo = B.EGMoE(tau).fit(tr[feats], ytr, atr, tr.time.values)
    preds["EG-MoE"] = mo.predict(te[feats], ate); print(f"  EG-MoE {time.time() - t0:.0f}s", flush=True)
    preds.update(nets(Ffull, tr, te, anchored=False))
    preds.update(nets(Ffull, tr, te, anchored=True))
    R = pd.DataFrame([dict(model=n, **metrics(yte, np.asarray(p), ate, tau, te.time.values)) for n, p in preds.items()])
    R.loc[R.model == "Persistence", ["DM_vs_pers", "P_vs_pers"]] = np.nan
    eg = np.clip(np.asarray(preds["EG-MoE"]), 0, None)        # Diebold-Mariano test of every model against EG-MoE
    dm = {n: dm_joint(yte - eg, yte - np.clip(np.asarray(p), 0, None), te.time.values, H) for n, p in preds.items() if n != "EG-MoE"}
    R["DM_vs_EGMoE"] = R.model.map(lambda n: dm[n][0] if n in dm else np.nan)
    R["P_vs_EGMoE"] = R.model.map(lambda n: dm[n][1] if n in dm else np.nan)
    pd.to_pickle(dict(time=te.time.values, y=yte, preds=preds), "../../results/aved_predictions.pkl")
    R = R.sort_values("R2", ascending=False)
    R.to_csv("../../results/aved_benchmark.csv", index=False)
    print(R.round(3).to_string(index=False))

    # validation-design audit at the second plant (hour-level random 30% hold-out, 5 repeats)
    Fa = F.copy(); rows = []
    for rep in range(5):
        msk = np.random.RandomState(RS + rep).rand(len(Fa)) < 0.3
        a, b = Fa[~msk], Fa[msk]; mda = np.nanmedian(a.target)
        pa = b.pers.fillna(mda).values
        for name in ["LightGBM", "Random forest", "Extra trees"]:
            p = np.clip(tab_models()[name].fit(a[feats], a.target.values).predict(b[feats]), 0, None)
            rows.append(dict(design="random", rep=rep, model=name, R2=r2_score(b.target, p),
                             Skill=1 - mean_squared_error(b.target, p) / mean_squared_error(b.target, pa)))
        rows.append(dict(design="random", rep=rep, model="Persistence", R2=r2_score(b.target, pa), Skill=0.0))
    for name in ["LightGBM", "Random forest", "Extra trees", "Persistence"]:
        r = R[R.model == name].iloc[0]; rows.append(dict(design="chrono", rep=0, model=name, R2=r.R2, Skill=r.Skill))
    S = pd.DataFrame(rows); S.to_csv("../../results/aved_split_inflation.csv", index=False)
    print(S.groupby(["design", "model"])[["R2", "Skill"]].agg(["mean", "min", "max"]).round(3))
    json.dump(info, open("../../results/aved_summary.json", "w"), indent=1); print(info)


if __name__ == "__main__":
    main()
