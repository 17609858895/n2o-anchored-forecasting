"""ML figure audit (nature-research-figures): train R2, forward-chained CV mean/min R2, test R2, train-test gap, RMSE.
H = 6 h, V1 split (train 2020-2021, test 2022). CV = 4 expanding-window folds over 5 time-ordered blocks of the
training period, with an H-hour buffer before each validation block. Output: tables/ml_audit_H6.csv"""
import sys, time, os, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import bench2 as B, deep as D

H = 6
F, feats = B.build(H)
cut = pd.Timestamp("2022-01-01")
TR = F[F.time < cut - pd.Timedelta(hours=H)]; TE = F[(F.time >= cut) & (F.time < pd.Timestamp("2023-01-01"))]
med = np.nanmedian(TR.target)


def deep_multi(tr, evals, epochs=12):
    """Same networks and settings as deep.run_deep, but returns predictions for several evaluation sets."""
    cols = D.COLS + ["yl0"]
    L = pd.read_pickle("long.pkl"); from prep import engineer
    E = engineer(L, 1, use_n2o_hist=True)[cols + ["street"]]
    E[cols] = E.groupby("street")[cols].ffill(limit=6)
    trmask = E.index.isin(tr.index)
    mu, sd = E.loc[trmask, cols].mean(), E.loc[trmask, cols].std() + 1e-6
    A = ((E[cols] - mu) / sd).fillna(0).values.astype(np.float32)
    start = E.groupby("street").apply(lambda d: d.index.min()).to_dict(); st = E.street.values

    def seqs(idx):
        idx = np.asarray(idx); ok = idx - np.vectorize(start.get)(st[idx]) >= D.SEQ - 1; idx = idx[ok]
        return idx, np.stack([A[i - D.SEQ + 1:i + 1] for i in idx])
    itr, Xtr = seqs(tr.index); ys = tr.target.std(); ytr = (tr.target.loc[itr].values / ys).astype(np.float32)
    ev = [seqs(e.index) for e in evals]
    out = {}
    for name, mk in [("LSTM", lambda d: D.RNN(d, nn.LSTM)), ("GRU", lambda d: D.RNN(d, nn.GRU)), ("TCN", D.TCN), ("Transformer", D.TF)]:
        torch.manual_seed(42); m = mk(len(cols)); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
        Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
        for ep in range(epochs):
            m.train(); perm = torch.randperm(len(Xt))
            for b in range(0, len(Xt), 256):
                j = perm[b:b + 256]; opt.zero_grad(); nn.functional.mse_loss(m(Xt[j]), yt[j]).backward(); opt.step()
        m.eval(); res = []
        for (ie, Xe), e in zip(ev, evals):
            with torch.no_grad():
                p = np.concatenate([m(torch.tensor(Xe[b:b + 1024])).numpy() for b in range(0, len(Xe), 1024)]) * ys
            full = pd.Series(np.nan, index=e.index); full.loc[ie] = p
            res.append(full.fillna(tr.target.median()).values)
        out[name] = res; print("   ", name, flush=True)
    return out


def fit_predict(tr, evals):
    """Returns {model: [pred for each eval set]} for all 21 models (EG-MoE included)."""
    ytr = tr.target.values; Xtr = tr[feats]; m_ = np.nanmedian(ytr); out = {}
    out["Persistence"] = [e.pers.fillna(m_).values for e in evals]
    out["SeasonalNaive"] = [e.pers_day.fillna(m_).values for e in evals]
    out["ClimMean"] = [np.full(len(e), ytr.mean()) for e in evals]
    for name, m in B.base_models().items():
        t0 = time.time()
        if name == "SVR":
            ii = np.random.RandomState(B.RS).choice(len(Xtr), min(8000, len(Xtr)), replace=False)
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVR(C=30, epsilon=1.0, gamma="scale")).fit(Xtr.iloc[ii], ytr[ii])
        else:
            m.fit(Xtr, ytr)
        out[name] = [m.predict(e[feats]) for e in evals]; print(f"   {name} {time.time() - t0:.0f}s", flush=True)
    out.update(deep_multi(tr, evals))
    tau = np.quantile(ytr, 0.8)
    mo = B.EGMoE(tau).fit(Xtr, ytr, tr.pers.fillna(m_).values, tr.time.values)
    out["EG-MoE"] = [mo.predict(e[feats], e.pers.fillna(m_).values) for e in evals]
    return out


r2 = lambda y, p: r2_score(y, np.clip(p, 0, None))
rmse = lambda y, p: mean_squared_error(y, np.clip(p, 0, None)) ** .5

if __name__ == "__main__":
    os.makedirs("tables", exist_ok=True)
    t0 = time.time()
    print("full fit", flush=True)
    full = fit_predict(TR, [TR, TE])
    rows = {m: dict(model=m, R2_train=r2(TR.target, p[0]), RMSE_train=rmse(TR.target, p[0]),
                    R2_test=r2(TE.target, p[1]), RMSE_test=rmse(TE.target, p[1])) for m, p in full.items()}
    pd.DataFrame(rows.values()).to_csv("tables/ml_audit_H6_partial.csv", index=False)
    blocks = np.array_split(np.sort(TR.time.unique()), 5); cv = {m: [] for m in rows}
    for i in range(1, 5):
        v0 = blocks[i][0]; v1 = blocks[i][-1]
        ftr = TR[TR.time < v0 - np.timedelta64(H, "h")]; fva = TR[(TR.time >= v0) & (TR.time <= v1)]
        print(f"fold {i}: train {len(ftr)} val {len(fva)} ({pd.Timestamp(v0):%Y-%m-%d} to {pd.Timestamp(v1):%Y-%m-%d})", flush=True)
        for m, p in fit_predict(ftr, [fva]).items():
            cv[m].append(r2(fva.target, p[0]))
        pd.DataFrame(cv).to_csv("tables/ml_audit_H6_cvfolds.csv", index=False)
    for m in rows:
        rows[m].update(CV_R2_mean=np.mean(cv[m]), CV_R2_min=np.min(cv[m]))
        rows[m]["Gap_train_test"] = rows[m]["R2_train"] - rows[m]["R2_test"]
    R = pd.DataFrame(rows.values())[["model", "R2_train", "CV_R2_mean", "CV_R2_min", "R2_test", "Gap_train_test", "RMSE_train", "RMSE_test"]]
    R.to_csv("tables/ml_audit_H6.csv", index=False)
    print(R.round(3).to_string(), f"\n{time.time() - t0:.0f}s")
