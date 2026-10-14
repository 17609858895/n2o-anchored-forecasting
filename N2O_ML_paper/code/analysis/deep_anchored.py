"""Residual-anchored sequence baselines: the network predicts y(t+1:t+H) - y(t) and the persistence anchor is added back.
Same architectures, inputs (48-h windows) and training settings as deep.py (12 epochs, AdamW 1e-3).
usage: python deep_anchored.py main   -> H6/V1 for LSTM/GRU/TCN/Transformer: test predictions + train R2 + forward-CV R2
       python deep_anchored.py other  -> GRU-A for H1/3/12/24 (V1), H6 V2 (test 2021) and H6 V3 (cross-street)"""
import sys, pickle, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error
import bench2 as B, deep as D
torch.set_num_threads(1)
NETS = {"LSTM": lambda d: D.RNN(d, nn.LSTM), "GRU": lambda d: D.RNN(d, nn.GRU), "TCN": D.TCN, "Transformer": D.TF}
L = pd.read_pickle("long.pkl"); from prep import engineer
COLS = D.COLS + ["yl0"]
E0 = engineer(L, 1, use_n2o_hist=True)[COLS + ["street"]]
E0[COLS] = E0.groupby("street")[COLS].ffill(limit=6)
START = E0.groupby("street").apply(lambda d: d.index.min()).to_dict(); ST = E0.street.values


def fit_eval(tr, evals, names, epochs=12):
    trmask = E0.index.isin(tr.index)
    mu, sd = E0.loc[trmask, COLS].mean(), E0.loc[trmask, COLS].std() + 1e-6
    A = ((E0[COLS] - mu) / sd).fillna(0).values.astype(np.float32)
    med = np.nanmedian(tr.target)

    def seqs(df):
        idx = np.asarray(df.index); ok = idx - np.vectorize(START.get)(ST[idx]) >= D.SEQ - 1; idx = idx[ok]
        return idx, np.stack([A[i - D.SEQ + 1:i + 1] for i in idx])
    itr, Xtr = seqs(tr)
    a_tr = tr.pers.loc[itr].fillna(med).values; r_tr = tr.target.loc[itr].values - a_tr
    rs = r_tr.std(); yt = torch.tensor((r_tr / rs).astype(np.float32)); Xt = torch.tensor(Xtr)
    ev = [seqs(e) for e in evals]; out = {}
    for name in names:
        torch.manual_seed(42); m = NETS[name](len(COLS)); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
        for ep in range(epochs):
            m.train(); perm = torch.randperm(len(Xt))
            for b in range(0, len(Xt), 256):
                j = perm[b:b + 256]; opt.zero_grad(); nn.functional.mse_loss(m(Xt[j]), yt[j]).backward(); opt.step()
        m.eval(); res = []
        for (ie, Xe), e in zip(ev, evals):
            with torch.no_grad():
                r = np.concatenate([m(torch.tensor(Xe[b:b + 2048])).numpy() for b in range(0, len(Xe), 2048)]) * rs
            anchor = e.pers.fillna(med).values
            full = pd.Series(np.nan, index=e.index); full.loc[ie] = r
            res.append(anchor + full.fillna(0).values)   # rows without a full 48-h window fall back to persistence
        out[name + "-A"] = res; print("   ", name, flush=True)
    return out


def split(F, H, scenario="chrono", test_year=2022):
    cut = pd.Timestamp(f"{test_year}-01-01")
    tr = F[F.time < cut - pd.Timedelta(hours=H)]; te = F[(F.time >= cut) & (F.time < pd.Timestamp(f"{test_year + 1}-01-01"))]
    if scenario == "xstreet": tr = tr[tr.street == 1]; te = te[te.street == 2]
    return tr, te


def add_to_pickle(fn, preds, y):
    d = pickle.load(open(fn, "rb")); assert len(d["y"]) == len(y) and np.allclose(d["y"], y)
    d["preds"].update(preds); pickle.dump(d, open(fn, "wb"))


r2 = lambda y, p: r2_score(y, np.clip(p, 0, None))
if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "main":
        H = 6; F, feats = B.build(H); TR, TE = split(F, H)
        full = fit_eval(TR, [TR, TE], list(NETS))
        test = {k: v[1] for k, v in full.items()}
        for fn in ["pred_H6_chrono_hist1.pkl", "pred_H6_chrono_h1_2022.pkl"]: add_to_pickle(fn, test, TE.target.values)
        rows = {k: dict(model=k, R2_train=r2(TR.target, v[0]), RMSE_train=mean_squared_error(TR.target, np.clip(v[0], 0, None)) ** .5,
                        R2_test=r2(TE.target, v[1]), RMSE_test=mean_squared_error(TE.target, np.clip(v[1], 0, None)) ** .5) for k, v in full.items()}
        print(pd.DataFrame(rows.values()).round(3), flush=True)
        blocks = np.array_split(np.sort(TR.time.unique()), 5); cv = {k: [] for k in rows}
        for i in range(1, 5):
            v0, v1 = blocks[i][0], blocks[i][-1]
            ftr = TR[TR.time < v0 - np.timedelta64(H, "h")]; fva = TR[(TR.time >= v0) & (TR.time <= v1)]
            for k, p in fit_eval(ftr, [fva], list(NETS)).items(): cv[k].append(r2(fva.target, p[0]))
            print("fold", i, {k: round(v[-1], 3) for k, v in cv.items()}, flush=True)
        for k in rows:
            rows[k].update(CV_R2_mean=np.mean(cv[k]), CV_R2_min=np.min(cv[k]), Gap_train_test=rows[k]["R2_train"] - rows[k]["R2_test"])
        R = pd.DataFrame(rows.values())[["model", "R2_train", "CV_R2_mean", "CV_R2_min", "R2_test", "Gap_train_test", "RMSE_train", "RMSE_test"]]
        R.to_csv("tables/ml_audit_H6_anchored_deep.csv", index=False); print(R.round(3))
    else:
        for H, sc, ty in [(1, "chrono", 2022), (3, "chrono", 2022), (12, "chrono", 2022), (24, "chrono", 2022), (6, "chrono", 2021), (6, "xstreet", 2022)]:
            F, feats = B.build(H); tr, te = split(F, H, sc, ty)
            p = fit_eval(tr, [te], ["GRU"])["GRU-A"][0]
            fn = f"pred_H{H}_{sc}_h1_{ty}.pkl"; add_to_pickle(fn, {"GRU-A": p}, te.target.values)
            print(fn, round(r2(te.target, p), 4), flush=True)
    print("DONE")
