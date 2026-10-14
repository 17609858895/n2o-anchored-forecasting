"""Inputs for Fig. 2-7 (everything except DML). Writes figdata/*.pkl|csv."""
import numpy as np, pandas as pd, pickle, warnings, os
warnings.filterwarnings("ignore")
from scipy import stats
from scipy.optimize import minimize
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, f1_score, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.calibration import calibration_curve
import lightgbm as lgb
from bench2 import build, LG, REG_VARS
os.makedirs("figdata", exist_ok=True)
rng = np.random.default_rng(0)
L = pd.read_pickle("long.pkl")
mo = pickle.load(open("egmoe_H6.pkl", "rb")); feats = pickle.load(open("feats.pkl", "rb"))
tr = pd.read_pickle("tr_H6.pkl"); te = pd.read_pickle("te_H6.pkl")
OUT = {}

from stats_joint import dm_joint, week_id  # calendar-week blocks shared by both streets

# ---------------- Fig 3: regimes ----------------
Xtr = tr[feats].reset_index(drop=True); pi_tr = mo._regime(Xtr); lab_tr = pi_tr.argmax(1)
Xte = te[feats].reset_index(drop=True); pi_te = mo._regime(Xte); lab_te = pi_te.argmax(1)
Ztr = mo.sc.transform(Xtr[REG_VARS])
sub = rng.choice(len(Ztr), 6000, replace=False)
import umap
emb = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(Ztr[sub])
cent = pd.DataFrame(Ztr, columns=REG_VARS).groupby(lab_tr).mean()
ytr = tr.target.values
reg_stats = pd.DataFrame({"regime": lab_tr, "y": ytr}).groupby("regime").y.agg(["mean", "median", "count", lambda v: (v > mo.tau).mean()])
reg_stats.columns = ["mean", "median", "n", "event_rate"]
# transition matrix on hourly sequence (train period, within street, consecutive hours only)
seq = pd.DataFrame({"t": tr.time.values, "s": tr.street.values, "r": lab_tr}).sort_values(["s", "t"])
nxt = seq.groupby("s").shift(-1); ok = (nxt.t - seq.t) == pd.Timedelta(hours=1)
TM = pd.crosstab(seq.r[ok], nxt.r[ok].astype(int), normalize="index")
# k selection and bootstrap stability
ks = list(range(2, 11)); bic, sil = [], []
zs = Ztr[rng.choice(len(Ztr), 8000, replace=False)]
for k in ks:
    gm = GaussianMixture(k, covariance_type="full", random_state=0, n_init=1).fit(Ztr)
    bic.append(gm.bic(Ztr)); sil.append(silhouette_score(zs, gm.predict(zs), sample_size=4000, random_state=0))
ari = {}
for k in [4, 5, 6, 7, 8]:
    ref = GaussianMixture(k, covariance_type="full", random_state=0, n_init=2).fit(Ztr).predict(Ztr)
    vals = []
    for b in range(10):
        ii = rng.choice(len(Ztr), len(Ztr), replace=True)
        gb = GaussianMixture(k, covariance_type="full", random_state=b, n_init=1).fit(Ztr[ii])
        vals.append(adjusted_rand_score(ref, gb.predict(Ztr)))
    ari[k] = vals
OUT["fig3"] = dict(emb=emb, emb_lab=lab_tr[sub], emb_y=ytr[sub], cent=cent, reg_stats=reg_stats, TM=TM,
                   ks=ks, bic=bic, sil=sil, ari=ari, lab_te=lab_te, te_time=te.time.values, te_street=te.street.values,
                   te_y=te.target.values, lab_tr=lab_tr, ytr=ytr, tau=mo.tau)
pickle.dump(OUT, open("figdata/fig3.pkl", "wb")); print("regimes done", flush=True)

# ---------------- Fig 4: full benchmark ----------------
A = pickle.load(open("pred_H6_chrono_hist1.pkl", "rb")); B = pickle.load(open("pred_H6_chrono_h1_2022.pkl", "rb"))
P = {k: np.clip(v, 0, None) for k, v in A["preds"].items() if not k.startswith("[")}
P["EG-MoE"] = np.clip(B["preds"]["EG-MoE"], 0, None)
y = B["y"]; tm = pd.to_datetime(B["time"]); st = B["street"]; tau = B["tau"]; jan = np.asarray(tm.month == 1)
o = np.lexsort((tm.values, st))
def dm(e1, e2, h):
    d = e1 ** 2 - e2 ** 2; n = len(d); db = d.mean(); Lg = 4 * h
    gm = [np.mean((d[k:] - db) * (d[:n - k] - db)) for k in range(Lg + 1)]
    v = gm[0] + 2 * sum((1 - k / (Lg + 1)) * gm[k] for k in range(1, Lg + 1)); s_ = db / np.sqrt(v / n)
    return s_, 2 * (1 - stats.norm.cdf(abs(s_)))
pers = P["Persistence"]; rows = []
for m, p in P.items():
    s_, pv = dm_joint(y - p, y - pers, tm, 6) if m != "Persistence" else (0, 1)
    rows.append(dict(model=m, R2=r2_score(y, p), R2_jan=r2_score(y[jan], p[jan]), R2_nonjan=r2_score(y[~jan], p[~jan]),
                     RMSE=mean_squared_error(y, p) ** .5, MAE=mean_absolute_error(y, p), R2_log=r2_score(np.log1p(y), np.log1p(p)),
                     F1=f1_score(y > tau, p > tau), absCum=abs(100 * (p.sum() - y.sum()) / y.sum()),
                     Skill=1 - mean_squared_error(y, p) / mean_squared_error(y, pers), DM=s_, DM_p=pv,
                     sd=p.std(), corr=np.corrcoef(y, p)[0, 1] if p.std() > 0 else 0))
M4 = pd.DataFrame(rows)
# entropy-weight TOPSIS on benefit (R2, R2_jan, R2_nonjan, R2_log, F1) and cost (RMSE, MAE, absCum)
ben = ["R2", "R2_nonjan", "R2_log", "F1"]; cost = ["RMSE", "MAE", "absCum"]
Mt = M4[M4.model != "ClimMean"].set_index("model")[ben + cost].copy()
Nm = Mt.copy()
for c in ben: Nm[c] = (Mt[c] - Mt[c].min()) / (Mt[c].max() - Mt[c].min())
for c in cost: Nm[c] = (Mt[c].max() - Mt[c]) / (Mt[c].max() - Mt[c].min())
Pn = (Nm + 1e-9) / (Nm + 1e-9).sum(); E = -(Pn * np.log(Pn)).sum() / np.log(len(Pn)); wts = (1 - E) / (1 - E).sum()
V = Nm * wts; dp = np.sqrt(((V - V.max()) ** 2).sum(1)); dn = np.sqrt(((V - V.min()) ** 2).sum(1))
M4 = M4.merge(pd.DataFrame({"model": V.index, "TOPSIS": (dn / (dp + dn)).values}), on="model", how="left")
# weekly block bootstrap RMSE
wk = week_id(tm, st); uw = np.unique(wk); idxw = {w: np.where(wk == w)[0] for w in uw}
boot = {m: [] for m in P}
for b in range(300):
    ws = rng.choice(uw, len(uw), replace=True); ii = np.concatenate([idxw[w] for w in ws])
    for m in P: boot[m].append(mean_squared_error(y[ii], P[m][ii]) ** .5)
OUT["fig4"] = dict(M=M4, wts=wts, boot=pd.DataFrame(boot), ysd=y.std())
M4.to_csv("figdata/fig4_metrics.csv", index=False)
print("benchmark done", flush=True)

# ---------------- Fig 5 ----------------
hor = {}
for H in [1, 3, 6, 12, 24]:
    d = pickle.load(open(f"pred_H{H}_chrono_h1_2022.pkl", "rb")); yy = d["y"]; t5 = pd.to_datetime(d["time"]); s5 = d["street"]
    PP = {k: np.clip(v, 0, None) for k, v in d["preds"].items()}
    wk5 = week_id(t5, s5); u5 = np.unique(wk5); ix5 = {w: np.where(wk5 == w)[0] for w in u5}
    res = {}
    for m in ["EG-MoE", "GRU-A", "LightGBM", "[expert] resid", "SeasonalNaive", "ExtraTrees", "CatBoost", "XGBoost", "Ridge", "Persistence"]:
        sk = 1 - mean_squared_error(yy, PP[m]) / mean_squared_error(yy, PP["Persistence"]); bs = []
        for b in range(300):
            ws = rng.choice(u5, len(u5), replace=True); ii = np.concatenate([ix5[w] for w in ws])
            bs.append(1 - mean_squared_error(yy[ii], PP[m][ii]) / mean_squared_error(yy[ii], PP["Persistence"][ii]))
        res[m] = dict(skill=sk, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), R2=r2_score(yy, PP[m]))
    hor[H] = res
scen = {}
for key, fn in [("V1", "pred_H6_chrono_h1_2022.pkl"), ("V2", "pred_H6_chrono_h1_2021.pkl"), ("V3", "pred_H6_xstreet_h1_2022.pkl")]:
    d = pickle.load(open(fn, "rb")); yy = d["y"]
    scen[key] = {m: dict(R2=r2_score(yy, np.clip(d["preds"][m], 0, None)), RMSE=mean_squared_error(yy, np.clip(d["preds"][m], 0, None)) ** .5)
                 for m in ["EG-MoE", "GRU-A", "[expert] resid", "LightGBM", "Persistence"]}
v2 = pickle.load(open("pred_H6_chrono_h1_2021.pkl", "rb"))
tr20 = tr[tr.time < pd.Timestamp("2021-01-01") - pd.Timedelta(hours=6)]
# train-period in-sample predictions of the final EG-MoE (for the train/test scatter)
p_tr = mo.predict(tr[feats], tr.pers.fillna(np.nanmedian(tr.target)).values)
# rolling Mondrian conformal on log1p residuals, episode-probability bins, realised residuals delayed by H
lab = pickle.load(open("lab_H6.pkl", "rb"))
O = lab["O"]; Fm = np.column_stack([np.clip(O[n], 0, None) for n in mo.names])
p_oof = (Fm * mo.gate_weights(O, lab["pi_tr"][O["idx"]])).sum(1); y_oof = lab["ytr"][O["idx"]]
bins_of = lambda p: np.digitize(p, [0.15, 0.5])
r_oof = np.abs(np.log1p(y_oof) - np.log1p(np.clip(p_oof, 0, None))); b_oof = bins_of(O["p"])
ypred = np.clip(B["preds"]["EG-MoE"], 0, None); bte = bins_of(B["p"])
conf = {}
for alpha in [0.1, 0.2]:
    lo = np.zeros(len(y)); hi = np.zeros(len(y)); cover = np.zeros(len(y), bool)
    for s in [1, 2]:
        ii = np.where(st == s)[0]; ii = ii[np.argsort(tm.values[ii])]
        hist = {b: list(r_oof[b_oof == b][-2000:]) for b in range(3)}
        times = tm.values[ii]; pend = []
        for j, i in enumerate(ii):
            while pend and pend[0][0] <= times[j]:
                _, bb, rr = pend.pop(0); hist[bb].append(rr); hist[bb] = hist[bb][-720:] if len(hist[bb]) > 720 else hist[bb]
            q = np.quantile(hist[bte[i]], 1 - alpha)
            lo[i] = np.expm1(max(np.log1p(ypred[i]) - q, 0)); hi[i] = np.expm1(np.log1p(ypred[i]) + q)
            cover[i] = lo[i] <= y[i] <= hi[i]
            pend.append((times[j] + np.timedelta64(6, "h"), bte[i], abs(np.log1p(y[i]) - np.log1p(ypred[i]))))
    conf[alpha] = dict(lo=lo, hi=hi, cover=cover)
# static (non-adaptive) split conformal for comparison
stat = {}
for alpha in [0.1, 0.2]:
    q = {b: np.quantile(r_oof[b_oof == b], 1 - alpha) for b in range(3)}
    qq = np.array([q[b] for b in bte]); l_ = np.expm1(np.maximum(np.log1p(ypred) - qq, 0)); h_ = np.expm1(np.log1p(ypred) + qq)
    stat[alpha] = dict(lo=l_, hi=h_, cover=(y >= l_) & (y <= h_))
OUT["fig5"] = dict(hor=hor, scen=scen, v2=v2, v2_trainmax=tr20.target.max(), tr_y=tr.target.values, tr_p=p_tr,
                   te=B, conf=conf, stat=stat, monthly=pd.read_csv("monthly_r2_H6.csv"))
print("fig5 done", flush=True)

# ---------------- Fig 6: ablation from lab pickle ----------------
T_ = lab["T"]; yte = lab["yte"]; pit = lab["pi_te"]; pio = lab["pi_tr"][O["idx"]]; yo = lab["ytr"][O["idx"]]
names = mo.names
def simplex(Fm_, y_, sw=None):
    sw = np.ones(len(y_)) if sw is None else sw
    f = lambda z: np.sum(sw * (Fm_ @ (np.exp(z) / np.exp(z).sum()) - y_) ** 2)
    z = minimize(f, np.zeros(Fm_.shape[1]), method="L-BFGS-B").x; return np.exp(z) / np.exp(z).sum()
def gated(use, episode=True, regime=True, lam=500):
    FO = np.column_stack([np.clip(O[n], 0, None) for n in use]); FT = np.column_stack([np.clip(T_[n], 0, None) for n in use])
    wg = simplex(FO, yo); pred = np.zeros(len(yte))
    bins = [0, .15, .5, 1.01] if episode else [0, 1.01]
    for b in range(len(bins) - 1):
        mo_ = (O["p"] >= bins[b]) & (O["p"] < bins[b + 1]); mt = (T_["p"] >= bins[b]) & (T_["p"] < bins[b + 1])
        nb = mo_.sum(); wb = (nb * simplex(FO[mo_], yo[mo_]) + lam * wg) / (nb + lam) if nb > 0 else wg
        if regime:
            Wk = []
            for k in range(pio.shape[1]):
                sw = pio[mo_, k]; nk = sw.sum(); wk_ = simplex(FO[mo_], yo[mo_], sw) if nk > 50 else wb
                Wk.append((nk * wk_ + lam * wb) / (nk + lam))
            pred[mt] = (FT[mt] * (pit[mt] @ np.array(Wk))).sum(1)
        else:
            pred[mt] = FT[mt] @ wb
    return pred
def met(p):
    p = np.clip(p, 0, None); return dict(RMSE=mean_squared_error(yte, p) ** .5, MAE=mean_absolute_error(yte, p), absCum=abs(100 * (p.sum() - yte.sum()) / yte.sum()),
                                         R2=r2_score(yte, p), R2_nonjan=r2_score(yte[~jan], p[~jan]))
abl = {"EG-MoE (full)": gated(names), "No regime gate": gated(names, regime=False), "No episode gate": gated(names, episode=False),
       "Global simplex stacking": gated(names, episode=False, regime=False),
       "Equal-weight average": np.column_stack([np.clip(T_[n], 0, None) for n in names]).mean(1),
       "No shrinkage (λ = 0)": gated(names, lam=1e-6), "No anchor/residual experts": gated(["raw", "log", "tweedie"])}
for n in names: abl[f"Expert only: {n}"] = T_[n]
ABL = pd.DataFrame({k: met(v) for k, v in abl.items()}).T
OUT["fig6"] = dict(ABL=ABL, W=mo.W, wglobal=mo.w_global, names=names, Wte=B["W"], p=B["p"], pi=B["pi"])
print(ABL.round(3), flush=True)
# ops-only classifier for ROC/PR
Fo, f0 = build(6, hist=False); cut = pd.Timestamp("2022-01-01")
a_tr = Fo[Fo.time < cut - pd.Timedelta(hours=6)]; a_te = Fo[Fo.time >= cut]
c_ops = lgb.LGBMClassifier(**LG).fit(a_tr[f0], a_tr.target > mo.tau); p_ops = c_ops.predict_proba(a_te[f0])[:, 1]
ev = y > tau
OUT["fig6"].update(roc_gate=roc_curve(ev, B["p"]), roc_ops=roc_curve(a_te.target > mo.tau, p_ops),
                   pr_gate=precision_recall_curve(ev, B["p"]), pr_ops=precision_recall_curve(a_te.target > mo.tau, p_ops),
                   auc_gate=roc_auc_score(ev, B["p"]), auc_ops=roc_auc_score(a_te.target > mo.tau, p_ops),
                   ap_gate=average_precision_score(ev, B["p"]), ap_ops=average_precision_score(a_te.target > mo.tau, p_ops),
                   base=ev.mean(), calib=calibration_curve(ev, B["p"], n_bins=10, strategy="quantile"))
# lead times: onset on hourly observed y (tau), warning when p_t >= 0.5 in the preceding 24 h
Lh = L.set_index(["street", "time"]).y
pser = pd.Series(B["p"], index=pd.MultiIndex.from_arrays([st, tm]))
leads, missed, onsets = [], 0, 0
for s in [1, 2]:
    ys = Lh.loc[s]; ys = ys[(ys.index >= "2022-01-02") & (ys.index < "2023-01-01")]
    above = ys > tau; prev24 = above.shift(1).rolling(24, min_periods=24).max()
    on = ys.index[(above) & (prev24 == 0)]
    ps = pser.loc[s]
    for u in on:
        win = ps[(ps.index >= u - pd.Timedelta(hours=24)) & (ps.index <= u - pd.Timedelta(hours=1))]
        if len(win) < 12: continue
        onsets += 1; hit = win[win >= 0.5]
        if len(hit): leads.append((u - hit.index.min()) / pd.Timedelta(hours=1))
        else: missed += 1
OUT["fig6"].update(leads=np.array(leads), missed=missed, onsets=onsets, tm=tm, st=st, y=y)
print("lead", onsets, missed, np.median(leads) if leads else None, flush=True)
pickle.dump(OUT, open("figdata/fig2to6.pkl", "wb"))

# ---------------- Fig 7: SHAP + ALE ----------------
import shap
Xa = Xte.assign(**{f"regime{k}": pi_te[:, k] for k in range(mo.k)})
Xs = Xa.sample(3000, random_state=0)
expl = shap.TreeExplainer(mo.E["resid"]); sv = expl.shap_values(Xs)
Xi = Xs.sample(800, random_state=1); siv = expl.shap_interaction_values(Xi)
def ale(model, X, f, bins=20):
    q = np.unique(np.nanquantile(X[f], np.linspace(0, 1, bins + 1))); x = X[f].values
    k = np.clip(np.searchsorted(q, x, side="right") - 1, 0, len(q) - 2); eff = []
    for j in range(len(q) - 1):
        S = X[k == j]
        if len(S) == 0: eff.append(0); continue
        lo_, hi_ = S.copy(), S.copy(); lo_[f] = q[j]; hi_[f] = q[j + 1]; eff.append(np.mean(model.predict(hi_) - model.predict(lo_)))
    a = np.concatenate([[0], np.cumsum(eff)]); return q, a - a.mean()
def pdp(model, X, f, grid):
    o_ = []
    for v in grid: Z = X.copy(); Z[f] = v; o_.append(model.predict(Z).mean())
    o_ = np.array(o_); return o_ - o_.mean()
ALE = {}
for f in ["T_in", "DO2", "NO3", "Supern", "Cl_m72", "Q_out"]:
    q, a = ale(mo.E["resid"], Xs, f); bs = []
    for b in range(20):
        Xb = Xs.sample(len(Xs), replace=True, random_state=b)
        qb, ab = ale(mo.E["resid"], Xb, f); bs.append(np.interp(q, qb, ab))
    ALE[f] = dict(q=q, a=a, lo=np.percentile(bs, 2.5, axis=0), hi=np.percentile(bs, 97.5, axis=0), pdp=pdp(mo.E["resid"], Xs, f, q), x=Xs[f].values)
pickle.dump(dict(sv=sv, Xs=Xs, siv=siv, Xi=Xi, ALE=ALE, lab=Xs[[f"regime{k}" for k in range(mo.k)]].values.argmax(1)), open("figdata/fig7.pkl", "wb"))
print("fig7 done", flush=True)
