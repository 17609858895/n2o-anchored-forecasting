import numpy as np, pandas as pd, warnings, pickle, sys; warnings.filterwarnings("ignore")
from bench2 import build, EGMoE, metrics
H=int(sys.argv[1]) if len(sys.argv)>1 else 6
F,feats=build(H); cut=pd.Timestamp("2022-01-01")
tr=F[F.time<cut-pd.Timedelta(hours=H)]; te=F[F.time>=cut]; med=np.nanmedian(tr.target)
Xtr,Xte,ytr,yte=tr[feats].reset_index(drop=True),te[feats].reset_index(drop=True),tr.target.values,te.target.values
atr,ate=tr.pers.fillna(med).values,te.pers.fillna(med).values; tau=np.quantile(ytr,.8)
mo=EGMoE(tau); pi=mo._regime(Xtr,fit=True); Xa=Xtr.assign(**{f"regime{k}":pi[:,k] for k in range(mo.k)})
order=np.argsort(tr.time.values); cuts=np.array_split(order,5); oof=[]
for i in range(1,5):
    a=np.concatenate(cuts[:i]); v=cuts[i]; E=mo._fit_experts(Xa.iloc[a],ytr[a],atr[a]); P=mo._pred_experts(E,Xa.iloc[v],atr[v]); P["idx"]=v; P["fold"]=np.full(len(v),i); oof.append(P)
O={k:np.concatenate([p[k] for p in oof]) for k in oof[0]}
pit=mo._regime(Xte); Xb=Xte.assign(**{f"regime{k}":pit[:,k] for k in range(mo.k)})
E=mo._fit_experts(Xa,ytr,atr); T=mo._pred_experts(E,Xb,ate)
pickle.dump(dict(O=O,T=T,pi_tr=pi,pi_te=pit,ytr=ytr,yte=yte,tau=tau,ate=ate,time_te=te.time.values,street_te=te.street.values),open(f"lab_H{H}.pkl","wb"))
print("saved")
