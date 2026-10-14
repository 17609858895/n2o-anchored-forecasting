import numpy as np, pickle, pandas as pd, sys
from scipy.optimize import minimize, nnls
from bench2 import metrics
H=int(sys.argv[1]) if len(sys.argv)>1 else 6
D=pickle.load(open(f"lab_H{H}.pkl","rb")); O,T=D["O"],D["T"]; ytr,yte,tau,ate=D["ytr"],D["yte"],D["tau"],D["ate"]
names=["anchor","resid","raw","log","tweedie"]
yo=ytr[O["idx"]]; pio=D["pi_tr"][O["idx"]]; pit=D["pi_te"]
FO=np.column_stack([np.clip(O[n],0,None) for n in names]); FT=np.column_stack([np.clip(T[n],0,None) for n in names])
def simplex_w(Fm,y,sw=None):
    k=Fm.shape[1]; sw=np.ones(len(y)) if sw is None else sw
    f=lambda z: np.sum(sw*(Fm@ (np.exp(z)/np.exp(z).sum()) - y)**2)
    z=minimize(f,np.zeros(k),method="L-BFGS-B").x; return np.exp(z)/np.exp(z).sum()
res={}
res["Persistence"]=metrics(yte,ate,tau,ate)
for i,n in enumerate(names): res["exp_"+n]=metrics(yte,FT[:,i],tau,ate)
res["mean5"]=metrics(yte,FT.mean(1),tau,ate)
late=O["fold"]>=3
for tag,m in [("all",np.ones(len(yo),bool)),("late",late)]:
    wg=simplex_w(FO[m],yo[m]); res[f"global_simplex_{tag}"]=metrics(yte,FT@wg,tau,ate); print(tag,"global w",wg.round(3))
    # episode gated: bins of p
    bins=[0,.15,.5,1.01]; pred=np.zeros(len(yte)); 
    for b in range(3):
        mo=m&(O["p"]>=bins[b])&(O["p"]<bins[b+1]); mt=(T["p"]>=bins[b])&(T["p"]<bins[b+1])
        n=mo.sum(); w=simplex_w(FO[mo],yo[mo]); lam=500; w=(n*w+lam*wg)/(n+lam); pred[mt]=FT[mt]@w; print(tag,"bin",b,n,w.round(3))
    res[f"episode_gated_{tag}"]=metrics(yte,pred,tau,ate)
    # regime x episode soft gating: weights = sum_k pi_k * w_k (each w_k simplex, shrunk), per episode bin
    pred2=np.zeros(len(yte))
    for b in range(3):
        mo=m&(O["p"]>=bins[b])&(O["p"]<bins[b+1]); mt=(T["p"]>=bins[b])&(T["p"]<bins[b+1])
        wb=simplex_w(FO[mo],yo[mo]); nb=mo.sum(); wb=(nb*wb+500*wg)/(nb+500)
        Wk=[]
        for k in range(pio.shape[1]):
            sw=pio[mo,k]; nk=sw.sum()
            wk=simplex_w(FO[mo],yo[mo],sw) if nk>50 else wb; Wk.append((nk*wk+500*wb)/(nk+500))
        Wk=np.array(Wk); Wx=pit[mt]@Wk; pred2[mt]=(FT[mt]*Wx).sum(1)
    res[f"regime_episode_gated_{tag}"]=metrics(yte,pred2,tau,ate)
print(pd.DataFrame(res).T.round(3))
