import pandas as pd, pickle, sys
from bench2 import run
SUB=["Ridge","ExtraTrees","XGBoost","LightGBM","CatBoost"]
jobs=[dict(H=6),dict(H=1),dict(H=3),dict(H=12),dict(H=24),dict(H=6,test_year=2021),dict(H=6,scenario="xstreet"),dict(H=6,hist=False)]
allR=[]
for j in jobs:
    kw=dict(H=6,scenario="chrono",hist=True,deep=False,models=SUB,test_year=2022); kw.update(j)
    print("JOB",kw,flush=True)
    R,out=run(**kw); R["test_year"]=kw["test_year"]; allR.append(R)
    tag=f"H{kw['H']}_{kw['scenario']}_h{int(kw['hist'])}_{kw['test_year']}"
    pickle.dump(dict(preds=out["preds"],tau=out["tau"],y=out["te"].target.values,time=out["te"].time.values,street=out["te"].street.values,
                     W=out["P"]["W"],p=out["P"]["p"],pi=out["pi"]),open(f"pred_{tag}.pkl","wb"))
    if kw["H"]==6 and kw["scenario"]=="chrono" and kw["hist"] and kw["test_year"]==2022:
        pickle.dump(out["model"],open("egmoe_H6.pkl","wb")); out["te"].to_pickle("te_H6.pkl"); out["tr"].to_pickle("tr_H6.pkl")
        pickle.dump(out["feats"],open("feats.pkl","wb"))
    pd.concat(allR).to_csv("pilot_all.csv",index=False)
print("DONE")
