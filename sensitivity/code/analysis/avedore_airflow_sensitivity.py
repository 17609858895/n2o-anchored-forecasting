"""Forecast an airflow-weighted dissolved-N2O index; this is NOT an emission flux.

Run with Python from N2O_ML_paper/code/analysis. Outputs are written under sensitivity/results.
The frozen reference modules retain the supplied Avedore benchmark settings.
"""
from pathlib import Path
import hashlib, json, sys, time, platform, shutil
import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[2]
ORIGINAL = PKG.parent / 'N2O_ML_paper'
OUT = PKG / 'results/avedore_airflow'
REF = Path(__file__).parent / 'reference'
OUT.mkdir(parents=True, exist_ok=True)
REF.mkdir(exist_ok=True)
manifest = {}
for name in ['aved_replication.py', 'bench2.py', 'deep.py', 'stats_joint.py', 'prep.py']:
    src = ORIGINAL / 'code/analysis' / name
    dst = REF / name
    if not dst.exists():
        shutil.copy2(src, dst)
    manifest[name] = hashlib.sha256(dst.read_bytes()).hexdigest()
sys.path.insert(0, str(REF))
import aved_replication as A
import lightgbm as lgb
import torch, sklearn, scipy, xgboost, catboost
from stats_joint import dm_joint


def save_preds(payload):
    pd.to_pickle(payload, OUT / 'predictions.pkl')


def main():
    started = time.time()
    df = pd.read_csv(ORIGINAL / 'data/avedore/aved_raw.csv.gz')
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.rename(columns={c: A.REN[c.replace('BIOLOGY.', '').replace(' value', '')]
                            for c in df.columns if c != 'time'})
    bad_air = int((df.Air1 < 0).sum())
    saturated = int((df.N2O >= 12).sum())
    df.loc[df.N2O >= 12, 'N2O'] = np.nan
    df['N2O'] = df.N2O.clip(lower=0)
    df.loc[df.Air1 < 0, 'Air1'] = np.nan
    for c in ['NH4', 'NO3', 'DO1', 'DO2', 'PO4']:
        df[c] = df[c].clip(lower=0)
    df.loc[df.Q_in < 0, 'Q_in'] = np.nan
    # Multiply contemporaneous 2-min samples BEFORE aggregation. Both must be valid.
    df['Index'] = df.Air1 * df.N2O
    valid_pairs = int(df.Index.notna().sum())
    h = df.set_index('time').resample('1h').mean()
    h.index = h.index.tz_convert('Europe/Copenhagen')
    h['ProductOfHourlyMeans'] = h.Air1 * h.N2O
    h.to_csv(OUT / 'hourly_index.csv.gz', compression='gzip')
    hp = h.reset_index()
    # Reuse all 75 predictors. Histories become log(1 + index / one index unit).
    # This /100 exactly cancels the reference feature builder's factor of 100.
    hp['N2O'] = hp['Index'] / 100
    Ffull, feats = A.features(hp)
    for c in ['target', 'pers', 'pers_day']:
        Ffull[c] *= 100
    direct = h.Index.iloc[::-1].rolling(6, min_periods=5).mean().iloc[::-1].shift(-1)
    np.testing.assert_allclose(Ffull.target, direct.to_numpy(), equal_nan=True)
    F = Ffull.dropna(subset=['target'])
    tr = F[F.time < A.SPLIT - pd.Timedelta(hours=A.H)]
    te = F[F.time >= A.SPLIT]
    assert tr.time.max() + pd.Timedelta(hours=6) < te.time.min()
    assert len(feats) == 75
    med = float(np.nanmedian(tr.target))
    ytr, yte = tr.target.values, te.target.values
    atr, ate = tr.pers.fillna(med).values, te.pers.fillna(med).values
    tau = float(np.quantile(ytr, .8))
    info = dict(target='airflow-weighted dissolved-N2O index',
                formula='hourly mean of contemporaneous 2-min Air1 * cleaned dissolved N2O',
                interpretation='sensitivity index only; not measured or calibrated off-gas emission',
                index_units='(Nm3 h-1)(mg N L-1)',
                reference_module_sha256=manifest, raw_records=len(df), valid_raw_pairs=valid_pairs,
                removed_saturated=saturated, negative_airflow_removed=bad_air,
                hours=len(h), valid_index_hours=int(h.Index.notna().sum()),
                n_train=len(tr), n_test=len(te), tau=tau,
                train_max=float(ytr.max()), test_max=float(yte.max()),
                n_tabular_inputs=len(feats), n_sequence_inputs=24, sequence_hours=48,
                epochs=12, seed=42, torch_cpu_threads=torch.get_num_threads(),
                missing_training_anchors=int(tr.pers.isna().sum()),
                missing_test_anchors=int(te.pers.isna().sum()),
                train_origin_start=str(tr.time.min()), train_origin_end=str(tr.time.max()),
                test_origin_start=str(te.time.min()), test_origin_end=str(te.time.max()),
                hourly_product_comparison=dict(
                    pearson=float(h[['Index','ProductOfHourlyMeans']].corr().iloc[0,1]),
                    mean_absolute_difference=float((h.Index-h.ProductOfHourlyMeans).abs().mean())),
                environment={m.__name__: m.__version__ for m in
                             [numpy_module(), pd, torch, lgb, sklearn, scipy, xgboost, catboost]})
    info['environment']['python'] = platform.python_version()
    (OUT/'summary.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps({k: info[k] for k in ['n_train','n_test','tau','train_max','test_max','negative_airflow_removed']}), flush=True)

    if (OUT/'predictions.pkl').exists():
        payload = pd.read_pickle(OUT/'predictions.pkl')
        np.testing.assert_allclose(payload['y'], yte)
        assert np.array_equal(payload['time'], te.time.values)
        preds = payload['preds']
    else:
        preds = {'Persistence': ate, 'Seasonal naive': te.pers_day.fillna(med).values,
                 'Climatology': np.full(len(te), ytr.mean())}
        payload = dict(time=te.time.values, y=yte, preds=preds)
        save_preds(payload)
    for name, m in A.tab_models().items():
        if name in preds:
            continue
        t0 = time.time()
        preds[name] = m.fit(tr[feats], ytr).predict(te[feats])
        save_preds(payload)
        print(f'{name}: {time.time()-t0:.0f} s', flush=True)
    if 'Anchored LightGBM' not in preds:
        preds['Anchored LightGBM'] = ate + lgb.LGBMRegressor(**A.B.LG).fit(tr[feats], ytr-atr).predict(te[feats])
        save_preds(payload)
    A.B.REG_VARS = A.REG
    if 'EG-MoE' not in preds:
        t0 = time.time()
        mo = A.B.EGMoE(tau).fit(tr[feats], ytr, atr, tr.time.values)
        preds['EG-MoE'] = mo.predict(te[feats], ate)
        save_preds(payload)
        print(f'EG-MoE: {time.time()-t0:.0f} s', flush=True)
    for anchored in [False, True]:
        names = [n + ('-A' if anchored else '') for n in ['LSTM','GRU','TCN','Transformer']]
        if all(n in preds for n in names):
            continue
        preds.update(A.nets(Ffull, tr, te, anchored, epochs=12))
        save_preds(payload)
    assert len(preds) == 19
    for name, p in preds.items():
        assert len(p) == len(te) and np.isfinite(p).all(), name
    R = pd.DataFrame([dict(model=n, **A.metrics(yte, np.asarray(p), ate, tau, te.time.values))
                      for n,p in preds.items()])
    eg = np.clip(preds['EG-MoE'], 0, None)
    dm = {n: dm_joint(yte-eg, yte-np.clip(p, 0, None), te.time.values, 6)
          for n,p in preds.items() if n != 'EG-MoE'}
    R['DM_vs_EGMoE'] = R.model.map(lambda n: dm[n][0] if n in dm else np.nan)
    R['P_vs_EGMoE'] = R.model.map(lambda n: dm[n][1] if n in dm else np.nan)
    for col in ['P_vs_pers', 'P_vs_EGMoE']:
        order = R[col].dropna().sort_values().index
        R.loc[order, col+'_Holm'] = np.maximum.accumulate(
            R.loc[order,col].values * (len(order)-np.arange(len(order)))).clip(0,1)
    R = R.sort_values('R2', ascending=False)
    R.to_csv(OUT/'benchmark.csv', index=False)
    print(R[['model','R2','RMSE','Skill','P_vs_pers_Holm']].round(4).to_string(index=False), flush=True)

    rows = []
    for rep in range(5):
        mask = np.random.RandomState(42+rep).rand(len(F)) < .3
        a,b = F[~mask],F[mask]
        pa = b.pers.fillna(np.nanmedian(a.target)).values
        for name in ['LightGBM','Random forest','Extra trees']:
            p = np.clip(A.tab_models()[name].fit(a[feats], a.target).predict(b[feats]), 0, None)
            rows.append(dict(design='random',rep=rep,model=name,
                             R2=A.r2_score(b.target,p),
                             Skill=1-A.mean_squared_error(b.target,p)/A.mean_squared_error(b.target,pa)))
        rows.append(dict(design='random',rep=rep,model='Persistence',R2=A.r2_score(b.target,pa),Skill=0.0))
        pd.DataFrame(rows).to_csv(OUT/'split_audit_partial.csv',index=False)
        print(f'Random-split repeat {rep+1}/5 complete',flush=True)
    for name in ['LightGBM','Random forest','Extra trees','Persistence']:
        r = R[R.model==name].iloc[0]
        rows.append(dict(design='chrono',rep=0,model=name,R2=r.R2,Skill=r.Skill))
    pd.DataFrame(rows).to_csv(OUT/'split_audit.csv',index=False)
    info['elapsed_seconds'] = time.time()-started
    info['complete'] = True
    (OUT/'summary.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(f'Completed in {time.time()-started:.0f} s',flush=True)


def numpy_module():
    return np


if __name__ == '__main__':
    main()
