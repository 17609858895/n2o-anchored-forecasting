"""Independently recompute saved Hofen sensitivity metrics without fitting models."""
from pathlib import Path
import argparse, hashlib, json, itertools
import numpy as np
import pandas as pd

PKG=Path(__file__).resolve().parents[2]
ROOT=PKG.parent
OUT=PKG/'results/baseline_training_sensitivity'

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-count',type=int,default=32)
    args=parser.parse_args()
    files=sorted((OUT/'jobs').glob('*/result.json'))
    assert len(files)==args.expected_count, (len(files),args.expected_count)
    design=json.loads((OUT/'predeclared_design.json').read_text(encoding='utf-8'))
    expected=set()
    for section in ['primary_configs','additional_initialization_configs']:
        spec=design[section]
        for architecture,variant,seed,formulation in itertools.product(spec['architectures'],spec['variants'],spec['seeds'],spec['formulations']):
            expected.add(f'{variant}_{architecture}_seed{seed}_{formulation}')
    assert all(f.parent.name in expected for f in files)
    if args.expected_count==32:assert {f.parent.name for f in files}==expected
    records=[]
    for file in files:
        job=file.parent
        r=json.loads(file.read_text(encoding='utf-8'))
        c=json.loads((job/'configuration.json').read_text(encoding='utf-8'))
        assert r['config_complete'] is True
        for key in ['architecture','formulation','variant','seed']:
            assert r[key]==c[key]
        assert c['max_epochs']==60 and c['extended_epochs']==60
        assert c['min_epochs']==20 and c['patience']==15
        assert c['device']=='cuda' and c['rnn_cudnn']=='on'
        assert c['environment']['deterministic_algorithms'] and c['environment']['tf32_enabled'] is False
        assert c['windows_cuda_teardown_workaround'] is True
        assert c['script_sha256']==sha(PKG/'code/analysis/baseline_training_sensitivity.py')
        for rel,expected in c['source_hashes'].items():
            assert sha(ROOT/rel)==expected
        raw=ROOT/'N2O_ML_paper/data/bd08c9ec-e189-482f-91cb-4269c21f7a44-n2oemissionsoperatingdatahofen/raw_data/qgl_hof_raw.csv'
        assert sha(raw)==c['data']['raw_sha256']
        for name,expected in r['artifact_sha256'].items():
            assert sha(job/name)==expected
        curves=pd.read_csv(job/'learning_curve.csv')
        outer=pd.read_csv(job/'outer_learning_curve.csv')
        assert curves.epoch.tolist()==list(range(1,r['evaluated_epochs']+1))
        assert int(curves.loc[curves.is_best,'epoch'].iloc[0])==r['selected_epoch']
        assert int(curves.validation_rmse.idxmin())+1==r['selected_epoch']
        assert outer.epoch.tolist()==list(range(1,61)) and outer.validation_rmse.isna().all()
        assert r['outer_evaluated_epochs']==60 and r['extended_epoch']==60
        for prefix,name in [('checkpoint','checkpoint_preprocessing.json'),('outer','outer_preprocessing.json')]:
            normalization=json.loads((job/name).read_text(encoding='utf-8'))
            assert normalization['fit_input_sha256']==r[f'{prefix}_input_sha256']
        with np.load(job/'test_predictions.npz') as p:
            y=p['target'];anchor=p['persistence']
            assert len(y)==r['n_test']==c['data']['n_test']==13785
            assert np.isfinite(y).all() and np.isfinite(anchor).all()
            assert pd.to_datetime(p['time']).min()>=pd.Timestamp('2022-01-01')
            assert pd.to_datetime(p['time']).max()<pd.Timestamp('2023-01-01')
            for label,prefix in [('selected','test'),('epoch12','test_epoch12'),('extended','test_extended')]:
                prediction=p[label]
                assert len(prediction)==len(y) and np.isfinite(prediction).all() and (prediction>=0).all()
                sse=float(np.sum((y-prediction)**2))
                independent=dict(R2=1-sse/float(np.sum((y-y.mean())**2)),
                                 RMSE=np.sqrt(sse/len(y)),
                                 Skill=1-sse/float(np.sum((y-anchor)**2)),
                                 MAE=np.mean(np.abs(y-prediction)),
                                 CumErr=100*(prediction.sum()-y.sum())/y.sum())
                for metric,value in independent.items():
                    if f'{prefix}_{metric}' in r:
                        assert np.isclose(value,r[f'{prefix}_{metric}'],rtol=1e-10,atol=1e-10),(job.name,label,metric)
                records.append(dict(job=job.name,checkpoint=label,**independent))
            if r['selected_epoch']==12: assert np.array_equal(p['selected'],p['epoch12'])
            if r['selected_epoch']==60: assert np.array_equal(p['selected'],p['extended'])
    paired_count=0
    for f in files:
        if not f.parent.name.endswith('_direct'):continue
        a=f.parent.with_name(f.parent.name[:-6]+'anchored')
        if not (a/'result.json').exists():continue
        direct=json.loads(f.read_text(encoding='utf-8'));anchored=json.loads((a/'result.json').read_text(encoding='utf-8'))
        for field in ['initial_weights_sha256','outer_initial_weights_sha256','checkpoint_input_sha256','outer_input_sha256']:
            assert direct[field]==anchored[field]
        for name in ['checkpoint_preprocessing.json','outer_preprocessing.json']:
            assert json.loads((f.parent/name).read_text(encoding='utf-8'))==json.loads((a/name).read_text(encoding='utf-8'))
        for name in ['learning_curve.csv','outer_learning_curve.csv']:
            dcurve=pd.read_csv(f.parent/name);acurve=pd.read_csv(a/name);n=min(len(dcurve),len(acurve))
            assert dcurve.batch_order_sha256.iloc[:n].tolist()==acurve.batch_order_sha256.iloc[:n].tolist()
        paired_count+=1
    pd.DataFrame(records).to_csv(OUT/'independent_metric_recalculation.csv',index=False)
    report=dict(passed=True,verified_configs=len(files),verified_prediction_sets=len(records),verified_pairs=paired_count,
                checks=['configuration/source/data/artifact hashes','completion and fixed budgets',
                        'validation-only best checkpoint','full-refit training curves',
                        'saved test-array metrics recomputed independently','paired inputs/initialization/shared batch-order'],test_tuning_performed=False)
    (OUT/'independent_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':main()
