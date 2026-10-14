"""Hofen training-budget and scale sensitivity for matched sequence models.

Run with the research Python that supplies PyTorch (D:/Anaconda/python.exe on
the current host). All new caches, curves and forecasts stay in the dedicated
sensitivity directory. Historical code/results and publication files are read-only.
"""
from pathlib import Path
import argparse, gc, hashlib, importlib.util, json, os, sys, time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

PKG = Path(__file__).resolve().parents[2]
ROOT = PKG.parent
ORIGINAL = ROOT / 'N2O_ML_paper'
OUT = PKG / 'results/baseline_training_sensitivity'
SHUTDOWN_WORKAROUND = False
H, SEQ = 6, 48
INNER_BOUNDARY = pd.Timestamp('2021-07-01')
TEST_BOUNDARY = pd.Timestamp('2022-01-01')
VARIANTS = {
    'legacy': 'log-current-emission input; uncentred original-scale output divided by its training SD',
    'centered': 'log-current-emission input; target/residual centred and scaled using training statistics',
    'raw_anchor': 'raw-current-emission input; target/residual centred and scaled using training statistics',
}

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def dump(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def state_hash(state):
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode()); h.update(state[key].cpu().numpy().tobytes())
    return h.hexdigest()

def array_hash(values):
    values = np.ascontiguousarray(values)
    return hashlib.sha256(memoryview(values).cast('B')).hexdigest()

def portable_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

def initialize_model(deep, architecture, channels, seed, device, rnn_cudnn):
    # Initialize on CPU before transfer so paired initial weights are identical.
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    # Recurrent backend is predeclared and identical within each matched pair.
    torch.backends.cudnn.enabled = rnn_cudnn == 'on' if architecture in ['LSTM','GRU'] else True
    model = model_factory(deep, architecture, channels)
    initial_hash = state_hash(model.state_dict())
    return model.to(device), initial_hash

def metrics(y, p, anchor):
    p = np.clip(np.asarray(p), 0, None)
    mse = mean_squared_error(y, p)
    return dict(R2=float(r2_score(y, p)), RMSE=float(np.sqrt(mse)),
                MAE=float(mean_absolute_error(y, p)),
                Skill=float(1 - mse / mean_squared_error(y, anchor)),
                CumErr=float(100 * (p.sum() - y.sum()) / y.sum()))

def data(prep, deep):
    raw = ORIGINAL / 'data/bd08c9ec-e189-482f-91cb-4269c21f7a44-n2oemissionsoperatingdatahofen/raw_data/qgl_hof_raw.csv'
    prep.RAW = str(raw)
    long = prep.load_long()
    assert int(long.y.notna().sum()) == 44227
    features = prep.engineer(long, 1, use_n2o_hist=True)
    g = long.groupby('street')['y']
    target = g.transform(lambda x: x.iloc[::-1].rolling(H, min_periods=5).mean().iloc[::-1].shift(-1))
    frame = long[['time', 'street']].copy()
    frame['target'], frame['anchor'] = target, long.y
    frame = frame.dropna(subset=['target'])
    full_train = frame[frame.time < TEST_BOUNDARY - pd.Timedelta(hours=H)]
    inner_train = full_train[full_train.time < INNER_BOUNDARY - pd.Timedelta(hours=H)]
    validation = full_train[full_train.time >= INNER_BOUNDARY]
    test = frame[(frame.time >= TEST_BOUNDARY) & (frame.time < pd.Timestamp('2023-01-01'))]
    assert inner_train.time.max() + pd.Timedelta(hours=H) < validation.time.min()
    assert full_train.time.max() + pd.Timedelta(hours=H) < test.time.min()
    assert set(inner_train.index).isdisjoint(validation.index)
    assert set(full_train.index).isdisjoint(test.index)
    cols = deep.COLS + ['yl0']
    e = features[cols + ['street']].copy()
    # All forward filling is causal and stays within each street.
    e[cols] = e.groupby('street')[cols].ffill(limit=6)
    starts = long.groupby('street').apply(lambda x: x.index.min(), include_groups=False).to_dict()
    eligible = np.asarray(long.index) - long.street.map(starts).to_numpy() >= SEQ - 1
    manifest = dict(raw_sha256=sha(raw), H=H, sequence_hours=SEQ, feature_channels=cols,
                    target='mean of following six hourly emissions, at least five observed',
                    outer_training='2020-2021, purged by six hours before 2022',
                    checkpoint_fit='2020-June 2021, purged by six hours before July 2021',
                    checkpoint_validation='July-December 2021, excluding the six-hour outer-boundary buffer',
                    held_out_test='2022', n_inner_train=len(inner_train), n_validation=len(validation),
                    n_outer_train=len(full_train), n_test=len(test),
                    inner_train_max=str(inner_train.time.max()), validation_min=str(validation.time.min()),
                    outer_train_max=str(full_train.time.max()), test_min=str(test.time.min()),
                    purge_checks_passed=True, missing_targets_imputed=False,
                    checkpoint_selection_limit='The July-December validation period does not contain a January peak; the fixed extended outer endpoint is evaluated separately.')
    return long, e, cols, eligible, inner_train, validation, full_train, test, manifest

def stage_data(long, base, cols, eligible, fit, evaluate, variant):
    e = base.copy()
    if variant == 'raw_anchor':
        e['yl0'] = long.y
        e['yl0'] = e.groupby('street')['yl0'].ffill(limit=6)
    mu, sd = e.loc[fit.index, cols].mean(), e.loc[fit.index, cols].std() + 1e-6
    values = ((e[cols] - mu) / sd).fillna(0).to_numpy(dtype=np.float32)
    med = float(fit.target.median())
    def pack(df):
        ie = df.index.to_numpy()[eligible[df.index.to_numpy()]]
        x = np.stack([values[i-SEQ+1:i+1] for i in ie])
        return dict(frame=df, indices=ie, X=torch.from_numpy(x),
                    y=df.target.loc[ie].to_numpy(),
                    anchor=df.anchor.loc[ie].fillna(med).to_numpy(),
                    full_y=df.target.to_numpy(), full_anchor=df.anchor.fillna(med).to_numpy())
    fit_pack, eval_pack = pack(fit), pack(evaluate)
    return fit_pack, eval_pack, dict(input_mean=mu.to_dict(), input_sd=sd.to_dict(), fallback_median=med,
                                   fit_input_sha256=array_hash(fit_pack['X'].numpy()),
                                   evaluation_input_sha256=array_hash(eval_pack['X'].numpy()))

def target_scale(fit_pack, formulation, variant):
    z = fit_pack['y'] if formulation == 'direct' else fit_pack['y'] - fit_pack['anchor']
    # The legacy direct implementation used pandas ddof=1 on the entire fit pool;
    # the legacy anchored implementation used numpy ddof=0 on eligible sequences.
    scale = float(fit_pack['frame'].target.std()) if variant == 'legacy' and formulation == 'direct' else float(np.std(z))
    centre = 0.0 if variant == 'legacy' else float(np.mean(z))
    assert np.isfinite(scale) and scale > 0
    return torch.from_numpy(((z-centre)/scale).astype(np.float32)), centre, scale

def model_factory(deep, architecture, channels):
    if architecture == 'LSTM': return deep.RNN(channels, torch.nn.LSTM)
    if architecture == 'GRU': return deep.RNN(channels, torch.nn.GRU)
    if architecture == 'TCN': return deep.TCN(channels)
    return deep.TF(channels)

def predict(model, pack, centre, scale, formulation, eval_batch):
    model.eval()
    output = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for i in range(0, len(pack['X']), eval_batch):
            output.append(model(pack['X'][i:i+eval_batch].to(device)).cpu().numpy())
    p = np.concatenate(output).astype(np.float64) * scale + centre
    if formulation == 'anchored': p += pack['anchor']
    return np.clip(p, 0, None)

def full_prediction(model, pack, centre, scale, formulation, med, eval_batch):
    fallback = np.full(len(pack['frame']), med) if formulation == 'direct' else pack['full_anchor'].copy()
    position = pack['frame'].index.get_indexer(pack['indices'])
    fallback[position] = predict(model, pack, centre, scale, formulation, eval_batch)
    return np.clip(fallback, 0, None)

def subset(pack, n):
    idx = np.random.default_rng(0).choice(len(pack['X']), min(n, len(pack['X'])), replace=False)
    return dict(X=pack['X'][idx], y=pack['y'][idx], anchor=pack['anchor'][idx])

def fit_inner(deep, args, architecture, formulation, variant, seed, train, val, directory):
    model, initial_hash = initialize_model(deep, architecture, train['X'].shape[-1], seed, args.device, args.rnn_cudnn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    y, centre, scale = target_scale(train, formulation, variant)
    generator = torch.Generator().manual_seed(seed)
    train_curve = subset(train, args.curve_train_n)
    best, best_epoch, best_state, stale, curves, stopped = np.inf, 0, None, 0, [], False
    start = time.monotonic()
    for epoch in range(1, args.max_epochs + 1):
        model.train(); permutation = torch.randperm(len(y), generator=generator)
        for pos in range(0, len(y), args.batch_size):
            j = permutation[pos:pos+args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(train['X'][j].to(args.device)), y[j].to(args.device))
            assert torch.isfinite(loss), 'Non-finite training loss'
            loss.backward(); optimizer.step()
        tp = predict(model, train_curve, centre, scale, formulation, args.eval_batch)
        vp = predict(model, val, centre, scale, formulation, args.eval_batch)
        trmse = float(np.sqrt(mean_squared_error(train_curve['y'], tp)))
        vrmse = float(np.sqrt(mean_squared_error(val['y'], vp)))
        improved = vrmse < best
        if improved:
            best, best_epoch, best_state, stale = vrmse, epoch, portable_state(model), 0
        else: stale += 1
        curves.append(dict(architecture=architecture, formulation=formulation, variant=variant,
                           variant_description=VARIANTS[variant], seed=seed, epoch=epoch,
                           stage='checkpoint_selection', train_rmse=trmse, validation_rmse=vrmse, is_best=False,
                           batch_order_sha256=array_hash(permutation.numpy()),
                           n_train_curve=len(train_curve['y']), elapsed_seconds=time.monotonic()-start))
        pd.DataFrame(curves).to_csv(directory/'learning_curve.csv', index=False)
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps(dict(job=directory.name, epoch=epoch, validation_rmse=round(vrmse,4),
                                  best_epoch=best_epoch, elapsed_seconds=round(time.monotonic()-start,1))), flush=True)
        if epoch >= args.min_epochs and stale >= args.patience:
            stopped = True; break
    for row in curves: row['is_best'] = row['epoch'] == best_epoch
    pd.DataFrame(curves).to_csv(directory/'learning_curve.csv', index=False)
    torch.save(best_state, directory/'inner_selected.pt')
    return dict(selected_epoch=best_epoch, evaluated_epochs=len(curves),
                validation_RMSE=best, reached_max=len(curves)==args.max_epochs,
                earlystop=stopped, patience=args.patience, initial_weights_sha256=initial_hash,
                curve_train_n=len(train_curve['y']), elapsed_inner_seconds=time.monotonic()-start,
                checkpoint_target_center=centre, checkpoint_target_scale=scale)

def fit_outer(deep, args, architecture, formulation, variant, seed, train, test, selected, directory):
    model, initial_hash = initialize_model(deep, architecture, train['X'].shape[-1], seed, args.device, args.rnn_cudnn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    y, centre, scale = target_scale(train, formulation, variant)
    generator = torch.Generator().manual_seed(seed)
    states = {}; curves = []; start = time.monotonic()
    train_curve = subset(train, args.curve_train_n)
    outer_epochs = max(12, selected, args.extended_epochs)
    for epoch in range(1, outer_epochs + 1):
        model.train(); permutation = torch.randperm(len(y), generator=generator)
        for pos in range(0, len(y), args.batch_size):
            j = permutation[pos:pos+args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(train['X'][j].to(args.device)), y[j].to(args.device))
            assert torch.isfinite(loss)
            loss.backward(); optimizer.step()
        if epoch in {12, selected, args.extended_epochs}: states[epoch] = portable_state(model)
        tp = predict(model, train_curve, centre, scale, formulation, args.eval_batch)
        curves.append(dict(architecture=architecture, formulation=formulation, variant=variant,
                           variant_description=VARIANTS[variant], seed=seed, epoch=epoch,
                           stage='full_training_refit', train_rmse=float(np.sqrt(mean_squared_error(train_curve['y'], tp))),
                           validation_rmse=float('nan'), is_best=epoch==selected,
                           batch_order_sha256=array_hash(permutation.numpy()),
                           is_epoch12=epoch==12, is_extended_endpoint=epoch==args.extended_epochs,
                           n_train_curve=len(train_curve['y']), elapsed_seconds=time.monotonic()-start))
        pd.DataFrame(curves).to_csv(directory/'outer_learning_curve.csv', index=False)
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps(dict(job=directory.name, stage='outer', epoch=epoch,
                                  elapsed_seconds=round(time.monotonic()-start,1))), flush=True)
    # The test set is accessed only after training/checkpoint selection is complete.
    results={}; pred={}
    for label, ep in [('epoch12',12), ('selected',selected), ('extended',args.extended_epochs)]:
        model.load_state_dict(states[ep])
        p=full_prediction(model,test,centre,scale,formulation,float(train['frame'].target.median()),args.eval_batch)
        pred[label]=p; results[label]=metrics(test['full_y'],p,test['full_anchor'])
    model.load_state_dict(states[selected])
    train_p=full_prediction(model,train,centre,scale,formulation,float(train['frame'].target.median()),args.eval_batch)
    train_metrics=metrics(train['full_y'],train_p,train['full_anchor'])
    torch.save(states[selected], directory/'outer_selected.pt')
    torch.save(states[args.extended_epochs], directory/'outer_extended.pt')
    np.savez_compressed(directory/'test_predictions.npz', time=test['frame'].time.to_numpy(dtype='datetime64[ns]'),
                        street=test['frame'].street.to_numpy(), target=test['full_y'],
                        persistence=test['full_anchor'], epoch12=pred['epoch12'], selected=pred['selected'], extended=pred['extended'])
    return dict(test_R2=results['selected']['R2'],test_RMSE=results['selected']['RMSE'],
                test_Skill=results['selected']['Skill'],test_MAE=results['selected']['MAE'],
                test_CumErr=results['selected']['CumErr'],test_epoch12_R2=results['epoch12']['R2'],
                test_epoch12_RMSE=results['epoch12']['RMSE'],test_epoch12_Skill=results['epoch12']['Skill'],
                test_extended_R2=results['extended']['R2'],test_extended_RMSE=results['extended']['RMSE'],
                test_extended_Skill=results['extended']['Skill'],test_extended_MAE=results['extended']['MAE'],
                test_extended_CumErr=results['extended']['CumErr'],extended_epoch=args.extended_epochs,
                outer_evaluated_epochs=outer_epochs,
                outer_train_R2=train_metrics['R2'],outer_train_RMSE=train_metrics['RMSE'],
                outer_initial_weights_sha256=initial_hash,outer_target_center=centre,outer_target_scale=scale,
                n_train=len(train['frame']),n_test=len(test['frame']),
                n_train_sequences=len(train['X']),n_test_sequences=len(test['X']),
                elapsed_outer_seconds=time.monotonic()-start)

def aggregate():
    records=[]; curves=[]
    for f in sorted((OUT/'jobs').glob('*/result.json')):
        record=json.loads(f.read_text(encoding='utf-8'));records.append(record)
        curves.append(pd.read_csv(f.parent/'learning_curve.csv'))
        curves.append(pd.read_csv(f.parent/'outer_learning_curve.csv'))
    summary=pd.DataFrame(records)
    if len(summary):
        summary.to_csv(OUT/'summary.csv',index=False)
        pd.concat(curves,ignore_index=True).to_csv(OUT/'learning_curves.csv',index=False)
        pairs=[]
        for keys,g in summary.groupby(['architecture','variant','seed']):
            if set(g.formulation)!={'direct','anchored'}:continue
            d=g[g.formulation=='direct'].iloc[0];a=g[g.formulation=='anchored'].iloc[0]
            assert d.initial_weights_sha256==a.initial_weights_sha256
            assert d.outer_initial_weights_sha256==a.outer_initial_weights_sha256
            for field in ['checkpoint_input_sha256','outer_input_sha256']:
                assert d[field]==a[field]
            for curve_file in ['learning_curve.csv','outer_learning_curve.csv']:
                paired_curves = [pd.read_csv(OUT/'jobs'/f'{keys[1]}_{keys[0]}_seed{keys[2]}_{mode}'/curve_file)
                                 for mode in ['direct','anchored']]
                shared = min(len(c) for c in paired_curves)
                assert paired_curves[0].batch_order_sha256.iloc[:shared].tolist()==paired_curves[1].batch_order_sha256.iloc[:shared].tolist()
            pairs.append(dict(architecture=keys[0],variant=keys[1],seed=int(keys[2]),
                              direct_R2=d.test_R2,anchored_R2=a.test_R2,delta_R2=a.test_R2-d.test_R2,
                              direct_Skill=d.test_Skill,anchored_Skill=a.test_Skill,
                              direct_epoch12_R2=d.test_epoch12_R2,anchored_epoch12_R2=a.test_epoch12_R2,
                              delta_epoch12_R2=a.test_epoch12_R2-d.test_epoch12_R2,
                              direct_extended_R2=d.test_extended_R2,anchored_extended_R2=a.test_extended_R2,
                              delta_extended_R2=a.test_extended_R2-d.test_extended_R2,
                              direct_extended_Skill=d.test_extended_Skill,anchored_extended_Skill=a.test_extended_Skill,
                              direct_selected_epoch=int(d.selected_epoch),anchored_selected_epoch=int(a.selected_epoch),
                              paired_initialization_verified=True, paired_input_verified=True,
                              shared_epoch_batch_order_verified=True))
        pd.DataFrame(pairs).to_csv(OUT/'paired_comparisons.csv',index=False)
    return len(records)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architectures',nargs='+',default=['LSTM','GRU','TCN','Transformer'],choices=['LSTM','GRU','TCN','Transformer'])
    parser.add_argument('--variants',nargs='+',default=list(VARIANTS),choices=list(VARIANTS))
    parser.add_argument('--seeds',nargs='+',type=int,default=[42])
    parser.add_argument('--max-epochs',type=int,default=60)
    parser.add_argument('--extended-epochs',type=int,default=60)
    parser.add_argument('--min-epochs',type=int,default=20)
    parser.add_argument('--patience',type=int,default=15)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--rnn-cudnn',choices=['on','off'],default='on')
    parser.add_argument('--windows-cuda-teardown-workaround',action='store_true',
                        help='After successful completion only, synchronize/flush then terminate this Python process without DLL teardown.')
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--eval-batch',type=int,default=512)
    parser.add_argument('--curve-train-n',type=int,default=4096)
    parser.add_argument('--pilot-epochs',type=int,default=0)
    parser.add_argument('--pilot-tag',default='')
    parser.add_argument('--aggregate-only',action='store_true')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);(OUT/'jobs').mkdir(exist_ok=True)
    if args.aggregate_only:print('Completed jobs:',aggregate());return
    requested_device = args.device
    if args.device == 'auto': args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested, but the imported PyTorch cannot access CUDA.')
    if args.windows_cuda_teardown_workaround:
        assert os.name=='nt' and args.device=='cuda'
        global SHUTDOWN_WORKAROUND
        SHUTDOWN_WORKAROUND=True
    assert args.extended_epochs >= 12 and args.max_epochs >= 1
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    prep=load_module('hofen_sensitivity_prep',ORIGINAL/'code/analysis/prep.py')
    deep=load_module('hofen_sensitivity_deep',ORIGINAL/'code/analysis/deep.py')
    torch.set_num_threads(args.threads)
    long,e,cols,eligible,inner,val,outer,test,manifest=data(prep,deep)
    source_files=[ORIGINAL/'code/analysis'/n for n in ['prep.py','bench2.py','deep.py','deep_anchored.py']]
    source_hashes={str(f.relative_to(ROOT)):sha(f) for f in source_files}
    dump(OUT/'data_manifest.json',manifest)
    config=dict(vars(args),H=H,sequence_hours=SEQ,learning_rate=1e-3,weight_decay=1e-4,
                checkpoint_selection='lowest July-December 2021 original-scale validation RMSE; never test',
                outer_refit='reinitialize on all purged 2020-2021 origins; independently report epoch 12, validation-selected epoch, and predeclared extended endpoint',
                outer_validation_rmse='not defined: the inner validation observations enter the full 2020-2021 refit',
                learning_curve_training_subset='4096 fixed randomly selected eligible training sequences, independent of formulation; no test tuning',
                batch_rng='independent torch.Generator with identical paired seed',
                raw_input='replace log(1+current emission) channel by raw current emission; no extra channel',
                seed_scope='one paired seed for initial full architecture/scale audit; extra seeds selectable',
                source_hashes=source_hashes,script_sha256=sha(__file__),data=manifest,
                requested_device=requested_device,
                environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,
                                 device=args.device,cuda_available=torch.cuda.is_available(),cuda_version=torch.version.cuda,
                                 cudnn_version=torch.backends.cudnn.version(),
                                 gpu_name=torch.cuda.get_device_name(0) if args.device=='cuda' else None,
                                 deterministic_algorithms=True,tf32_enabled=False,
                                 threads=args.threads,cpu_count=os.cpu_count()))
    if args.pilot_epochs:
        args.max_epochs=args.pilot_epochs;args.min_epochs=args.pilot_epochs
        directory=OUT/'pilots'/f'{args.architectures[0]}_{args.pilot_epochs}epochs_{args.device}_{args.threads}threads_rnncudnn{args.rnn_cudnn}_seed{args.seeds[0]}';directory.mkdir(parents=True,exist_ok=True)
        if args.pilot_tag:
            directory=directory.with_name(directory.name+'_'+args.pilot_tag);directory.mkdir(parents=True,exist_ok=True)
        dump(directory/'configuration.json',dict(config,max_epochs=args.max_epochs,min_epochs=args.min_epochs))
        tr,va,_=stage_data(long,e,cols,eligible,inner,val,'legacy')
        timing=fit_inner(deep,args,args.architectures[0],'direct','legacy',args.seeds[0],tr,va,directory)
        model,_=initialize_model(deep,args.architectures[0],tr['X'].shape[-1],args.seeds[0],args.device,args.rnn_cudnn)
        model.load_state_dict(torch.load(directory/'inner_selected.pt',map_location='cpu',weights_only=True))
        vp=predict(model,va,timing['checkpoint_target_center'],timing['checkpoint_target_scale'],'direct',args.eval_batch)
        np.savez_compressed(directory/'validation_predictions.npz',target=va['y'],prediction=vp,persistence=va['anchor'])
        dump(directory/'timing.json',timing);print(json.dumps(timing),flush=True);return
    dump(OUT/'run_config.json',config)
    for variant in args.variants:
        for architecture in args.architectures:
            for seed in args.seeds:
                for formulation in ['direct','anchored']:
                    job=f'{variant}_{architecture}_seed{seed}_{formulation}'
                    directory=OUT/'jobs'/job;directory.mkdir(exist_ok=True)
                    if (directory/'result.json').exists():
                        previous=json.loads((directory/'configuration.json').read_text(encoding='utf-8'))
                        assert all(previous[k]==getattr(args,k) for k in ['max_epochs','min_epochs','extended_epochs','patience','device','rnn_cudnn','batch_size','eval_batch','curve_train_n'])
                        assert previous['source_hashes']==source_hashes and previous['data']==manifest
                        print('Resume: completed',job,flush=True);continue
                    dump(directory/'configuration.json',dict(config,architecture=architecture,formulation=formulation,variant=variant,seed=seed))
                    tr,va,normalization=stage_data(long,e,cols,eligible,inner,val,variant)
                    dump(directory/'checkpoint_preprocessing.json',normalization)
                    checkpoint_input_sha256=normalization['fit_input_sha256']
                    inner_result=fit_inner(deep,args,architecture,formulation,variant,seed,tr,va,directory)
                    del tr,va;gc.collect()
                    tr,te,normalization=stage_data(long,e,cols,eligible,outer,test,variant)
                    dump(directory/'outer_preprocessing.json',normalization)
                    outer_result=fit_outer(deep,args,architecture,formulation,variant,seed,tr,te,inner_result['selected_epoch'],directory)
                    result=dict(config_complete=True,architecture=architecture,formulation=formulation,variant=variant,seed=seed,
                                n_checkpoint_train=len(inner),n_validation=len(val),max_epochs=args.max_epochs,
                                checkpoint_input_sha256=checkpoint_input_sha256,
                                outer_input_sha256=normalization['fit_input_sha256'],
                                min_epochs=args.min_epochs,**inner_result,**outer_result)
                    result['artifact_sha256']={name:sha(directory/name) for name in
                                               ['inner_selected.pt','outer_selected.pt','outer_extended.pt','test_predictions.npz']}
                    dump(directory/'result.json',result);del tr,te;gc.collect()
                    if args.device=='cuda':torch.cuda.empty_cache()
                    aggregate()
                    completed=[json.loads(f.read_text(encoding='utf-8')) for f in sorted((OUT/'jobs').glob('*/result.json'))]
                    dump(OUT/'progress.json',dict(config_complete=all(r['config_complete'] for r in completed),
                         completed_configs=len(completed),planned_this_run=len(args.variants)*len(args.architectures)*len(args.seeds)*2,
                         completed=[dict(architecture=r['architecture'],formulation=r['formulation'],variant=r['variant'],seed=r['seed'],
                                         selected_epoch=r['selected_epoch'],evaluated_epochs=r['evaluated_epochs']) for r in completed]))
                    print('Finished',job,'selected epoch',inner_result['selected_epoch'],flush=True)
    print('Completed jobs:',aggregate(),flush=True)

def completed_process_exit():
    # Invoked only after main() returns normally. Python/training exceptions do
    # not reach this function. Windows ExitProcess/os._exit still calls the
    # faulty cuDNN DLL detach; TerminateProcess deliberately bypasses it.
    import ctypes
    torch.cuda.synchronize()
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    kernel.GetProcessId.argtypes=[ctypes.c_void_p];kernel.GetProcessId.restype=ctypes.c_uint32
    kernel.TerminateProcess.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
    handle=kernel.GetCurrentProcess()
    assert kernel.GetProcessId(handle)==os.getpid(), 'Refuse to terminate any process other than this Python worker.'
    print('All requested work completed; CUDA synchronized. Explicit Windows process termination bypasses the documented cuDNN DLL-teardown fault.',flush=True)
    sys.stdout.flush();sys.stderr.flush()
    kernel.TerminateProcess(handle,0)
    raise RuntimeError('TerminateProcess unexpectedly returned without terminating this worker.')

if __name__=='__main__':
    main()
    if SHUTDOWN_WORKAROUND:completed_process_exit()
