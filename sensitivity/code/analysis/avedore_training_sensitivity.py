"""Predeclared training sensitivity on Avedore dissolved N2O, not off-gas flux.

No original data, analysis, benchmark, or manuscript is overwritten. Run first
with --prepare-only to validate the frozen baseline denominator; then run the
24 architecture/encoding/formulation configurations. Checkpoints are selected
only on the local 2022-12-12--2023-06-11 validation interval. Epoch 12 and the
fixed epoch-60 endpoint are reported independently, never used for selection.

Example (temporary PYTHONPATH must point at ../.runtime_cuda/packages):
  D:/Anaconda/python.exe -X utf8 avedore_training_sensitivity.py --prepare-only
  D:/Anaconda/python.exe -X utf8 avedore_training_sensitivity.py --device cuda \
      --windows-cuda-teardown-workaround
"""
from pathlib import Path
import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time

# Must precede importing PyTorch, including imports in reference modules.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch

PKG = Path(__file__).resolve().parents[2]
ROOT = PKG.parent
REF = Path(__file__).parent / 'reference'
RAW = ROOT / 'N2O_ML_paper/data/avedore/aved_raw.csv.gz'
OUT = PKG / 'results/avedore_training_sensitivity'
H, SEQ = 6, 48
INNER = pd.Timestamp('2022-12-12', tz='Europe/Copenhagen')
VARIANTS = {
    'legacy': 'legacy zero-centered target/residual with log1p(100c) current concentration input',
    'centered': 'mean-centered target/residual with log1p(100c) current concentration input',
    'raw_anchor': 'mean-centered target/residual with raw current concentration replacing log1p(100c)',
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())


def frozen_sources():
    sources = [RAW, Path(__file__).parent / 'baseline_training_sensitivity.py']
    sources += [REF / name for name in ['aved_replication.py', 'bench2.py', 'deep.py', 'stats_joint.py', 'prep.py']]
    sources += [ROOT / 'N2O_ML_paper/code/analysis' / name for name in
                ['aved_replication.py', 'bench2.py', 'deep.py', 'stats_joint.py', 'prep.py']]
    return {str(path.relative_to(ROOT)): sha(path) for path in sources}


def assert_frozen(expected):
    current = {name: sha(ROOT / name) for name in expected}
    if current != expected:
        raise RuntimeError('Frozen raw data or analysis source changed during this run.')


def target_scale(fit, formulation, variant):
    """Match A.nets: both direct/A scale on eligible sequences, NumPy ddof=0."""
    values = fit['y'] if formulation == 'direct' else fit['y'] - fit['anchor']
    center = 0.0 if variant == 'legacy' else float(np.mean(values))
    scale = float(np.std(values, ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Nonpositive fit-only eligible-sequence target scale.')
    return torch.from_numpy(((values - center) / scale).astype(np.float32)), center, scale


def data(A):
    A.RAW = str(RAW)
    h, info = A.load()
    frame, tab_cols = A.features(h)
    assert np.array_equal(frame.index.to_numpy(), np.arange(len(frame)))
    assert len(tab_cols) == 75
    assert A.SEQ == SEQ and A.H == H
    expected_outer = pd.Timestamp('2023-06-12', tz='Europe/Copenhagen')
    assert A.SPLIT == expected_outer
    all_origins = frame.dropna(subset=['target']).index.to_numpy(dtype=int)
    checkpoint = all_origins[(frame.time.loc[all_origins] < INNER - pd.Timedelta(hours=H)).to_numpy()]
    validation = all_origins[((frame.time.loc[all_origins] >= INNER) &
                             (frame.time.loc[all_origins] < A.SPLIT - pd.Timedelta(hours=H))).to_numpy()]
    outer = all_origins[(frame.time.loc[all_origins] < A.SPLIT - pd.Timedelta(hours=H)).to_numpy()]
    test = all_origins[(frame.time.loc[all_origins] >= A.SPLIT).to_numpy()]
    assert len(outer) == 8397 and len(test) == 8702, (len(outer), len(test))
    assert frame.time.loc[checkpoint].max() + pd.Timedelta(hours=H) < frame.time.loc[validation].min()
    assert frame.time.loc[validation].max() + pd.Timedelta(hours=H) < frame.time.loc[test].min()
    cols = [name for name in A.SENS if name != 'DOsp2'] + ['hour_sin', 'hour_cos', 'yl0']
    assert len(cols) == 24 and cols[-1] == 'yl0'
    eligible = np.arange(len(frame)) >= SEQ - 1
    windows = {key: len(idx[eligible[idx]]) for key, idx in
               [('checkpoint_train', checkpoint), ('validation', validation), ('outer_train', outer), ('test', test)]}
    segments = {}
    for key, idx in [('checkpoint_train', checkpoint), ('validation', validation), ('outer_train', outer), ('test', test)]:
        segments[key] = dict(n_origins=len(idx), n_sequences=windows[key],
                             origin_start=str(frame.time.loc[idx].min()), origin_end=str(frame.time.loc[idx].max()))
    info.update(site='Avedore', measured_signal='dissolved N2O concentration in tank 1',
                target='mean next 6 hourly dissolved N2O concentrations, at least 5 valid hours',
                target_units='mg N L-1', timezone='Europe/Copenhagen',
                inner_boundary=str(INNER), outer_boundary=str(A.SPLIT), purge_hours=H,
                n_train=len(outer), n_test=len(test), n_tabular_inputs=75, n_sequence_inputs=24,
                sequence_hours=SEQ, sequence_columns=cols, segments=segments,
                missing_training_anchors=int(frame.pers.loc[outer].isna().sum()),
                missing_test_anchors=int(frame.pers.loc[test].isna().sum()),
                feature_hash=hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest())
    return h, frame, cols, eligible, checkpoint, validation, outer, test, info


def stage_data(frame, cols, eligible, fit_origins, evaluation_origins, variant, baseline):
    E = frame[cols].copy()
    if variant == 'raw_anchor':
        # Replace the same channel, preserving the architecture/input count.
        E['yl0'] = frame.pers.to_numpy()
    E = E.ffill(limit=6)  # Causal across the uninterrupted hourly single-line series.
    mu = E.loc[fit_origins].mean()
    sd = E.loc[fit_origins].std(ddof=1) + 1e-6
    values = ((E - mu) / sd).fillna(0).to_numpy(dtype=np.float32)
    median = float(np.nanmedian(frame.target.loc[fit_origins]))

    def pack(idx):
        ie = idx[eligible[idx]]
        # CPU windows only; shared fit functions transfer one minibatch at a time.
        X = np.stack([values[i-SEQ+1:i+1] for i in ie])
        full = frame.loc[idx, ['target', 'time', 'street']].copy()
        full['anchor'] = frame.pers.loc[idx].fillna(median).to_numpy(dtype=float)
        return dict(X=torch.from_numpy(X), y=frame.target.loc[ie].to_numpy(dtype=float),
                    anchor=frame.pers.loc[ie].fillna(median).to_numpy(dtype=float), frame=full,
                    indices=ie, full_y=full.target.to_numpy(dtype=float),
                    full_anchor=full.anchor.to_numpy(dtype=float), fallback_median=median)

    tr, ev = pack(fit_origins), pack(evaluation_origins)
    preprocessing = dict(input_mean=mu.to_dict(), input_sd=sd.to_dict(), input_sd_ddof=1,
                         input_sd_epsilon=1e-6, fit_origin_count=len(fit_origins),
                         fit_sequence_count=len(tr['X']), fallback_median=median,
                         fit_input_sha256=baseline.array_hash(tr['X'].numpy()),
                         evaluation_input_sha256=baseline.array_hash(ev['X'].numpy()),
                         target_scale_pool='eligible fit sequences only', target_scale_ddof=0,
                         current_concentration_channel='raw c' if variant == 'raw_anchor' else 'log1p(100c)',
                         sequence_columns=cols)
    preprocessing['expected_target_scales'] = {
        mode: dict(center=target_scale(tr, mode, variant)[1], scale=target_scale(tr, mode, variant)[2])
        for mode in ['direct', 'anchored']}
    return tr, ev, preprocessing


def aggregate():
    records, curves, pairs = [], [], []
    for path in sorted((OUT / 'jobs').glob('*/result.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        records.append(record)
        for filename in ['learning_curve.csv', 'outer_learning_curve.csv']:
            curve = pd.read_csv(path.parent / filename)
            curve['site'], curve['target_units'] = 'Avedore', 'mg N L-1'
            curves.append(curve)
    if not records:
        return 0
    summary = pd.DataFrame(records)
    summary.to_csv(OUT / 'summary.csv', index=False)
    pd.concat(curves, ignore_index=True).to_csv(OUT / 'learning_curves.csv', index=False)
    for keys, group in summary.groupby(['architecture', 'variant', 'seed']):
        if set(group.formulation) != {'direct', 'anchored'}:
            continue
        d, a = [group[group.formulation == mode].iloc[0] for mode in ['direct', 'anchored']]
        for field in ['initial_weights_sha256', 'outer_initial_weights_sha256',
                      'checkpoint_input_sha256', 'outer_input_sha256']:
            assert d[field] == a[field], (keys, field)
        for filename in ['learning_curve.csv', 'outer_learning_curve.csv']:
            pair = [pd.read_csv(OUT/'jobs'/f'{keys[1]}_{keys[0]}_seed{keys[2]}_{mode}'/filename)
                    for mode in ['direct', 'anchored']]
            shared = min(len(curve) for curve in pair)
            assert pair[0].batch_order_sha256.iloc[:shared].tolist() == pair[1].batch_order_sha256.iloc[:shared].tolist()
        pairs.append(dict(site='Avedore', target_units='mg N L-1', architecture=keys[0], variant=keys[1], seed=int(keys[2]),
                          direct_R2=d.test_R2, anchored_R2=a.test_R2, delta_R2=a.test_R2-d.test_R2,
                          direct_Skill=d.test_Skill, anchored_Skill=a.test_Skill,
                          direct_epoch12_R2=d.test_epoch12_R2, anchored_epoch12_R2=a.test_epoch12_R2,
                          delta_epoch12_R2=a.test_epoch12_R2-d.test_epoch12_R2,
                          direct_extended_R2=d.test_extended_R2, anchored_extended_R2=a.test_extended_R2,
                          delta_extended_R2=a.test_extended_R2-d.test_extended_R2,
                          direct_extended_Skill=d.test_extended_Skill, anchored_extended_Skill=a.test_extended_Skill,
                          direct_selected_epoch=int(d.selected_epoch), anchored_selected_epoch=int(a.selected_epoch),
                          paired_initialization_verified=True, paired_input_verified=True,
                          shared_epoch_batch_order_verified=True))
    pd.DataFrame(pairs).to_csv(OUT / 'paired_comparisons.csv', index=False)
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--aggregate-only', action='store_true')
    parser.add_argument('--architectures', nargs='+', default=['LSTM', 'GRU', 'TCN', 'Transformer'],
                        choices=['LSTM', 'GRU', 'TCN', 'Transformer'])
    parser.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--max-epochs', type=int, default=60)
    parser.add_argument('--extended-epochs', type=int, default=60)
    parser.add_argument('--min-epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--rnn-cudnn', choices=['on', 'off'], default='on')
    parser.add_argument('--windows-cuda-teardown-workaround', action='store_true')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--eval-batch', type=int, default=512)
    parser.add_argument('--curve-train-n', type=int, default=4096)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'jobs').mkdir(exist_ok=True)
    if args.aggregate_only:
        print('Completed jobs:', aggregate(), flush=True)
        return False, None
    assert args.max_epochs == 60 and args.extended_epochs == 60 and args.min_epochs == 20 and args.patience == 15
    source_hashes = frozen_sources()
    freeze_file = OUT / 'frozen_sources.json'
    if freeze_file.exists():
        if json.loads(freeze_file.read_text(encoding='utf-8')) != source_hashes:
            raise RuntimeError('Frozen source hashes differ from the previously prepared run.')
    else:
        dump(freeze_file, source_hashes)
    baseline = load_module('avedore_private_baseline_fit', Path(__file__).parent / 'baseline_training_sensitivity.py')
    baseline.target_scale, baseline.VARIANTS = target_scale, VARIANTS
    sys.path.insert(0, str(REF))
    try:
        A = load_module('avedore_frozen_reference', REF / 'aved_replication.py')
    finally:
        sys.path.remove(str(REF))
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    h, frame, cols, eligible, inner, val, outer, test, manifest = data(A)
    dump(OUT / 'data_manifest.json', manifest)
    design = dict(site='Avedore', target_units='mg N L-1', measured_signal='dissolved N2O concentration',
                  selection='minimum original-scale validation RMSE; no test-based selection',
                  inner_validation=manifest['segments']['validation'], outer_refit_epochs=60,
                  reported_endpoints=['epoch12', 'validation-selected epoch', 'fixed epoch60'],
                  target_scale='eligible fit target/residual NumPy std(ddof=0) in all variants and both formulations',
                  centering='legacy: zero; centered/raw_anchor: eligible fit target/residual mean',
                  input_statistics='fit-origin means and pandas std(ddof=1)+1e-6 only; causal ffill limit6; fill remaining normalized gaps with zero',
                  paired_comparison='same CPU initial weights and independent same-seed batch-order generator',
                  variant_definitions=VARIANTS, raw_input='replace yl0 by raw current c; no additional channel',
                  source_hashes=source_hashes, design_sha256=sha(__file__))
    dump(OUT / 'design.json', design)
    if args.prepare_only:
        probes = {}
        for key, fit, evaluation in [('checkpoint', inner, val), ('outer', outer, test)]:
            tr, ev, stats = stage_data(frame, cols, eligible, fit, evaluation, 'legacy', baseline)
            scales = {mode: dict(center=target_scale(tr, mode, 'legacy')[1],
                                std_ddof0=target_scale(tr, mode, 'legacy')[2]) for mode in ['direct', 'anchored']}
            probes[key] = dict(preprocessing=stats, target_scales=scales,
                               fit_tensor_shape=list(tr['X'].shape), evaluation_tensor_shape=list(ev['X'].shape))
            del tr, ev
            gc.collect()
        assert_frozen(source_hashes)
        dump(OUT / 'prepare_verification.json', dict(passed=True, n_train=8397, n_test=8702,
                                                     source_hashes_unchanged=True, probes=probes))
        print(json.dumps(dict(prepared=True, n_train=8397, n_test=8702, segments=manifest['segments']), ensure_ascii=False), flush=True)
        return False, baseline
    requested_device = args.device
    args.device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable.')
    if args.windows_cuda_teardown_workaround:
        assert os.name == 'nt' and args.device == 'cuda'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    config = dict(vars(args), H=H, sequence_hours=SEQ, learning_rate=1e-3, weight_decay=1e-4,
                  checkpoint_selection=design['selection'], data=manifest, source_hashes=source_hashes,
                  script_sha256=sha(__file__), requested_device=requested_device, design=design,
                  environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
                                   device=args.device, cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
                                   gpu_name=torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
                                   deterministic_algorithms=True, tf32_enabled=False, threads=args.threads))
    config_file = OUT / 'run_config.json'
    if config_file.exists():
        previous = json.loads(config_file.read_text(encoding='utf-8'))
        for field in ['max_epochs', 'extended_epochs', 'min_epochs', 'patience', 'device', 'rnn_cudnn',
                      'batch_size', 'eval_batch', 'curve_train_n', 'source_hashes', 'data', 'script_sha256']:
            assert previous[field] == config[field], field
    dump(config_file, config)
    start = time.monotonic()
    for variant in args.variants:
        for architecture in args.architectures:
            for seed in args.seeds:
                for formulation in ['direct', 'anchored']:
                    job = f'{variant}_{architecture}_seed{seed}_{formulation}'
                    directory = OUT / 'jobs' / job
                    directory.mkdir(exist_ok=True)
                    if (directory / 'result.json').exists():
                        print('Resume: completed', job, flush=True)
                        continue
                    dump(directory / 'configuration.json', dict(config, architecture=architecture, formulation=formulation,
                                                                 variant=variant, seed=seed))
                    tr, va, normalization = stage_data(frame, cols, eligible, inner, val, variant, baseline)
                    dump(directory / 'checkpoint_preprocessing.json', normalization)
                    checkpoint_input_sha256 = normalization['fit_input_sha256']
                    inner_result = baseline.fit_inner(A.D, args, architecture, formulation, variant, seed, tr, va, directory)
                    del tr, va
                    gc.collect()
                    tr, te, normalization = stage_data(frame, cols, eligible, outer, test, variant, baseline)
                    dump(directory / 'outer_preprocessing.json', normalization)
                    outer_result = baseline.fit_outer(A.D, args, architecture, formulation, variant, seed, tr, te,
                                                      inner_result['selected_epoch'], directory)
                    with np.load(directory / 'test_predictions.npz', allow_pickle=False) as saved:
                        endpoint12 = baseline.metrics(saved['target'], saved['epoch12'], saved['persistence'])
                    outer_result.update(test_epoch12_MAE=endpoint12['MAE'], test_epoch12_CumErr=endpoint12['CumErr'])
                    result = dict(config_complete=True, site='Avedore', target_units='mg N L-1',
                                  architecture=architecture, formulation=formulation, variant=variant, seed=seed,
                                  n_checkpoint_train=len(inner), n_validation=len(val), max_epochs=args.max_epochs,
                                  checkpoint_input_sha256=checkpoint_input_sha256,
                                  outer_input_sha256=normalization['fit_input_sha256'], min_epochs=args.min_epochs,
                                  **inner_result, **outer_result)
                    result['artifact_sha256'] = {name: sha(directory / name) for name in
                                                 ['inner_selected.pt', 'outer_selected.pt', 'outer_extended.pt', 'test_predictions.npz']}
                    dump(directory / 'result.json', result)
                    del tr, te
                    gc.collect()
                    if args.device == 'cuda':
                        torch.cuda.empty_cache()
                    assert_frozen(source_hashes)
                    n = aggregate()
                    completed = [json.loads(path.read_text(encoding='utf-8')) for path in sorted((OUT / 'jobs').glob('*/result.json'))]
                    dump(OUT / 'progress.json', dict(completed_configs=n, planned_this_run=len(args.variants)*len(args.architectures)*len(args.seeds)*2,
                                                     elapsed_seconds=time.monotonic()-start,
                                                     completed=[{key: record[key] for key in
                                                                 ['architecture', 'formulation', 'variant', 'seed', 'selected_epoch', 'evaluated_epochs']}
                                                                for record in completed]))
                    print('Finished', job, 'selected epoch', inner_result['selected_epoch'], flush=True)
    n = aggregate()
    assert_frozen(source_hashes)
    dump(OUT / 'completion.json', dict(completed=True, completed_configs=n, elapsed_seconds=time.monotonic()-start,
                                       frozen_sources_unchanged=True))
    print('Completed jobs:', n, flush=True)
    return args.windows_cuda_teardown_workaround, baseline


if __name__ == '__main__':
    workaround, module = main()
    # Exceptions cannot reach this success-only workaround. It affects this PID only.
    if workaround:
        module.completed_process_exit()
