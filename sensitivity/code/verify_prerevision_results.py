"""Verify saved forecasts, paired controls and targets without fitting models.

This CPU-only audit reads frozen raw data and saved predictions. It does not
import fitting code or select a model from held-out test performance.
"""
from pathlib import Path
import argparse, hashlib, itertools, json
import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
OUT = PKG / 'qa/prerevision_sensitivity_20261015'
ENDPOINTS = [('epoch12', 'test_epoch12_'), ('selected', 'test_'), ('extended', 'test_extended_')]

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def source_path(relative):
    """Resolve recorded paths while preserving every original byte hash.

    A published package may rename the workspace package folder from 最新版
    to sensitivity. Only that explicitly declared prefix is aliased; source
    filenames, data files and the recorded digests are never changed.
    """
    parts = str(relative).replace('\\', '/').split('/')
    assert not Path(relative).is_absolute() and '..' not in parts
    if parts[0] == '最新版':
        return PKG.joinpath(*parts[1:])
    return ROOT.joinpath(*parts)

def metrics(y, prediction, anchor, clip_negative=False):
    y, prediction, anchor = map(lambda v: np.asarray(v, dtype=float), (y, prediction, anchor))
    assert len(y) == len(prediction) == len(anchor) > 0
    assert np.isfinite(y).all() and np.isfinite(prediction).all() and np.isfinite(anchor).all()
    if clip_negative:
        prediction = np.maximum(prediction, 0)  # Cleaning benchmark stores pre-clipping model output.
    else:
        assert (prediction >= 0).all(), 'Training NPZ forecasts must already be nonnegative.'
    sse = float(np.sum((y - prediction) ** 2))
    return dict(R2=1-sse/float(np.sum((y-y.mean())**2)), RMSE=np.sqrt(sse/len(y)),
                MAE=np.mean(np.abs(y-prediction)), Skill=1-sse/float(np.sum((y-anchor)**2)),
                CumErr=100*(prediction.sum()-y.sum())/y.sum())

def future_observed_mean(values):
    """Direct shifted arrays, independent of fitting-code rolling."""
    values = np.asarray(values, dtype=float)
    future = np.full((len(values), 6), np.nan)
    for lead in range(1, 7):
        future[:-lead, lead-1] = values[lead:]
    count = np.isfinite(future).sum(axis=1)
    total = np.nansum(future, axis=1)
    target = np.divide(total, count, out=np.full(len(values), np.nan), where=count >= 5)
    return target, count

def reconstructed_tasks():
    hofen_raw = ROOT / 'N2O_ML_paper/data/bd08c9ec-e189-482f-91cb-4269c21f7a44-n2oemissionsoperatingdatahofen/raw_data/qgl_hof_raw.csv'
    columns = ['_time'] + [f'_f_n2o_emissions_h{i}' for i in range(1, 7)]
    raw = pd.read_csv(hofen_raw, usecols=columns)
    timestamp = pd.to_datetime(raw['_time'].str[:19]).dt.round('h')
    frames = []
    for street in (1, 2):
        cols = [f'_f_n2o_emissions_h{i}' for i in range(3*street-2, 3*street+1)]
        values = raw[cols].clip(lower=0).sum(axis=1, min_count=3).to_numpy() * 1000
        frame = pd.DataFrame(dict(time=timestamp, street=street, observed=values)).sort_values('time')
        frame['target'], frame['target_observed_hours'] = future_observed_mean(frame.observed)
        frames.append(frame)
    full = pd.concat(frames, ignore_index=True).dropna(subset=['target'])
    boundary, inner = pd.Timestamp('2022-01-01'), pd.Timestamp('2021-07-01')
    train = full[full.time < boundary-pd.Timedelta(hours=6)]
    test = full[(full.time >= boundary) & (full.time < pd.Timestamp('2023-01-01'))].copy()
    test['persistence'] = test.observed.fillna(train.target.median())
    assert len(train) == 30283 and len(test) == 13785
    assert int((train.time < inner-pd.Timedelta(hours=6)).sum()) == 23139
    assert int((train.time >= inner).sum()) == 7132
    assert (test.target_observed_hours >= 5).all()
    assert train.time.max()+pd.Timedelta(hours=6) < test.time.min()

    ave_raw = ROOT / 'N2O_ML_paper/data/avedore/aved_raw.csv.gz'
    signal = 'BIOLOGY.LINE 3 TANK 1.N2O value'
    raw = pd.read_csv(ave_raw, usecols=['time', signal])
    raw['time'] = pd.to_datetime(raw.time, utc=True)
    raw.loc[raw[signal] >= 12, signal] = np.nan
    raw[signal] = raw[signal].clip(lower=0)
    hourly = raw.set_index('time')[signal].resample('1h').mean()
    full = pd.DataFrame(dict(time=hourly.index, street=1, observed=hourly.to_numpy()))
    full['target'], full['target_observed_hours'] = future_observed_mean(full.observed)
    full = full.dropna(subset=['target'])
    boundary = pd.Timestamp('2023-06-12', tz='Europe/Copenhagen').tz_convert('UTC')
    inner = pd.Timestamp('2022-12-12', tz='Europe/Copenhagen').tz_convert('UTC')
    train = full[full.time < boundary-pd.Timedelta(hours=6)]
    test_ave = full[full.time >= boundary].copy()
    test_ave['persistence'] = test_ave.observed.fillna(train.target.median())
    assert len(train) == 8397 and len(test_ave) == 8702
    assert int((train.time < inner-pd.Timedelta(hours=6)).sum()) == 4108
    assert int((train.time >= inner).sum()) == 4283
    assert (test_ave.target_observed_hours >= 5).all()
    assert train.time.max()+pd.Timedelta(hours=6) < test_ave.time.min()
    test_ave['time'] = test_ave.time.dt.tz_localize(None)  # NPZ datetime64 stores UTC without zone tag.
    return {'Hofen': test, 'Avedore': test_ave}

def expected_jobs(site):
    primary = {f'{variant}_{architecture}_seed42_{formulation}'
               for architecture, variant, formulation in itertools.product(
                   ['LSTM', 'GRU', 'TCN', 'Transformer'], ['legacy', 'centered', 'raw_anchor'], ['direct', 'anchored'])}
    if site == 'Hofen':
        primary.update(f'{variant}_GRU_seed{seed}_{formulation}'
                       for variant, seed, formulation in itertools.product(
                           ['legacy', 'raw_anchor'], [43, 44], ['direct', 'anchored']))
    return primary

def verify_training(site, directory, reconstructed, require_complete=True):
    summary = pd.read_csv(directory / 'summary.csv')
    aggregate_curves = pd.read_csv(directory / 'learning_curves.csv')
    aggregate_pairs = pd.read_csv(directory / 'paired_comparisons.csv')
    files = sorted((directory / 'jobs').glob('*/result.json'))
    names, expected = {p.parent.name for p in files}, expected_jobs(site)
    assert names <= expected
    if require_complete:
        assert names == expected, (site, sorted(expected-names))
    assert len(summary) == len(files) and not summary.duplicated(['architecture', 'variant', 'seed', 'formulation']).any()
    records, periods, results = [], [], {}
    for path in files:
        job = path.parent
        config, result = read_json(job / 'configuration.json'), read_json(path)
        results[job.name] = result
        assert result['config_complete'] is True and result['extended_epoch'] == result['outer_evaluated_epochs'] == 60
        for key in ('architecture', 'formulation', 'variant', 'seed'):
            assert config[key] == result[key]
        selected_row = summary[(summary.architecture == result['architecture']) &
                               (summary.formulation == result['formulation']) &
                               (summary.variant == result['variant']) & (summary.seed == result['seed'])]
        assert len(selected_row) == 1
        for key, value in result.items():
            if key.startswith('test_') or key in ['selected_epoch', 'extended_epoch', 'n_test']:
                assert np.isclose(float(selected_row.iloc[0][key]), value, rtol=1e-12, atol=1e-12), (job.name, key)
        assert config['max_epochs'] == config['extended_epochs'] == 60
        assert config['min_epochs'] == 20 and config['patience'] == 15
        assert config['environment']['deterministic_algorithms'] and not config['environment']['tf32_enabled']
        script = 'baseline_training_sensitivity.py' if site == 'Hofen' else 'avedore_training_sensitivity.py'
        assert config['script_sha256'] == sha(PKG / 'code/analysis' / script)
        for relative, digest in config['source_hashes'].items():
            assert sha(source_path(relative)) == digest, (site, relative)
        for filename, digest in result['artifact_sha256'].items():
            assert sha(job / filename) == digest, (job.name, filename)
        data = config['data']
        if site == 'Hofen':
            assert data['purge_checks_passed'] and not data['missing_targets_imputed']
            assert pd.Timestamp(data['inner_train_max'])+pd.Timedelta(hours=6) < pd.Timestamp(data['validation_min'])
            assert pd.Timestamp(data['outer_train_max'])+pd.Timedelta(hours=6) < pd.Timestamp(data['test_min'])
            raw = ROOT / 'N2O_ML_paper/data/bd08c9ec-e189-482f-91cb-4269c21f7a44-n2oemissionsoperatingdatahofen/raw_data/qgl_hof_raw.csv'
            assert sha(raw) == data['raw_sha256']
        else:
            assert data['purge_hours'] == 6 and data['n_sequence_inputs'] == 24
            s = data['segments']
            assert pd.Timestamp(s['checkpoint_train']['origin_end'])+pd.Timedelta(hours=6) < pd.Timestamp(s['validation']['origin_start'])
            assert pd.Timestamp(s['validation']['origin_end'])+pd.Timedelta(hours=6) < pd.Timestamp(s['test']['origin_start'])
        inner = pd.read_csv(job / 'learning_curve.csv')
        outer = pd.read_csv(job / 'outer_learning_curve.csv')
        assert inner.epoch.tolist() == list(range(1, result['evaluated_epochs']+1))
        assert int(inner.loc[inner.validation_rmse.idxmin(), 'epoch']) == result['selected_epoch']
        assert int(inner.loc[inner.is_best, 'epoch'].iloc[0]) == result['selected_epoch']
        assert np.isclose(inner.validation_rmse.min(), result['validation_RMSE'], rtol=1e-12, atol=1e-12)
        assert outer.epoch.tolist() == list(range(1, 61)) and outer.validation_rmse.isna().all()
        for saved_curve in [inner, outer]:
            stage = saved_curve.stage.iloc[0]
            published_curve = aggregate_curves[(aggregate_curves.architecture == result['architecture']) &
                                               (aggregate_curves.formulation == result['formulation']) &
                                               (aggregate_curves.variant == result['variant']) &
                                               (aggregate_curves.seed == result['seed']) &
                                               (aggregate_curves.stage == stage)].sort_values('epoch')
            assert len(published_curve) == len(saved_curve)
            for column in ['epoch', 'train_rmse', 'validation_rmse']:
                np.testing.assert_allclose(published_curve[column], saved_curve[column], rtol=1e-12, atol=1e-12, equal_nan=True)
            for column in ['is_best', 'batch_order_sha256']:
                assert published_curve[column].tolist() == saved_curve[column].tolist()
        for prefix, filename in [('checkpoint', 'checkpoint_preprocessing.json'), ('outer', 'outer_preprocessing.json')]:
            norm = read_json(job / filename)
            assert norm['fit_input_sha256'] == result[f'{prefix}_input_sha256'] and norm['evaluation_input_sha256']
        with np.load(job / 'test_predictions.npz', allow_pickle=False) as payload:
            n = 13785 if site == 'Hofen' else 8702
            assert len(payload['target']) == n == result['n_test'] == data['n_test']
            assert np.array_equal(payload['time'], reconstructed.time.to_numpy(dtype='datetime64[ns]'))
            assert np.array_equal(payload['street'], reconstructed.street.to_numpy())
            np.testing.assert_allclose(payload['target'], reconstructed.target, rtol=1e-12, atol=1e-10)
            np.testing.assert_allclose(payload['persistence'], reconstructed.persistence, rtol=1e-12, atol=1e-10)
            for label, prefix in ENDPOINTS:
                values = metrics(payload['target'], payload[label], payload['persistence'])
                for name, measured in values.items():
                    if prefix+name in result:
                        assert np.isclose(measured, result[prefix+name], rtol=1e-10, atol=1e-10), (job.name, label, name)
                records.append(dict(site=site, job=job.name, architecture=result['architecture'],
                                    formulation=result['formulation'], variant=result['variant'], seed=result['seed'],
                                    checkpoint=label, n=n, **values))
                if site == 'Hofen':
                    january = pd.to_datetime(payload['time']).month == 1
                    for period, mask in [('all', np.ones(n, dtype=bool)), ('January', january), ('February-December', ~january)]:
                        periods.append(dict(architecture=result['architecture'], formulation=result['formulation'],
                                            variant=result['variant'], seed=result['seed'], checkpoint=label,
                                            period=period, n=int(mask.sum()),
                                            **metrics(payload['target'][mask], payload[label][mask], payload['persistence'][mask])))
            if result['selected_epoch'] == 12:
                assert np.array_equal(payload['selected'], payload['epoch12'])
            if result['selected_epoch'] == 60:
                assert np.array_equal(payload['selected'], payload['extended'])
    pairs = 0
    for name, direct in results.items():
        if not name.endswith('_direct'):
            continue
        anchored_name = name[:-6]+'anchored'
        if anchored_name not in results:
            assert not require_complete
            continue
        anchored = results[anchored_name]
        for field in ['initial_weights_sha256', 'outer_initial_weights_sha256', 'checkpoint_input_sha256', 'outer_input_sha256']:
            assert direct[field] == anchored[field], (name, field)
        for filename in ['checkpoint_preprocessing.json', 'outer_preprocessing.json']:
            assert read_json(directory / 'jobs' / name / filename) == read_json(directory / 'jobs' / anchored_name / filename)
        for filename in ['learning_curve.csv', 'outer_learning_curve.csv']:
            d = pd.read_csv(directory / 'jobs' / name / filename)
            a = pd.read_csv(directory / 'jobs' / anchored_name / filename)
            shared = min(len(d), len(a))
            assert d.batch_order_sha256.iloc[:shared].tolist() == a.batch_order_sha256.iloc[:shared].tolist()
        pair = aggregate_pairs[(aggregate_pairs.architecture == direct['architecture']) &
                               (aggregate_pairs.variant == direct['variant']) & (aggregate_pairs.seed == direct['seed'])]
        assert len(pair) == 1
        pair = pair.iloc[0]
        for prefix, suffix in [('test_', ''), ('test_epoch12_', '_epoch12'), ('test_extended_', '_extended')]:
            for label, result_row in [('direct', direct), ('anchored', anchored)]:
                assert np.isclose(pair[f'{label}{suffix}_R2'], result_row[prefix+'R2'], rtol=1e-12, atol=1e-12)
            assert np.isclose(pair[f'delta{suffix}_R2'], anchored[prefix+'R2']-direct[prefix+'R2'], rtol=1e-12, atol=1e-12)
        for key in ['paired_initialization_verified', 'paired_input_verified', 'shared_epoch_batch_order_verified']:
            assert bool(pair[key])
        pairs += 1
    assert len(aggregate_pairs) == pairs
    pd.DataFrame(records).to_csv(directory / 'independent_metric_recalculation.csv', index=False)
    if periods:
        pd.DataFrame(periods).to_csv(directory / 'period_metrics.csv', index=False)
        baseline = []
        january = pd.to_datetime(reconstructed.time).dt.month == 1
        for period, mask in [('all', np.ones(len(reconstructed), dtype=bool)), ('January', january), ('February-December', ~january)]:
            baseline.append(dict(period=period, n=int(np.asarray(mask).sum()),
                                 **metrics(reconstructed.target[mask], reconstructed.persistence[mask], reconstructed.persistence[mask])))
        pd.DataFrame(baseline).to_csv(directory / 'period_persistence_metrics.csv', index=False)
    report = dict(passed=True, complete=names == expected, site=site, verified_configs=len(files),
                  verified_prediction_sets=len(records), verified_pairs=pairs,
                  targets_and_persistence_reconstructed_from_raw=True, six_hour_purges_verified=True,
                  target_minimum_observed_hours=5, target_imputation=False,
                  validation_only_checkpoint_selection=True, paired_initialization_inputs_and_batch_orders=True,
                  source_data_script_and_saved_artifact_hashes_match=True, test_tuning_performed=False,
                  recorded_source_path_alias={'最新版': PKG.name})
    (directory / 'independent_verification.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report

def verify_cleaning(directory=None):
    directory = directory or PKG / 'results/avedore_cleaning_sensitivity'
    info = read_json(directory / 'summary.json')
    assert info['complete'] and info['case_count'] == 5
    own = pd.read_csv(directory / 'own_origin_benchmarks.csv')
    assert len(own) == 40
    frames, count, negative_outputs = {}, 0, {}
    for case in info['cases']:
        frame = pd.read_csv(directory / case / 'predictions.csv.gz').set_index('origin_id')
        assert frame.index.is_unique
        frames[case] = frame
        for row in own[own.scenario == case].itertuples():
            negative_outputs[f'{case}/{row.model}'] = int((frame[row.model] < 0).sum())
            values = metrics(frame.target, frame[row.model], frame.Persistence, clip_negative=True)
            for name, value in values.items():
                assert np.isclose(value, getattr(row, name), rtol=1e-9, atol=1e-9), (case, row.model, name)
            count += 1
    common = pd.read_csv(directory / 'common_test_origins.csv').origin_id.to_numpy()
    assert len(common) == info['common_test_origins'] == 7581
    assert set(common) == set.intersection(*(set(frame.index) for frame in frames.values()))
    common_metrics = pd.read_csv(directory / 'common_origin_benchmarks.csv')
    assert len(common_metrics) == 80
    for row in common_metrics.itertuples():
        f = frames[row.scenario].loc[common]
        reference = frames['baseline'].loc[common] if row.scope == 'common_origins_fixed_baseline_target' else f
        values = metrics(reference.target, f[row.model], reference.Persistence, clip_negative=True)
        for name, value in values.items():
            assert np.isclose(value, getattr(row, name), rtol=1e-9, atol=1e-9), (row.scenario, row.model, row.scope, name)
        count += 1
    p = info['quality_provenance']
    assert p['timestamp_alignment_exact'] and p['quality_definition'] == {'0': 'Good quality', '1': 'Bad quality'}
    assert p['n2o_negative_values'] == 164041 and p['n2o_negative_good_quality'] == 158485
    for kind in ['quality', 'value']:
        assert sha(source_path(p[kind+'_file'])) == p[kind+'_file_sha256']
    return dict(passed=True, cases=5, forecast_sets_recomputed=count, common_origins=7581,
                quality_and_value_raw_file_hashes_match=True, quality_alignment_exact=True,
                scoring_negative_output_rule='Clip model forecasts at zero, as prespecified in the cleaning benchmark.',
                negative_preclipping_forecasts_by_case_and_model=negative_outputs)

def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-partial', action='store_true', help='Interim audit only; complete is explicitly false.')
    parser.add_argument('--results-root', type=Path, default=PKG/'results',
                        help='Saved result root; public archives can use ../reported_results.')
    parser.add_argument('--output-root', type=Path, default=OUT, help='Directory for the combined audit report.')
    args = parser.parse_args()
    OUT = args.output_root.resolve()
    results_root = args.results_root.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = reconstructed_tasks()
    training = {
        'Hofen': verify_training('Hofen', results_root / 'baseline_training_sensitivity', tasks['Hofen'], not args.allow_partial),
        'Avedore': verify_training('Avedore', results_root / 'avedore_training_sensitivity', tasks['Avedore'], not args.allow_partial),
    }
    cleaning = verify_cleaning(results_root / 'avedore_cleaning_sensitivity')
    complete = all(item['complete'] for item in training.values())
    result = dict(complete=complete, training_configs=sum(item['verified_configs'] for item in training.values()),
                  saved_training_forecast_sets_recomputed=sum(item['verified_prediction_sets'] for item in training.values()),
                  training=training, cleaning=cleaning,
                  cleaning_own_and_common_forecast_sets_recomputed=cleaning['forecast_sets_recomputed'],
                  paired_initialization_and_inputs_match=True, data_source_hashes_match=True,
                  time_purge_checks_passed=True, missing_targets_imputed=False, common_test_origins=7581,
                  all_metric_checks='PASS' if complete else 'INTERIM PASS; configurations outstanding',
                  recorded_source_path_alias={'最新版': PKG.name},
                  verification_code_sha256=sha(__file__))
    (OUT / 'saved_results_verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))

if __name__ == '__main__':
    main()

