"""Prespecified Avedore dissolved-N2O cleaning sensitivity, independently refitted.

Run with the analysis environment: python avedore_cleaning_sensitivity.py
No retained manuscript predictions or original inputs are overwritten. The same
rounded value columns are used in every scenario; official V1 supplies quality
flags only, avoiding a simultaneous change of measurement precision.
"""
from pathlib import Path
import os
for _name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_name] = "2"
import argparse, gc, hashlib, json, platform, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb
import sklearn
from threadpoolctl import threadpool_limits

PKG = Path(__file__).resolve().parents[2]
ORIGINAL = PKG.parent / "N2O_ML_paper"
OUT = PKG / "results/avedore_cleaning_sensitivity"
REF = ORIGINAL / "code/analysis"
sys.path.insert(0, str(REF))
import aved_replication as A
import deep as D
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
A.B.LG = dict(A.B.LG, n_jobs=2)
A.B.REG_VARS = A.REG

VALUE_FILE = ORIGINAL / "data/avedore/aved_raw.csv.gz"
QUALITY_FILE = ORIGINAL / "data/avedore/provenance/mendeley_v1_original.csv.gz"
QUALITY_SOURCE = "https://arxiv.org/html/2407.05959v1#S2.SS2.SSS5"
SEED, EPOCHS, RANDOM_REPEATS = 42, 12, 5
CASES = {
    "baseline": dict(negative="zero", extreme="exclude", quality="ignore"),
    "negative_missing": dict(negative="missing", extreme="exclude", quality="ignore"),
    "keep_extreme": dict(negative="zero", extreme="retain", quality="ignore"),
    "source_quality": dict(negative="zero", extreme="exclude", quality="per_signal_good"),
    "quality_negative_missing": dict(negative="missing", extreme="exclude", quality="per_signal_good"),
}
MODELS = ["Persistence", "LightGBM", "Anchored LightGBM", "EG-MoE", "GRU", "GRU-A", "Transformer", "Transformer-A"]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def input_data():
    values = pd.read_csv(VALUE_FILE)
    values["time"] = pd.to_datetime(values["time"], utc=True)
    quality = pd.read_csv(QUALITY_FILE, usecols=lambda c: c == "time" or c.endswith(" quality"))
    quality["time"] = pd.to_datetime(quality["time"], utc=True)
    assert len(values) == len(quality) == 831557
    assert np.array_equal(values.time.values, quality.time.values)
    assert values.time.is_monotonic_increasing
    offset = 0
    for chunk in pd.read_csv(QUALITY_FILE, usecols=lambda c: c.endswith(" value"), chunksize=100000):
        expected = values.iloc[offset:offset+len(chunk)][chunk.columns].to_numpy()
        np.testing.assert_allclose(chunk.round(4).to_numpy(), expected, rtol=0, atol=1e-12, equal_nan=True)
        offset += len(chunk)
    assert offset == len(values)
    named = {}
    rows = []
    for col in values.columns:
        if col == "time":
            continue
        qcol = col[:-6] + " quality" if col.endswith(" value") else None
        assert qcol in quality, col
        q = quality[qcol]
        codes = set(q.dropna().unique())
        assert codes <= {0.0, 1.0}, (qcol, codes)
        short = A.REN[col.replace("BIOLOGY.", "").replace(" value", "")]
        named[short] = q
        y = values[col]
        rows.append(dict(signal=short, value_column=col, quality_column=qcol,
                         value_nonmissing=int(y.notna().sum()), good_flag=int(q.eq(0).sum()),
                         bad_flag=int(q.eq(1).sum()), missing_flag=int(q.isna().sum()),
                         good_nonmissing_values=int((q.eq(0) & y.notna()).sum()),
                         negative_values=int(y.lt(0).sum()),
                         negative_values_good=int((y.lt(0) & q.eq(0)).sum()),
                         observed_quality_codes=";".join(map(str, sorted(codes)))))
    qframe = pd.DataFrame(named)
    values = values.rename(columns={c: A.REN[c.replace("BIOLOGY.", "").replace(" value", "")]
                                   for c in values if c != "time"})
    pd.DataFrame(rows).to_csv(OUT / "quality_flag_audit.csv", index=False)
    ext = values.N2O.ge(12)
    write_json(OUT / "quality_provenance.json", {
        "official_paper": "https://arxiv.org/html/2407.05959v1",
        "section": "2.2.5 Quality Assessment", "quality_definition": {"0": "Good quality", "1": "Bad quality"},
        "rule": "Keep each signal only when its own quality field equals 0; missing or bad quality masks that signal, without dropping the time grid.",
        "interpretation": "Source flags concern sensor maintenance, constant signals, active controls and propagated quality of estimates; good quality does not certify nonnegative concentration or the 12 mg N/L plateau.",
        "value_precision": "Identical retained value columns, rounded to four decimal places, across all rules.",
        "quality_file": str(QUALITY_FILE.relative_to(PKG.parent)), "quality_file_sha256": digest(QUALITY_FILE),
        "value_file": str(VALUE_FILE.relative_to(PKG.parent)), "value_file_sha256": digest(VALUE_FILE),
        "rows": len(values), "timestamp_alignment_exact": True, "all_value_columns_match_official_after_rounding": True,
        "n2o_negative_values": int(values.N2O.lt(0).sum()),
        "n2o_negative_good_quality": int((values.N2O.lt(0) & qframe.N2O.eq(0)).sum()),
        "extreme_values_ge12": int(ext.sum()), "extreme_good_quality": int((ext & qframe.N2O.eq(0)).sum()),
        "extreme_dates": values.loc[ext, "time"].astype(str).tolist(),
    })
    del quality
    gc.collect()
    return values, qframe


def prepare_case(values, quality, case):
    rule = CASES[case]
    df = values.copy()
    if rule["quality"] == "per_signal_good":
        for col in quality:
            df.loc[~quality[col].eq(0), col] = np.nan
    quality_removed_n2o = int((values.N2O.notna() & df.N2O.isna()).sum())
    excluded_extreme = int(df.N2O.ge(12).sum()) if rule["extreme"] == "exclude" else 0
    if rule["extreme"] == "exclude":
        df.loc[df.N2O.ge(12), "N2O"] = np.nan
    excluded_negative = int(df.N2O.lt(0).sum()) if rule["negative"] == "missing" else 0
    if rule["negative"] == "missing":
        df.loc[df.N2O.lt(0), "N2O"] = np.nan
    else:
        df["N2O"] = df.N2O.clip(lower=0)
    for col in ["NH4", "NO3", "DO1", "DO2", "PO4"]:
        df[col] = df[col].clip(lower=0)
    df.loc[df.Q_in.lt(0), "Q_in"] = np.nan
    h = df.set_index("time").resample("1h").mean()
    count = df.set_index("time")["N2O"].resample("1h").count()
    h.index = h.index.tz_convert("Europe/Copenhagen")
    count.index = h.index
    full, feats = A.features(h.reset_index())
    assert len(feats) == 75 and np.array_equal(full.index, np.arange(len(full)))
    direct = h.N2O.iloc[::-1].rolling(6, min_periods=5).mean().iloc[::-1].shift(-1)
    np.testing.assert_allclose(full.target, direct.to_numpy(), equal_nan=True)
    valid = full.dropna(subset=["target"]).copy()
    tr = valid[valid.time < A.SPLIT - pd.Timedelta(hours=6)].copy()
    te = valid[valid.time >= A.SPLIT].copy()
    assert tr.time.max() + pd.Timedelta(hours=6) < te.time.min()
    assert len(tr) > 100 and len(te) > 100
    cover = dict(case=case, **rule, raw_records=len(df),
                 scenario=case, quality_excluded_n2o_values=quality_removed_n2o,
                 excluded_extreme_ge12_values=excluded_extreme, excluded_negative_values=excluded_negative,
                 raw_n2o_before=int(values.N2O.notna().sum()), raw_n2o_retained=int(df.N2O.notna().sum()),
                 raw_n2o_zero=int(df.N2O.eq(0).sum()), raw_n2o_max=float(df.N2O.max()),
                 hourly_records=len(h), hourly_valid_n2o=int(h.N2O.notna().sum()),
                 hourly_zero_n2o=int(h.N2O.eq(0).sum()), hourly_mean=float(h.N2O.mean()),
                 hourly_valid_readings_median=float(count[count.gt(0)].median()),
                 n_train=len(tr), n_test=len(te), n_target_origins=len(valid),
                 train_start=str(tr.time.min()), train_end=str(tr.time.max()),
                 test_start=str(te.time.min()), test_end=str(te.time.max()),
                 missing_train_anchors=int(tr.pers.isna().sum()), missing_test_anchors=int(te.pers.isna().sum()),
                 train_mean=float(tr.target.mean()), test_mean=float(te.target.mean()),
                 train_max=float(tr.target.max()), test_max=float(te.target.max()),
                 training_tau=float(tr.target.quantile(.8)),
                 all_missing_train_features=[c for c in feats if tr[c].isna().all()],
                 n_tabular_features=len(feats), n_sequence_inputs=24)
    pd.DataFrame(dict(time=h.index.astype(str), observed_concentration=h.N2O.values,
                      n2o_valid_two_minute_records=count.values)).to_csv(OUT / case / "hourly_target.csv.gz", index=False)
    pd.DataFrame({"feature": feats, "train_missing_fraction": tr[feats].isna().mean().values,
                  "test_missing_fraction": te[feats].isna().mean().values}).to_csv(OUT / case / "feature_coverage.csv", index=False)
    write_json(OUT / case / "coverage.json", cover)
    return dict(full=full, valid=valid, feats=feats, tr=tr, te=te, coverage=cover)


def prediction_metrics(y, p, anchor):
    p = np.clip(np.asarray(p), 0, None)
    mse = A.mean_squared_error(y, p)
    bias = 100 * (p.sum() - np.asarray(y).sum()) / np.asarray(y).sum()
    return dict(n=len(y), n_test=len(y), R2=A.r2_score(y, p), RMSE=mse ** .5,
                MAE=A.mean_absolute_error(y, p), Skill=1 - mse / A.mean_squared_error(y, anchor),
                CumErr=bias, cumulative_bias_percent=bias)


def network(data, name, anchored):
    full, tr, te = data["full"], data["tr"], data["te"]
    cols = [c for c in A.SENS if c != "DOsp2"] + ["hour_sin", "hour_cos", "yl0"]
    x = full[cols].ffill(limit=6)
    mask = full.index.isin(tr.index)
    mu, sd = x.loc[mask].mean(), x.loc[mask].std() + 1e-6
    arr = ((x - mu) / sd).fillna(0).to_numpy(dtype=np.float32)
    med = float(tr.target.median())
    def windows(frame):
        idx = np.asarray(frame.index)
        idx = idx[idx >= 47]
        return idx, np.stack([arr[i-47:i+1] for i in idx])
    itr, xtr = windows(tr)
    ite, xte = windows(te)
    atr = tr.pers.loc[itr].fillna(med).values
    residual = tr.target.loc[itr].values - (atr if anchored else 0)
    scale = residual.std()
    assert scale > 0
    xt, yt = torch.tensor(xtr), torch.tensor((residual / scale).astype(np.float32))
    torch.manual_seed(SEED)
    model = D.RNN(len(cols), nn.GRU) if name == "GRU" else D.TF(len(cols))
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    losses = []
    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(len(xt))
        total = 0.
        for start in range(0, len(xt), 256):
            j = permutation[start:start+256]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xt[j]), yt[j])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(j)
        losses.append(total / len(xt))
    model.eval()
    with torch.no_grad():
        pred = np.concatenate([model(torch.tensor(xte[b:b+512])).numpy() for b in range(0, len(xte), 512)]) * scale
    series = pd.Series(np.nan, index=te.index)
    series.loc[ite] = pred
    final = te.pers.fillna(med).values + series.fillna(0).values if anchored else series.fillna(med).values
    del model, xt, yt, xtr, xte, arr
    gc.collect()
    return final, dict(target_scale=float(scale), n_training_sequences=len(itr), n_test_sequences=len(ite),
                       epochs=EPOCHS, epoch_training_losses=losses, prediction_batch_size=512)


def fit_case(case, data, source_key):
    tr, te, features = data["tr"], data["te"], data["feats"]
    med = tr.target.median()
    atr, ate = tr.pers.fillna(med).values, te.pers.fillna(med).values
    cache, infofile = OUT / case / "predictions.csv.gz", OUT / case / "model_fits.json"
    info = json.loads(infofile.read_text(encoding="utf-8")) if infofile.exists() else {"source_key": source_key, "fits": {}}
    assert info["source_key"] == source_key, "Cached computation differs from current prespecified implementation."
    if cache.exists():
        stored = pd.read_csv(cache).set_index("origin_id")
        assert np.array_equal(stored.index, te.index)
        np.testing.assert_allclose(stored.target, te.target)
        predictions = {n: stored[n].values for n in MODELS if n in stored}
    else:
        predictions = {"Persistence": ate}
    def save():
        frame = pd.DataFrame(dict(origin_id=te.index, time=te.time.astype(str).values, target=te.target.values, **predictions))
        frame.to_csv(cache, index=False)
        write_json(infofile, info)
    save()
    for name in MODELS[1:]:
        if name in predictions:
            continue
        t0 = time.time()
        if name == "LightGBM":
            predictions[name] = lgb.LGBMRegressor(**A.B.LG).fit(tr[features], tr.target).predict(te[features])
            details = {}
        elif name == "Anchored LightGBM":
            predictions[name] = ate + lgb.LGBMRegressor(**A.B.LG).fit(tr[features], tr.target.values-atr).predict(te[features])
            details = {}
        elif name == "EG-MoE":
            estimator = A.B.EGMoE(float(tr.target.quantile(.8))).fit(tr[features], tr.target.values, atr, tr.time.values)
            predictions[name] = estimator.predict(te[features], ate)
            details = {}
        else:
            predictions[name], details = network(data, name.split("-")[0], name.endswith("-A"))
        assert len(predictions[name]) == len(te) and np.isfinite(predictions[name]).all()
        info["fits"][name] = dict(elapsed_seconds=time.time()-t0, **details)
        save()
        metrics = prediction_metrics(te.target, predictions[name], ate)
        print(case, name, f'{time.time()-t0:.1f}s', json.dumps(metrics), flush=True)
    rows = [dict(case=case, scenario=case, model=n, scope="own_available_origins", **prediction_metrics(te.target, predictions[n], ate)) for n in MODELS]
    pd.DataFrame(rows).to_csv(OUT / case / "benchmark.csv", index=False)
    return predictions


def random_splits(case, data):
    full, valid, feats = data["full"], data["valid"], data["feats"]
    rows = []
    for rep in range(RANDOM_REPEATS):
        # Origin-indexed draws remain paired between cleaning rules despite
        # different target coverage. All rules use the same complete hourly grid.
        draw = np.random.RandomState(SEED + rep).rand(len(full))
        mask = draw[valid.index] < .3
        tr, te = valid[~mask], valid[mask]
        anchor = te.pers.fillna(tr.target.median()).values
        pred = lgb.LGBMRegressor(**A.B.LG).fit(tr[feats], tr.target).predict(te[feats])
        for name, prediction in [("LightGBM", pred), ("Persistence", anchor)]:
            rows.append(dict(case=case, scenario=case, design="random_hours", repetition=rep, rep=rep, seed=SEED+rep,
                             n_train=len(tr), **prediction_metrics(te.target, prediction, anchor)))
            rows[-1]["model"] = name
        print(case, "random repeat", rep+1, flush=True)
    rows += [dict(case=case, scenario=case, design="held_out_year", repetition=0, rep=0, seed=SEED,
                  n_train=len(data["tr"]), model=n,
                  **prediction_metrics(data["te"].target, data["predictions"][n], data["predictions"]["Persistence"]))
             for n in ["LightGBM", "Persistence"]]
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / case / "validation_design.csv", index=False)
    return frame


def summaries(all_data):
    cases = list(all_data)
    common = sorted(set.intersection(*(set(d["te"].index) for d in all_data.values())))
    base = all_data["baseline"]
    base_anchor = pd.Series(base["predictions"]["Persistence"], index=base["te"].index)
    rows, pairs, targets = [], [], []
    for case, d in all_data.items():
        y, own_anchor = d["te"].target.loc[common], d["te"].pers.fillna(d["tr"].target.median()).loc[common].values
        baseline_y = base["te"].target.loc[common]
        delta = y.to_numpy() - baseline_y.to_numpy()
        targets.append(dict(case=case, n_common=len(common), mean_own_target=float(y.mean()),
                            mean_baseline_target=float(baseline_y.mean()), target_difference_MAE=float(np.abs(delta).mean()),
                            target_difference_max_abs=float(np.abs(delta).max())))
        for name in MODELS:
            prediction = pd.Series(d["predictions"][name], index=d["te"].index).loc[common].values
            rows.append(dict(case=case, scenario=case, model=name, scope="common_origins_own_target", **prediction_metrics(y, prediction, own_anchor)))
            rows.append(dict(case=case, scenario=case, model=name, scope="common_origins_fixed_baseline_target",
                             **prediction_metrics(baseline_y, prediction, base_anchor.loc[common].values)))
        own = pd.read_csv(OUT / case / "benchmark.csv")
        for name in ["LightGBM", "GRU", "Transformer"]:
            direct = own[own.model.eq(name)].iloc[0]
            anchored_name = "Anchored LightGBM" if name == "LightGBM" else name + "-A"
            anchored = own[own.model.eq(anchored_name)].iloc[0]
            pairs.append(dict(case=case, family=name, direct_R2=direct.R2, anchored_R2=anchored.R2,
                              anchoring_R2_gain=anchored.R2-direct.R2, direct_skill=direct.Skill, anchored_skill=anchored.Skill))
    pd.DataFrame(rows).to_csv(OUT / "common_origin_benchmarks.csv", index=False)
    pd.DataFrame(pairs).to_csv(OUT / "anchoring_pairs.csv", index=False)
    pd.DataFrame(targets).to_csv(OUT / "common_target_differences.csv", index=False)
    pd.concat([pd.read_csv(OUT / c / "benchmark.csv") for c in cases], ignore_index=True).to_csv(OUT / "own_origin_benchmarks.csv", index=False)
    validation = pd.concat([pd.read_csv(OUT / c / "validation_design.csv") for c in cases], ignore_index=True)
    validation.to_csv(OUT / "validation_design.csv", index=False)
    pd.DataFrame([d["coverage"] for d in all_data.values()]).to_csv(OUT / "coverage_summary.csv", index=False)
    pd.DataFrame({"origin_id": common, "time": base["te"].time.loc[common].astype(str).values}).to_csv(OUT / "common_test_origins.csv", index=False)
    summary = {
        "complete": True, "case_count": len(cases), "models_per_case": len(MODELS), "common_test_origins": len(common),
        "cases": CASES, "coverage": {c: d["coverage"] for c, d in all_data.items()},
        "quality_provenance": json.loads((OUT / "quality_provenance.json").read_text(encoding="utf-8")),
        "same_nominal_split": str(A.SPLIT), "training_buffer_hours": 6,
        "target": "following-six-hour mean dissolved N2O, at least five observed hourly values",
        "units": "mg N L-1", "sequence_families": ["GRU", "Transformer"],
        "not_evaluated_here": ["LSTM", "TCN", "aeration-weighted index"],
        "seed": SEED, "epochs": EPOCHS, "random_hour_repeats": RANDOM_REPEATS,
        "random_design_models": ["LightGBM", "Persistence"],
        "random_design_mask": "same seed-indexed complete-hour-grid draws shared across cleaning rules",
        "quality_definition_source": QUALITY_SOURCE,
        "interpretation_boundaries": {
            "own_available_origins": "Refitted models against each rule's own concentration target and available times; these are different data tasks across rules.",
            "common_origins_own_target": "Time support is identical, while concentration targets and each-rule anchors may still differ.",
            "common_origins_fixed_baseline_target": "All forecasts evaluated against the baseline-defined target and baseline persistence on common times; auxiliary robustness, not truth independent of the baseline cleaning rule.",
            "validation_design": "Random hours also change training size and period/episode coverage; the comparison does not isolate leakage alone.",
            "quality_flags": "Source good flags coexist with negative readings and the 12 mg N/L plateau; flags do not establish physical validity.",
            "no_cleaning_selection": "All five rules were specified before fitting; none is selected using test performance.",
        },
        "environment": dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                            torch=torch.__version__, lightgbm=lgb.__version__, sklearn=sklearn.__version__, cpu_threads=2),
    }
    write_json(OUT / "summary.json", summary)
    print(pd.DataFrame(pairs).round(4).to_string(index=False), flush=True)
    print("COMMON ORIGINS", len(common), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        (OUT / case).mkdir(exist_ok=True)
    source_key = digest(Path(__file__)) + digest(VALUE_FILE) + digest(QUALITY_FILE)
    write_json(OUT / "prespecified_design.json", {
        "cases": CASES, "models": MODELS, "seed": SEED, "epochs": EPOCHS, "random_repeats": RANDOM_REPEATS,
        "reference_sha256": {n: digest(REF / n) for n in ["aved_replication.py", "bench2.py", "deep.py", "stats_joint.py", "prep.py"]},
        "source_key": source_key, "method": "independent refit for every prespecified cleaning rule",
        "nominal_split": str(A.SPLIT), "training_buffer_hours": 6, "source_quality_rule": "each signal quality == 0",
    })
    values, quality = input_data()
    all_data = {c: prepare_case(values, quality, c) for c in CASES}
    print(pd.DataFrame([d["coverage"] for d in all_data.values()])[["case", "raw_n2o_retained", "hourly_valid_n2o", "n_train", "n_test"]].to_string(index=False), flush=True)
    del values, quality
    gc.collect()
    if args.prepare_only:
        pd.DataFrame([d["coverage"] for d in all_data.values()]).to_csv(OUT / "coverage_summary.csv", index=False)
        return
    started = time.time()
    with threadpool_limits(limits=2):
        for case, data in all_data.items():
            data["predictions"] = fit_case(case, data, source_key)
            random_splits(case, data)
    summaries(all_data)
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    summary["elapsed_seconds"] = time.time() - started
    write_json(OUT / "summary.json", summary)


if __name__ == "__main__":
    main()
