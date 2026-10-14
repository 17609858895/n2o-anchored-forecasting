# Anchoring wastewater N2O forecasts to current measurements

Research data, analysis code and saved robustness results for *Anchoring machine learning forecasts of episodic wastewater N2O emissions to current measurements*.

Hofen supplies hourly off-gas emissions from 2020–2022. Avedore supplies a separately fitted dissolved-N2O series. This second task tests forecasting-method behavior; it is not cross-plant transfer of a trained off-gas model or replication of emission physics. The aeration-weighted index is an auxiliary target without independent emission-flux calibration. Retrospective dosing/aeration illustrations do not establish experimentally validated emission reductions.

## Evidence added before submission

`reported_summaries/` contains aggregate metrics, paired comparisons, prespecified designs and quality/runtime audits from the completed local experiments. All raw inputs and fitting/export/verification code are public. Individual trained checkpoints, per-origin forecast arrays and plot exports are not distributed in this public package; the reproduction commands create them locally under `sensitivity/results/`.

- Hofen: 24 architecture/scale configurations at seed 42, plus eight paired GRU configurations at seeds 43/44. Direct and anchored models share initialization, inputs and epoch-wise batch orders. Checkpoints are selected on July–December 2021; full 2020–2021 refits report epochs 12, 60 and the validation-selected epoch on 2022.
- Avedore: 24 architecture/scale configurations at seed 42, with separate chronological checkpoint selection and full refitting. The original analysis uses a log-transformed current-concentration channel; the raw-input variant replaces that channel without adding information. It does not estimate off-gas emissions.
- Cleaning: five prespecified rules with eight models per rule. Results include each rule's own available origins, 7,581 common origins with rule-specific targets, and auxiliary scoring against the original cleaning rule's targets. Identical time support does not make these target definitions identical; the fixed-target analysis is not independent ground truth.

The stronger comparisons limit the anchoring claim. For Hofen's raw-current-input Transformer, direct/anchored R2 is 0.8435/0.8152 at epoch 12 and 0.8133/0.7829 at validation-selected checkpoints. Raw-input GRU selected gains vary across seeds 42/43/44 (0.0102, 0.3718 and 0.0259); all three anchored epoch-60 runs have negative skill against persistence. Avedore's raw-input GRU and Transformer selected gains shrink to 0.0229 and 0.0450. Anchoring is conditional on formulation, training and signal behavior, rather than universally superior.

The cleaning results also retain weak cases. Under good-quality flags plus removal of negative readings, GRU-A has skill 0.0172 and cumulative concentration-target bias +31.0%; anchored LightGBM is worse than its direct counterpart. Of the negative retained concentration readings, 96.6% have source flag 0 (Good), and all 28 readings at 12 mg N/L are also flagged Good. Historical source comments describe negative values as a zero offset and the upper threshold as saturation; these comments are not verified sensor diagnoses. The frozen numerical sources are preserved, while the cleaning analysis explicitly evaluates the alternative rules.

The additional export scripts regenerate Tables S12–S16 and Figs S21–S23 after the corresponding fits, including vector PDFs, 600-dpi PNGs and every standalone panel. Figure colors use the selected `au_nimno_soft_8` palette. These are quantitative data plots.

## Data

- Hofen raw data and variable metadata: `N2O_ML_paper/data/bd08c9ec-e189-482f-91cb-4269c21f7a44-n2oemissionsoperatingdatahofen/`. Source DOI: https://doi.org/10.25678/000DBM. Please also cite Froemelt et al. (2025), https://doi.org/10.1016/j.wroa.2025.100336. The original provider LICENSE.txt is included (CC BY 4.0); DOI metadata also records CC0. Source attribution is retained, and the provider license text is preserved without alteration.
- Avedore analysis input and complete V1 source: `N2O_ML_paper/data/avedore/`. Source DOI: https://doi.org/10.17632/xmbxhscgpr.1 (CC BY 4.0). See its provenance README for rounding and quality-flag handling.
- `file_manifest.json` records SHA-256 hashes of the included data/code. No dataset is re-licensed by this repository.

## Installation and checks

Use Python 3.11 or newer in an isolated environment. Install the scientific dependencies:

```bash
python -m pip install -r requirements.txt
python validate_release.py
python run_reproduction.py prepare
```

The dependency list is not an invented historical lock. The recorded robustness environments are Python 3.12.3, NumPy 1.26.4, pandas 2.2.2 and scikit-learn 1.5.1. The training fits use isolated PyTorch 2.13.0+cu126, CUDA 12.6, an NVIDIA RTX 4060 Laptop GPU, deterministic FP32 and disabled TF32. Cleaning fits use PyTorch 2.13.0+cpu and LightGBM 4.6.0 with two CPU threads. Their per-fit records retain further environment details. Historical primary-model fits lack a complete saved environment lock; library versions, hardware and random training can affect numerical replication.

## Reproduction

The runner sets the required working directory. Historical analyses:

```bash
python run_reproduction.py hofen
python run_reproduction.py avedore
python run_reproduction.py airflow
```

`prepare` generates Hofen long.pkl. `hofen` runs temporal/horizon benchmarks, H=6 comparisons, gate components/ablations and paired sequence networks. `avedore` runs concentration models and gate weights; `airflow` runs the aeration-weighted analysis. These are full training jobs. The individual original modules retain diagnostics, DML, campaign sampling and feature interpretation.

Additional prespecified robustness fits write fresh results, checkpoints and forecasts under `sensitivity/results/`, separately from the published summaries:

```bash
python run_reproduction.py training_hofen --device cpu
python run_reproduction.py training_avedore --device cpu
python run_reproduction.py cleaning
```

Hofen defaults to seed 42 for all architecture/scale pairs and seeds 43/44 for GRU original-scale/raw-input pairs only. `--seeds 42` runs its primary 24 configurations; `--seeds 43 44` runs its eight additional configurations. Avedore and cleaning use seed 42 only. Add `--dry-run` to any entry point to inspect its commands without fitting. CUDA can be requested with `--device cuda` after installing a compatible PyTorch build.

On the recorded Windows machine, cuDNN recurrent calculations and saving completed but normal DLL teardown crashed. The optional `--windows-cuda-teardown-workaround`, used together with `--device cuda`, synchronizes CUDA and flushes output only after successful completion, checks the worker's own PID and terminates that same process to bypass DLL teardown. Exceptions cannot enter the success path. This is an explicit abrupt exit, not normal library cleanup. Do not use this flag on Linux; it is rejected there. CPU/GPU numerical comparisons, repeated-fit checks and independent saved-array/hash checks are documented in `reported_summaries/baseline_training_sensitivity/`. No installed CUDA environment is included.

## Verify new fits and reproduce the additional displays

After completing the additional fits, the independent CPU-only verifier reconstructs targets and persistence from raw observations, checks time purges and frozen hashes, then recalculates 168 training forecast sets and 120 cleaning comparisons without fitting or test-set selection:

```bash
python sensitivity/code/verify_prerevision_results.py --results-root sensitivity/results --output-root generated_outputs/verification
python sensitivity/code/make_prerevision_tables.py --results-root sensitivity/results --output-dir generated_outputs/tables
python sensitivity/code/plot_prerevision_sensitivity.py all --results-root sensitivity/results --outputs-root generated_outputs/figures
```

Verification regenerates each result directory's independent recalculation/audit files and Hofen period summaries from saved arrays; it preserves fitting configurations, completion records, curves, PT states and NPZ forecasts. The included local audit records document the completed local fits; checking an individual saved prediction or trained state requires regenerating the fit outputs. The verifier accepts the explicit folder alias `最新版` to `sensitivity` while preserving recorded source/data digests. Export scripts alone gained explicit result/output path options for portability; fitting and reference sources retain their frozen bytes. Cosmetic recombination of historical figures and explanatory schematics are outside this release.

Primary modules are prep.py, bench2.py, driver.py, meta_lab.py, meta2.py, deep.py, deep_anchored.py and stats_joint.py. Historical plotting modules in `N2O_ML_paper/code/figures` use their corresponding analysis/figdata caches. Additional robustness scripts are in `sensitivity/code/analysis`; their immutable source/reference hashes and per-configuration artifact hashes are retained with the results.

All original data remain unchanged byte-for-byte. Release checks cover every tracked-file checksum, Python syntax, lightweight raw-data preprocessing, the runner and the independent saved-result audit. The robustness experiments summarized here were fully run locally; publication validation does not claim a fresh full rerun of every historical analysis or statistical convergence of every direct network.
