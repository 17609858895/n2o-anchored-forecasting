# Current-measurement anchoring for wastewater N2O forecasts

Research data and analysis code for *Current-measurement anchoring enables machine-learning forecasts to outperform persistence and track extreme wastewater N2O emissions*.

Hofen supplies hourly off-gas emissions from 2020–2022. Avedore supplies a separate dissolved-N2O series, used to test anchoring and validation-design sensitivity. The aeration-weighted index is a sensitivity target and is not an independently calibrated emission flux.

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

Dependency names are supplied without inventing an original environment lock. The exact environment used for the additional aeration-weighted analysis is recorded by that script at runtime. Historical primary-model fits do not have a complete saved environment lock; numerical agreement may depend on dependency versions and stochastic training.

## Reproduction

The runner sets the required working directory and uses the included original analysis modules:

```bash
python run_reproduction.py hofen
python run_reproduction.py avedore
python run_reproduction.py airflow
```

`prepare` generates Hofen long.pkl. `hofen` runs the temporal/horizon benchmarks, the H=6 model comparison, gate components/ablations and paired sequence networks. `avedore` runs concentration models and gate weights. `airflow` runs the aeration-weighted sensitivity analysis. These are full training jobs, not a quick demonstration; use the original individual modules for diagnostics, DML, campaign sampling, feature interpretation and additional figure-data exports. Generated caches are written locally.

Primary modules: prep.py, bench2.py, driver.py, meta_lab.py, meta2.py, deep.py, deep_anchored.py and stats_joint.py. The remaining analysis modules retain the corresponding diagnostic calculations. Source plotting modules in code/figures reproduce quantitative plots after their analysis/figdata caches have been generated; cosmetic manuscript recombination and AI-generated explanatory schematics are outside this data/code release.

The numerical source code is retained; the sensitivity script's machine-specific Python command in its docstring is replaced with a portable invocation. All data are unchanged byte-for-byte. Validation covers file checksums, source syntax and lightweight data preparation, not a new full model-training replication.
