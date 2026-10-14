"""Verify released file hashes and syntax; smoke-test input preprocessing without fitting models."""
from pathlib import Path
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import sys

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--skip-preprocessing',action='store_true')
args=parser.parse_args()
manifest=json.loads((ROOT/'file_manifest.json').read_text(encoding='utf-8'))
for entry in manifest:
    path=ROOT/entry['path']
    with path.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
    assert digest==entry['sha256'],entry['path']
    if path.suffix=='.py':compile(path.read_text(encoding='utf-8-sig'),str(path),'exec')
original=ROOT/'N2O_ML_paper/data/avedore/provenance/mendeley_v1_original.csv.gz'
with gzip.open(original,'rb') as handle:original_sha=hashlib.file_digest(handle,'sha256').hexdigest()
assert original_sha=='d1cf007ddd79fb7924ac0e06612d3fb57a5c5027bc7cc24069eae06bc6e8bd07'
results={'files_verified':len(manifest),'source_syntax':'PASS','original_avedore_v1_sha256':original_sha}
if not args.skip_preprocessing:
    import pandas as pd
    analysis=ROOT/'N2O_ML_paper/code/analysis'
    spec=importlib.util.spec_from_file_location('study_prep',analysis/'prep.py')
    prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
    previous=os.getcwd()
    try:
        os.chdir(analysis)
        long=prep.load_long()
        assert int(long.y.notna().sum())==44227
        features=prep.engineer(long,1)
        assert len(features)==len(long) and {'target','y_now','street','time'} <= set(features.columns)
    finally:os.chdir(previous)
    avedore=ROOT/'N2O_ML_paper/data/avedore/aved_raw.csv.gz'
    first_column=pd.read_csv(avedore,nrows=0).columns[0]
    rows=sum(len(frame) for frame in pd.read_csv(avedore,usecols=[first_column],chunksize=100000))
    assert rows==831557
    results.update(hofen_valid_street_hours=44227,hofen_feature_rows=len(features),avedore_input_rows=rows,model_training_rerun=False)
print(json.dumps(results))
