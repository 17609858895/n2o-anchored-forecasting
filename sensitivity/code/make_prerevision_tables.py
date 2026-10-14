"""Create supplement table inputs from completed sensitivity results only."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

PKG=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results-root',type=Path,default=PKG/'results')
parser.add_argument('--output-dir',type=Path,default=PKG/'results/prerevision_tables')
args=parser.parse_args()
OUT=args.output_dir.resolve()
OUT.mkdir(parents=True,exist_ok=True)
TRAIN=args.results_root.resolve()/'baseline_training_sensitivity'
AVEDORE_TRAIN=args.results_root.resolve()/'avedore_training_sensitivity'
CLEAN=args.results_root.resolve()/'avedore_cleaning_sensitivity'
SCENARIOS=['baseline','negative_missing','keep_extreme','source_quality','quality_negative_missing']
LABELS={'baseline':'Original','negative_missing':'Negative missing','keep_extreme':'Keep 12','source_quality':'Source quality','quality_negative_missing':'Quality + negative missing'}
MODELS=['Persistence','LightGBM','Anchored LightGBM','EG-MoE','GRU','GRU-A','Transformer','Transformer-A']

def number(value,n=3):
 assert np.isfinite(value),value
 return f'{value:.{n}f}'

def markdown(spec):
 lines=[spec['caption'],'','| '+' | '.join(spec['headers'])+' |','| '+' | '.join(['---']*len(spec['headers']))+' |']
 lines+=['| '+' | '.join(row)+' |' for row in spec['rows']]
 if spec.get('note'):lines+=['','Note. '+spec['note']]
 return '\n'.join(lines)

tables=[]
training=pd.read_csv(TRAIN/'summary.csv')
assert len(training)==32, f'Hofen requires 24 primary and 8 additional configurations, found {len(training)}'
assert training.config_complete.all() and (training.extended_epoch==60).all()
assert not training.duplicated(['architecture','variant','formulation','seed']).any()
primary=training[training.seed==42]
assert len(primary)==24 and set(primary.variant)=={'legacy','centered','raw_anchor'}
assert primary.config_complete.all() and (primary.extended_epoch==60).all()
architectures=['LSTM','GRU','TCN','Transformer']
variants=['legacy','centered','raw_anchor']
variant_names={'legacy':'Original scales','centered':'Centered output','raw_anchor':'Raw current value'}
rows=[]
for variant in variants:
 for architecture in architectures:
  for form in ['direct','anchored']:
   r=primary[(primary.variant==variant)&(primary.architecture==architecture)&(primary.formulation==form)].iloc[0]
   rows.append([architecture+('-A' if form=='anchored' else ''),variant_names[variant],str(int(r.seed)),str(int(r.selected_epoch)),number(r.test_epoch12_R2),number(r.test_extended_R2),number(r.test_R2),number(r.test_Skill)])
extra=training[training.seed!=42]
assert len(extra)==8 and set(extra.seed)=={43,44} and set(extra.architecture)=={'GRU'}
assert set(extra.variant)=={'legacy','raw_anchor'}
for r in extra.sort_values(['variant','seed','formulation']).itertuples():
 rows.append([r.architecture+('-A' if r.formulation=='anchored' else ''),variant_names[r.variant],str(int(r.seed)),str(int(r.selected_epoch)),number(r.test_epoch12_R2),number(r.test_extended_R2),number(r.test_R2),number(r.test_Skill)])
for start in range(0,len(rows),16):
 tables.append(dict(number='S12',caption='Table S12'+(' (continued)' if start else '')+'. Network training and scale sensitivity.',
  headers=['Model','Scale configuration','Seed','Selected epoch','R^2^, 12 epochs','R^2^, 60 epochs','R^2^, selected','Skill, selected'],rows=rows[start:start+16],
  note='R^2^, coefficient of determination; skill, relative MSE reduction against persistence; A, output anchoring.'))

info=json.loads((CLEAN/'summary.json').read_text(encoding='utf-8'))
assert info['complete'] and info['case_count']==5
coverage=pd.read_csv(CLEAN/'coverage_summary.csv').set_index('case')
tables.append(dict(number='S13',caption='Table S13. Observation coverage after cleaning.',
 headers=['Cleaning rule','Raw N~2~O readings','Valid hourly N~2~O','Training origins','Test origins'],
 rows=[[LABELS[c]]+[f'{int(coverage.loc[c,k]):,}' for k in ['raw_n2o_retained','hourly_valid_n2o','n_train','n_test']] for c in SCENARIOS],note='N~2~O, nitrous oxide.'))
own=pd.read_csv(CLEAN/'own_origin_benchmarks.csv')
common=pd.read_csv(CLEAN/'common_origin_benchmarks.csv')
assert len(own)==40 and len(common)==80
rows=[]
for case in SCENARIOS:
 for model in MODELS:
  a=own[(own.scenario==case)&(own.model==model)].iloc[0]
  b=common[(common.scenario==case)&(common.model==model)&(common.scope=='common_origins_own_target')].iloc[0]
  c=common[(common.scenario==case)&(common.model==model)&(common.scope=='common_origins_fixed_baseline_target')].iloc[0]
  rows.append([LABELS[case],model,number(a.R2),number(b.R2),number(c.R2),number(b.Skill),number(b.RMSE),number(b.CumErr,1)])
for start in range(0,len(rows),16):
 tables.append(dict(number='S14',caption='Table S14'+(' (continued)' if start else '')+'. Forecast sensitivity to cleaning rules.',
  headers=['Cleaning rule','Model','Own-origin R^2^','Common-origin R^2^','Fixed-target R^2^','Common-origin skill','RMSE','Bias (%)'],rows=rows[start:start+16],
  note='RMSE, mg N L^−1^; bias, cumulative target bias; A, output anchoring; EG-MoE, episode- and regime-gated mixture of experts.'))
design=pd.read_csv(CLEAN/'validation_design.csv')
rows=[]
for case in SCENARIOS:
 r=design[(design.scenario==case)&(design.model=='LightGBM')]
 chrono=r[r.design=='held_out_year'].iloc[0]
 random=r[r.design=='random_hours'];assert len(random)==5
 rows.append([LABELS[case],number(chrono.R2),number(chrono.Skill),number(random.R2.mean())+f' [{random.R2.min():.3f}, {random.R2.max():.3f}]',number(random.Skill.mean())+f' [{random.Skill.min():.3f}, {random.Skill.max():.3f}]'])
tables.append(dict(number='S15',caption='Table S15. Validation-design sensitivity after cleaning.',headers=['Cleaning rule','Held-out-year R^2^','Held-out-year skill','Random R^2^ mean [range]','Random skill mean [range]'],rows=rows,note='R^2^, coefficient of determination; skill, relative MSE reduction against persistence.'))
avedore_training=pd.read_csv(AVEDORE_TRAIN/'summary.csv')
assert len(avedore_training)==24 and set(avedore_training.variant)==set(variants)
assert avedore_training.config_complete.all() and (avedore_training.extended_epoch==60).all()
rows=[]
for variant in variants:
 for architecture in architectures:
  for form in ['direct','anchored']:
   r=avedore_training[(avedore_training.variant==variant)&(avedore_training.architecture==architecture)&(avedore_training.formulation==form)].iloc[0]
   rows.append([architecture+('-A' if form=='anchored' else ''),variant_names[variant],str(int(r.seed)),str(int(r.selected_epoch)),number(r.test_epoch12_R2),number(r.test_extended_R2),number(r.test_R2),number(r.test_Skill)])
for start in range(0,len(rows),16):
 tables.append(dict(number='S16',caption='Table S16'+(' (continued)' if start else '')+'. Dissolved-signal network training sensitivity.',
  headers=['Model','Scale configuration','Seed','Selected epoch','R^2^, 12 epochs','R^2^, 60 epochs','R^2^, selected','Skill, selected'],rows=rows[start:start+16],
  note='R^2^, coefficient of determination; skill, relative MSE reduction against persistence; A, output anchoring.'))
(OUT/'tables.json').write_text(json.dumps(tables,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(OUT/'tables.md').write_text('\n\n'.join(markdown(t) for t in tables)+'\n',encoding='utf-8')
print(json.dumps(dict(table_numbers=5,table_pages=len(tables),hofen_training_configs=len(training),avedore_training_configs=len(avedore_training),cleaning_fits=len(own),common_test_origins=info['common_test_origins'])))
