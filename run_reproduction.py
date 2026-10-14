"""Run historical analyses or the prespecified additional robustness fits."""
from pathlib import Path
import argparse,subprocess,sys
ROOT=Path(__file__).resolve().parent
ANALYSIS=ROOT/'N2O_ML_paper/code/analysis'
steps={
 'prepare':[['prep.py']],
 'hofen':[['driver.py'],['bench2.py','6','chrono','1','1'],['meta_lab.py','6'],['meta2.py','6'],['deep_anchored.py','main'],['deep_anchored.py','other']],
 'avedore':[['aved_replication.py'],['aved_gate_weights.py']],
 'airflow':[[str(ROOT/'sensitivity/code/analysis/avedore_airflow_sensitivity.py')]],
}
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('step',choices=[*steps,'training_hofen','training_avedore','cleaning'])
parser.add_argument('--dry-run',action='store_true')
parser.add_argument('--device',choices=['cpu','cuda','auto'],default='cpu',help='Device for additional training fits only.')
parser.add_argument('--seeds',nargs='+',type=int,choices=[42,43,44],help='Hofen: 42 all architectures; 43/44 GRU only. Avedore/cleaning: 42.')
parser.add_argument('--windows-cuda-teardown-workaround',action='store_true',help='Windows CUDA success-only termination workaround; affects the spawned Python worker itself.')
args=parser.parse_args()
training_args=['--max-epochs','60','--extended-epochs','60','--min-epochs','20','--patience','15','--threads','2','--device',args.device]
if args.windows_cuda_teardown_workaround:
 if sys.platform!='win32' or args.device=='cpu':parser.error('The teardown workaround requires Windows and a CUDA device.')
 training_args+=['--windows-cuda-teardown-workaround']
additional={}
seeds=args.seeds or [42,43,44]
commands=[]
script=str(ROOT/'sensitivity/code/analysis/baseline_training_sensitivity.py')
if 42 in seeds:commands.append([script,*training_args,'--seeds','42'])
extras=[str(seed) for seed in seeds if seed in [43,44]]
if extras:commands.append([script,*training_args,'--architectures','GRU','--variants','legacy','raw_anchor','--seeds',*extras])
additional['training_hofen']=commands
additional['training_avedore']=[[str(ROOT/'sensitivity/code/analysis/avedore_training_sensitivity.py'),*training_args,'--seeds','42']]
additional['cleaning']=[[str(ROOT/'sensitivity/code/analysis/avedore_cleaning_sensitivity.py')]]
if args.step in ['training_avedore','cleaning'] and args.seeds and args.seeds!=[42]:
 parser.error('The prespecified Avedore and cleaning analyses use seed 42 only.')
if args.seeds and args.step not in ['training_hofen','training_avedore','cleaning']:
 parser.error('Seed selection applies only to the additional robustness analyses.')
if args.step=='cleaning' and args.device!='cpu':parser.error('Cleaning sensitivity uses the prespecified CPU implementation.')
if args.step not in ['training_hofen','training_avedore'] and args.windows_cuda_teardown_workaround:
 parser.error('The teardown workaround applies only to additional CUDA training fits.')
steps.update(additional)
if not args.dry_run:
 for output_dir in [ANALYSIS/'tables',ROOT/'N2O_ML_paper/results']:
  output_dir.mkdir(parents=True,exist_ok=True)
for command in steps[args.step]:
 script=Path(command[0]);script=script if script.is_absolute() else ANALYSIS/script
 assert script.is_file(),script
 print('Running:',script.name,*command[1:],flush=True)
 if not args.dry_run:subprocess.run([sys.executable,str(script),*command[1:]],cwd=ANALYSIS,check=True)
