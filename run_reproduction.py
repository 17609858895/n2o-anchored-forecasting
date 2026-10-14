"""Run original analysis modules from their required working directory."""
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
parser.add_argument('step',choices=steps)
parser.add_argument('--dry-run',action='store_true')
args=parser.parse_args()
for command in steps[args.step]:
 script=Path(command[0]);script=script if script.is_absolute() else ANALYSIS/script
 assert script.is_file(),script
 print('Running:',script.name,*command[1:],flush=True)
 if not args.dry_run:subprocess.run([sys.executable,str(script),*command[1:]],cwd=ANALYSIS,check=True)
