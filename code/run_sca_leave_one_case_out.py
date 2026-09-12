"""Leave-one-case-out SCA evaluation with native-rate features."""
import os,sys,json
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT/'EAAI_v25_reproducibility_package'/'code')); sys.path.insert(0,str(ROOT.parent/'revision_stage5_20260912'))
os.environ['SCA_FEATURE_MODE']='native'; os.environ['SCA_FIELD_ROOT']=str(ROOT/'sca_field_raw')
from run_sca_strict_case_retraining import load_test_only,choose_threshold,case_metrics
from run_hust_zero_shot_models import DirectMLP,predict,aggregate,metrics,seed_all,train as train_direct
from composition_experiment import CompositionModel,train as train_comp
OUT=ROOT/'sca_leave_one_case_out_results'
def fit(name,rows,seed):
 seed_all(seed)
 if name=='direct': m=DirectMLP(100); train_direct(m,rows,epochs=60,lr=.001)
 else: m=CompositionModel(100,decoder_steps=5,use_reference=False); train_comp(m,rows,epochs=60,lr=.001,seed=seed,weights={'recon':.1,'align':0,'union':.1,'coh':.001})
 return m
def cv_threshold(name,rows,seed):
 groups=sorted({r['group'] for r in rows}); folds=[set(groups[i::max(2,min(4,len(groups)))]) for i in range(max(2,min(4,len(groups))))]; cross=[]
 for j,fold in enumerate(folds):
  tr=[r for r in rows if r['group'] not in fold]; va=[r for r in rows if r['group'] in fold]; m=fit(name,tr,seed+j*1000); cross.extend(aggregate(va,predict(m,va)))
 return choose_threshold(cross)
def main():
 rows=load_test_only(); cases=sorted({r['case'] for r in rows}); allres=[]
 for name in ['direct','candidate']:
  case_res=[]
  for held in cases:
   tr=[r for r in rows if r['case']!=held]; te=[r for r in rows if r['case']==held]
   runs=[]
   for seed in [11,22,33]:
    th=cv_threshold(name,tr,seed); m=fit(name,tr,seed+999); pr=predict(m,te); pred=aggregate(te,pr); mm=metrics(pred,th); mm.update({'held_case':held,'seed':seed,'threshold':th,'record_fault_recall':float(np.mean([np.any(np.asarray(p)>=th) for p,r in zip(pr,te) if np.sum(r['y'])>0])) if any(np.sum(r['y'])>0 for r in te) else None}); runs.append(mm)
   case_res.append({'held_case':held,'runs':runs,'mean_macro_f1':float(np.mean([r['macro_f1'] for r in runs])),'mean_jaccard':float(np.mean([r['mean_set_jaccard'] for r in runs])),'mean_exact':float(np.mean([r['exact_set_match'] for r in runs]))})
  allres.append({'method':name,'cases':case_res})
 OUT.mkdir(exist_ok=True); (OUT/'results.json').write_text(json.dumps({'protocol':'leave-one-case-out; no records from held case used for fitting or threshold; native-rate SCA features; 3 seeds','methods':allres},ensure_ascii=False,indent=2)); print(json.dumps(allres,ensure_ascii=False))
if __name__=='__main__': main()
