"""Same-protocol Random Forest strong baseline for HUST unseen-combination scoring."""
import json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.ensemble import RandomForestClassifier
ROOT=Path(__file__).resolve().parents[1]
PK=ROOT/'eaai_three_reviews_20260910'/'packet'; sys.path.insert(0,str(PK))
from run_hust_zero_shot_models import find_files,make_rows,aggregate,metrics
DATA=Path(r'C:\Users\FGHJ8\Desktop\BH-SCI\BH第一篇\实验数据\data_acquisition_20260907\raw\HUST_Hanoi_Bearing_v2\HUST bearing dataset')
MAN=ROOT/'review_v11_20260908'/'packet'/'experiment'/'artifacts'/'hust_zero_shot_v1'
OUT=ROOT/'revision_stage12_20260912'/'hust_rf_results'

def probs(model,rows):
 x=np.asarray([np.r_[r['x'],r['c']] for r in rows]); pp=model.predict_proba(x); return np.column_stack([a[:,1] if a.shape[1]>1 else np.zeros(len(x)) for a in pp])
def th_select(items):
 best=(-1,.5)
 for t in np.linspace(.1,.9,81):
  m=metrics(items,float(t)); h=[i for i in items if sum(i['y'])==0]; far=np.mean([np.any(np.asarray(i['p'])>=t) for i in h]); score=m['macro_f1'] if far<=.2 else m['macro_f1']-2*(far-.2)
  if score>best[0]: best=(score,float(t))
 return best[1]
def main():
 dev=json.loads((MAN/'development_manifest.json').read_text()); vault=json.loads((MAN/'scoring_vault'/'compound_manifest.json').read_text()); fm=find_files(DATA); dr=make_rows(dev,fm,False,'envelope_order100'); sr=make_rows(vault,fm,True,'envelope_order100'); by=defaultdict(list)
 for r in dr: by[r['group']].append(r)
 groups=sorted(by); folds=[set(groups[i::4]) for i in range(4)]; out=[]
 for seed in [11,22,33,44,55]:
  cross=[]
  for fold in folds:
   tr=[r for r in dr if r['group'] not in fold]; va=[r for r in dr if r['group'] in fold]
   m=RandomForestClassifier(n_estimators=400,max_features='sqrt',min_samples_leaf=2,class_weight='balanced_subsample',random_state=seed,n_jobs=1); m.fit(np.asarray([np.r_[r['x'],r['c']] for r in tr]),np.asarray([r['y'] for r in tr])); cross.extend(aggregate(va,probs(m,va)))
  th=th_select(cross); m=RandomForestClassifier(n_estimators=400,max_features='sqrt',min_samples_leaf=2,class_weight='balanced_subsample',random_state=seed,n_jobs=1); m.fit(np.asarray([np.r_[r['x'],r['c']] for r in dr]),np.asarray([r['y'] for r in dr])); pred=aggregate(sr,probs(m,sr)); mm=metrics(pred,th); mm.update({'seed':seed,'threshold':th,'predictions':pred}); out.append(mm)
 result={'method':'RandomForest multi-output strong baseline','protocol':'HUST development-only group cross-fitted threshold; compound scoring vault read only after freeze','runs':out}
 for k in ['macro_f1','mean_set_jaccard','exact_set_match']:
  v=[r[k] for r in out]; result['mean_'+k]=float(np.mean(v)); result['std_'+k]=float(np.std(v,ddof=1))
 OUT.mkdir(parents=True,exist_ok=True); (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)); print({k:result[k] for k in result if k.startswith(('mean_','std_'))})
if __name__=='__main__': main()
