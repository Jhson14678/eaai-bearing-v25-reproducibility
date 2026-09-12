"""Target-domain adaptation with strict chronological holdout on SCA field cases."""
from pathlib import Path
import sys,json,os
from collections import defaultdict
from datetime import datetime
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'revision_stage12_20260912')); sys.path.insert(0,str(ROOT/'revision_stage5_20260912')); sys.path.insert(0,str(ROOT/'eaai_three_reviews_20260910'/'packet'))
from run_sca_field_external import load_field,choose_threshold
from run_hust_zero_shot_models import find_files,make_rows,predict,aggregate,metrics,seed_all
from composition_experiment import CompositionModel,train as train_comp
HUST=Path(os.environ.get('HUST_ROOT', 'data/HUST bearing dataset')); MAN=ROOT/'review_v11_20260908'/'packet'/'experiment'/'artifacts'/'hust_zero_shot_v1'; OUT=ROOT/'revision_stage12_20260912'/'sca_field_results'
def main():
 devrec=json.loads((MAN/'development_manifest.json').read_text(encoding='utf8')); dev=make_rows(devrec,find_files(HUST),False,'envelope_order100'); field=load_field(); groups=defaultdict(list)
 for r in field: groups[r['group']].append(r)
 adapt=[]; hold=[]
 for g,rs in groups.items():
  rs=sorted(rs,key=lambda r:r.get('timestamp',''))
  if 'train' in g: adapt.extend(rs); continue
  n=max(1,int(len(rs)*.70)); adapt.extend(rs[:n]); hold.extend(rs[n:])
 print('adapt',len(adapt),'holdout',len(hold),flush=True)
 runs=[]
 for seed in [11,22,33,44,55]:
  seed_all(seed); m=CompositionModel(100,decoder_steps=5,use_reference=False); train_comp(m,dev,epochs=35,lr=.002,seed=seed,weights={'recon':.1,'align':0,'union':.1,'coh':.001});
  # supervised target-domain update uses only chronological adaptation segment
  train_comp(m,adapt,epochs=20,lr=.0005,seed=seed+1000,weights={'recon':.1,'align':0,'union':0,'coh':.001})
  pa=aggregate(adapt,predict(m,adapt)); best=(-1,.5)
  for th in np.linspace(.1,.9,81):
   mm=metrics(pa,float(th)); h=[i for i in pa if np.sum(i['y'])==0]; far=np.mean([np.any(np.asarray(i['p'])>=th) for i in h]) if h else 1; score=mm['macro_f1'] if far<=.2 else mm['macro_f1']-2*(far-.2)
   if score>best[0]: best=(score,float(th))
  th=best[1]; ph=predict(m,hold); pred=aggregate(hold,ph); mm=metrics(pred,th); mm.update({'seed':seed,'threshold':th});
  # Persist every held-out record so the reported field metrics are auditable.
  recs=[]
  for r,pv in zip(hold,ph):
   recs.append({'group':r.get('group'),'timestamp':r.get('timestamp'),'event_date':r.get('event_date'),'y':list(map(float,r['y'])),'probability':list(map(float,pv)),'prediction':list(map(int,(np.asarray(pv)>=th).astype(int)))})
  (OUT/f'predictions_seed_{seed}.json').write_text(json.dumps({'seed':seed,'threshold':th,'n_records':len(recs),'records':recs},ensure_ascii=False),encoding='utf8')
  yy=np.asarray([r['y'] for r in hold]); pb=(np.asarray(ph)>=th).astype(float); healthy=(yy.sum(1)==0); faulty=~healthy
  mm['healthy_false_alarm_rate']=float(np.mean(np.any(pb[healthy]>0.5,axis=1))) if healthy.any() else None
  mm['fault_file_recall']=float(np.mean(np.any(pb[faulty]>0.5,axis=1))) if faulty.any() else None
  # event lead time: earliest positive in each held-out group vs documented toDate
  leads=[]
  for g,rs in defaultdict(list).items(): pass
  by=defaultdict(list)
  for r,pv in zip(hold,ph): by[r['group']].append((r,pv))
  for g,vals in by.items():
   pos=[r for r,pv in vals if np.any(pv>=th)];
   if not pos: continue
   try: leads.append((datetime.fromisoformat(pos[0]['event_date'].replace('Z','')),datetime.fromisoformat(pos[0]['timestamp'].replace('Z',''))))
   except Exception: pass
  mm['event_lead_days_mean']=float(np.mean([(a-b).total_seconds()/86400 for a,b in leads])) if leads else None; mm['event_lead_cases']=len(leads); runs.append(mm)
 out={'dataset':'SCA bearing dataset V1','doi':'10.17632/tdn96mkkpt.1','protocol':'HUST pretraining then supervised target-domain update on first 70% chronological measurements per test group; case1 post-replacement train data included only in adaptation; final 30% held out','runs':runs,'mean_macro_f1':float(np.mean([r['macro_f1'] for r in runs])),'std_macro_f1':float(np.std([r['macro_f1'] for r in runs],ddof=1)),'mean_jaccard':float(np.mean([r['mean_set_jaccard'] for r in runs])),'mean_exact':float(np.mean([r['exact_set_match'] for r in runs])),'mean_healthy_false_alarm_rate':float(np.mean([r['healthy_false_alarm_rate'] for r in runs])),'mean_fault_file_recall':float(np.mean([r['fault_file_recall'] for r in runs])),'mean_event_lead_days':float(np.mean([r['event_lead_days_mean'] for r in runs if r['event_lead_days_mean'] is not None])) if any(r['event_lead_days_mean'] is not None for r in runs) else None}
 (OUT/'adaptation_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8'); print(json.dumps({k:out[k] for k in ['mean_macro_f1','std_macro_f1','mean_jaccard','mean_exact','mean_event_lead_days']},ensure_ascii=False))
if __name__=='__main__': main()
