"""Strict case-level retraining for the proposed non-negative candidate model."""
from pathlib import Path
import sys, json
import numpy as np
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'revision_stage12_20260912')); sys.path.insert(0,str(ROOT/'revision_stage5_20260912')); sys.path.insert(0,str(ROOT/'eaai_three_reviews_20260910'/'packet'))
from run_sca_strict_case_retraining import load_test_only, choose_threshold, case_metrics
from run_hust_zero_shot_models import predict, aggregate, metrics, seed_all
from composition_experiment import CompositionModel, train as train_comp
OUT=ROOT/'revision_stage12_20260912'/'sca_field_results_strict_candidate'; OUT.mkdir(parents=True,exist_ok=True)

def main():
    rows=load_test_only(); by=defaultdict(list)
    for r in rows: by[r['case']].append(r)
    train=[]; tune=[]; hold=[]
    for case,rs in by.items():
        rs=sorted(rs,key=lambda r:r['timestamp']); n=len(rs); a=int(n*.56); b=int(n*.70); train.extend(rs[:a]); tune.extend(rs[a:b]); hold.extend(rs[b:])
    print('rows',len(rows),'train',len(train),'tune',len(tune),'hold',len(hold),flush=True)
    runs=[]
    for seed in [11,22,33,44,55]:
        seed_all(seed); m=CompositionModel(100,decoder_steps=5,use_reference=False)
        train_comp(m,train,epochs=100,lr=.001,seed=seed,weights={'recon':.1,'align':0,'union':.1,'coh':.001})
        pt=predict(m,tune); th=choose_threshold(aggregate(tune,pt)); ph=predict(m,hold); pred=aggregate(hold,ph); mm=metrics(pred,th); yy=np.asarray([r['y'] for r in hold]); pb=ph>=th; healthy=yy.sum(1)==0; faulty=~healthy
        mm.update({'seed':seed,'threshold':th,'healthy_false_alarm_rate':float(np.mean(np.any(pb[healthy],axis=1))),'fault_file_recall':float(np.mean(np.any(pb[faulty],axis=1))),'holdout_n':len(hold),'holdout_healthy_n':int(healthy.sum()),'holdout_fault_n':int(faulty.sum())}); mm.update(case_metrics(hold,ph,th)); runs.append(mm)
        rec=[]
        for r,p in zip(hold,ph): rec.append({'case':r['case'],'pos':r['pos'],'group':r['group'],'timestamp':r['timestamp'],'event_date':r['event_date'],'y':list(map(float,r['y'])),'probability':list(map(float,p)),'prediction':list(map(int,(np.asarray(p)>=th).astype(int)))})
        (OUT/f'holdout_predictions_seed_{seed}.json').write_text(json.dumps({'seed':seed,'threshold':th,'records':rec},ensure_ascii=False),encoding='utf8')
    out={'protocol':'same strict case-level 56/14/30 temporal split as DirectMLP; candidate CompositionModel from random initialization; no post-replacement train data','runs':runs}
    for key in ['macro_f1','mean_set_jaccard','exact_set_match','healthy_false_alarm_rate','fault_file_recall','fault_event_recall','event_lead_days_mean']:
        vals=[r[key] for r in runs if r.get(key) is not None]; out['mean_'+key]=float(np.mean(vals)); out['std_'+key]=float(np.std(vals,ddof=1)) if len(vals)>1 else None
    (OUT/'candidate_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8'); print(json.dumps({k:out[k] for k in out if k.startswith('mean_') or k.startswith('std_')},ensure_ascii=False))
if __name__=='__main__': main()
