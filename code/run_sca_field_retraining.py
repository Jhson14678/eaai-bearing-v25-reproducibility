"""Target-site supervised retraining with strict chronological holdout."""
from pathlib import Path
import sys, json
from collections import defaultdict
from datetime import datetime
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'revision_stage12_20260912')); sys.path.insert(0,str(ROOT/'eaai_three_reviews_20260910'/'packet'))
from run_sca_field_external import load_field
from run_hust_zero_shot_models import DirectMLP, train as train_direct, predict, aggregate, metrics, seed_all
OUT=ROOT/'revision_stage12_20260912'/'sca_field_results_retraining'; OUT.mkdir(parents=True,exist_ok=True)

def choose_threshold(items):
    best=(-1,.5)
    for th in np.linspace(.1,.9,81):
        mm=metrics(items,float(th)); h=[i for i in items if np.sum(i['y'])==0]
        far=np.mean([np.any(np.asarray(i['p'])>=th) for i in h]) if h else 1.0
        score=mm['macro_f1'] if far<=.2 else mm['macro_f1']-2*(far-.2)
        if score>best[0]: best=(score,float(th))
    return best[1]

def main():
    field=load_field(); groups=defaultdict(list)
    for r in field: groups[r['group']].append(r)
    adapt=[]; hold=[]
    for g,rs in groups.items():
        rs=sorted(rs,key=lambda r:r.get('timestamp',''))
        if 'train' in g:
            adapt.extend(rs)
        else:
            n=max(1,int(len(rs)*.70)); adapt.extend(rs[:n]); hold.extend(rs[n:])
    print('adapt',len(adapt),'holdout',len(hold),flush=True)
    runs=[]
    for seed in [11,22,33,44,55]:
        seed_all(seed); m=DirectMLP(100)
        train_direct(m,adapt,epochs=100,lr=.001)
        pa=predict(m,adapt); th=choose_threshold(aggregate(adapt,pa))
        ph=predict(m,hold); pred=aggregate(hold,ph); mm=metrics(pred,th); yy=np.asarray([r['y'] for r in hold]); pb=ph>=th; healthy=yy.sum(1)==0; faulty=~healthy
        mm.update({'seed':seed,'threshold':th,'healthy_false_alarm_rate':float(np.mean(np.any(pb[healthy],axis=1))), 'fault_file_recall':float(np.mean(np.any(pb[faulty],axis=1)))})
        by=defaultdict(list)
        for r,pv in zip(hold,ph): by[r['group']].append((r,pv))
        event_leads=[]; fault_events=0; detected_fault_events=0; healthy_events=0; false_healthy_events=0
        for g,vals in by.items():
            truth=np.asarray([v[0]['y'] for v in vals]); predg=np.asarray([v[1] for v in vals])>=th
            is_fault=bool(np.any(truth.sum(1)>0)); detected=bool(np.any(predg))
            if is_fault:
                fault_events += 1; detected_fault_events += int(detected)
                if detected:
                    first=next(v[0] for v in vals if np.any(v[1]>=th))
                    try:
                        event_leads.append((datetime.fromisoformat(first['event_date'].replace('Z','')), datetime.fromisoformat(first['timestamp'].replace('Z',''))))
                    except Exception:
                        pass
            else:
                healthy_events += 1; false_healthy_events += int(detected)
        mm['fault_event_recall']=float(detected_fault_events/fault_events) if fault_events else None
        mm['healthy_event_false_alarm_rate']=float(false_healthy_events/healthy_events) if healthy_events else None
        mm['event_lead_days_mean']=float(np.mean([(a-b).total_seconds()/86400 for a,b in event_leads])) if event_leads else None
        mm['event_lead_cases']=len(event_leads)
        runs.append(mm)
        def recs(rows,probs,include_pred):
            out=[]
            for r,pv in zip(rows,probs):
                x={'group':r.get('group'),'timestamp':r.get('timestamp'),'event_date':r.get('event_date'),'y':list(map(float,r['y'])),'probability':list(map(float,pv))}
                if include_pred: x['prediction']=list(map(int,(np.asarray(pv)>=th).astype(int)))
                out.append(x)
            return out
        (OUT/f'adaptation_predictions_seed_{seed}.json').write_text(json.dumps({'seed':seed,'threshold':th,'n_records':len(adapt),'records':recs(adapt,pa,False)},ensure_ascii=False),encoding='utf8')
        (OUT/f'holdout_predictions_seed_{seed}.json').write_text(json.dumps({'seed':seed,'threshold':th,'n_records':len(hold),'records':recs(hold,ph,True)},ensure_ascii=False),encoding='utf8')
    out={'dataset':'SCA bearing dataset V1','doi':'10.17632/tdn96mkkpt.1','protocol':'target-site DirectMLP retraining on first 70% chronological measurements of each test group plus case1 post-replacement healthy train data; final 30% held out; threshold selected on adaptation segment','runs':runs}
    for key in ['macro_f1','mean_set_jaccard','exact_set_match','healthy_false_alarm_rate','fault_file_recall','fault_event_recall','healthy_event_false_alarm_rate','event_lead_days_mean']:
        vals=[r[key] for r in runs if r.get(key) is not None]; out['mean_'+key]=float(np.mean(vals)); out['std_'+key]=float(np.std(vals,ddof=1)) if len(vals)>1 else None
    (OUT/'retraining_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:out[k] for k in out if k.startswith('mean_') or k.startswith('std_')},ensure_ascii=False))
if __name__=='__main__': main()
