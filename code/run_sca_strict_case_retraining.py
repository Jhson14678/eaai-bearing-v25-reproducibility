"""Strict case-level temporal replay without post-replacement future data.

Only case*_test.mat files are used. DS/FS records from the same case share one
chronological split, preventing a sensor from leaking future information into
the paired sensor. The first 56% is training, the next 14% tunes the threshold,
and the final 30% is a completely held-out time segment.
"""
from pathlib import Path
import sys, json
from collections import defaultdict
from datetime import datetime
import numpy as np
import torch
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'revision_stage12_20260912')); sys.path.insert(0,str(ROOT/'eaai_three_reviews_20260910'/'packet'))
from run_sca_field_external import feat
from run_hust_zero_shot_models import DirectMLP, train as train_direct, predict, aggregate, metrics, seed_all
FIELD=ROOT/'revision_stage12_20260912'/'sca_field_raw'; OUT=ROOT/'revision_stage12_20260912'/'sca_field_results_strict'; OUT.mkdir(parents=True,exist_ok=True)

def load_test_only():
    rows=[]
    for p in sorted(FIELD.glob('case*_test.mat')):
        d=loadmat(p,squeeze_me=True); case=p.stem.split('_')[0]; to_date=str(d.get('toDate','')).strip()
        for pos in ['DS','FS']:
            if pos not in d: continue
            rec=d[pos][()] if isinstance(d[pos],np.ndarray) and d[pos].shape==() else d[pos]
            raw=np.asarray(rec['rawData'],dtype=np.float32); labels=np.asarray(rec['label']).reshape(-1); rpm=np.asarray(rec['RPM']).reshape(-1); sr=np.asarray(rec['samplingRate']).reshape(-1); times=np.asarray(rec['time']).reshape(-1)
            for i,(x,y,rr,ss,tm) in enumerate(zip(raw,labels,rpm,sr,times)):
                if int(y)<0 or float(ss)<=0 or float(rr)<=0: continue
                yy=np.zeros(3,dtype=np.float32)
                if int(y) in (1,2,3): yy[{1:0,3:1,2:2}[int(y)]]=1.0
                speed=float(rr)/60.0; li=int(np.argmin(np.abs(np.asarray([24.9,25.0,25.1])-speed))); c=np.zeros(3,dtype=np.float32); c[li]=1.0
                rows.append({'x':feat(x,float(ss),speed),'c':c,'y':yy,'file':f'{case}_{pos}_{i:03d}','case':case,'group':f'{case}_{pos}','pos':pos,'state':('N' if int(y)==0 else str(int(y))),'timestamp':str(tm).strip(),'event_date':to_date})
    return rows

def choose_threshold(items):
    best=(-1,.5)
    for th in np.linspace(.1,.9,81):
        mm=metrics(items,float(th)); h=[i for i in items if np.sum(i['y'])==0]; far=np.mean([np.any(np.asarray(i['p'])>=th) for i in h]) if h else 1.0
        score=mm['macro_f1'] if far<=.2 else mm['macro_f1']-2*(far-.2)
        if score>best[0]: best=(score,float(th))
    return best[1]

def case_metrics(rows, probs, th):
    by=defaultdict(list)
    for r,p in zip(rows,probs): by[r['case']].append((r,p))
    event_leads=[]; fault_cases=0; detected_fault=0
    for case,vals in by.items():
        truth=np.asarray([v[0]['y'] for v in vals]); pred=np.asarray([v[1] for v in vals])>=th
        is_fault=bool(np.any(truth.sum(1)>0)); detected=bool(np.any(pred))
        if is_fault:
            fault_cases+=1; detected_fault+=int(detected)
            if detected:
                first=next(v[0] for v in vals if np.any(v[1]>=th))
                try: event_leads.append((datetime.fromisoformat(first['event_date'].replace('Z','')),datetime.fromisoformat(first['timestamp'].replace('Z',''))))
                except Exception: pass
    return {'fault_event_recall':float(detected_fault/fault_cases) if fault_cases else None,'fault_event_n':fault_cases,'fault_event_detected':detected_fault,'event_lead_days_mean':float(np.mean([(a-b).total_seconds()/86400 for a,b in event_leads])) if event_leads else None,'event_lead_days':[(a-b).total_seconds()/86400 for a,b in event_leads]}

def main():
    rows=load_test_only(); by=defaultdict(list)
    for r in rows: by[r['case']].append(r)
    train=[]; tune=[]; hold=[]
    split_info={}
    for case,rs in by.items():
        rs=sorted(rs,key=lambda r:r['timestamp']); n=len(rs); a=int(n*.56); b=int(n*.70); train.extend(rs[:a]); tune.extend(rs[a:b]); hold.extend(rs[b:]); split_info[case]={'n':n,'train':a,'tune':b-a,'hold':n-b,'first':rs[0]['timestamp'],'adapt_end':rs[b-1]['timestamp'],'hold_start':rs[b]['timestamp'],'last':rs[-1]['timestamp'],'toDate':rs[0]['event_date']}
    print('rows',len(rows),'train',len(train),'tune',len(tune),'hold',len(hold),flush=True)
    runs=[]
    for seed in [11,22,33,44,55]:
        seed_all(seed); m=DirectMLP(100); train_direct(m,train,epochs=100,lr=.001); pt=predict(m,tune); th=choose_threshold(aggregate(tune,pt)); ph=predict(m,hold); pred=aggregate(hold,ph); mm=metrics(pred,th); yy=np.asarray([r['y'] for r in hold]); pb=ph>=th; healthy=yy.sum(1)==0; faulty=~healthy
        mm.update({'seed':seed,'threshold':th,'healthy_false_alarm_rate':float(np.mean(np.any(pb[healthy],axis=1))) if healthy.any() else None,'fault_file_recall':float(np.mean(np.any(pb[faulty],axis=1))) if faulty.any() else None,'holdout_n':len(hold),'holdout_healthy_n':int(healthy.sum()),'holdout_fault_n':int(faulty.sum())}); mm.update(case_metrics(hold,ph,th)); runs.append(mm)
        def pack(rs,ps,include_pred):
            out=[]
            for r,p in zip(rs,ps):
                q={'case':r['case'],'pos':r['pos'],'group':r['group'],'timestamp':r['timestamp'],'event_date':r['event_date'],'y':list(map(float,r['y'])),'probability':list(map(float,p))}
                if include_pred:q['prediction']=list(map(int,(np.asarray(p)>=th).astype(int)))
                out.append(q)
            return out
        (OUT/f'train_tune_predictions_seed_{seed}.json').write_text(json.dumps({'seed':seed,'threshold':th,'tune_records':pack(tune,pt,False)},ensure_ascii=False),encoding='utf8')
        (OUT/f'holdout_predictions_seed_{seed}.json').write_text(json.dumps({'seed':seed,'threshold':th,'holdout_records':pack(hold,ph,True)},ensure_ascii=False),encoding='utf8')
    out={'dataset':'SCA bearing dataset V1','doi':'10.17632/tdn96mkkpt.1','protocol':'test-only case-level chronological split; no post-replacement train data; first 56% train, next 14% threshold tuning, final 30% holdout; DS/FS from each case share the same split','split_info':split_info,'runs':runs}
    for key in ['macro_f1','mean_set_jaccard','exact_set_match','healthy_false_alarm_rate','fault_file_recall','fault_event_recall','event_lead_days_mean']:
        vals=[r[key] for r in runs if r.get(key) is not None]; out['mean_'+key]=float(np.mean(vals)); out['std_'+key]=float(np.std(vals,ddof=1)) if len(vals)>1 else None
    (OUT/'strict_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8'); print(json.dumps({k:out[k] for k in out if k.startswith('mean_') or k.startswith('std_')},ensure_ascii=False))
if __name__=='__main__': main()
