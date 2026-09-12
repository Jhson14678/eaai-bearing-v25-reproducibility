"""Strict case-level temporal replay without post-replacement future data.

Only case*_test.mat files are used. DS/FS records from the same case share one
chronological split, preventing a sensor from leaking future information into
the paired sensor. The first 56% is training, the next 14% tunes the threshold,
and the final 30% is a completely held-out time segment.
"""
from pathlib import Path
import sys, json, os
from collections import defaultdict
from datetime import datetime
import numpy as np
import torch
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parents[1]
CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE))
if not (CODE/'run_hust_zero_shot_models.py').exists():
    sys.path.insert(0,str(ROOT/'eaai_three_reviews_20260910'/'packet'))
from scipy.signal import hilbert, resample_poly, butter, sosfiltfilt
from run_hust_zero_shot_models import DirectMLP, train as train_direct, predict, aggregate, metrics, seed_all
default_field=ROOT/'revision_stage12_20260912'/'sca_field_raw' if (ROOT/'revision_stage12_20260912'/'sca_field_raw').exists() else ROOT/'sca_field_raw'
default_out=ROOT/'revision_stage12_20260912'/'sca_field_results_strict' if (ROOT/'revision_stage12_20260912').exists() else ROOT/'results'/'strict_case_retraining'
FIELD=Path(os.environ.get('SCA_FIELD_ROOT',str(default_field))); OUT=Path(os.environ.get('SCA_OUTPUT_ROOT',str(default_out))); OUT.mkdir(parents=True,exist_ok=True)

def _normalize(v):
    v=np.asarray(v,dtype=np.float32)
    return (v-v.mean())/(v.std()+1e-6)

def feat_native(x,sr,shaft):
    """Extract order-envelope features without reconstructing high frequencies.

    Each record contributes up to ten native one-second windows. A conservative
    0.5--0.45*Nyquist band-pass is applied before the Hilbert transform; orders
    above the measurable band are zero-filled rather than extrapolated.
    """
    sr=max(float(sr),1.0); shaft=max(float(shaft),1e-6)
    x=np.asarray(x,dtype=np.float64).reshape(-1); n=max(1,int(round(sr)))
    centers=np.linspace(.10,20.0,100); fs=[]
    try:
        hi=min(0.45*sr,0.45*sr) # Hz; below Nyquist with guard band
        lo=min(0.5,0.25*shaft)
        if hi>lo*1.05 and hi < sr/2:
            sos=butter(4,[lo/(sr/2),hi/(sr/2)],btype='bandpass',output='sos')
        else:
            sos=None
    except Exception:
        sos=None
    for i in range(min(10,x.size//n)):
        w=x[i*n:(i+1)*n]; w=w-w.mean()
        if sos is not None:
            try: w=sosfiltfilt(sos,w)
            except Exception: pass
        env=np.abs(hilbert(w)); raw=np.abs(np.fft.rfft(env)); freq=np.fft.rfftfreq(n,d=1.0/sr)
        order=freq/shaft; valid=(order>=centers[0])&(order<=min(20.0,0.45*sr/shaft))
        s=np.zeros_like(centers,dtype=np.float64)
        if np.any(valid): s=np.interp(centers,order[valid],np.log1p(raw[valid]),left=0.0,right=0.0)
        fs.append(_normalize(s))
    return np.mean(fs,axis=0).astype(np.float32) if fs else np.zeros(100,dtype=np.float32)

def feat_upsample_legacy(x,sr,shaft):
    """Legacy common-grid feature retained only for sensitivity comparison."""
    z=resample_poly(np.asarray(x,dtype=np.float64).reshape(-1),51200,max(1,int(round(float(sr))))); fs=[]
    for i in range(min(10,z.size//51200)):
        w=z[i*51200:(i+1)*51200]; w=w-w.mean(); raw=np.abs(np.fft.rfft(np.abs(hilbert(w)))); freq=np.fft.rfftfreq(51200,d=1.0/51200.0); order=freq/max(float(shaft),1e-6); centers=np.linspace(.10,20.0,100); s=np.interp(centers,order,np.log1p(raw),left=0.0,right=0.0).astype(np.float32); fs.append(_normalize(s))
    return np.mean(fs,axis=0).astype(np.float32) if fs else np.zeros(100,dtype=np.float32)

FEATURE_MODE=os.environ.get('SCA_FEATURE_MODE','native').lower()
feat=feat_native if FEATURE_MODE=='native' else feat_upsample_legacy

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
                speed=float(rr)/60.0; c=np.asarray([1.,0.,0.],dtype=np.float32)  # SCA has no HUST load labels; no load-code prior is used.
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
        vals=sorted(vals,key=lambda v: v[0]['timestamp'])
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
        rs=sorted(rs,key=lambda r:r['timestamp']); n=len(rs); ts=np.array([datetime.fromisoformat(r['timestamp'].replace('Z','')) for r in rs]); t0,t1=ts[0],ts[-1]; t56=t0+(t1-t0)*.56; t70=t0+(t1-t0)*.70
        tr=[r for r,t in zip(rs,ts) if t<=t56]; tu=[r for r,t in zip(rs,ts) if t>t56 and t<=t70]; ho=[r for r,t in zip(rs,ts) if t>t70]
        if not tr or not tu or not ho: raise RuntimeError(f'empty time split for {case}')
        train.extend(tr); tune.extend(tu); hold.extend(ho)
        split_info[case]={'n':n,'train':len(tr),'tune':len(tu),'hold':len(ho),'first':rs[0]['timestamp'],'train_end':tr[-1]['timestamp'],'tune_end':tu[-1]['timestamp'],'hold_start':ho[0]['timestamp'],'last':rs[-1]['timestamp'],'toDate':rs[0]['event_date'],'feature_mode':FEATURE_MODE}
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
    out={'dataset':'SCA bearing dataset V1','doi':'10.17632/tdn96mkkpt.1','protocol':'test-only case-level chronological split; no post-replacement train data; elapsed-time 56% train, next 14% threshold tuning, final 30% holdout; DS/FS from each case share the same split; native-rate band-limited order-envelope features by default','feature_mode':FEATURE_MODE,'split_info':split_info,'runs':runs}
    for key in ['macro_f1','mean_set_jaccard','exact_set_match','healthy_false_alarm_rate','fault_file_recall','fault_event_recall','event_lead_days_mean']:
        vals=[r[key] for r in runs if r.get(key) is not None]; out['mean_'+key]=float(np.mean(vals)); out['std_'+key]=float(np.std(vals,ddof=1)) if len(vals)>1 else None
    (OUT/'strict_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8'); print(json.dumps({k:out[k] for k in out if k.startswith('mean_') or k.startswith('std_')},ensure_ascii=False))
if __name__=='__main__': main()
