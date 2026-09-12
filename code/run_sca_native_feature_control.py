"""Native-rate, no-load-code control for the strict case-level replay."""
from pathlib import Path
import sys, json
from collections import defaultdict
from datetime import datetime
import numpy as np
from scipy.io import loadmat
from scipy.signal import hilbert
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'eaai_three_reviews_20260910'/'packet'))
from run_hust_zero_shot_models import DirectMLP,train as train_direct,predict,aggregate,metrics,seed_all
FIELD=ROOT/'revision_stage12_20260912'/'sca_field_raw'; OUT=ROOT/'revision_stage12_20260912'/'sca_field_results_native_control'; OUT.mkdir(parents=True,exist_ok=True)

def native_feat(x,sr,shaft):
    x=np.asarray(x,dtype=np.float64).reshape(-1); x=x-x.mean(); env=np.abs(hilbert(x)); raw=np.abs(np.fft.rfft(env)); freq=np.fft.rfftfreq(x.size,d=1.0/max(float(sr),1e-6)); order=freq/max(float(shaft),1e-6); centers=np.linspace(.10,20.0,100); z=np.interp(centers,order,np.log1p(raw),left=np.log1p(raw[0]),right=np.log1p(raw[-1])).astype(np.float32); return ((z-z.mean())/(z.std()+1e-6)).astype(np.float32)

def load_rows():
    rows=[]
    for p in sorted(FIELD.glob('case*_test.mat')):
        d=loadmat(p,squeeze_me=True); case=p.stem.split('_')[0]; todate=str(d.get('toDate','')).strip()
        for pos in ['DS','FS']:
            if pos not in d: continue
            rec=d[pos][()] if isinstance(d[pos],np.ndarray) and d[pos].shape==() else d[pos]; raw=np.asarray(rec['rawData'],dtype=np.float32); labels=np.asarray(rec['label']).reshape(-1); rpm=np.asarray(rec['RPM']).reshape(-1); sr=np.asarray(rec['samplingRate']).reshape(-1); times=np.asarray(rec['time']).reshape(-1)
            for i,(x,y,rr,ss,tm) in enumerate(zip(raw,labels,rpm,sr,times)):
                if int(y)<0 or float(ss)<=0 or float(rr)<=0: continue
                yy=np.zeros(3,dtype=np.float32)
                if int(y) in (1,2,3): yy[{1:0,3:1,2:2}[int(y)]]=1
                rows.append({'x':native_feat(x,float(ss),float(rr)/60.0),'c':np.asarray([1.,0.,0.],dtype=np.float32),'y':yy,'file':f'{case}_{pos}_{i:03d}','case':case,'group':f'{case}_{pos}','pos':pos,'state':('N' if int(y)==0 else str(int(y))),'timestamp':str(tm).strip(),'event_date':todate})
    return rows

def choose(items):
    best=(-1,.5)
    for th in np.linspace(.1,.9,81):
        mm=metrics(items,float(th)); h=[i for i in items if np.sum(i['y'])==0]; far=np.mean([np.any(np.asarray(i['p'])>=th) for i in h]) if h else 1.; score=mm['macro_f1'] if far<=.2 else mm['macro_f1']-2*(far-.2)
        if score>best[0]: best=(score,float(th))
    return best[1]

def main():
    rows=load_rows(); by=defaultdict(list)
    for r in rows: by[r['case']].append(r)
    tr=[]; tu=[]; ho=[]
    for c,rs in by.items():
        rs=sorted(rs,key=lambda r:r['timestamp']); n=len(rs); a=int(n*.56); b=int(n*.70); tr.extend(rs[:a]); tu.extend(rs[a:b]); ho.extend(rs[b:])
    runs=[]
    for seed in [11,22,33,44,55]:
        seed_all(seed); m=DirectMLP(100); train_direct(m,tr,epochs=100,lr=.001); pt=predict(m,tu); th=choose(aggregate(tu,pt)); ph=predict(m,ho); pi=aggregate(ho,ph); mm=metrics(pi,th); y=np.asarray([r['y'] for r in ho]); b=ph>=th; h=y.sum(1)==0; f=~h; mm.update({'seed':seed,'threshold':th,'healthy_false_alarm_rate':float(np.mean(np.any(b[h],axis=1))),'fault_file_recall':float(np.mean(np.any(b[f],axis=1))),'holdout_n':len(ho)}); runs.append(mm)
    out={'protocol':'native-rate full-record envelope-order features, constant condition input, case-level 56/14/30 split','runs':runs}
    for k in ['macro_f1','mean_set_jaccard','exact_set_match','healthy_false_alarm_rate','fault_file_recall']:
        v=[r[k] for r in runs]; out['mean_'+k]=float(np.mean(v)); out['std_'+k]=float(np.std(v,ddof=1))
    (OUT/'native_control_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8'); print(json.dumps({k:out[k] for k in out if k.startswith('mean_') or k.startswith('std_')},ensure_ascii=False))
if __name__=='__main__': main()
