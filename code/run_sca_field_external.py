"""Real industrial field validation using the SCA pulp-mill bearing dataset.

SCA provides continuous date-stamped measurements before bearing replacement,
normal post-replacement training records, and labels 0/1/2/3 for normal,
inner-ring/ball/outer-ring faults. HUST is used for development only.
"""
from pathlib import Path
import sys,json,os,random
from collections import defaultdict
import numpy as np
from datetime import datetime
from scipy.io import loadmat
from scipy.signal import hilbert,resample_poly
import torch
ROOT=Path(__file__).resolve().parents[1]; PK=ROOT/'eaai_three_reviews_20260910'/'packet'; sys.path.insert(0,str(PK)); sys.path.insert(0,str(ROOT/'revision_stage5_20260912'))
from run_hust_zero_shot_models import find_files,make_rows,predict,aggregate,metrics,DirectMLP,train as train_direct,seed_all
from composition_experiment import CompositionModel,train as train_comp
HUST=Path(os.environ.get('HUST_ROOT', 'data/HUST bearing dataset'))
MAN=ROOT/'review_v11_20260908'/'packet'/'experiment'/'artifacts'/'hust_zero_shot_v1'; FIELD=ROOT/'revision_stage12_20260912'/'sca_field_raw'; OUT=ROOT/'revision_stage12_20260912'/'sca_field_results'; OUT.mkdir(parents=True,exist_ok=True)

def feat(x,sr,shaft):
    z=resample_poly(np.asarray(x,dtype=np.float64).reshape(-1),51200,max(1,int(round(float(sr)))))
    fs=[]
    for i in range(min(10,z.size//51200)):
        w=z[i*51200:(i+1)*51200]; w=w-w.mean(); raw=np.abs(np.fft.rfft(np.abs(hilbert(w)))); freq=np.fft.rfftfreq(51200,d=1.0/51200.0); order=freq/max(float(shaft),1e-6); centers=np.linspace(.10,20.0,100); s=np.interp(centers,order,np.log1p(raw)).astype(np.float32); fs.append((s-s.mean())/(s.std()+1e-6))
    return np.mean(fs,axis=0).astype(np.float32) if fs else np.zeros(100,dtype=np.float32)

def load_field():
    rows=[]
    for p in sorted(FIELD.glob('case*_test.mat')) + [FIELD/'case1_train.mat']:
        d=loadmat(p,squeeze_me=True); case=p.stem
        for pos in ['DS','FS']:
            if pos not in d: continue
            rec=d[pos][()] if isinstance(d[pos],np.ndarray) and d[pos].shape==() else d[pos]
            names=rec.dtype.names
            raw=np.asarray(rec['rawData'],dtype=np.float32); labels=np.asarray(rec['label']).reshape(-1); rpm=np.asarray(rec['RPM']).reshape(-1); sr=np.asarray(rec['samplingRate']).reshape(-1); times=np.asarray(rec['time']).reshape(-1)
            for i,(x,y,rr,ss,tm) in enumerate(zip(raw,labels,rpm,sr,times)):
                if int(y)<0 or float(ss)<=0 or float(rr)<=0: continue
                yy=np.zeros(3,dtype=np.float32)
                if int(y) in (1,2,3): yy[{1:0,3:1,2:2}[int(y)]]=1.0
                # Map variable-speed field measurements to the nearest HUST condition code.
                speed=float(rr)/60.0; li=int(np.argmin(np.abs(np.asarray([24.9,25.0,25.1])-speed)))
                c=np.zeros(3,dtype=np.float32); c[li]=1.0
                rows.append({'x':feat(x,float(ss),speed),'c':c,'y':yy,'file':f'{case}_{pos}_{i:03d}','group':f'{case}_{pos}','state':('N' if int(y)==0 else str(int(y))),'timestamp':str(tm).strip(),'event_date':str(d.get('toDate','')).strip()})
    return rows

def folds(dev):
    by=defaultdict(list)
    for r in dev: by[r['state']].append(r['group'])
    fs=[set() for _ in range(4)]
    for gs in by.values():
        for i,g in enumerate(sorted(set(gs))): fs[i%4].add(g)
    return fs

def choose_threshold(dev,name,seed):
    cross=[]
    for fold in folds(dev):
        tr=[r for r in dev if r['group'] not in fold]; va=[r for r in dev if r['group'] in fold]
        seed_all(seed); m=DirectMLP(100) if name=='direct' else CompositionModel(100,decoder_steps=5,use_reference=False)
        if name=='direct': train_direct(m,tr,epochs=35,lr=.002)
        else: train_comp(m,tr,epochs=35,lr=.002,seed=seed,weights={'recon':.1,'align':0,'union':.1,'coh':.001})
        cross.extend(aggregate(va,predict(m,va)))
    best=(-1,.66)
    for th in np.linspace(.1,.9,81):
        mm=metrics(cross,float(th)); h=[i for i in cross if np.sum(i['y'])==0]; far=np.mean([np.any(np.asarray(i['p'])>=th) for i in h]) if h else 1; score=mm['macro_f1'] if far<=.2 else mm['macro_f1']-2*(far-.2)
        if score>best[0]: best=(score,float(th))
    return best[1]

def main():
    devrec=json.loads((MAN/'development_manifest.json').read_text(encoding='utf8')); dev=make_rows(devrec,find_files(HUST),False,'envelope_order100'); field=load_field(); field_train=[r for r in field if 'train' in r['file']]; field_test=[r for r in field if 'test' in r['file']]; print('field rows',len(field),'train',len(field_train),'test',len(field_test),flush=True)
    methods=[]
    for name in ['direct','candidate']:
        runs=[]
        for seed in [11,22,33,44,55]:
            th=choose_threshold(dev,name,seed); seed_all(seed); m=DirectMLP(100) if name=='direct' else CompositionModel(100,decoder_steps=5,use_reference=False)
            if name=='direct': train_direct(m,dev,epochs=35,lr=.002)
            else: train_comp(m,dev,epochs=35,lr=.002,seed=seed,weights={'recon':.1,'align':0,'union':.1,'coh':.001})
            ptest=predict(m,field_test); pred=aggregate(field_test,ptest); mm=metrics(pred,th); mm.update({'seed':seed,'threshold':th});
            ptr=predict(m,field_train); train_items=aggregate(field_train,ptr); cal_th=float(np.quantile(np.max(np.asarray([i['p'] for i in train_items]),axis=1),.95)); cal=metrics(pred,cal_th); mm['healthy_baseline_calibrated_threshold']=cal_th; mm['healthy_baseline_calibrated']=cal; runs.append(mm)
        methods.append({'method':name,'runs':runs,'mean_macro_f1':float(np.mean([r['macro_f1'] for r in runs])),'std_macro_f1':float(np.std([r['macro_f1'] for r in runs],ddof=1)),'mean_jaccard':float(np.mean([r['mean_set_jaccard'] for r in runs])),'std_jaccard':float(np.std([r['mean_set_jaccard'] for r in runs],ddof=1)),'mean_exact':float(np.mean([r['exact_set_match'] for r in runs])),'calibrated_mean_macro_f1':float(np.mean([r['healthy_baseline_calibrated']['macro_f1'] for r in runs])),'calibrated_mean_jaccard':float(np.mean([r['healthy_baseline_calibrated']['mean_set_jaccard'] for r in runs])),'calibrated_mean_exact':float(np.mean([r['healthy_baseline_calibrated']['exact_set_match'] for r in runs]))})
    out={'dataset':'SCA bearing dataset, Mendeley Data V1','doi':'10.17632/tdn96mkkpt.1','article_doi':'10.3390/data8070115','source':'pulp mill field measurements 2019-2022; case 1 test plus case 1 train','field_rows':len(field),'train_rows':len(field_train),'test_rows':len(field_test),'label_mapping':'0 healthy; 1 inner-ring; 2 ball; 3 outer-ring; -1 excluded','methods':methods,'calibration':'95th percentile of file-level maximum probability on post-replacement healthy train data; no fault labels used for threshold calibration'}
    (OUT/'results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8'); print(json.dumps({m['method']:{k:m[k] for k in ['mean_macro_f1','std_macro_f1','mean_jaccard','std_jaccard','mean_exact']} for m in methods},ensure_ascii=False))
if __name__=='__main__': main()
