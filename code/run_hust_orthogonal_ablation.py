"""Fixed-protocol orthogonal ablation for HUST composition model."""
import json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'eaai_three_reviews_20260910'/'packet')); sys.path.insert(0,str(ROOT/'revision_stage5_20260912'))
from run_hust_zero_shot_models import find_files,make_rows,aggregate,predict,metrics
from composition_experiment import CompositionModel,train as train_comp,seed_all
DATA=Path(r'C:\Users\FGHJ8\Desktop\BH-SCI\BH第一篇\实验数据\data_acquisition_20260907\raw\HUST_Hanoi_Bearing_v2\HUST bearing dataset'); MAN=ROOT/'review_v11_20260908'/'packet'/'experiment'/'artifacts'/'hust_zero_shot_v1'; OUT=ROOT/'revision_stage12_20260912'/'hust_orthogonal_ablation'
VARIANTS=['full','without_reconstruction','without_union','without_coherence','one_decoder_step','without_alignment_loss']
def choose(items):
 best=(-1,.5)
 for t in np.linspace(.1,.9,81):
  mm=metrics(items,float(t)); h=[i for i in items if sum(i['y'])==0]; far=np.mean([np.any(np.asarray(i['p'])>=t) for i in h]); score=mm['macro_f1'] if far<=.2 else mm['macro_f1']-2*(far-.2)
  if score>best[0]: best=(score,float(t))
 return best[1]
def fit(rows,seed,var):
 seed_all(seed); m=CompositionModel(len(rows[0]['x']),decoder_steps=1 if var=='one_decoder_step' else 5,use_reference=True); w={'recon':0 if var=='without_reconstruction' else .1,'align':0 if var=='without_alignment_loss' else .01,'union':0 if var=='without_union' else .1,'coh':0 if var=='without_coherence' else .001}; train_comp(m,rows,epochs=35,lr=.002,seed=seed,weights=w); return m
def main():
 dev=json.loads((MAN/'development_manifest.json').read_text()); vault=json.loads((MAN/'scoring_vault'/'compound_manifest.json').read_text()); fm=find_files(DATA); dr=make_rows(dev,fm,False,'envelope_order100'); sr=make_rows(vault,fm,True,'envelope_order100'); groups=sorted({r['group'] for r in dr}); folds=[set(groups[i::4]) for i in range(4)]; allres=[]
 for var in VARIANTS:
  runs=[]
  for seed in [11,22,33]:
   cross=[]
   for fold in folds:
    tr=[r for r in dr if r['group'] not in fold]; va=[r for r in dr if r['group'] in fold]; m=fit(tr,seed+len(cross),var); cross.extend(aggregate(va,predict(m,va)))
   th=choose(cross); m=fit(dr,seed+999,var); pred=aggregate(sr,predict(m,sr)); mm=metrics(pred,th); mm.update({'seed':seed,'threshold':th}); runs.append(mm)
  allres.append({'variant':var,'runs':runs,'mean_macro_f1':float(np.mean([r['macro_f1'] for r in runs])),'std_macro_f1':float(np.std([r['macro_f1'] for r in runs],ddof=1)),'mean_jaccard':float(np.mean([r['mean_set_jaccard'] for r in runs])),'std_jaccard':float(np.std([r['mean_set_jaccard'] for r in runs],ddof=1)),'mean_exact':float(np.mean([r['exact_set_match'] for r in runs]))})
 OUT.mkdir(parents=True,exist_ok=True); (OUT/'results.json').write_text(json.dumps({'protocol':'same HUST development group cross-fit; 35 epochs, lr 0.002, 3 seeds; one mechanism changed at a time','variants':allres},ensure_ascii=False,indent=2)); print(json.dumps([{k:x[k] for k in ['variant','mean_macro_f1','std_macro_f1','mean_jaccard','mean_exact']} for x in allres],ensure_ascii=False))
if __name__=='__main__': main()
