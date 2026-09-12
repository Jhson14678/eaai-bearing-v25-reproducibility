import json,glob
from collections import defaultdict
from datetime import datetime
import numpy as np

def first_run(vals,k):
    vals=list(vals)
    for i in range(len(vals)-k+1):
        if all(vals[i:i+k]): return i
    return None

def main():
    outs=[]
    for path in sorted(glob.glob('revision_stage12_20260912/sca_field_results_strict/holdout_predictions_seed_*.json')):
        d=json.load(open(path,encoding='utf8')); th=d['threshold']; by=defaultdict(list)
        for r in d['holdout_records']: by[(r['case'],r['pos'])].append(r)
        for key in by: by[key].sort(key=lambda r:r['timestamp'])
        for k in [1,2,3]:
            fault_cases=det_fault=healthy_groups=false_h=0; leads=[]
            for (case,pos),rs in by.items():
                truth=np.asarray([sum(r['y'])>0 for r in rs]); pred=np.asarray([max(r['probability'])>=th for r in rs])
                is_fault=bool(truth.any()); idx=first_run(pred,k); alarm=idx is not None
                if is_fault:
                    fault_cases+=1; det_fault+=int(alarm)
                    if alarm:
                        # first persistent alert in this sensor stream
                        try:
                            a=datetime.fromisoformat(rs[idx]['event_date'].replace('Z','')); b=datetime.fromisoformat(rs[idx]['timestamp'].replace('Z','')); leads.append((a-b).total_seconds()/86400)
                        except Exception: pass
                else:
                    healthy_groups+=1; false_h+=int(alarm)
            outs.append({'seed':int(path.split('_')[-1].split('.')[0]),'k':k,'fault_event_recall':det_fault/fault_cases if fault_cases else None,'fault_events':fault_cases,'detected_fault_events':det_fault,'healthy_sensor_event_far':false_h/healthy_groups if healthy_groups else None,'healthy_sensor_groups':healthy_groups,'lead_days_mean':float(np.mean(leads)) if leads else None,'lead_days':leads})
    out={'protocol':'post-hoc persistence sensitivity on strict holdout predictions; no thresholds changed','results':outs}
    open('revision_stage12_20260912/sca_field_results_strict/alarm_persistence_sensitivity.json','w',encoding='utf8').write(json.dumps(out,ensure_ascii=False,indent=2)); print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__': main()
