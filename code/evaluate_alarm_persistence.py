import json,glob,os
from collections import defaultdict
from datetime import datetime
import numpy as np

def first_run(records,k,max_gap_hours=24.0):
    vals=[bool(r['positive']) for r in records]
    for i in range(len(vals)-k+1):
        if all(vals[i:i+k]):
            ok=True
            for j in range(i,i+k-1):
                t0=datetime.fromisoformat(records[j]['timestamp'].replace('Z','')); t1=datetime.fromisoformat(records[j+1]['timestamp'].replace('Z',''))
                if (t1-t0).total_seconds()>max_gap_hours*3600: ok=False; break
            if ok: return i
    return None

def main():
    outs=[]
    root=os.environ.get('SCA_RESULTS_ROOT','revision_stage12_20260912/sca_field_results_native_primary')
    max_gap=float(os.environ.get('SCA_PERSIST_MAX_GAP_HOURS','24'))
    for path in sorted(glob.glob(os.path.join(root,'holdout_predictions_seed_*.json'))):
        d=json.load(open(path,encoding='utf8')); th=d['threshold']; by=defaultdict(list)
        for r in d['holdout_records']: by[(r['case'],r['pos'])].append(r)
        for key in by: by[key].sort(key=lambda r:r['timestamp'])
        for k in [1,2,3]:
            fault_case_alarm={}; healthy_groups=false_h=0; leads=[]
            for (case,pos),rs in by.items():
                truth=np.asarray([sum(r['y'])>0 for r in rs]); pred=np.asarray([max(r['probability'])>=th for r in rs]); records=[{'timestamp':r['timestamp'],'positive':bool(v),'event_date':r['event_date']} for r,v in zip(rs,pred)]
                is_fault=bool(truth.any()); idx=first_run(records,k,max_gap); alarm=idx is not None
                if is_fault:
                    fault_case_alarm[case]=fault_case_alarm.get(case,False) or alarm
                    if alarm:
                        # first persistent alert in this sensor stream
                        try:
                            a=datetime.fromisoformat(rs[idx]['event_date'].replace('Z','')); b=datetime.fromisoformat(rs[idx]['timestamp'].replace('Z','')); leads.append((a-b).total_seconds()/86400)
                        except Exception: pass
                else:
                    healthy_groups+=1; false_h+=int(alarm)
            fault_cases=len(fault_case_alarm); det_fault=sum(fault_case_alarm.values())
            outs.append({'seed':int(path.split('_')[-1].split('.')[0]),'k':k,'max_gap_hours':max_gap,'fault_event_recall':det_fault/fault_cases if fault_cases else None,'fault_events':fault_cases,'detected_fault_events':det_fault,'healthy_sensor_event_far':false_h/healthy_groups if healthy_groups else None,'healthy_sensor_groups':healthy_groups,'lead_days_mean':float(np.mean(leads)) if leads else None,'lead_days':leads})
    out={'protocol':'post-hoc persistence sensitivity on strict holdout predictions; no thresholds changed; case-level fault event aggregation; consecutive records must be within max gap','max_gap_hours':max_gap,'results':outs}
    out_path=os.path.join(root,'alarm_persistence_sensitivity.json'); open(out_path,'w',encoding='utf8').write(json.dumps(out,ensure_ascii=False,indent=2)); print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__': main()
