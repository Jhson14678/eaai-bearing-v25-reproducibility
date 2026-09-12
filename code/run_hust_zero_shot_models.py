"""Run an auditable HUST zero-shot multi-label experiment.

This first executable confirmation uses a fixed log-spectrum representation and
two matched PyTorch models: a direct multi-label MLP and the manuscript's
condition-anchor plus non-negative component-dictionary head. Compound labels
are read only from the scoring vault after all training-side decisions freeze.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import hilbert
from torch import nn


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def find_files(root: Path) -> dict[str, Path]:
    return {p.name: p for p in root.rglob("*.mat")}


def features_for_file(path: Path, mode: str, window: int = 51200, max_windows: int = 10) -> np.ndarray:
    obj = loadmat(path, squeeze_me=True)
    x = np.asarray(obj["data"], dtype=np.float32).reshape(-1)
    shaft = float(np.asarray(obj.get("fs", 1.0)).reshape(-1)[0])
    n = min(max_windows, x.size // window)
    out = []
    for i in range(n):
        z = x[i * window : (i + 1) * window].astype(np.float64)
        z -= z.mean()
        base = np.abs(hilbert(z)) if mode == 'envelope_order100' else z
        raw = np.abs(np.fft.rfft(base))
        if mode in ('order100','envelope_order100'):
            freq = np.fft.rfftfreq(window, d=1.0/51200.0)
            order = freq / max(shaft, 1e-6)
            centers = np.linspace(0.10, 20.0, 100)
            spec = np.interp(centers, order, np.log1p(raw)).astype(np.float32)
        else:
            spec = np.log1p(raw)[1:257].astype(np.float32)
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        out.append(spec)
    if not out:
        raise ValueError(f"no complete 1-s window in {path}")
    return np.asarray(out, dtype=np.float32)


class DirectMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim + 3, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU())
        self.out = nn.Linear(64, 3)

    def forward(self, x, c):
        return self.out(self.net(torch.cat([x, c], dim=1)))


class DictionaryModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim + 3, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU())
        self.anchor = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, 64))
        self.dictionary = nn.Parameter(torch.randn(64, 3) * 0.05)
        self.scale = nn.Parameter(torch.ones(3))
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, x, c):
        h = self.encoder(torch.cat([x, c], dim=1))
        r = h - self.anchor(c)
        p = self.dictionary / (torch.linalg.vector_norm(self.dictionary, dim=0, keepdim=True) + 1e-6)
        coeff = torch.relu(r @ p)
        return coeff * torch.nn.functional.softplus(self.scale) + self.bias


def make_rows(records, file_map, include_compounds: bool, mode: str):
    rows = []
    for rec in records:
        if (rec["state"] in {"IO", "IB", "OB"}) != include_compounds:
            continue
        xwin = features_for_file(file_map[rec["file"]], 'order100' if mode == 'order200_file' else mode)
        x = np.concatenate([xwin.mean(axis=0), xwin.std(axis=0)])[None, :] if mode == 'order200_file' else xwin
        load_idx = {"00": 0, "02": 1, "04": 2}[rec["load_code"]]
        c = np.zeros((x.shape[0], 3), dtype=np.float32)
        c[:, load_idx] = 1.0
        y = np.zeros(3, dtype=np.float32)
        for lab in rec["labels"]:
            y[{"I": 0, "O": 1, "B": 2}[lab]] = 1.0
        for j in range(x.shape[0]):
            rows.append({"x": x[j], "c": c[j], "y": y, "file": rec["file"], "group": rec["physical_group"], "state": rec["state"]})
    return rows


def _union_rows(rows):
    """Build leakage-free single-label pair augmentations.

    The pairs are created only from development rows, never from the compound
    scoring vault.  Features are mixed in the normalized log-spectrum space;
    the union target is the element-wise OR of the two single-label targets.
    This is used as a compositional-consistency regularizer, not as a source of
    additional test labels.
    """
    by_load = defaultdict(dict)
    for r in rows:
        active = np.flatnonzero(r["y"] > 0.5)
        if active.size != 1:
            continue
        by_load[int(np.argmax(r["c"]))][int(active[0])] = r
    out = []
    for load, label_rows in sorted(by_load.items()):
        labels = sorted(label_rows)
        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                ra, rb = label_rows[a], label_rows[b]
                y = np.maximum(ra["y"], rb["y"]).astype(np.float32)
                out.append({
                    "x": (0.5 * (ra["x"] + rb["x"])).astype(np.float32),
                    "c": ra["c"].copy(),
                    "y": y,
                    "file": f"union::{ra['file']}+{rb['file']}",
                    "group": f"union::{ra['group']}+{rb['group']}",
                    "state": "synthetic_union",
                })
    return out


def train(model, rows, epochs=35, lr=2e-3, union_weight=0.0):
    model.train()
    x = torch.tensor(np.stack([r["x"] for r in rows]))
    c = torch.tensor(np.stack([r["c"] for r in rows]))
    y = torch.tensor(np.stack([r["y"] for r in rows]))
    pos = y.sum(0)
    neg = y.shape[0] - pos
    weight = neg / torch.clamp(pos, min=1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    union = _union_rows(rows) if union_weight > 0 else []
    ux = torch.tensor(np.stack([r["x"] for r in union])) if union else None
    uc = torch.tensor(np.stack([r["c"] for r in union])) if union else None
    uy = torch.tensor(np.stack([r["y"] for r in union])) if union else None
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(x, c), y, pos_weight=weight)
        if union:
            union_weight_vec = torch.ones(3)
            union_loss = nn.functional.binary_cross_entropy_with_logits(model(ux, uc), uy, pos_weight=union_weight_vec)
            loss = loss + float(union_weight) * union_loss
        loss.backward(); opt.step()


@torch.no_grad()
def predict(model, rows):
    model.eval()
    x = torch.tensor(np.stack([r["x"] for r in rows]))
    c = torch.tensor(np.stack([r["c"] for r in rows]))
    return torch.sigmoid(model(x, c)).cpu().numpy()


def aggregate(rows, probs):
    bucket = defaultdict(list)
    for r, p in zip(rows, probs):
        bucket[r["file"]].append((r, p))
    out = []
    for file, vals in sorted(bucket.items()):
        r = vals[0][0]
        out.append({"file": file, "group": r["group"], "state": r["state"], "y": r["y"].tolist(), "p": np.mean([v[1] for v in vals], axis=0).tolist()})
    return out


def f1(y, pred):
    tp = np.sum(y * pred, axis=0); fp = np.sum((1-y) * pred, axis=0); fn = np.sum(y * (1-pred), axis=0)
    return (2*tp / np.maximum(2*tp+fp+fn, 1e-12)).tolist()


def metrics(items, threshold):
    y = np.asarray([i["y"] for i in items], dtype=np.float64)
    p = np.asarray([i["p"] for i in items], dtype=np.float64)
    pred = (p >= threshold).astype(float)
    fs = f1(y, pred)
    js = []
    exact = []
    for a,b in zip(y,pred):
        inter = np.sum((a>0.5)&(b>0.5)); union = np.sum((a>0.5)|(b>0.5))
        js.append(1.0 if union == 0 else float(inter/union)); exact.append(float(np.array_equal(a,b)))
    return {"threshold": threshold, "per_component_f1": fs, "macro_f1": float(np.mean(fs)), "mean_set_jaccard": float(np.mean(js)), "exact_set_match": float(np.mean(exact)), "n_files": len(items), "n_groups": len({i["group"] for i in items})}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',type=Path,required=True); ap.add_argument('--manifest-root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--seeds',default='11,22,33,44,55'); ap.add_argument('--feature-mode',choices=['logfft','order100','envelope_order100','order200_file'],default='logfft'); ap.add_argument('--union-weight',type=float,default=0.0); args=ap.parse_args()
    seed_list=[int(x) for x in args.seeds.split(',')]
    dev=json.loads((args.manifest_root/'development_manifest.json').read_text(encoding='utf-8'))
    vault=json.loads((args.manifest_root/'scoring_vault'/'compound_manifest.json').read_text(encoding='utf-8'))
    file_map=find_files(args.data_root); args.output.mkdir(parents=True,exist_ok=True)
    dev_rows=make_rows(dev,file_map,False,args.feature_mode); score_rows=make_rows(vault,file_map,True,args.feature_mode)
    input_dim=len(dev_rows[0]['x'])
    # Four stratified group folds. Each fold holds one physical group from each
    # available state whenever possible; all threshold decisions use only these
    # cross-fitted healthy/single predictions.
    by_state=defaultdict(list)
    for r in dev: by_state[r['state']].append(r['physical_group'])
    for s in by_state: by_state[s]=sorted(set(by_state[s]))
    folds=[set() for _ in range(4)]
    for s, groups in by_state.items():
        for i,g in enumerate(groups): folds[i%4].add(g)
    results={}
    for kind, ctor in [('direct',lambda:DirectMLP(input_dim)),('dictionary',lambda:DictionaryModel(input_dim))]:
        seed_results=[]; cross_rows=[]
        for seed in seed_list:
            seed_all(seed)
            # Cross-fitted validation predictions for threshold selection.
            cross=[]
            for fold in folds:
                tr=[r for r in dev_rows if r['group'] not in fold]; va=[r for r in dev_rows if r['group'] in fold]
                m=ctor(); train(m,tr,union_weight=args.union_weight if kind == 'dictionary' else 0.0); cross.extend(aggregate(va,predict(m,va)))
            # Choose a single threshold on file-level cross-fitted predictions.
            best=(0.0,0.5)
            for th in np.linspace(0.10,0.90,81):
                mm=metrics(cross,float(th)); health=[i for i in cross if np.sum(i['y'])==0]
                hfp=float(np.mean([np.any(np.asarray(i['p'])>=th) for i in health])) if health else 1.0
                score=mm['macro_f1'] if hfp<=0.20 else mm['macro_f1']-2.0*(hfp-0.20)
                if score>best[0]: best=(score,float(th))
            threshold=best[1]
            threshold_metrics=metrics(cross,threshold)
            health=[i for i in cross if np.sum(i['y'])==0]
            threshold_metrics['health_false_alarm']=float(np.mean([np.any(np.asarray(i['p'])>=threshold) for i in health])) if health else float('nan')
            seed_all(seed); m=ctor(); train(m,dev_rows,union_weight=args.union_weight if kind == 'dictionary' else 0.0); pred=aggregate(score_rows,predict(m,score_rows)); mm=metrics(pred,threshold); mm['seed']=seed; mm['threshold_selection']=threshold_metrics; mm['predictions']=pred; seed_results.append(mm)
        results[kind]={'seeds':seed_results,'mean_macro_f1':float(np.mean([r['macro_f1'] for r in seed_results])),'mean_jaccard':float(np.mean([r['mean_set_jaccard'] for r in seed_results])),'mean_exact_set_match':float(np.mean([r['exact_set_match'] for r in seed_results]))}
    (args.output/'hust_zero_shot_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    meta={'dataset':'HUST Hanoi Bearing v2','development_files':len({r['file'] for r in dev_rows}),'scoring_files':len({r['file'] for r in score_rows}),'window_seconds':1.0,'feature_mode':args.feature_mode,'feature_dimension':input_dim,'seeds':seed_list,'union_weight':args.union_weight,'union_pairs_training_only':True,'compound_labels_used_for_training':False,'label_firewall':True}
    (args.output/'run_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:{q:v for q,v in val.items() if q.startswith('mean_')} for k,val in results.items()},ensure_ascii=False))


if __name__=='__main__': main()
