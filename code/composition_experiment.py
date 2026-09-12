"""Executable condition-residual composition model for the HUST protocol.

The implementation is intentionally separate from the legacy exploratory
scripts.  It uses only development rows for optimization and creates union
pairs from singleton development rows.  Compound rows are loaded only by the
post-fit scoring path.
"""
from __future__ import annotations
import argparse, json, math, random, sys
from collections import defaultdict
from pathlib import Path
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

LEGACY = Path(os.environ.get('LEGACY_ROOT', '.'))
if str(LEGACY) not in sys.path: sys.path.insert(0, str(LEGACY))
from run_hust_zero_shot_models import find_files, make_rows, metrics, aggregate, predict, features_for_file

def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

class CompositionModel(nn.Module):
    def __init__(self, input_dim: int, decoder_steps: int = 5, sparsity: float = .001, use_reference: bool = True):
        super().__init__()
        self.decoder_steps = int(decoder_steps); self.sparsity = float(sparsity); self.use_reference = bool(use_reference)
        self.encoder = nn.Sequential(nn.Linear(input_dim + 3, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU())
        self.anchor = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, 64))
        self.dictionary = nn.Parameter(torch.randn(64, 3) * .05)
        self.rho = nn.Parameter(torch.tensor(0.0))
        self.gamma = nn.Parameter(torch.ones(3))
        self.bias = nn.Parameter(torch.zeros(3))

    def normalized_dictionary(self):
        return self.dictionary / (torch.linalg.vector_norm(self.dictionary, dim=0, keepdim=True) + 1e-6)

    def decode_residual(self, residual, steps=None):
        p = self.normalized_dictionary()
        nstep = self.decoder_steps if steps is None else int(steps)
        # Spectral norm is recomputed during training; frozen inference can cache it.
        eta = torch.sigmoid(self.rho) / (torch.linalg.matrix_norm(p, ord=2) ** 2 + 1e-6)
        a = torch.zeros((residual.shape[0], 3), dtype=residual.dtype, device=residual.device)
        for _ in range(nstep):
            a = F.relu(a + eta * (residual @ p - a @ (p.T @ p)) - eta * self.sparsity)
        logits = a * F.softplus(self.gamma) + self.bias
        return logits, a, p, eta

    def forward(self, x, c, return_state=False, steps=None):
        z = self.encoder(torch.cat([x, c], dim=1)); r = z - self.anchor(c) if self.use_reference else z
        logits, coeff, p, eta = self.decode_residual(r, steps=steps)
        if return_state: return logits, {'z': z, 'r': r, 'a': coeff, 'p': p, 'eta': eta}
        return logits

def _rows_to_tensors(rows):
    return (torch.tensor(np.stack([r['x'] for r in rows]), dtype=torch.float32),
            torch.tensor(np.stack([r['c'] for r in rows]), dtype=torch.float32),
            torch.tensor(np.stack([r['y'] for r in rows]), dtype=torch.float32))

def _grouped_alignment(state, rows):
    """Return index groups keyed by state/load, equalizing records before cells."""
    cells = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        cells[(r['state'], int(np.argmax(r['c'])))][r['group']].append(i)
    return cells

def alignment_loss(residual, rows):
    cells = _grouped_alignment(None, rows); by_state = defaultdict(list)
    for (state, load), groups in cells.items():
        if len(groups) < 1: continue
        group_means = [residual[idx].mean(0) for idx in groups.values()]
        cell_mean = torch.stack(group_means).mean(0)
        by_state[state].append(cell_mean)
    terms=[]
    for vals in by_state.values():
        if len(vals) >= 2:
            state_mean=torch.stack(vals).mean(0)
            terms.extend([((v-state_mean)**2).mean() for v in vals])
    return torch.stack(terms).mean() if terms else residual.new_zeros(())

def coherence_loss(p):
    gram = p.T @ p
    off = gram - torch.diag(torch.diag(gram))
    return (off**2).sum() / 6.0

def union_pairs(rows, rng, max_pairs=9):
    """Sample one group-pair uniformly, then one window per selected group."""
    groups=defaultdict(lambda: defaultdict(list))
    for r in rows:
        active=np.flatnonzero(np.asarray(r['y'])>.5)
        if len(active)==1 and r['state'] in {'I','O','B'}:
            load=int(np.argmax(r['c'])); groups[load][int(active[0])].append(r)
    out=[]
    for load in sorted(groups):
        labels=sorted(groups[load])
        for ia,a in enumerate(labels):
            for b in labels[ia+1:]:
                ga={r['group'] for r in groups[load][a]}; gb={r['group'] for r in groups[load][b]}
                group_pairs=[(ga0,gb0) for ga0 in sorted(ga) for gb0 in sorted(gb) if ga0!=gb0]
                if not group_pairs: continue
                ga0,gb0=group_pairs[int(rng.integers(len(group_pairs)))]
                ra=rng.choice([r for r in groups[load][a] if r['group']==ga0])
                rb=rng.choice([r for r in groups[load][b] if r['group']==gb0])
                y=np.maximum(ra['y'],rb['y']).astype(np.float32)
                out.append({'x':(.5*(ra['x']+rb['x'])).astype(np.float32),'c':ra['c'].copy(),'y':y,
                            'file':f'union::{ra["file"]}+{rb["file"]}','group':f'union::{ra["group"]}+{rb["group"]}',
                            'state':'synthetic_union','_sources':(ra,rb)})
    return out[:max_pairs]

def loss_terms(model, rows, union, weights=None):
    weights=weights or {'recon':.1,'align':.01,'union':.1,'coh':.001}
    x,c,y=_rows_to_tensors(rows); logits,st=model(x,c,return_state=True)
    pos=y.sum(0); neg=y.shape[0]-pos; posw=neg/torch.clamp(pos,min=1.)
    supervised=F.binary_cross_entropy_with_logits(logits,y,pos_weight=posw)
    recon=((st['r']-st['a']@st['p'].T)**2).mean()
    align=alignment_loss(st['r'],rows)
    coh=coherence_loss(st['p'])
    union_cls=logit_add=torch.zeros((),dtype=logits.dtype)
    if union:
        # A synthetic union is defined in latent residual space.  We therefore
        # decode r_a+r_b directly, rather than averaging raw feature vectors.
        union_logits=[]; add_terms=[]; union_targets=[]
        for item in union:
            ra,rb=item['_sources']
            _,sa=model(*_rows_to_tensors([ra])[:2],return_state=True)
            _,sb=model(*_rows_to_tensors([rb])[:2],return_state=True)
            ul,ua,_,_=model.decode_residual(sa['r']+sb['r'])
            union_logits.append(ul[0]); union_targets.append(torch.tensor(item['y'],dtype=ul.dtype,device=ul.device))
            add_terms.append((ua-sa['a']-sb['a']).abs().mean())
        union_logits=torch.stack(union_logits); union_targets=torch.stack(union_targets)
        union_cls=F.binary_cross_entropy_with_logits(union_logits,union_targets)
        add= torch.stack(add_terms).mean() if add_terms else union_cls.new_zeros(())
        union_loss=union_cls+add
    else: union_loss=union_cls
    total=supervised+weights['recon']*recon+weights['align']*align+weights['union']*union_loss+weights['coh']*coh
    return total, {'supervised':supervised.detach(),'recon':recon.detach(),'align':align.detach(),'union':union_loss.detach(),'coh':coh.detach()}

def make_union_pairs(rows,rng,max_pairs=9):
    return union_pairs(rows,rng,max_pairs=max_pairs)

def train(model, rows, epochs=35, lr=.002, seed=11, weights=None):
    model.train(); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4); logs=[]
    for epoch in range(epochs):
        rng=np.random.default_rng(seed*100003+epoch)
        union=make_union_pairs(rows,rng)
        opt.zero_grad(set_to_none=True); total, parts=loss_terms(model,rows,union,weights); total.backward(); opt.step()
        logs.append({'epoch':epoch+1,'total':float(total.detach()),**{k:float(v) for k,v in parts.items()},'union_pairs':len(union)})
    return logs

def run_smoke(data_root,manifest_root,out,epochs=3):
    from run_hust_zero_shot_models import find_files,make_rows
    dev=json.loads((manifest_root/'development_manifest.json').read_text(encoding='utf-8'))
    fm=find_files(data_root); rows=make_rows(dev,fm,False,'envelope_order100')
    seed_all(11); model=CompositionModel(len(rows[0]['x']))
    logs=train(model,rows,epochs=epochs,seed=11); model.eval()
    x,c,y=_rows_to_tensors(rows); logits,st=model(x,c,return_state=True)
    out.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),out/'smoke_model.pt')
    (out/'smoke_report.json').write_text(json.dumps({'rows':len(rows),'input_dim':len(rows[0]['x']),'epochs':epochs,'loss_log':logs,'finite_logits':bool(torch.isfinite(logits).all()),'finite_coeff':bool(torch.isfinite(st['a']).all()),'decoder_steps':model.decoder_steps},indent=2),encoding='utf-8')
    print(json.dumps({'rows':len(rows),'last_loss':logs[-1]['total'],'finite_logits':bool(torch.isfinite(logits).all()),'finite_coeff':bool(torch.isfinite(st['a']).all())}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--manifest-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--epochs',type=int,default=3)
    a=ap.parse_args();run_smoke(a.data_root,a.manifest_root,a.output,a.epochs)
