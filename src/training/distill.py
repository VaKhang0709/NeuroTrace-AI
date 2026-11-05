"""Frozen-PRO binary knowledge distillation. Never retrain/step the teacher.

Binary KL uses TWO outcomes [not-PD,PD], not softmax over one scalar (zero KL).
Original teacher train/validation parents are reused; no new partition or split.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
from src.models.tremorformer_family import TremorFormer,CONTRACT
from src.models.v6_tremorformer import TremorFormer as Pro
from src.training.family_data import load_context,windows,sha256
from src.training.v6_train import seed_all
from src.evaluation.v6_eval import aggregate,metrics


def distillation_loss(student_logits,teacher_logits,labels,temperature=3.,alpha=.5):
    if temperature<=0 or not 0<=alpha<=1: raise ValueError('Need positive T and alpha in [0,1]')
    if student_logits.shape!=teacher_logits.shape or labels.shape!=student_logits.shape:
        raise ValueError('KD needs matching binary logit/label vectors')
    if not torch.all((labels==0)|(labels==1)): raise ValueError('KD clinical labels must be known')
    s=student_logits.float()/temperature; t=teacher_logits.detach().float()/temperature
    logp=torch.stack([F.logsigmoid(-s),F.logsigmoid(s)],dim=-1)
    q=torch.stack([torch.sigmoid(-t),torch.sigmoid(t)],dim=-1)
    kd=F.kl_div(logp,q,reduction='batchmean')*temperature**2
    bce=F.binary_cross_entropy_with_logits(student_logits.float(),labels.float())
    return alpha*kd+(1-alpha)*bce


class FrozenONNXTeacher(nn.Module):
    """Exact production graph, no trainable parameters or gradients. Cache once.

    Uses CPU ORT so a separate GPU runtime package/provider is not required.
    Caching these targets avoids 80 epochs of repeated 15.78M teacher inference.
    """
    def __init__(self,path):
        super().__init__()
        from src.export.export_family import session,check_signature
        self.session=session(path); check_signature(self.session); self.eval()
    def train(self,mode=True):
        super().train(False); return self
    @torch.no_grad()
    def forward(self,x,device_id):
        values=self.session.run(['logits'],dict(trajectory=x.detach().cpu().numpy().astype(np.float32),
                    device_id=device_id.detach().cpu().numpy().astype(np.int64)))[0]
        return torch.from_numpy(values).to(x.device)


def teacher_targets(teacher,data,batch_size=8):
    teacher.eval(); result=[]
    with torch.no_grad():
        for start in range(0,len(data['X']),batch_size):
            x=torch.from_numpy(data['X'][start:start+batch_size]); d=torch.from_numpy(data['device_id'][start:start+batch_size])
            result.extend(teacher(x,d).cpu().tolist())
    values=np.asarray(result,dtype=np.float32)
    if not np.isfinite(values).all(): raise ValueError('Nonfinite frozen teacher targets')
    return values


def verify_torch_teacher(metadata,teacher,data):
    if 'model_state_dict' not in metadata: return None # hash-bound ONNX-only metadata; export still needs actual .pth
    if metadata['model_config']!=Pro().config: raise ValueError('PRO architecture differs from original teacher')
    model=Pro(**metadata['model_config']).eval(); model.load_state_dict(metadata['model_state_dict'],strict=True)
    for p in model.parameters(): p.requires_grad_(False)
    maximum=0.
    with torch.no_grad():
        for start in range(0,min(16,len(data['X'])),4):
            x=torch.from_numpy(data['X'][start:start+4]); d=torch.from_numpy(data['device_id'][start:start+4])
            maximum=max(maximum,float((model(x,d)['pd_logits']-teacher(x,d)).abs().max()))
    if maximum>=1e-5: raise ValueError(f'Teacher checkpoint differs from production ONNX: {maximum}')
    return maximum


def _loader(data,soft=None,batch_size=8,shuffle=False):
    values=[torch.from_numpy(data[k]) for k in ('X','device_id','label')]
    if soft is not None: values.append(torch.from_numpy(soft))
    return DataLoader(TensorDataset(*values),batch_size=batch_size,shuffle=shuffle,num_workers=0)


def _sync(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize()


def profile(model,train_loader,val_loader,device):
    """Development-only timing, no optimizer update, never inspect holdout."""
    model.to(device); model.train(); costs=[]
    for i,(x,d,y,soft) in enumerate(train_loader):
        x,d,y,soft=[v.to(device) for v in (x,d,y,soft)]; _sync(device); start=time.perf_counter()
        out=model(x,d); loss=distillation_loss(out['pd_logits'],soft,y)+.1*F.cross_entropy(out['device_logits'],d)
        loss.backward(); model.zero_grad(set_to_none=True); _sync(device)
        if i: costs.append(time.perf_counter()-start)
        if i>=3: break
    model.eval(); x,d,_=next(iter(val_loader)); x=x.to(device); d=d.to(device)
    _sync(device); start=time.perf_counter()
    with torch.no_grad(): model(x,d)
    _sync(device); val_cost=time.perf_counter()-start
    return max(costs or [val_cost])*len(train_loader)+val_cost*len(val_loader)


@torch.no_grad()
def validation(model,loader,parents,device):
    model.eval(); probs=[]; y=[]
    for x,d,label in loader:
        probs.extend(torch.sigmoid(model(x.to(device),d.to(device))['pd_logits']).cpu().tolist()); y.extend(label.tolist())
    _,yy,pp=aggregate(y,probs,parents)
    return metrics(yy,pp)


def run(args):
    start=time.monotonic(); seed_all(args.seed)
    if args.temperature<=0 or not 0<=args.alpha<=1: raise ValueError('Invalid KD temperature/alpha')
    if args.epochs<1 or args.batch_size<1 or args.device_weight<0: raise ValueError('Invalid training settings')
    records,part,meta,context=load_context(args.records,args.partition,args.scaler,args.teacher_metadata,args.teacher_onnx)
    context['records_sha256']=sha256(args.records)
    if args.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no claimed T4 run')
    device=args.device
    train=windows(records,meta['training_subjects']); valid=windows(records,meta['validation_subjects'])
    print('Original locked partition preserved; development-only train/validation windows:',len(train['X']),len(valid['X']),flush=True)
    teacher=FrozenONNXTeacher(args.teacher_onnx)
    print('Teacher checkpoint preflight max error:',verify_torch_teacher(meta,teacher,valid),flush=True)
    soft=teacher_targets(teacher,train,args.batch_size) # only original TRAIN parents, never holdout.
    train_loader=_loader(train,soft,args.batch_size,True); val_loader=_loader(valid,batch_size=args.batch_size)
    root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    settings=dict(epochs=args.epochs,batch_size=args.batch_size,temperature=args.temperature,alpha=args.alpha,
                  device_weight=args.device_weight,seed=args.seed,USE_SYNTHETIC=False,feature_contract=CONTRACT)
    timing={}
    for tier in ('nano','base'):
        probe=TremorFormer(tier); timing[tier]=profile(probe,train_loader,val_loader,device); del probe
    projection=sum(timing.values())*args.epochs*1.3+args.reserve_seconds+time.monotonic()-start
    print('Measured development-only seconds/epoch:',timing,'projected total seconds with 30% margin:',projection,flush=True)
    status=dict(settings=settings,context=context,profile_seconds_per_epoch=timing,projected_seconds=projection,complete=False)
    (root/'run_status.json').write_text(json.dumps(status,indent=2))
    if args.max_seconds and projection>args.max_seconds:
        raise RuntimeError('Projected run exceeds remaining time/CU budget; no reduced epoch count, data cap, or fabricated completion')
    for tier in ('nano','base'):
        seed_all(args.seed); model=TremorFormer(tier).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.05)
        amp=device.startswith('cuda'); scaler=torch.amp.GradScaler('cuda',enabled=amp)
        best=-math.inf; best_ck=None; durations=[]
        with (root/f'{tier}_epochs.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['epoch','loss','val_subject_auc','val_pr_auc','seconds']); writer.writeheader()
            for epoch in range(args.epochs):
                if args.max_seconds and time.monotonic()-start+max(durations or [timing[tier]])+args.reserve_seconds>args.max_seconds:
                    raise RuntimeError('Time/CU budget reached; incomplete students NOT accepted/exported')
                epoch_start=time.monotonic(); model.train(); losses=[]
                for x,d,y,soft_batch in train_loader:
                    if args.max_seconds and time.monotonic()-start+args.reserve_seconds>args.max_seconds:
                        raise RuntimeError('Time/CU guard stopped an incomplete epoch; no accepted artifacts')
                    x,d,y,soft_batch=[v.to(device) for v in (x,d,y,soft_batch)]
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type='cuda' if amp else 'cpu',enabled=amp):
                        out=model(x,d)
                        loss=distillation_loss(out['pd_logits'],soft_batch,y,args.temperature,args.alpha)
                        loss=loss+args.device_weight*F.cross_entropy(out['device_logits'],d) # encoder GRL retained.
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite KD loss')
                    scaler.scale(loss).backward(); scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(),1.)
                    scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
                vm=validation(model,val_loader,valid['subject'],device); auc=vm['roc_auc']
                if auc is None or not np.isfinite(auc): raise ValueError('Invalid subject-validation AUC')
                durations.append(time.monotonic()-epoch_start)
                row=dict(epoch=epoch+1,loss=float(np.mean(losses)),val_subject_auc=auc,val_pr_auc=vm['pr_auc'],seconds=durations[-1])
                writer.writerow(row); f.flush(); print(tier,json.dumps(row),flush=True)
                if auc>best:
                    best=auc
                    best_ck=dict(model_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        model_config=model.config,family_tier=tier,feature_contract=CONTRACT,context=context,
                        training_subjects=sorted(meta['training_subjects']),validation_subjects=sorted(meta['validation_subjects']),
                        ssl_subjects=[],selected_epoch=epoch+1,val_auc=auc,settings=settings,
                        epochs_completed=epoch+1,epochs_requested=args.epochs,complete=False)
                torch.save(dict(best_ck,epochs_completed=epoch+1),root/f'{tier}_best.pth')
            best_ck.update(epochs_completed=args.epochs,complete=True)
            torch.save(best_ck,root/f'{tier}_best.pth')
        del model,optimizer
        if amp: torch.cuda.empty_cache()
    status.update(complete=True,elapsed_seconds=time.monotonic()-start)
    (root/'run_status.json').write_text(json.dumps(status,indent=2)); return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('records','partition','scaler','teacher-metadata','teacher-onnx'): p.add_argument('--'+name,required=True)
    p.add_argument('--output',default='models/v6/family'); p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--batch-size',type=int,default=8); p.add_argument('--temperature',type=float,default=3.)
    p.add_argument('--alpha',type=float,default=.5); p.add_argument('--device-weight',type=float,default=.1)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--device',default='cuda')
    p.add_argument('--max-seconds',type=float,default=1800); p.add_argument('--reserve-seconds',type=float,default=240)
    print(json.dumps(run(p.parse_args()),indent=2))

if __name__=='__main__': main()
