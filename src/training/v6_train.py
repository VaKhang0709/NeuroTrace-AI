"""Parent-disjoint 5-fold V6 training. No automatic test selection or label inference.

Input: list of versioned unified trajectories and a private JSON partition with
`development_subjects`, `holdout_subjects`, optional `unlabeled_subjects`.
Holdout is never loaded into an optimizer or SSL loader. All outputs are private.
"""
import argparse
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from src.data_loaders.schema import Trajectory,DOMAINS,TASKS
from src.preprocessing.v6_pipeline import transform,BASE_CONTRACT
from src.augmentation.v6_augmentations import Augmenter,mix_batch,DEFAULT_PROBABILITIES
from src.models.v6_tremorformer import TremorFormer
from src.pretraining.v6_mae_pretrain import pretrain


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)


def validate_partition(records,partition):
    dev=set(partition['development_subjects']); test=set(partition['holdout_subjects'])
    unlabeled=set(partition.get('unlabeled_subjects',[]))
    if dev & test or unlabeled & (dev|test): raise ValueError('Partition parent overlap')
    known={t.subject_id for t in records}
    if not dev or not test or not (dev|test|unlabeled)<=known: raise ValueError('Invalid/missing partition parents')
    labels={}
    for tr in records:
        if tr.synthetic: raise ValueError('Source records must be real; augment synthetic only inside train fold')
        if tr.label>=0:
            if tr.subject_id in labels and labels[tr.subject_id]!=tr.label: raise ValueError('Conflicting subject diagnosis')
            labels[tr.subject_id]=tr.label
    if not dev<=set(labels) or not test<=set(labels): raise ValueError('All clinical partition parents need diagnosis')
    if unlabeled & set(labels): raise ValueError('Unlabeled partition has labeled parents')
    return dev,test,unlabeled

class TrajectoryDataset(Dataset):
    def __init__(self,records,allowed_subjects,augmenter=None,seed=42,contract=BASE_CONTRACT):
        self.records=list(records); self.subjects=sorted({t.subject_id for t in records})
        if not set(self.subjects)<=set(allowed_subjects): raise ValueError('Dataset has forbidden parent')
        self.contract=contract; self.augmenter=augmenter; self.rng=np.random.default_rng(seed)
        self.cached=[]; self.index=[]
        # Fail loudly on short/bad trials rather than silently changing clinical cohort.
        for i,tr in enumerate(records):
            windows,_=transform(tr,contract=self.contract); self.cached.append(windows)
            self.index.extend((i,j) for j in range(len(windows)))
        if not self.index: raise ValueError('Empty dataset')
    def __len__(self): return len(self.index)
    def __getitem__(self,index):
        i,j=self.index[index]; tr=self.records[i]; x=self.cached[i][j].copy()
        if self.augmenter:
            tr,_=self.augmenter(tr)
            windows,_=transform(tr,contract=self.contract); x=windows[min(j,len(windows)-1)].copy()
            # Explicit feature dropout, not merely native-pressure -> proxy fallback.
            if self.rng.random()<self.augmenter.p['pressure_dropout']:
                x[[2,11,17] if self.contract==BASE_CONTRACT else [2,11,18]]=0
        task=TASKS.index(tr.task_id) if tr.task_id in TASKS else -1
        return dict(x=torch.from_numpy(x),label=torch.tensor(tr.label,dtype=torch.float32),
                    domain=torch.tensor(DOMAINS.index(tr.domain)),task=torch.tensor(task),
                    age=torch.tensor(tr.age_bucket),updrs=torch.tensor(tr.updrs,dtype=torch.float32),
                    synthetic=torch.tensor(tr.synthetic),subject=tr.subject_id)

class MultiTaskLoss(nn.Module):
    """Uncertainty-weighted active tasks only. +CE(device) WITH GRL, never -CE."""
    def __init__(self):
        super().__init__(); self.log_variances=nn.Parameter(torch.zeros(5))
    def forward(self,out,batch):
        losses=[]
        clinical=batch['label']>=0
        losses.append(F.binary_cross_entropy_with_logits(out['pd_logits'][clinical],batch['label'][clinical]) if clinical.any() else None)
        mask=torch.isfinite(batch['updrs']) & ~batch['synthetic']
        # Normalize UPDRS Part III to 0..1 loss scale; exported head is multiplied by 132.
        losses.append(F.mse_loss(out['updrs'][mask],batch['updrs'][mask]/132.) if mask.any() else None)
        for name,key in [('task_logits','task'),('age_logits','age')]:
            mask=(batch[key]>=0) & ~batch['synthetic']
            losses.append(F.cross_entropy(out[name][mask],batch[key][mask]) if mask.any() else None)
        mask=~batch['synthetic']
        losses.append(F.cross_entropy(out['device_logits'][mask],batch['domain'][mask]) if mask.any() else None)
        total=out['pd_logits'].sum()*0
        for i,loss in enumerate(losses):
            if loss is not None:
                logvar=self.log_variances[i].clamp(-5,5)
                total=total+torch.exp(-logvar)*loss+logvar
        return total

@torch.no_grad()
def predict(model,loader,device):
    model.eval(); probabilities=[]; labels=[]; parents=[]; domains=[]; embeddings=[]
    for batch in loader:
        out=model(batch['x'].to(device),batch['domain'].to(device))
        probabilities.extend(torch.sigmoid(out['pd_logits']).cpu().tolist())
        labels.extend(batch['label'].tolist()); parents.extend(batch['subject']); domains.extend(batch['domain'].tolist())
        embeddings.extend(out['features'].cpu().tolist())
    return dict(probability=np.array(probabilities),label=np.array(labels),subject=np.array(parents),
                domain=np.array(domains),features=np.array(embeddings))

def subject_auc(pred):
    ids=sorted(set(pred['subject'])); y=[]; p=[]
    for sid in ids:
        mask=pred['subject']==sid; labels=np.unique(pred['label'][mask])
        if len(labels)!=1: raise ValueError('Conflicting parent labels')
        y.append(labels[0]); p.append(pred['probability'][mask].mean())
    return float(roc_auc_score(y,p)) if len(set(y))==2 else float('nan')


def train_fold(model,train_loader,val_loader,output,epochs=200,patience=20,
               effective_batch=256,device='cpu',seed=42,probabilities=None):
    from torch.utils.tensorboard import SummaryWriter
    if set(train_loader.dataset.subjects)&set(val_loader.dataset.subjects): raise ValueError('Parent leakage')
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    model.to(device); criterion=MultiTaskLoss().to(device)
    opt=torch.optim.AdamW(list(model.parameters())+list(criterion.parameters()),lr=3e-4,weight_decay=.05)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=25,T_mult=2)
    amp=str(device).startswith('cuda'); scaler=torch.amp.GradScaler('cuda',enabled=amp)
    rng=np.random.default_rng(seed); accumulation=max(1,int(np.ceil(effective_batch/train_loader.batch_size)))
    best=-float('inf'); stale=0; top=[]; writer=SummaryWriter(str(output/'tensorboard'))
    with (output/'epochs.csv').open('w',newline='') as f:
        log=csv.DictWriter(f,fieldnames=['epoch','loss','val_subject_auc','val_pr_auc','val_sensitivity','val_specificity','val_ece','lr','synthetic_windows']); log.writeheader()
        for epoch in range(epochs):
            if epoch<5:
                for group in opt.param_groups: group['lr']=3e-4*(epoch+1)/5
            else: scheduler.step(epoch-5)
            model.train(); criterion.train(); opt.zero_grad(set_to_none=True); losses=[]; synthetic_windows=0
            for step,batch in enumerate(train_loader):
                batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
                batch=mix_batch(batch,rng,probabilities)
                synthetic_windows+=int(batch['synthetic'].sum())
                group_start=(step//accumulation)*accumulation
                group_size=min(accumulation,len(train_loader)-group_start)
                with torch.autocast(device_type='cuda' if amp else 'cpu',enabled=amp):
                    loss=criterion(model(batch['x'],batch['domain']),batch)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                scaler.scale(loss/group_size).backward(); losses.append(float(loss.detach()))
                if (step+1)%accumulation==0 or step+1==len(train_loader):
                    scaler.unscale_(opt); nn.utils.clip_grad_norm_(list(model.parameters())+list(criterion.parameters()),1.)
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            from src.evaluation.v6_eval import aggregate,metrics
            validation_prediction=predict(model,val_loader,device)
            _,vy,vp=aggregate(validation_prediction['label'],validation_prediction['probability'],validation_prediction['subject'])
            vm=metrics(vy,vp); val=vm['roc_auc']
            if val is None or not np.isfinite(val): raise ValueError('Validation fold lacks both diagnoses; cannot select on AUC')
            checkpoint=dict(model_state_dict=model.state_dict(),model_config=model.config,feature_contract=getattr(model,'feature_contract',BASE_CONTRACT),
                 loss_state=criterion.state_dict(),optimizer_state=opt.state_dict(),scheduler_state=scheduler.state_dict(),
                 scaler_state=scaler.state_dict(),epoch=epoch,val_auc=val,seed=seed,
                 training_subjects=train_loader.dataset.subjects,
                 synthetic_parent_subjects=getattr(train_loader.dataset,'provenance_subjects',[]),validation_subjects=val_loader.dataset.subjects,
                 ssl_subjects=getattr(model,'ssl_subjects',[]),
                 rng_state=dict(torch=torch.get_rng_state(),python=random.getstate()),
                 augmentation_probabilities=dict(probabilities or DEFAULT_PROBABILITIES))
            torch.save(checkpoint,output/'last.pth')
            path=output/f'epoch_{epoch:03d}.pth'; torch.save(checkpoint,path); top.append((val,path))
            top.sort(key=lambda item:item[0],reverse=True)
            for _,removed in top[3:]: removed.unlink(missing_ok=True)
            top=top[:3]
            row=dict(epoch=epoch,loss=float(np.mean(losses)),val_subject_auc=val,val_pr_auc=vm['pr_auc'],val_sensitivity=vm['sensitivity'],
                     val_specificity=vm['specificity'],val_ece=vm['ece'],lr=opt.param_groups[0]['lr'],synthetic_windows=synthetic_windows)
            log.writerow(row); f.flush()
            for key in ('loss','val_subject_auc','val_pr_auc','val_sensitivity','val_specificity','val_ece','lr'): writer.add_scalar(key,row[key],epoch)
            if val>best+1e-5: best=val; stale=0
            else: stale+=1
            if stale>=patience: break
    writer.close()
    return top[0][1]


def run(records,partition,output,epochs=200,ssl_epochs=100,batch_size=4,seed=42,
        device=None,model_config=None,synthetic_injection=False,augmentation_probabilities=None,feature_contract=BASE_CONTRACT,expanded_manifest=None,synthetic_fraction=.25):
    seed_all(seed); dev,holdout,unlabeled=validate_partition(records,partition)
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    clinical=[tr for tr in records if tr.subject_id in dev and tr.label>=0]
    subjects=sorted({tr.subject_id for tr in clinical})
    if len(subjects)<5: raise ValueError('Need >=5 development subjects; each validation fold needs both labels')
    folds=GroupKFold(n_splits=5); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for fold,(train_i,val_i) in enumerate(folds.split(subjects,groups=subjects)):
        train={subjects[i] for i in train_i}; valid={subjects[i] for i in val_i}
        assert not train&valid and not (train|valid|unlabeled)&holdout
        seed_all(seed+fold)
        folder=output/f'fold_{fold}'; folder.mkdir(parents=True,exist_ok=True)
        model=TremorFormer(**(model_config or {})); model.feature_contract=feature_contract
        if ssl_epochs:
            ssl_records=[tr for tr in records if tr.subject_id in train|unlabeled]
            ds=TrajectoryDataset(ssl_records,train|unlabeled,contract=feature_contract)
            pretrain(model,DataLoader(ds,batch_size=batch_size,shuffle=True),train|unlabeled,
                     folder/'v6_pretrained_backbone.pth',epochs=ssl_epochs,device=device,seed=seed+fold)
        model.ssl_subjects=sorted(train|unlabeled) if ssl_epochs else []
        train_records=[tr for tr in clinical if tr.subject_id in train]
        probabilities=dict(DEFAULT_PROBABILITIES)
        probabilities.update(augmentation_probabilities or {})
        probabilities['tremor_injection']=probabilities.get('tremor_injection',.1) if synthetic_injection else 0.
        if synthetic_injection and probabilities['tremor_injection']==0: probabilities['tremor_injection']=.1
        donors=[tr for tr in train_records if tr.label==1 and tr.domain=='wacom'] if synthetic_injection else []
        aug=Augmenter(train,donors,probabilities,seed+fold)
        train_ds=TrajectoryDataset(train_records,train,aug,seed+fold,contract=feature_contract)
        val_ds=TrajectoryDataset([tr for tr in clinical if tr.subject_id in valid],valid,contract=feature_contract)
        sampler=None
        if expanded_manifest:
            from src.data_loaders.synthetic_manifest_loader import SyntheticManifestDataset,CombinedTrainingDataset
            synthetic_ds=SyntheticManifestDataset(expanded_manifest,train,contract=feature_contract)
            assert set(synthetic_ds.provenance_subjects)<=train
            if not 0<synthetic_fraction<=.5: raise ValueError('Synthetic fraction must be in (0,.5]')
            if len(synthetic_ds):
                from torch.utils.data import WeightedRandomSampler
                nreal=len(train_ds); nsynth=len(synthetic_ds)
                weights=np.r_[np.full(nreal,(1-synthetic_fraction)/nreal),np.full(nsynth,synthetic_fraction/nsynth)]
                train_ds=CombinedTrainingDataset(train_ds,synthetic_ds)
                sampler=WeightedRandomSampler(torch.as_tensor(weights),num_samples=int(np.ceil(nreal/(1-synthetic_fraction))),replacement=True)
        best=train_fold(model,DataLoader(train_ds,batch_size=batch_size,shuffle=sampler is None,sampler=sampler,num_workers=0),
              DataLoader(val_ds,batch_size=batch_size,num_workers=0),folder,epochs=epochs,
              device=device,seed=seed+fold,probabilities=probabilities)
        import shutil
        shutil.copyfile(best,folder.parent/f'fold_{fold}_best.pth')
        manifest.append(dict(fold=fold,best_checkpoint=str(best),train_subjects=sorted(train),
                             validation_subjects=sorted(valid),holdout_subjects=sorted(holdout)))
        (output/'fold_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',required=True,help='Private JSON list of unified trajectories')
    p.add_argument('--partition',required=True); p.add_argument('--output',default='models/v6_runs')
    p.add_argument('--epochs',type=int,default=200); p.add_argument('--ssl-epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=4); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--synthetic-injection',action='store_true')
    p.add_argument('--config',help='Optional JSON model and augmentation overrides')
    a=p.parse_args(); config=json.loads(Path(a.config).read_text()) if a.config else {}
    records=[Trajectory.from_dict(v) for v in json.loads(Path(a.records).read_text())]
    run(records,json.loads(Path(a.partition).read_text()),a.output,a.epochs,a.ssl_epochs,a.batch_size,a.seed,
        synthetic_injection=a.synthetic_injection,model_config=config.get('model'),
        augmentation_probabilities=config.get('augmentation_probabilities'))

if __name__=='__main__': main()
